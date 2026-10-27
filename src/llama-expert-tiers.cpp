#include "llama-expert-tiers.h"
#include "llama-model.h"
#include "llama-expert-remote.h"
#include "hash/hash.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "llama-impl.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <string>
#include <cstring>
#include <cstdint>
#include <limits>
#include <chrono>
#include <map>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

struct llama_tier_runtime {
    struct device_workspace {
        ggml_backend_t backend = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_threadpool_t threadpool = nullptr;
        ~device_workspace() {
            if (backend) { ggml_backend_synchronize(backend); }
            ggml_backend_buffer_free(buffer);
            ggml_backend_free(backend);
            ggml_threadpool_free(threadpool);
        }
    };
    struct host_staging {
        ggml_backend_buffer_t buffer = nullptr;
        ~host_staging() { ggml_backend_buffer_free(buffer); }
    };
    std::unique_ptr<llama_remote_peer> remote;
    std::vector<float> remote_reciprocal, remote_exponential;
    std::string calibration;
    std::map<const float *, std::string> calibration_ids;
    std::map<const ggml_tensor *, std::string> remote_identities;
    size_t bytes, staging_bytes;
    std::map<ggml_backend_dev_t, std::unique_ptr<device_workspace>> devices;
    std::map<ggml_backend_dev_t, std::unique_ptr<host_staging>> staging;
    llama_tier_runtime(size_t bytes, size_t staging_bytes) : bytes(bytes), staging_bytes(staging_bytes) {}
    void * stage(ggml_tensor * tensor) {
        if (ggml_backend_buffer_is_host(tensor->buffer)) { return nullptr; }
        auto * device = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
        auto & host = staging[device];
        if (!host) {
            auto * buft = ggml_backend_dev_host_buffer_type(device);
            if (!buft) { throw std::runtime_error("expert tiers: device has no pinned host staging"); }
            host = std::make_unique<host_staging>();
            host->buffer = ggml_backend_buft_alloc_buffer(buft, staging_bytes);
            if (!host->buffer) { throw std::bad_alloc(); }
        }
        return ggml_backend_buffer_get_base(host->buffer);
    }
    device_workspace & get(ggml_backend_dev_t device, int threads) {
        auto & workspace = devices[device];
        if (!workspace) {
            workspace = std::make_unique<device_workspace>();
            workspace->backend = ggml_backend_dev_init(device, nullptr);
            if (!workspace->backend) { throw std::runtime_error("expert tiers: cannot initialize backend"); }
            if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                const int n_threads = std::max(1, threads);
                auto params = ggml_threadpool_params_default(n_threads);
                workspace->threadpool = ggml_threadpool_new(&params);
                if (!workspace->threadpool) { throw std::runtime_error("expert tiers: cannot create CPU worker pool"); }
                ggml_backend_cpu_set_n_threads(workspace->backend, n_threads);
                ggml_backend_cpu_set_threadpool(workspace->backend, workspace->threadpool);
            }
            workspace->buffer = ggml_backend_alloc_buffer(workspace->backend, bytes);
            if (!workspace->buffer) { throw std::bad_alloc(); }
        }
        return *workspace;
    }
};

namespace {
constexpr int64_t pair_tile = 256;
std::atomic<uint64_t> capture_sequence{0};

struct bank_graph {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * tokens = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * activated = nullptr;
    std::array<ggml_cpu_tier_quant_policy, 3> quant_policy{};
    ggml_cpu_tier_swiglu_params swiglu{};
    int64_t pairs = 0;
    int64_t logical_tokens = 0;
    const ggml_tensor * source = nullptr;
    const void * source_data = nullptr;
    ggml_backend_buffer_t source_buffer = nullptr;
    ~bank_graph() {
        if (backend) { ggml_backend_synchronize(backend); }
        ggml_free(ctx);
    }
};
}

struct llama_tier_dispatch {
    const llama_expert_tiers & layer;
    float clamp;
    bool glm;
    int threads;
    ggml_backend_dev_t layer_device;
    bool instrument;
    bool cuda_arithmetic;
    bool remote_policy_active = false;
    std::array<uint32_t, 3> remote_policy{};
    int remote_warps = 1;
    const float * reciprocal_table = nullptr;
    const float * exponential_table = nullptr;
    std::string dump_directory;
    ggml_backend_sched_dispatch boundary;
    std::shared_ptr<llama_tier_runtime> runtime;
    std::map<int64_t, std::unique_ptr<bank_graph>> hot, cold;
    std::vector<int32_t> source_map;
    uint64_t hot_pairs = 0, cold_pairs = 0, cold_input_bytes = 0, cold_output_bytes = 0;
    uint64_t hot_jobs = 0, cold_jobs = 0, cold_skips = 0;
    double body_ms = 0, control_ms = 0;
    std::vector<uint8_t> scatter_coverage;

    llama_tier_dispatch(const llama_expert_tiers & layer, float clamp, bool glm, int threads, ggml_backend_dev_t layer_device, std::shared_ptr<llama_tier_runtime> runtime) : layer(layer), clamp(clamp), glm(glm), threads(threads), layer_device(layer_device), instrument(std::getenv("CNM_TIER_TRACE") != nullptr), cuda_arithmetic(std::getenv("CNM_TIER_CUDA_ARITH") != nullptr), runtime(std::move(runtime)) {
        if (const char * endpoint = std::getenv("CNM_TIER_REMOTE")) {
            if (cuda_arithmetic && (!layer_device || !std::getenv("CNM_TIER_CUDA_SFU"))) { throw std::runtime_error("expert remote: exact policy needs a CUDA router and SFU calibration"); }
            if (!this->runtime->remote) { this->runtime->remote = std::make_unique<llama_remote_peer>(endpoint); }
        }
        if (const char * path = std::getenv("CNM_TIER_DUMP")) { dump_directory = path; }
        if (cuda_arithmetic && std::getenv("CNM_TIER_CUDA_SFU") && layer_device) {
            for (const auto * bank : {&layer.hot, &layer.cold}) {
                if (!bank->down || !ggml_backend_buffer_is_host(bank->down->buffer)) { continue; }
                const auto * gate = bank->gate ? bank->gate : bank->gate_up;
                if (!ggml_is_quantized(bank->down->type) && (!gate || !ggml_is_quantized(gate->type))) { continue; }
                using table_t = const float * (*)(ggml_backend_dev_t);
                auto reg = ggml_backend_dev_backend_reg(layer_device);
                auto rcp = reinterpret_cast<table_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_tier_reciprocal"));
                auto ex2 = reinterpret_cast<table_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_tier_exponential"));
                reciprocal_table = rcp ? rcp(layer_device) : nullptr;
                exponential_table = ex2 ? ex2(layer_device) : nullptr;
                if (!reciprocal_table || !exponential_table) { throw std::runtime_error("expert tiers: SFU policy requires a qualified CUDA device"); }
                break;
            }
        }
        boundary = {execute, this};
        int32_t count = 0;
        for (const auto * bank : {&layer.hot, &layer.cold}) {
            for (int32_t id : bank->ids) { count = std::max(count, id + 1); }
        }
        source_map.assign(count, std::numeric_limits<int32_t>::min());
        for (size_t i = 0; i < layer.hot.ids.size(); ++i) { source_map[layer.hot.ids[i]] = int32_t(i); }
        for (size_t i = 0; i < layer.cold.ids.size(); ++i) { source_map[layer.cold.ids[i]] = -int32_t(i) - 1; }
    }

    ggml_tensor * projection(bank_graph & graph, ggml_tensor * weight, ggml_tensor * input, int index, bool cpu) {
        const bool supported = weight->type == GGML_TYPE_IQ3_S || weight->type == GGML_TYPE_IQ3_XXS ||
            weight->type == GGML_TYPE_IQ4_XS || weight->type == GGML_TYPE_IQ4_NL ||
            weight->type == GGML_TYPE_IQ2_S || weight->type == GGML_TYPE_IQ2_XS || weight->type == GGML_TYPE_Q2_0;
        if (cpu && (layer_device || remote_policy_active) && cuda_arithmetic && ggml_is_quantized(weight->type) && !supported) {
            throw std::runtime_error("expert tiers: unsupported CPU/CUDA compatibility weight type");
        }
        if (cpu && (layer_device || remote_policy_active) && supported && cuda_arithmetic) {
            if (remote_policy_active) { graph.quant_policy[index].kind = remote_policy[index]; }
            else {
                using query_t = int (*)(ggml_backend_dev_t, ggml_type, int64_t, int64_t);
                auto query = reinterpret_cast<query_t>(ggml_backend_reg_get_proc_address(
                        ggml_backend_dev_backend_reg(layer_device), "ggml_backend_cuda_tier_quant_policy"));
                if (!query) { throw std::runtime_error("expert tiers: CUDA arithmetic policy unavailable"); }
                graph.quant_policy[index].kind = query(layer_device, weight->type, graph.source->ne[1], layer.hot.ids.size() + layer.cold.ids.size());
            }
            if (!graph.quant_policy[index].kind) { throw std::runtime_error("expert tiers: unsupported CUDA arithmetic policy"); }
            graph.quant_policy[index].reciprocal = reciprocal_table;
            graph.quant_policy[index].mmvq_warps = remote_policy_active ? remote_warps : (reciprocal_table && graph.source->ne[1] == 1 ? 4 : 1);
            ggml_tensor * args[] = {weight, input, graph.ids};
            return ggml_custom_4d(graph.ctx, GGML_TYPE_F32, weight->ne[1], 1, graph.pairs, 1,
                    args, 3, ggml_cpu_tier_quant_matmul, GGML_N_TASKS_MAX, &graph.quant_policy[index]);
        }
        return ggml_mul_mat_id(graph.ctx, weight, input, graph.ids);
    }

    std::unique_ptr<bank_graph> prepare(const llama_expert_bank & bank, ggml_tensor * activations, int64_t n) {
        auto g = std::make_unique<bank_graph>();
        auto * dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(bank.down->buffer));
        if (ggml_backend_buffer_is_host(bank.down->buffer)) { dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU); }
        if (!dev) { throw std::runtime_error("expert tiers: weight backend has no device"); }
        const bool cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
        if (!cpu && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            throw std::runtime_error("expert tiers: only CPU and CUDA banks are qualified");
        }
        if (!cpu && std::string(ggml_backend_dev_name(dev)).find("CUDA") != 0) {
            throw std::runtime_error("expert tiers: non-CUDA GPU bank is unsupported");
        }
        if (!cpu && !dump_directory.empty() && !std::getenv("GGML_CUDA_DISABLE_FUSION")) {
            throw std::runtime_error("expert tiers: CUDA projection dumps require GGML_CUDA_DISABLE_FUSION=1");
        }
        for (auto * weight : {bank.down, bank.up, bank.gate, bank.gate_up}) {
            auto * weight_dev = weight ? ggml_backend_buft_get_device(ggml_backend_buffer_get_type(weight->buffer)) : nullptr;
            if (weight && ggml_backend_buffer_is_host(weight->buffer)) { weight_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU); }
            if (weight && weight_dev != dev) {
                throw std::runtime_error("expert tiers: bank projections must share a device");
            }
        }
        auto & workspace = runtime->get(dev, threads);
        g->backend = workspace.backend;
        g->ctx = ggml_init({ggml_tensor_overhead() * 96 + ggml_graph_overhead_custom(128, false), nullptr, true});
        if (!g->ctx) { throw std::bad_alloc(); }
        auto * ctx = g->ctx;
        g->pairs = n;
        g->logical_tokens = activations->ne[1];
        g->source = activations;
        g->source_data = activations->data;
        g->source_buffer = activations->buffer;
        g->ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, n);
        ggml_set_input(g->ids);
        ggml_tensor * input;
        // GPU gathers stay on the layer device. CPU inputs contain selected pairs only.
        if (cpu) {
            g->input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, activations->ne[0], 1, n);
            ggml_set_input(g->input);
            input = g->input;
        } else {
            g->tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
            ggml_set_input(g->tokens);
            // The outer prefix is already complete. Reference its allocation as a leaf.
            auto * external = ggml_dup_tensor(ctx, activations);
            external->data = activations->data;
            external->buffer = activations->buffer;
            input = ggml_reshape_3d(ctx, ggml_get_rows(ctx, external, g->tokens), activations->ne[0], 1, n);
        }
        ggml_tensor * gate, * up;
        if (bank.gate_up) {
            auto * merged = projection(*g, bank.gate_up, input, 0, cpu);
            int64_t ff = merged->ne[0] / 2;
            gate = ggml_view_3d(ctx, merged, ff, 1, n, merged->nb[1], merged->nb[2], 0);
            up = ggml_view_3d(ctx, merged, ff, 1, n, merged->nb[1], merged->nb[2], ff * merged->nb[0]);
        } else {
            up = projection(*g, bank.up, input, 1, cpu);
            gate = projection(*g, bank.gate, input, 0, cpu);
        }
        ggml_tensor * activated;
        if (bank.gate && clamp > 1e-6f) {
            activated = glm ? ggml_swiglu_clamp(ctx, gate, up, clamp) :
                ggml_mul(ctx, ggml_clamp(ctx, ggml_silu(ctx, gate), -INFINITY, clamp), ggml_clamp(ctx, up, -clamp, clamp));
        } else { activated = ggml_swiglu_split(ctx, gate, up); }
        if (g->quant_policy[0].reciprocal) {
            g->swiglu = {g->quant_policy[0].reciprocal, exponential_table, bank.gate ? clamp : 0.0f, glm};
            ggml_tensor * args[] = {gate, up};
            activated = ggml_custom_4d(ctx, GGML_TYPE_F32, gate->ne[0], 1, n, 1,
                args, 2, ggml_cpu_tier_swiglu, GGML_N_TASKS_MAX, &g->swiglu);
        }
        g->gate = gate; g->up = up; g->activated = activated;
        g->output = projection(*g, bank.down, activated, 2, cpu);
        ggml_set_output(g->output);
        g->graph = ggml_new_graph_custom(ctx, 128, false);
        ggml_build_forward_expand(g->graph, g->output);
        for (int i = 0; i < ggml_graph_n_nodes(g->graph); ++i) {
            auto * node = ggml_graph_node(g->graph, i);
            if (node->op == GGML_OP_MUL_MAT_ID) { ggml_mul_mat_id_set_batch_hint(node, activations->ne[1]); }
        }
        for (int i = 0; i < ggml_graph_n_nodes(g->graph); ++i) {
            auto * node = ggml_graph_node(g->graph, i);
            if (!ggml_backend_supports_op(g->backend, node)) { throw std::runtime_error("expert tiers: unsupported bank operation"); }
        }
        const size_t required = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, ggml_backend_buffer_get_type(workspace.buffer));
        if (!required || required > ggml_backend_buffer_get_size(workspace.buffer)) {
            throw std::runtime_error("expert tiers: prepared graph exceeds reserved tile workspace");
        }
        auto allocator = ggml_tallocr_new(workspace.buffer);
        for (auto * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
            if (tensor->view_src) { ggml_backend_view_init(tensor); }
            else if (!tensor->data && ggml_tallocr_alloc(&allocator, tensor) != GGML_STATUS_SUCCESS) { throw std::bad_alloc(); }
        }
        return g;
    }

    void dump_bank(const llama_expert_bank & bank, const bank_graph & g, const std::vector<int64_t> & pairs,
            const std::vector<int32_t> & local, size_t begin, int64_t k) {
        if (dump_directory.empty()) { return; }
        std::filesystem::create_directories(dump_directory);
        const std::string prefix = dump_directory + "/" + bank.down->name + ".t" + std::to_string(g.logical_tokens) + ".capture" + std::to_string(capture_sequence.fetch_add(1, std::memory_order_relaxed));
        std::ofstream mapping(prefix + ".pairs");
        for (int64_t i = 0; i < g.pairs; ++i) {
            const auto q = pairs[begin + i];
            mapping << q << " " << q / k << " " << q % k << " " << bank.ids.at(local[begin + i]) << " " << local[begin + i] << "\n";
        }
        std::ofstream policy(prefix + ".policy");
        policy << ggml_backend_name(g.backend) << " logical_tokens=" << g.logical_tokens << " packed_pairs=" << g.pairs << "\n";
        policy << "quant_policy=" << g.quant_policy[0].kind << "," << g.quant_policy[1].kind << "," << g.quant_policy[2].kind << " reciprocal=" << bool(g.quant_policy[0].reciprocal) << "\n";
        for (auto * weight : {bank.gate, bank.up, bank.gate_up, bank.down}) {
            if (weight) { policy << weight->name << " " << ggml_type_name(weight->type) << " " << weight->data << " " << weight->nb[2] << "\n"; }
        }
        for (const auto & item : std::vector<std::pair<const char *, ggml_tensor *>>{{"gate", g.gate}, {"up", g.up}, {"activated", g.activated}, {"down", g.output}}) {
            std::string path = prefix + "." + item.first;
            auto * tensor = item.second;
            std::ofstream shape(path + ".shape");
            shape << int(tensor->type);
            for (int i = 0; i < 4; ++i) { shape << " " << tensor->ne[i]; }
            for (int i = 0; i < 4; ++i) { shape << " " << tensor->nb[i]; }
            shape << "\n";
            std::vector<char> data(ggml_nbytes(tensor));
            ggml_backend_tensor_get(tensor, data.data(), 0, data.size());
            std::ofstream binary(path + ".bin", std::ios::binary);
            binary.write(data.data(), data.size());
            if (!shape || !binary || !mapping || !policy) { throw std::runtime_error("expert tiers: diagnostic write failed"); }
        }
    }

    void run_bank(const llama_expert_bank & bank, std::map<int64_t, std::unique_ptr<bank_graph>> & variants, const std::vector<int64_t> & original_pairs,
            const std::vector<int32_t> & original_local, ggml_tensor * dst) {
        if (original_pairs.empty()) { return; }
        auto * src = dst->src[0];
        const int64_t d = dst->ne[0], k = dst->ne[1];
        const size_t bytes = d * sizeof(float);
        const bool cpu = ggml_backend_buffer_is_host(bank.down->buffer);
        const bool group_cpu = cpu && !std::getenv("CNM_TIER_LEGACY_CPU_TILES");
        std::vector<int64_t> sorted_pairs;
        std::vector<int32_t> sorted_local;
        if (group_cpu) {
            sorted_pairs.resize(original_pairs.size()); sorted_local.resize(original_local.size());
            std::vector<size_t> order(original_pairs.size());
            for (size_t i = 0; i < order.size(); ++i) { order[i] = i; }
            std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return original_local[a] < original_local[b]; });
            for (size_t i = 0; i < order.size(); ++i) { sorted_pairs[i] = original_pairs[order[i]]; sorted_local[i] = original_local[order[i]]; }
        }
        const auto & pairs = group_cpu ? sorted_pairs : original_pairs;
        const auto & local = group_cpu ? sorted_local : original_local;
        // Group each expert and balance large groups so no tail switches from tiled dots to vec_dot.
        // Read each needed token once per bank invocation, including across tiles.
        std::map<int64_t, std::vector<float>> inputs;
        if (cpu) {
            for (auto q : pairs) {
                const int64_t token = q / k;
                if (!inputs.count(token)) {
                    auto & x = inputs[token]; x.resize(d);
                    if (auto * pinned = runtime->stage(src)) {
                        ggml_backend_tensor_get(src, pinned, token * src->nb[1], bytes);
                        std::memcpy(x.data(), pinned, bytes);
                        if (&bank == &layer.cold) { cold_input_bytes += bytes; }
                    } else { ggml_backend_tensor_get(src, x.data(), token * src->nb[1], bytes); }
                }
            }
        }
        const int64_t tile = group_cpu ? pair_tile : std::min(pair_tile, src->ne[1]);
        size_t group_end = 0;
        for (size_t begin = 0; begin < pairs.size();) {
            int64_t n = std::min<size_t>(tile, pairs.size() - begin);
            if (group_cpu) {
                if (begin == group_end) {
                    group_end = begin + 1;
                    while (group_end < pairs.size() && local[group_end] == local[begin]) { ++group_end; }
                }
                const int64_t remaining = group_end - begin;
                const int64_t tiles = (remaining + pair_tile - 1) / pair_tile;
                n = (remaining + tiles - 1) / tiles;
            }
            // Bound metadata variants; decode keeps every possible nonempty pair count.
            const size_t max_variants = k <= 16 && dst->ne[2] == 1 ? size_t(k) : 4;
            if (!variants.count(n) && variants.size() >= max_variants) { variants.erase(variants.begin()); }
            auto & cached = variants[n];
            if (!cached || cached->source != src || cached->source_data != src->data || cached->source_buffer != src->buffer || cached->logical_tokens != src->ne[1]) {
                cached = prepare(bank, src, n);
            }
            auto & g = *cached;
            std::vector<int32_t> ids(n), tokens(n);
            std::vector<float> packed(cpu ? n * d : 0);
            for (int64_t i = 0; i < n; ++i) {
                ids[i] = local[begin + i];
                tokens[i] = pairs[begin + i] / k;
                if (cpu) { std::copy(inputs.at(tokens[i]).begin(), inputs.at(tokens[i]).end(), packed.begin() + i * d); }
            }
            ggml_backend_tensor_set(g.ids, ids.data(), 0, ids.size() * sizeof(int32_t));
            if (cpu) { ggml_backend_tensor_set(g.input, packed.data(), 0, packed.size() * sizeof(float)); }
            else { ggml_backend_tensor_set(g.tokens, tokens.data(), 0, tokens.size() * sizeof(int32_t)); }
            const auto started = instrument ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            if (ggml_backend_graph_compute(g.backend, g.graph) != GGML_STATUS_SUCCESS) { throw std::runtime_error("expert tiers: bank execution failed"); }
            dump_bank(bank, g, pairs, local, begin, k);
            if (instrument) { body_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count(); }
            if (&bank == &layer.cold) { ++cold_jobs; } else { ++hot_jobs; }
            // Copy disjoint slot vectors without tier reduction or cancellation.
            auto * view_ctx = ggml_init({ggml_tensor_overhead() * 2 * n, nullptr, true});
            if (!view_ctx) { throw std::bad_alloc(); }
            for (int64_t i = 0; i < n; ++i) {
                if (!scatter_coverage.empty() && scatter_coverage.at(pairs[begin + i])++) {
                    throw std::runtime_error("expert tiers: duplicate output slot");
                }
                auto * from = ggml_view_1d(view_ctx, g.output, d, i * g.output->nb[2]);
                auto * to = ggml_view_1d(view_ctx, dst, d, pairs[begin + i] * bytes);
                ggml_backend_view_init(from);
                ggml_backend_view_init(to);
                if (cpu && !ggml_backend_buffer_is_host(dst->buffer)) {
                    auto * pinned = runtime->stage(dst);
                    ggml_backend_tensor_get(from, pinned, 0, bytes);
                    ggml_backend_tensor_set(to, pinned, 0, bytes);
                    if (&bank == &layer.cold) { cold_output_bytes += bytes; }
                } else { ggml_backend_tensor_copy(from, to); }
            }
            ggml_free(view_ctx);
            begin += n;
        }
    }

    std::string bank_identity() {
        auto & cached = runtime->remote_identities[layer.cold.down];
        if (!cached.empty()) { return cached; }
        std::string manifest = "ep-strict-slot-cpu-v1:" + std::to_string(clamp) + ":" + std::to_string(glm);
        for (auto id : layer.cold.ids) { manifest += ":" + std::to_string(id); }
        for (auto * weight : {layer.cold.gate, layer.cold.up, layer.cold.gate_up, layer.cold.down}) {
            if (!weight) { manifest += ":absent"; continue; }
            if (!ggml_backend_buffer_is_host(weight->buffer)) { throw std::runtime_error("expert remote: head cold shadow must be raw CPU weights"); }
            manifest += ":" + std::string(weight->name) + ":" + std::to_string(weight->type);
            for (auto n : weight->ne) { manifest += ":" + std::to_string(n); }
            manifest += ":" + hash_sha256_hex(weight->data, ggml_nbytes(weight));
        }
        cached = hash_sha256_hex(manifest.data(), manifest.size()); return cached;
    }

    llama_remote_call remote_send(ggml_tensor * dst, const std::vector<int64_t> & pairs, const std::vector<int32_t> & local) {
        llama_remote_call c; c.d = dst->ne[0]; c.k = dst->ne[1]; c.t = dst->ne[2];
        c.bank = layer.cold.down->name; c.identity = bank_identity(); c.clamp = clamp; c.glm = glm;
        bool quantized = false;
        for (auto * weight : {layer.cold.gate, layer.cold.up, layer.cold.gate_up, layer.cold.down}) { quantized |= weight && ggml_is_quantized(weight->type); }
        if (cuda_arithmetic && quantized) {
            using query_t = int (*)(ggml_backend_dev_t, ggml_type, int64_t, int64_t);
            auto query = reinterpret_cast<query_t>(ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(layer_device), "ggml_backend_cuda_tier_quant_policy"));
            if (!query || !reciprocal_table || !exponential_table) { throw std::runtime_error("expert remote: exact policy unavailable"); }
            std::array<ggml_tensor *, 3> weights{layer.cold.gate ? layer.cold.gate : layer.cold.gate_up, layer.cold.up, layer.cold.down};
            for (size_t i = 0; i < weights.size(); ++i) {
                if (weights[i] && ggml_is_quantized(weights[i]->type)) {
                    c.policies[i] = query(layer_device, weights[i]->type, c.t, layer.hot.ids.size() + layer.cold.ids.size());
                    if (!c.policies[i]) { throw std::runtime_error("expert remote: unsupported exact policy"); }
                }
            }
            c.warps = c.t == 1 ? 4 : 1;
            auto & calibration = runtime->calibration_ids[reciprocal_table];
            if (calibration.empty()) {
                calibration = hash_sha256_hex(reciprocal_table, (size_t(1) << 23) * sizeof(float)) + hash_sha256_hex(exponential_table, (size_t(1) << 24) * sizeof(float));
            }
            c.calibration = calibration;
        }
        std::vector<uint8_t> seen(c.t, 0);
        for (size_t i = 0; i < pairs.size(); ++i) {
            c.pairs.push_back(pairs[i]); c.ids.push_back(layer.cold.ids.at(local[i]));
            const uint32_t t = pairs[i] / c.k;
            if (!seen[t]++) {
                c.tokens.push_back(t); const size_t offset = c.input.size(); c.input.resize(offset + c.d);
                ggml_backend_tensor_get(dst->src[0], c.input.data() + offset, t * dst->src[0]->nb[1], c.d * sizeof(float));
            }
        }
        runtime->remote->send(c); return c;
    }

    void remote_receive(ggml_tensor * dst, llama_remote_call & c) {
        runtime->remote->receive(c);
        for (size_t i = 0; i < c.pairs.size(); ++i) {
            if (!scatter_coverage.empty() && scatter_coverage.at(c.pairs[i])++) { throw std::runtime_error("expert remote: duplicate scatter"); }
            ggml_backend_tensor_set(dst, c.output.data() + i * c.d, size_t(c.pairs[i]) * c.d * sizeof(float), c.d * sizeof(float));
        }
        if (instrument) { LLAMA_LOG_INFO("tiers remote bank=%s pairs=%zu tokens=%zu compute_ms=%.6f\n", c.bank.c_str(), c.pairs.size(), c.tokens.size(), c.compute_ns / 1e6); }
    }

    static ggml_status execute(ggml_tensor * dst, void * user) {
        auto & self = *static_cast<llama_tier_dispatch *>(user);
        try {
            const auto started = self.instrument ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            const uint64_t input_before = self.cold_input_bytes, output_before = self.cold_output_bytes;
            const int64_t k = dst->ne[1], t = dst->ne[2];
            auto * selected = dst->src[1];
            std::vector<int32_t> ids(k * t);
            const size_t control_bytes = ids.size() * sizeof(int32_t);
            if (auto * pinned = self.runtime->stage(selected)) {
                if (control_bytes > self.runtime->staging_bytes) { throw std::runtime_error("expert tiers: routing control exceeds reserved staging"); }
                ggml_backend_tensor_get(selected, pinned, 0, control_bytes);
                std::memcpy(ids.data(), pinned, control_bytes);
            } else { ggml_backend_tensor_get(selected, ids.data(), 0, control_bytes); }
            std::vector<int64_t> hot, cold;
            std::vector<int32_t> hot_ids, cold_ids;
            for (int64_t q = 0; q < k * t; ++q) {
                if (ids[q] < 0 || size_t(ids[q]) >= self.source_map.size() || self.source_map[ids[q]] == std::numeric_limits<int32_t>::min()) {
                    throw std::runtime_error("expert tiers: router selected absent expert");
                }
                const int32_t slot = self.source_map[ids[q]];
                if (slot >= 0) { hot.push_back(q); hot_ids.push_back(slot); }
                else { cold.push_back(q); cold_ids.push_back(-slot - 1); }
            }
            if (self.instrument) { self.control_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count(); }
            if (cold.empty()) { ++self.cold_skips; }
            if (self.instrument || !self.dump_directory.empty()) { self.scatter_coverage.assign(k * t, 0); }
            llama_remote_call remote;
            const bool use_remote = self.runtime->remote && !cold.empty();
            const auto invoke_started = std::chrono::steady_clock::now();
            if (use_remote) { remote = self.remote_send(dst, cold, cold_ids); }
            const auto hot_started = std::chrono::steady_clock::now();
            self.run_bank(self.layer.hot, self.hot, hot, hot_ids, dst);
            const auto hot_finished = std::chrono::steady_clock::now();
            if (use_remote) {
                self.remote_receive(dst, remote);
                if (self.instrument) {
                    const auto finished = std::chrono::steady_clock::now();
                    LLAMA_LOG_INFO("tiers remote timing bank=%s invoke_ms=%.6f hot_ms=%.6f join_ms=%.6f worker_ms=%.6f\n", remote.bank.c_str(),
                        std::chrono::duration<double, std::milli>(finished - invoke_started).count(), std::chrono::duration<double, std::milli>(hot_finished - hot_started).count(),
                        std::chrono::duration<double, std::milli>(finished - hot_finished).count(), remote.compute_ns / 1e6);
                }
            } else { self.run_bank(self.layer.cold, self.cold, cold, cold_ids, dst); }
            if (!self.scatter_coverage.empty() && std::find(self.scatter_coverage.begin(), self.scatter_coverage.end(), uint8_t(0)) != self.scatter_coverage.end()) {
                throw std::runtime_error("expert tiers: missing output slot");
            }
            self.hot_pairs += hot.size(); self.cold_pairs += cold.size();
            if (self.instrument) {
                LLAMA_LOG_INFO("tiers hot=%zu cold=%zu cold_input_bytes=%llu cold_output_bytes=%llu\n", hot.size(), cold.size(),
                    (unsigned long long) (self.cold_input_bytes - input_before), (unsigned long long) (self.cold_output_bytes - output_before));
                LLAMA_LOG_INFO("tiers counters hot_jobs=%llu cold_jobs=%llu cold_skips=%llu expert_weight_transfer_bytes=0 body_ms=%.3f control_ms=%.3f\n",
                    (unsigned long long) self.hot_jobs, (unsigned long long) self.cold_jobs, (unsigned long long) self.cold_skips, self.body_ms, self.control_ms);
            }
            return GGML_STATUS_SUCCESS;
        } catch (const std::exception & error) {
            LLAMA_LOG_ERROR("%s\n", error.what());
            return GGML_STATUS_FAILED;
        }
    }
};

std::shared_ptr<llama_tier_runtime> llama_tier_runtime_create(size_t bytes, size_t staging_bytes) {
    return std::make_shared<llama_tier_runtime>(bytes, staging_bytes);
}

std::shared_ptr<llama_tier_dispatch> llama_tier_dispatch_create(const llama_expert_tiers & layer, float clamp, bool glm, int threads, ggml_backend_dev_t layer_device, std::shared_ptr<llama_tier_runtime> runtime) {
    return std::make_shared<llama_tier_dispatch>(layer, clamp, glm, threads, layer_device, std::move(runtime));
}

ggml_tensor * llama_tier_dispatch_build(llama_tier_dispatch & dispatch, ggml_context * ctx, ggml_tensor * input, ggml_tensor * ids) {
    if (input->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 || !ggml_is_contiguous(input) || !ggml_is_contiguous(ids)) {
        throw std::runtime_error("expert tiers: dispatch requires contiguous F32 activations and I32 IDs");
    }
    return ggml_backend_sched_dispatch_new(ctx, input->ne[0], ids->ne[0], ids->ne[1], input, ids, &dispatch.boundary);
}

void llama_tier_remote_serve(llama_model & model, const char * endpoint, int threads) {
    if (std::getenv("CNM_TIER_REMOTE") || std::getenv("CNM_TIER_CUDA_ARITH")) { throw std::runtime_error("expert remote: worker requires local CPU arithmetic"); }
    struct request_tensors {
        ggml_context * ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ~request_tensors() { reset(); }
        void reset() { ggml_backend_buffer_free(buffer); ggml_free(ctx); buffer = nullptr; ctx = nullptr; }
    };
    struct worker_layer {
        std::shared_ptr<llama_tier_runtime> runtime;
        request_tensors storage, gpu_storage;
        ggml_tensor * input = nullptr;
        ggml_tensor * output = nullptr;
        std::unique_ptr<llama_tier_dispatch> dispatch;
    };
    size_t max_d = 0, max_ff = 0;
    for (const auto & layer : model.expert_tiers) {
        if (layer.cold.down) { max_d = std::max(max_d, size_t(layer.cold.down->ne[1])); max_ff = std::max(max_ff, size_t(layer.cold.down->ne[0])); }
    }
    auto worker_runtime = llama_tier_runtime_create(256 * (8 * max_ff + 3 * max_d) * sizeof(float) + 96 * 4096, 256 * max_d * sizeof(float));
    if (const char * directory = std::getenv("CNM_TIER_REMOTE_TABLES")) {
        auto load = [&](const char * name, size_t count, std::vector<float> & values) {
            std::ifstream file(std::string(directory) + "/" + name, std::ios::binary | std::ios::ate);
            if (!file || file.tellg() != std::streamoff(count * sizeof(float))) { throw std::runtime_error("expert remote: invalid SFU table file"); }
            values.resize(count); file.seekg(0); file.read(reinterpret_cast<char *>(values.data()), count * sizeof(float));
            if (!file) { throw std::runtime_error("expert remote: SFU table read failed"); }
        };
        load("reciprocal-table.f32", size_t(1) << 23, worker_runtime->remote_reciprocal);
        load("exponential-table.f32", size_t(1) << 24, worker_runtime->remote_exponential);
        worker_runtime->calibration = hash_sha256_hex(worker_runtime->remote_reciprocal.data(), worker_runtime->remote_reciprocal.size() * sizeof(float)) + hash_sha256_hex(worker_runtime->remote_exponential.data(), worker_runtime->remote_exponential.size() * sizeof(float));
    }
    struct gpu_bank {
        llama_expert_bank * target = nullptr;
        llama_expert_bank original;
        ggml_context * ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ~gpu_bank() { if (target) { *target = original; } ggml_backend_buffer_free(buffer); ggml_free(ctx); }
    };
    std::vector<std::unique_ptr<gpu_bank>> gpu_banks;
    size_t gpu_remaining = 0;
    if (const char * budget = std::getenv("CNM_TIER_REMOTE_GPU_BYTES")) {
        std::string value(budget);
        if (value.empty() || !std::all_of(value.begin(), value.end(), [](char c) { return c >= '0' && c <= '9'; })) { throw std::runtime_error("expert remote: invalid GPU budget"); }
        gpu_remaining = std::stoull(value);
    }
    auto * gpu = gpu_remaining ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) : nullptr;
    if (gpu_remaining && (!gpu || std::string(ggml_backend_dev_name(gpu)).find("CUDA") != 0)) { throw std::runtime_error("expert remote: CUDA GPU budget requested but CUDA unavailable"); }
    std::map<std::string, worker_layer> layers;
    for (size_t i = 0; i < model.expert_tiers.size(); ++i) {
        auto & layer = model.expert_tiers[i]; if (!layer.cold.down) { continue; }
        auto & w = layers[layer.cold.down->name];
        w.runtime = worker_runtime;
        w.dispatch = std::make_unique<llama_tier_dispatch>(layer, model.hparams.swiglu_clamp_exp[i], model.arch == LLM_ARCH_GLM5_NEXT, threads, nullptr, w.runtime);
        const auto identity = w.dispatch->bank_identity();
        size_t payload = 0;
        for (auto * weight : {layer.cold.gate, layer.cold.up, layer.cold.gate_up, layer.cold.down}) { if (weight) { payload += ggml_nbytes(weight); } }
        if (gpu && payload <= gpu_remaining) {
            auto bank = std::make_unique<gpu_bank>(); bank->original = layer.cold;
            bank->ctx = ggml_init({4 * ggml_tensor_overhead(), nullptr, true}); if (!bank->ctx) { throw std::bad_alloc(); }
            llama_expert_bank placed = layer.cold;
            auto clone = [&](ggml_tensor * weight) {
                if (!weight) { return weight; }
                auto * tensor = ggml_dup_tensor(bank->ctx, weight); ggml_set_name(tensor, weight->name); return tensor;
            };
            placed.gate = clone(placed.gate); placed.up = clone(placed.up); placed.gate_up = clone(placed.gate_up); placed.down = clone(placed.down);
            bank->buffer = ggml_backend_alloc_ctx_tensors_from_buft(bank->ctx, ggml_backend_dev_buffer_type(gpu));
            if (!bank->buffer) { throw std::bad_alloc(); }
            const size_t allocated = ggml_backend_buffer_get_size(bank->buffer);
            if (allocated <= gpu_remaining) {
                for (const auto & item : {std::make_pair(layer.cold.gate, placed.gate), std::make_pair(layer.cold.up, placed.up), std::make_pair(layer.cold.gate_up, placed.gate_up), std::make_pair(layer.cold.down, placed.down)}) {
                    if (item.first) { ggml_backend_tensor_copy(item.first, item.second); }
                }
                bank->target = &layer.cold; layer.cold = placed; gpu_remaining -= allocated;
                worker_runtime->remote_identities[placed.down] = identity;
                LLAMA_LOG_INFO("expert remote: resident CUDA bank=%s payload_bytes=%zu allocated_bytes=%zu remaining_bytes=%zu\n", placed.down->name, payload, allocated, gpu_remaining);
                gpu_banks.push_back(std::move(bank));
            }
        }
    }
    llama_remote_listen(endpoint, [&](llama_remote_call & c) {
        auto found = layers.find(c.bank);
        if (found == layers.end()) { throw std::runtime_error("expert remote: unknown bank"); }
        auto & state = found->second;
        auto & dispatch = *state.dispatch;
        if (c.identity != dispatch.bank_identity() || c.clamp != dispatch.clamp || c.glm != dispatch.glm || c.d != dispatch.layer.cold.down->ne[1]) { throw std::runtime_error("expert remote: bank payload, placement or policy mismatch"); }
        const bool exact = !c.calibration.empty();
        if (exact != dispatch.remote_policy_active || c.policies != dispatch.remote_policy || int(c.warps) != dispatch.remote_warps) { dispatch.cold.clear(); }
        dispatch.remote_policy_active = exact; dispatch.cuda_arithmetic = exact;
        if (dispatch.remote_policy_active) {
            if (c.calibration != worker_runtime->calibration) { throw std::runtime_error("expert remote: SFU calibration mismatch"); }
            dispatch.remote_policy = c.policies; dispatch.remote_warps = c.warps;
            dispatch.reciprocal_table = worker_runtime->remote_reciprocal.data(); dispatch.exponential_table = worker_runtime->remote_exponential.data();
        } else if (std::any_of(c.policies.begin(), c.policies.end(), [](uint32_t p) { return p != 0; })) { throw std::runtime_error("expert remote: missing numerical calibration"); }
        std::vector<int64_t> pairs(c.pairs.begin(), c.pairs.end()); std::vector<int32_t> local;
        for (auto id : c.ids) {
            const auto & ids = dispatch.layer.cold.ids;
            auto at = std::find(ids.begin(), ids.end(), int32_t(id));
            if (at == ids.end()) { throw std::runtime_error("expert remote: selected expert absent"); }
            local.push_back(at - ids.begin());
        }
        if (!state.output || state.output->ne[0] != c.d || state.output->ne[1] != c.k || state.output->ne[2] != c.t) {
            dispatch.cold.clear(); state.gpu_storage.reset(); state.storage.reset();
            auto & storage = state.storage;
            storage.ctx = ggml_init({4 * ggml_tensor_overhead(), nullptr, true});
            if (!storage.ctx) { throw std::bad_alloc(); }
            state.input = ggml_new_tensor_2d(storage.ctx, GGML_TYPE_F32, c.d, c.t);
            state.output = ggml_new_tensor_3d(storage.ctx, GGML_TYPE_F32, c.d, c.k, c.t); state.output->src[0] = state.input;
            storage.buffer = ggml_backend_alloc_ctx_tensors_from_buft(storage.ctx, ggml_backend_cpu_buffer_type());
            if (!storage.buffer) { throw std::bad_alloc(); }
            if (!ggml_backend_buffer_is_host(dispatch.layer.cold.down->buffer)) {
                auto & gpu_storage = state.gpu_storage;
                gpu_storage.ctx = ggml_init({ggml_tensor_overhead(), nullptr, true}); if (!gpu_storage.ctx) { throw std::bad_alloc(); }
                auto * staged = ggml_dup_tensor(gpu_storage.ctx, state.input);
                gpu_storage.buffer = ggml_backend_alloc_ctx_tensors_from_buft(gpu_storage.ctx, ggml_backend_dev_buffer_type(gpu));
                if (!gpu_storage.buffer) { throw std::bad_alloc(); }
                state.input = staged; state.output->src[0] = staged;
            }
        }
        auto * input = state.input; auto * output = state.output;
        for (size_t i = 0; i < c.tokens.size(); ++i) { ggml_backend_tensor_set(input, c.input.data() + i * c.d, size_t(c.tokens[i]) * c.d * sizeof(float), c.d * sizeof(float)); }
        auto started = std::chrono::steady_clock::now();
        dispatch.run_bank(dispatch.layer.cold, dispatch.cold, pairs, local, output);
        c.compute_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count();
        c.output.resize(size_t(c.d) * pairs.size());
        for (size_t i = 0; i < pairs.size(); ++i) { ggml_backend_tensor_get(output, c.output.data() + i * c.d, size_t(pairs[i]) * c.d * sizeof(float), c.d * sizeof(float)); }
    });
}
