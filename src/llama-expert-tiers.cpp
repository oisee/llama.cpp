#include "llama-expert-tiers.h"
#include "llama-model.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "llama-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <cstring>
#include <limits>
#include <chrono>
#include <map>
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
struct bank_graph {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * tokens = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    int64_t pairs = 0;
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
    bool instrument;
    ggml_backend_sched_dispatch boundary;
    std::shared_ptr<llama_tier_runtime> runtime;
    std::map<int64_t, std::unique_ptr<bank_graph>> hot, cold;
    std::vector<int32_t> source_map;
    uint64_t hot_pairs = 0, cold_pairs = 0, cold_input_bytes = 0, cold_output_bytes = 0;
    uint64_t hot_jobs = 0, cold_jobs = 0, cold_skips = 0;
    double body_ms = 0, control_ms = 0;

    llama_tier_dispatch(const llama_expert_tiers & layer, float clamp, bool glm, int threads, std::shared_ptr<llama_tier_runtime> runtime) : layer(layer), clamp(clamp), glm(glm), threads(threads), instrument(std::getenv("CNM_TIER_TRACE") != nullptr), runtime(std::move(runtime)) {
        boundary = {execute, this};
        int32_t count = 0;
        for (const auto * bank : {&layer.hot, &layer.cold}) {
            for (int32_t id : bank->ids) { count = std::max(count, id + 1); }
        }
        source_map.assign(count, std::numeric_limits<int32_t>::min());
        for (size_t i = 0; i < layer.hot.ids.size(); ++i) { source_map[layer.hot.ids[i]] = int32_t(i); }
        for (size_t i = 0; i < layer.cold.ids.size(); ++i) { source_map[layer.cold.ids[i]] = -int32_t(i) - 1; }
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
            auto * merged = ggml_mul_mat_id(ctx, bank.gate_up, input, g->ids);
            int64_t ff = merged->ne[0] / 2;
            gate = ggml_view_3d(ctx, merged, ff, 1, n, merged->nb[1], merged->nb[2], 0);
            up = ggml_view_3d(ctx, merged, ff, 1, n, merged->nb[1], merged->nb[2], ff * merged->nb[0]);
        } else {
            up = ggml_mul_mat_id(ctx, bank.up, input, g->ids);
            gate = ggml_mul_mat_id(ctx, bank.gate, input, g->ids);
        }
        ggml_tensor * activated;
        if (bank.gate && clamp > 1e-6f) {
            activated = glm ? ggml_swiglu_clamp(ctx, gate, up, clamp) :
                ggml_mul(ctx, ggml_clamp(ctx, ggml_silu(ctx, gate), -INFINITY, clamp), ggml_clamp(ctx, up, -clamp, clamp));
        } else { activated = ggml_swiglu_split(ctx, gate, up); }
        g->output = ggml_mul_mat_id(ctx, bank.down, activated, g->ids);
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

    void run_bank(const llama_expert_bank & bank, std::map<int64_t, std::unique_ptr<bank_graph>> & variants, const std::vector<int64_t> & pairs,
            const std::vector<int32_t> & local, ggml_tensor * dst) {
        if (pairs.empty()) { return; }
        auto * src = dst->src[0];
        const int64_t d = dst->ne[0], k = dst->ne[1];
        const size_t bytes = d * sizeof(float);
        const bool cpu = ggml_backend_buffer_is_host(bank.down->buffer);
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
        const int64_t tile = std::min(pair_tile, src->ne[1]);
        for (size_t begin = 0; begin < pairs.size(); begin += tile) {
            const int64_t n = std::min<size_t>(tile, pairs.size() - begin);
            // Bound metadata variants; decode keeps every possible nonempty pair count.
            const size_t max_variants = k <= 16 && dst->ne[2] == 1 ? size_t(k) : 4;
            if (!variants.count(n) && variants.size() >= max_variants) { variants.erase(variants.begin()); }
            auto & cached = variants[n];
            if (!cached || cached->source != src || cached->source_data != src->data || cached->source_buffer != src->buffer) {
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
            if (instrument) { body_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count(); }
            if (&bank == &layer.cold) { ++cold_jobs; } else { ++hot_jobs; }
            // Copy disjoint slot vectors without tier reduction or cancellation.
            auto * view_ctx = ggml_init({ggml_tensor_overhead() * 2 * n, nullptr, true});
            if (!view_ctx) { throw std::bad_alloc(); }
            for (int64_t i = 0; i < n; ++i) {
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
        }
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
            self.run_bank(self.layer.hot, self.hot, hot, hot_ids, dst);
            self.run_bank(self.layer.cold, self.cold, cold, cold_ids, dst);
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

std::shared_ptr<llama_tier_dispatch> llama_tier_dispatch_create(const llama_expert_tiers & layer, float clamp, bool glm, int threads, std::shared_ptr<llama_tier_runtime> runtime) {
    return std::make_shared<llama_tier_dispatch>(layer, clamp, glm, threads, std::move(runtime));
}

ggml_tensor * llama_tier_dispatch_build(llama_tier_dispatch & dispatch, ggml_context * ctx, ggml_tensor * input, ggml_tensor * ids) {
    if (input->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 || !ggml_is_contiguous(input) || !ggml_is_contiguous(ids)) {
        throw std::runtime_error("expert tiers: dispatch requires contiguous F32 activations and I32 IDs");
    }
    return ggml_backend_sched_dispatch_new(ctx, input->ne[0], ids->ne[0], ids->ne[1], input, ids, &dispatch.boundary);
}
