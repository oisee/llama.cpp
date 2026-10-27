#include "llama-expert-tiers.h"
#include "llama-model.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

static int uploads = 0, overlaps = 0;
static void log_capture(ggml_log_level, const char * text, void *) {
    if (std::strstr(text, "tiers prefill_offload")) { ++uploads; }
    if (std::strstr(text, "tiers decode_timing") && std::strstr(text, "overlap=1")) { ++overlaps; }
    std::fputs(text, stderr);
}

struct fixture {
    static constexpr int64_t d = 256, ff = 256, experts = 8, k = 4;
    ggml_context_ptr ctx{ggml_init({24 * ggml_tensor_overhead(), nullptr, true})};
    ggml_backend_buffer_ptr host_buffer, device_buffer;
    llama_expert_tiers tiers;
    llama_expert_bank full;
    std::vector<ggml_tensor *> host, device;
    fixture(ggml_backend_t cpu, ggml_backend_t backend, ggml_type type, bool merged) {
        tiers.tiered = true;
        tiers.hot.ids = {0, 2, 4, 6}; tiers.cold.ids = {1, 3, 5, 7};
        auto create = [&](llama_expert_bank & bank, int64_t n, bool gpu) {
            auto tensor = [&](int64_t width, int64_t rows, const char * name) {
                auto * w = ggml_new_tensor_3d(ctx.get(), type, width, rows, n);
                ggml_set_name(w, name);
                (gpu ? device : host).push_back(w);
                return w;
            };
            if (merged) { bank.gate_up = tensor(d, 2 * ff, "gate_up"); }
            else { bank.gate = tensor(d, ff, "gate"); bank.up = tensor(d, ff, "up"); }
            bank.down = tensor(ff, d, "down");
        };
        create(full, experts, false);
        create(tiers.hot, 4, true); create(tiers.cold, 4, false);
        auto buft = ggml_backend_get_default_buffer_type(cpu);
        host_buffer.reset(ggml_backend_buft_alloc_buffer(buft, 16 << 20));
        device_buffer.reset(ggml_backend_alloc_buffer(backend, 8 << 20));
        check(host_buffer && device_buffer, "fixture buffers");
        auto ha = ggml_tallocr_new(host_buffer.get()), da = ggml_tallocr_new(device_buffer.get());
        for (auto * w : host) { check(ggml_tallocr_alloc(&ha, w) == GGML_STATUS_SUCCESS, "host tensor"); }
        for (auto * w : device) { check(ggml_tallocr_alloc(&da, w) == GGML_STATUS_SUCCESS, "device tensor"); }
        for (auto * w : {full.gate, full.up, full.gate_up, full.down}) {
            if (!w) { continue; }
            std::vector<float> values(ggml_nelements(w));
            const float phase = w == full.down ? 2.1f : w == full.up ? 1.3f : 0.7f;
            for (size_t i = 0; i < values.size(); ++i) { values[i] = 0.07f * std::sin(float(i) * 0.123f + phase); }
            std::vector<char> raw(ggml_nbytes(w));
            check(ggml_quantize_chunk(type, values.data(), raw.data(), 0, w->ne[1] * experts, w->ne[0], nullptr) == raw.size(), "quantize");
            ggml_backend_tensor_set(w, raw.data(), 0, raw.size());
        }
        for (auto * bank : {&tiers.hot, &tiers.cold}) {
            for (auto pair : {std::make_pair(full.gate, bank->gate), std::make_pair(full.up, bank->up),
                    std::make_pair(full.gate_up, bank->gate_up), std::make_pair(full.down, bank->down)}) {
                if (!pair.first) { continue; }
                for (size_t i = 0; i < bank->ids.size(); ++i) {
                    ggml_backend_tensor_set(pair.second, (const char *) pair.first->data + bank->ids[i] * pair.first->nb[2], i * pair.second->nb[2], pair.first->nb[2]);
                }
            }
        }
    }
};

struct replay {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    std::shared_ptr<llama_tier_runtime> runtime;
    std::shared_ptr<llama_tier_dispatch> dispatch;
    ggml_backend_sched_t sched;
    ggml_cgraph * graph;
    ggml_tensor * x, * ids, * output;
    int64_t t;
    replay(fixture & f, ggml_backend_t cpu, ggml_backend_t backend, bool native, int64_t tokens,
            std::shared_ptr<llama_tier_runtime> shared, bool op_offload = true) : runtime(std::move(shared)), t(tokens) {
        ctx.reset(ggml_init({96 * ggml_tensor_overhead() + ggml_graph_overhead_custom(128, false), nullptr, true}));
        ggml_backend_t backends[] = {backend, cpu};
        sched = ggml_backend_sched_new(backends, nullptr, backend == cpu ? 1 : 2, 128, false, true);
        x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, fixture::d, t); ggml_set_input(x);
        ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, fixture::k, t); ggml_set_input(ids);
        if (native) {
            auto * input = ggml_reshape_3d(ctx.get(), x, fixture::d, 1, t);
            auto upload = [&](ggml_tensor * w) {
                if (!w) { return (ggml_tensor *) nullptr; }
                auto * copy = ggml_dup_tensor(ctx.get(), w);
                return copy;
            };
            auto * gate = upload(f.full.gate), * up = upload(f.full.up), * merged = upload(f.full.gate_up), * down = upload(f.full.down);
            ggml_tensor * g, * u;
            if (merged) {
                auto * both = ggml_mul_mat_id(ctx.get(), merged, input, ids);
                g = ggml_view_3d(ctx.get(), both, fixture::ff, fixture::k, t, both->nb[1], both->nb[2], 0);
                u = ggml_view_3d(ctx.get(), both, fixture::ff, fixture::k, t, both->nb[1], both->nb[2], fixture::ff * sizeof(float));
            } else { g = ggml_mul_mat_id(ctx.get(), gate, input, ids); u = ggml_mul_mat_id(ctx.get(), up, input, ids); }
            output = ggml_mul_mat_id(ctx.get(), down, ggml_swiglu_split(ctx.get(), g, u), ids);
            buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
            for (auto pair : {std::make_pair(f.full.gate, gate), std::make_pair(f.full.up, up),
                    std::make_pair(f.full.gate_up, merged), std::make_pair(f.full.down, down)}) {
                if (pair.first) { ggml_backend_tensor_copy(pair.first, pair.second); }
            }
        } else {
            dispatch = llama_tier_dispatch_create(f.tiers, 0, false, 2,
                backend == cpu ? nullptr : ggml_backend_get_device(backend), runtime, op_offload);
            output = llama_tier_dispatch_build(*dispatch, ctx.get(), x, ids);
            ggml_backend_sched_set_tensor_backend(sched, output, backend);
            buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        }
        ggml_set_output(output);
        graph = ggml_new_graph_custom(ctx.get(), 128, false); ggml_build_forward_expand(graph, output);
        check(ggml_backend_sched_alloc_graph(sched, graph), "outer allocation");
        std::vector<float> values(fixture::d * t);
        for (size_t i = 0; i < values.size(); ++i) { values[i] = 0.2f * std::cos(float(i) * 0.17f); }
        ggml_backend_tensor_set(x, values.data(), 0, values.size() * sizeof(float));
    }
    ~replay() {
        ggml_backend_sched_synchronize(sched);
        dispatch.reset();
        ggml_backend_sched_free(sched);
    }
    std::vector<float> run(int pattern) {
        std::vector<int32_t> selected(fixture::k * t);
        const int32_t sparse[] = {3, 7, 2, 6}, partial[] = {0, 2, 4, 1};
        for (size_t i = 0; i < selected.size(); ++i) {
            // Partial, full, and sparse cold unions, plus hot/cold endpoints.
            selected[i] = pattern == 0 ? partial[i % 4] : pattern == 1 ? int(i % 8) :
                pattern == 2 ? 2 * int(i % 4) : pattern == 3 ? 2 * int(i % 4) + 1 :
                sparse[i % 4];
        }
        ggml_backend_tensor_set(ids, selected.data(), 0, selected.size() * sizeof(int32_t));
        check(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "compute");
        std::vector<float> result(fixture::d * fixture::k * t);
        ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
        return result;
    }
};

int main(int argc, char ** argv) {
    try {
        const bool cuda = argc > 1 && std::strcmp(argv[1], "--cuda") == 0;
        ggml_backend_load_all();
        llama_log_set(log_capture, nullptr);
        ggml_backend_ptr cpu(ggml_backend_cpu_init()), gpu;
        if (cuda) {
            auto * device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
            check(device != nullptr, "CUDA device missing");
            gpu.reset(ggml_backend_dev_init(device, nullptr));
        }
        auto * backend = cuda ? gpu.get() : cpu.get();
        ggml_backend_cpu_set_n_threads(cpu.get(), 2);
        const char * flag = std::getenv("CNM_TIER_PREFILL_OFFLOAD");
        const bool enabled = cuda && !std::getenv("CNM_TIER_CUDA_ARITH") && (!flag || std::strcmp(flag, "0") != 0);
        const char * concurrent = std::getenv("CNM_TIER_OVERLAP");
        const bool overlap = cuda && (!concurrent || std::strcmp(concurrent, "0") != 0);
        const char * threshold = std::getenv("GGML_OP_OFFLOAD_MIN_BATCH");
        const int minimum = threshold ? std::atoi(threshold) : 32;
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_IQ3_S}) {
            for (bool merged : {false, true}) {
                fixture f(cpu.get(), backend, type, merged);
                auto shared = llama_tier_runtime_create(16 << 20, 2 << 20);
                for (int64_t tokens : {1, 31, 32, 64, 256}) {
                    replay native(f, cpu.get(), backend, true, tokens, shared);
                    replay tiers(f, cpu.get(), backend, false, tokens, shared);
                    for (int pattern : {0, 1, 4, 0, 2, 3, 1}) {
                        const auto reference = native.run(pattern);
                        const int before = uploads, overlap_before = overlaps;
                        const auto actual = tiers.run(pattern);
                        const bool expected = enabled && tokens > 1 && tokens >= minimum && pattern != 2;
                        check(uploads - before == int(expected), "offload threshold or exact-mode gate");
                        check(overlaps - overlap_before == int(overlap && tokens == 1 && pattern != 2 && pattern != 3), "decode overlap or empty-bank gate");
                        double error = 0;
                        for (size_t i = 0; i < actual.size(); ++i) {
                            check(std::isfinite(actual[i]) && std::isfinite(reference[i]), "nonfinite output");
                            error = std::max(error, std::abs(double(actual[i]) - reference[i]));
                        }
                        std::printf("type=%s merged=%d tokens=%lld pattern=%d offload=%d max_error=%.9g\n", ggml_type_name(type), int(merged), (long long) tokens, pattern, int(expected), error);
                        check(error < (type == GGML_TYPE_F32 ? 2e-5 : 0.005), "native comparison");
                    }
                }
                replay disabled(f, cpu.get(), backend, false, 64, shared, false);
                const int before = uploads;
                disabled.run(1);
                check(uploads == before, "global op_offload=false was ignored");
                if (cuda && type == GGML_TYPE_IQ3_S && std::getenv("CNM_TIER_CUDA_ARITH")) {
                    replay failure(f, cpu.get(), backend, false, 1, shared);
                    auto * weight = f.tiers.cold.gate ? f.tiers.cold.gate : f.tiers.cold.gate_up;
                    weight->type = GGML_TYPE_Q4_0;
                    bool rejected = false;
                    try { failure.run(0); } catch (const std::exception &) { rejected = true; }
                    weight->type = type;
                    check(rejected, "unsupported exact CPU projection was accepted");
                    failure.run(0);
                }
            }
        }
        ggml_quantize_free();
        std::puts("PASS: native oracle, threshold, exact/disable gates, changing unions, merged gate/up, shared workspace, decode overlap, worker failure recovery");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
