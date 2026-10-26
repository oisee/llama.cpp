// Dump named MoE intermediates with the public evaluation callback.
#include "llama.h"
#include "ggml-backend.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
struct Trace { std::string dir; std::ofstream index; int seq = 0; };
static bool callback(ggml_tensor * t, bool ask, void * user) {
    bool want = std::strstr(t->name, "ffn_moe_") || std::strstr(t->name, "ffn_norm-");
    if (std::strchr(t->name, ' ')) want = false;
    if (std::getenv("TRACE_LONG") && (std::strstr(t->name, "ffn_moe_gate-") || std::strstr(t->name, "ffn_moe_up-") || std::strstr(t->name, "ffn_moe_swiglu-") || std::strstr(t->name, "ffn_moe_down-") || std::strstr(t->name, "ffn_moe_weighted-"))) want = false;
    if (std::getenv("TRACE_TOKEN_FILE")) {
        want = std::strstr(t->name, "ffn_moe_topk-") || std::strstr(t->name, "ffn_moe_weights_norm-");
    }
    if (ask) return want;
    if (!want) return true;
    auto & s = *static_cast<Trace *>(user);
    std::string file = std::to_string(s.seq++) + ".bin";
    std::vector<char> bytes(ggml_nbytes(t));
    ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
    std::ofstream out(s.dir + "/" + file, std::ios::binary);
    out.write(bytes.data(), bytes.size());
    s.index << file << ' ' << t->name << ' ' << ggml_type_name(t->type);
    for (int i=0; i<4; ++i) s.index << ' ' << t->ne[i];
    for (int i=0; i<4; ++i) s.index << ' ' << t->nb[i];
    s.index << '\n';
    return bool(out) && bool(s.index);
}
int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    llama_backend_init();
    auto mp = llama_model_default_params();
    const bool gpu = std::getenv("TRACE_GPU") != nullptr;
    mp.n_gpu_layers = gpu ? 999 : 0;
    llama_model_tensor_buft_override overrides[] = {{"\\.ffn_.*_exps", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}};
    if (gpu) mp.tensor_buft_overrides = overrides;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    Trace trace{argv[2], std::ofstream(std::string(argv[2])+"/index.txt")};
    auto cp = llama_context_default_params();
    const bool full_chunk = std::getenv("TRACE_TOKEN_FILE") != nullptr;
    cp.n_ctx=full_chunk ? 4096 : 512; cp.n_batch=cp.n_ctx; cp.n_ubatch=cp.n_ctx;
    cp.n_threads=8; cp.n_threads_batch=8;
    cp.offload_kqv=gpu; cp.op_offload=gpu;
    cp.cb_eval=callback; cp.cb_eval_user_data=&trace;
    auto * ctx=llama_init_from_model(model, cp);
    if (!ctx) return 4;
    // Fixed teacher-forced token IDs, identical for every model.
    std::vector<llama_token> tokens{42, 14, 159, 26};
    if (full_chunk) {
        std::ifstream in(std::getenv("TRACE_TOKEN_FILE"), std::ios::binary);
        char header[20];
        in.read(header, sizeof(header));
        if (!in || std::memcmp(header, "_logits_", 8) != 0) return 8;
        tokens.resize(4096);
        in.read(reinterpret_cast<char *>(tokens.data()), tokens.size() * sizeof(llama_token));
        if (!in) return 9;
    } else if (std::getenv("TRACE_LONG")) {
        std::ifstream in("texts/heldout-own-body006.txt");
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        auto * v = llama_model_get_vocab(model);
        int n = -llama_tokenize(v,text.data(),text.size(),nullptr,0,true,false);
        tokens.resize(n);
        if (llama_tokenize(v,text.data(),text.size(),tokens.data(),n,true,false)!=n) return 7;
        tokens.resize(512);
    }
    std::vector<int8_t> all(tokens.size(), 1);
    auto batch=llama_batch_get_one(tokens.data(), tokens.size());
    batch.logits=all.data();
    if (llama_decode(ctx, batch)) return 5;
    auto * logits=llama_get_logits(ctx);
    int nv=llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::ofstream out(std::string(argv[2])+"/logits.f32", std::ios::binary);
    out.write((const char *)logits, tokens.size()*nv*sizeof(float));
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return out && trace.index ? 0 : 6;
}
