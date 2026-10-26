#pragma once
#include "ggml-backend.h"
#include <memory>

struct llama_expert_tiers;
struct llama_tier_dispatch;
struct llama_tier_runtime;
std::shared_ptr<llama_tier_runtime> llama_tier_runtime_create(size_t bytes, size_t staging_bytes);

std::shared_ptr<llama_tier_dispatch> llama_tier_dispatch_create(const llama_expert_tiers & layer, float clamp, bool glm, int threads, std::shared_ptr<llama_tier_runtime> runtime);
ggml_tensor * llama_tier_dispatch_build(llama_tier_dispatch & dispatch, ggml_context * ctx, ggml_tensor * input, ggml_tensor * ids);
