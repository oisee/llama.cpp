#pragma once

#include "ggml-backend.h"

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

struct llama_model;

// keeps recently and repeatedly used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // [CNM] cumulative counters, safe to read from any thread
    struct counters {
        uint64_t hits        = 0; // decode-path (ubatch <= 32) distinct experts found in the cache
        uint64_t misses      = 0; // decode-path distinct experts uploaded
        uint64_t bytes       = 0; // decode-path bytes uploaded
        uint64_t sel         = 0; // decode-path expert selections (top-k entries, not distinct)
        uint64_t sel_cold    = 0; // of sel, selections of experts with a nonzero offset in the current CNM bias
        uint64_t pf_uploaded = 0; // prefetched experts that were uploaded
        uint64_t pf_present  = 0; // prefetched experts that were already cached (only promoted)
        uint64_t pf_skipped  = 0; // prefetch pairs skipped: uncached layer, duplicate or over the slots of the group
        uint64_t pf_bytes    = 0; // bytes uploaded by prefetch
        uint64_t n_slots     = 0; // slots of all groups
        uint64_t n_groups    = 0;
    };
    counters get_counters() const;

    // [CNM] upload the experts of (layer, expert) pairs, in priority order (first = most important), and mark them
    // most recently used, the first pair ending at the MRU end; at most n_slots pairs per group are taken, an already
    // cached expert is only promoted; call only when no graph is running; returns the number of uploaded experts
    // throws std::invalid_argument for an expert index out of range
    int64_t prefetch(const int32_t * il_expert_pairs, size_t n_pairs);

    // [CNM] experts of layer il with a nonzero value count as cold selections in the counters, nullptr clears the layer
    void set_cold(int32_t il, const float * values, int64_t n);


private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
