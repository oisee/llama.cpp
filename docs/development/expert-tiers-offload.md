# Cold expert prefill offload

Fast tier dispatch streams the union of selected CPU experts to its layer CUDA device when the logical token batch reaches CUDA's `GGML_OP_OFFLOAD_MIN_BATCH` (default 32). The original weights remain in host memory. A growable temporary expert buffer is shared across layers on each context/device; only selected expert slices are copied, once per projection and layer invocation. Allocation failure falls back to the existing CPU path.

The outer scheduler cannot place the private bank graphs because dispatch executes them directly. This path uses the existing CUDA `supports_op` and `offload_op` policies and existing packed `MUL_MAT_ID` kernels, with compact expert IDs. It does not change expert weighting or slot reduction.

- `CNM_TIER_PREFILL_OFFLOAD=0` disables streaming; it defaults on for fast arithmetic.
- The caller's `op_offload=false` also disables streaming.
- `CNM_TIER_CUDA_ARITH` always preserves the CPU exact path, including when the streaming switch is explicitly enabled.
- Single-token decode keeps CPU cold experts regardless of threshold overrides.
- `CNM_TIER_TRACE=1` reports union size, uploaded weight bytes, buffer high water, upload/total time, and offload job counters.

Build and run the fixture-free native oracle test:

```sh
cmake --build build --target test-expert-tiers
CNM_TIER_TRACE=1 build/bin/test-expert-tiers
CNM_TIER_TRACE=1 build/bin/test-expert-tiers --cuda
CNM_TIER_TRACE=1 CNM_TIER_PREFILL_OFFLOAD=0 build/bin/test-expert-tiers --cuda
CNM_TIER_TRACE=1 GGML_OP_OFFLOAD_MIN_BATCH=64 build/bin/test-expert-tiers --cuda
CNM_TIER_TRACE=1 CNM_TIER_CUDA_ARITH=1 CNM_TIER_CUDA_SFU=1 build/bin/test-expert-tiers --cuda
```

The CPU and RTX 2070 tests cover F32/IQ3_S, separate/merged gate-up, batches 1/31/32/64/256, sparse and changing expert unions, empty selections, shared buffers, and global/specific switches. A small quantized Qwen fixture matches all-CUDA bitwise on three different 64-position excerpts; an F32 GLM fixture has mean KL around 3e-15, comparable to its native CPU control. Exact-mode prefill outputs are bitwise unchanged by the new switch on both fixtures. These are synthetic arithmetic checks, not real-model language-quality or i7 performance results.

# Decode overlap

`CNM_TIER_OVERLAP=1` (default) runs a nonempty CUDA hot bank on a persistent worker while the caller copies the cold input and computes CPU cold experts. CPU output remains in pinned host staging until the hot work completes, then is written to the original disjoint slots. The external expert reduction and both banks' tiling/arithmetic policies are unchanged. `CNM_TIER_OVERLAP=0` selects the original serial bank order. Empty banks and CPU-only layers do not create worker jobs.

The input copy uses the caller's CUDA per-thread stream, independent of the hot backend stream. Workspaces and pinned staging are initialized before submission; cold execution does not mutate GPU state. Failure paths wait for the worker before releasing captured inputs. The worker is shared across layers in a runtime and is joined before its backend workspaces are destroyed.

`CNM_TIER_TRACE=1` adds per-layer hot/cold/input/staging/scatter time, measured wall time, their sum and their maximum. Timing includes host launch and synchronization costs; it is not a CUDA-kernel-only duration. Sixteen teacher-forced comparisons against the preserved prefill engine cover both fixtures, fast/exact arithmetic, three texts and 256 decode positions: all 1,060,864 decode values match bitwise, as do prefill logits. Explicit CPU projection failure followed by recovery also passes.
