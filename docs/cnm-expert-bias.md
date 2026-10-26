# Runtime cold-expert bias

`CNM_EXPERT_BIAS=/path/plan.json` adds nonpositive F32 offsets to router logits before softmax and top-k. This experimental option supports uncut Qwen4Exp models with ungrouped softmax routers and no expert-selection bias. It rejects simultaneous `CNM_EXPERT_MASK`. Shared experts and model weights are untouched; the model owns small CPU constant tensors.

Scalar format (IDs name the cold experts):

```json
{"bias": -2, "layers": {"0": [1, 4, 7], "1": [0, 9]}}
```

Per-expert format:

```json
{"layers": {"0": {"1": -1, "4": -4, "7": "-inf"}}}
```

Omitted experts and layers have zero offset. An ID array requires a scalar `bias`. Use the string `"-inf"` because JSON has no infinity literal. Duplicate keys/IDs, invalid indices, positive/nonrepresentable values and fewer than top-k finite experts are rejected. Zero-only vectors add no graph operation.

For logits z and offsets b, p = softmax(z+b). Top-k selects from p; the existing code gathers these same probabilities and, when the model requests normalization, divides them by their selected sum (with the existing F16-minimum clamp). Any existing router scale is then applied. Thus normalized selected weights are mathematically exp(z_i+b_i)/sum_selected exp(z_j+b_j), subject to that existing clamp and floating-point rounding. No new renormalization is introduced.

For a fixed hot set H, `b_i=0` on H and `b_i=-inf` otherwise produces the same F32 constant vector as `CNM_EXPERT_MASK=H`. Both options allocate the same named tensors and use the same `ggml_add` before softmax. All subsequent graph operations, their shapes and their inputs are identical. With finite input logits, at least top-k finite hot experts and a deterministic backend/configuration, the selected IDs, selected probabilities, normalized weights and final logits are bit-identical. This is graph identity, not a claim that a finite large negative bias equals a mask, or that a physically compact cut is exact. Nonfinite input logits can invalidate routing for both paths.

`llama-perplexity` optionally accepts `CNM_ROUTING_TRACE=/path/trace.bin`. Each record is three native-endian I32 values (layer, top-k, tokens), followed by token-major I32 selected expert IDs. It uses an evaluation callback, which can inhibit graph fusion; apply it consistently to all compared runs. Use `--no-warmup` to exclude warmup IDs. The trace includes all forwarded prompt positions, including the context half not scored by KL. Normal executions have no callback or trace overhead.

Validation harness: compile `tests/cnm-expert-bias-trace.cpp` against the built llama/ggml libraries as `WORK/moe-trace`, place the existing deterministic 4-layer, 32-expert CNM Qwen4Exp fixture at `WORK/tiny.gguf`, and put its per-layer hot IDs in `WORK/inputs/tiny-hard.json`. Run `python3 tests/test-cnm-expert-bias.py WORK`. It compares every captured MoE intermediate and all final logits for hard/-inf, unbiased/zero, and scalar/per-expert plans; it also checks finite-bias sensitivity and invalid-plan rejection. The fixture has vocabulary size 259; the four fixed token IDs are 42, 14, 159, 26.

For repeatable CUDA comparisons at context 4096, configure with `-DGGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC=4096`. This exposes the existing source-level dispatch threshold as a CMake cache setting; empty preserves the backend default. The existing radix top-k gather uses atomic counters and can choose different tied indexed-attention entries even when repeating the same plan. With the threshold override, rows up to 4096 columns use bitonic sorting when they fit shared memory. The S14 sweep uses that override for both the unmasked reference and every candidate. The setting does not guarantee determinism at larger widths that fall back to radix.

Set `TRACE_TOKEN_FILE` to a saved reference's token-ID prefix (20-byte `_logits_` header followed by I32 IDs) to run the trace harness on its first 4096 tokens. This mode saves selected IDs, normalized weights and all final F32 logits, and exercises the indexed-attention cutoff that short prompts cannot reach. Compare hard and `-inf` with the same token file and CUDA threshold setting.
