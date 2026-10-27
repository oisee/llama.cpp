# Exact expert tiers and CUDA MMQ

`CNM_TIER_CUDA_ARITH=1 CNM_TIER_CUDA_SFU=1` must cover both the cold CPU body and the hot CUDA body. A correct cold body alone does not make a split model match its unsplit CUDA reference.

MMQ chooses Stream-K decomposition from its output tile count and the device SM count. Packing selected pairs into a smaller hot bank changes that tile count. The unsplit reference can compute a complete K sum in one block while the hot bank computes partial sums and adds them in a fixup kernel. This changes floating-point rounding before activation quantization.

A real K192p four-layer prefix reproduces this on an RTX 2070. Keeping 29 hot experts in layer 1 gives 145 gate/up output tiles on 36 SMs, below the 90 percent tiling threshold. The first mismatch is in the hot IQ2_S gate/up projections. Layer input, router, selected IDs, and all cold projections still match. Changing only the CUDA tiling policy removes the differences in every captured stage and in the final logits.

The existing `MUL_MAT_ID` logical-batch hint identifies packed bank graphs. In exact arithmetic mode these MMQ operations launch one complete K sum per output tile. Ordinary CUDA operations and the default tier mode retain their existing policy. The fix does not change routing, bank membership, quantization, activation functions, expert weighting, or the expert reduction graph.

`CNM_TIER_TRACE=1` logs the effective device, exact arithmetic/SFU state, AVX2 availability, and MMQ launch geometry. The geometry line includes the physical tile count, occupancy heuristic, block count, fixup decision, and `no_split_k` flag.

The Ada SFU admission uses the existing device-calibrated reciprocal and exponential tables. Qualifying those tables does not qualify an entire model graph. The full 48-layer, two-device Ada gate must be rerun with CUDA 13 after this change, including the matched native `--n-cpu-moe 24` comparison. No Ada model execution is claimed by the RTX 2070 reproduction.
