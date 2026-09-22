# CCCL ↔ EngineX Architecture Alignment

## Executive Summary

EngineX ships precompiled `.so` kernels — **zero `.cu` source files** are available.
The optimization surface is Python runtime params + Triton JIT kernels.

CCCL's value is NOT parameter values. It's the **architectural patterns** that
tell us which parameters matter, what their constraints are, and why.

## Three-Layer Architecture Mapping

### CCCL Layer → EngineX Layer → What We Control

| CCCL | EngineX | Controllable? |
|------|---------|--------------|
| `dispatch_reduce.cuh` (GridEvenShare work distribution) | `paged_attn.py` (V1/V2 dispatch, _PARTITION_SIZE) | **Yes** — Python runtime |
| `kernel_reduce.cuh` (kernel entry, atomic vs 2-phase) | `_C_flashattention.so` (paged_attention_v1/v2) | **No** — precompiled |
| `agent_reduce.cuh` (tile consumption, vectorized load) | internal to `.so` | **No** — precompiled |
| `tuning_reduce.cuh` (policy_selector) | `_custom_ops.py` (SMEM=49152) | **Partially** — SMEM limit |
| `dispatch_transform.cuh` (spread_out_items_per_thread) | `rmsnorm_kernels.py` (BLOCK_SIZE heuristic) | **Yes** — Triton autotune |
| `kernel_scan.cuh` (lookback/lookahead) | `prefix_prefill.py` (BLOCK_M/N, num_warps) | **Yes** — Triton config |
| `dispatch_scan.cuh` (tile init + scan kernel) | `triton_splitk.py` (split-K attention) | **Yes** — Triton config |

### Key CCCL Patterns We Apply

1. **GridEvenShare** (`grid_even_share.cuh`):
   - `max_blocks = sm_occupancy × sm_count × subscription_factor`
   - BI-V100: 1 × 16 × 5 = 80 max CTAs
   - Applied to: `paged_attn.py` _BI100_TARGET_TILES, V1/V2 threshold

2. **Compound Reduce** (`summary_statistics.cu`):
   - Accumulator = struct{m, l, o} (max, sum_exp, weighted_output)
   - unary_op: score_tile → partial softmax stats
   - binary_op: online softmax merge with correction factor
   - Applied to: `_forward_decode_pytorch` online softmax loop

3. **Two-Phase Reduce** (`kernel_reduce.cuh`):
   - Phase 1: each CTA reduces a partition → `d_block_reductions[blockIdx.x]`
   - Phase 2: single CTA reduces all block results
   - Applied to: paged_attention_v2 partition → merge

4. **spread_out_items_per_thread** (`dispatch_transform.cuh`):
   - Reduce items/thread when there aren't enough items to fill all SMs
   - `items = min(max, ceil_div(N, sm_count × threads × occupancy))`
   - Applied to: Triton kernel BLOCK_SIZE selection

5. **Lookback Delay** (`tuning_scan.cuh`):
   - 16 SMs → ~32 concurrent CTAs → tile_state fits in 6MB L2
   - Inter-CTA contention near zero → no_delay optimal
   - Applied to: scan-based operations (softmax denominator)

## BI-V100 Hardware Profile (Confirmed)

| Property | Value | Impact |
|----------|-------|--------|
| SM count | 16 | 3.1x fewer CTAs than spec (50) → larger tiles per CTA |
| SMEM | 48KB | Same as NVIDIA → CCCL SMEM constraints apply directly |
| HBM BW | 900 GB/s | BW/SM = 56 GB/s ≈ B200 level → bytes_in_flight = 64KB |
| L2 cache | 6MB | 8.3x smaller than SM100 → faster coherence, no_delay wins |
| Warp size | 32 | Same as NVIDIA → CCCL warp-level primitives work |

## Files Inventory

### Precompiled (CANNOT modify)
- `_C_flashattention.so` — paged_attention_v1, paged_attention_v2, reshape_and_cache
- `_C.so` — xformers attention backends
- `libtriton.so` — Triton compiler/runtime

### Triton JIT (CAN modify)
- `pkgs/triton/ops/flash_attention.py` — Flash Attention (head_dim ≤ 128 only)
- `pkgs/xformers/ops/fmha/triton_splitk.py` — Split-K attention (V2 pattern)
- `pkgs/xformers/ops/triton/rmsnorm_kernels.py` — RMSNorm
- `pkgs/xformers/ops/triton/rope_padded_kernels.py` — RoPE

### Python runtime (CAN modify)
- `paged_attn.py` — V1/V2 dispatch, _PARTITION_SIZE, decode fallback
- `prefix_prefill.py` — Prefill attention BLOCK_M/N/NUM_WARPS
- `vllm/_custom_ops.py` — SMEM=49152 (already fixed from 32768)
- `computility-run.yaml` — Server launch params
