# Decode Step Profile — 2026-09-23

## Environment
- Model: Qwen3.6-35B-A3B (MoE, 64 layers: 40 MoE + 24 linear attention)
- Hardware: 4× Iluvatar BI-V100 (32GB HBM each), TP=4
- Config: BI100_MOE_COREX_DIRECT_ROUTED=1, BI100_MOE_COREX_TOPK_SOFTMAX=1
- Current throughput: **p10=13.2 p50=13.3 avg=13.3 tok/s**
- Source: torch.profiler chrome trace (5.4MB, 19483 events, step 16 T=1)

## Decode Step Breakdown (80.2ms per token)

```
  25.3%   20274μs  ████████████  NCCL (fence+sync)
  13.8%   11033μs  ██████        Memcpy+Copy (DtoD)
  13.3%   10702μs  ██████        GEMM (attention q/k/v/o projections)
  11.1%    8873μs  █████         Elementwise (add, mul, legacy)
   7.3%    5880μs  ███           aten::to/_to_copy (dtype casts)
   4.6%    3694μs  ██            aten::item (GPU→CPU sync, 10 calls)
   4.5%    3640μs  ██            MoE fused (direct_w13 + w2_reduce + topk)
   3.4%    2759μs  █             GDN decode (linear attention)
   3.1%    2477μs  █             cudaLaunchKernel (host dispatch overhead)
   2.4%    1921μs  █             Activation (silu_and_mul)
   2.2%    1732μs  █             Norm (fused_add_rms_norm)
   1.5%    1219μs                Reduce
```

## Key Observations

### 1. NCCL Communication is #1 Bottleneck (25.3%, 20.3ms)
- 972× fenceWait + 486× fenceOps = 1458 fence operations per step
- Each TP=4 all-reduce after attention output and MoE output
- **Optimization**: `ix_full_bridge_fused_ar.so` exists in prebuilt but is
  NOT wired into the hot path. This fuses GEMM + all-reduce into one op,
  eliminating the fence pair (~50% of NCCL cost, ~10ms saving, ~12% speedup).

### 2. Memcpy + dtype Casts = 18.5% (16.9ms)
- 496 DtoD memcpy (4.5ms) + 243 kernelCopy (2.9ms) + 192 copy_ (2.4ms)
- 333 aten::to + 152 _to_copy = dtype cast round-trips (5.9ms)
- Root causes:
  - fp32↔fp16 conversions around NCCL all-reduce
  - Non-contiguous tensor reshapes triggering implicit copies
  - router_logits.float() and topk_weights.to(hidden_states.dtype)

### 3. Elementwise ops = 11.1% (8.9ms)
- 243× legacy::elementWiseKernel (2.8ms) — likely NCCL reduce scatter
- 93× aten::mul (1.4ms), 71× aten::add (1.1ms)
- Many are fuse-able with adjacent operations

### 4. aten::item GPU→CPU Sync = 4.6% (3.7ms)
- 10 calls × 370μs average
- Each .item() forces GPU→CPU synchronization
- Sources: likely GDN prefix cache operations and EP ghost expert filtering

### 5. MoE is Already Optimized (4.5%, 3.6ms)
- direct_w13: 1.4ms (40 calls, 35μs each)
- direct_w2_reduce: 1.1ms (40 calls, 27μs each)
- topk_gating_softmax: 1.2ms (40 calls, 29μs each)
- Silu activation: 1.5ms (included in MOE pipeline)
- **No further MoE optimization will have meaningful impact.**

### 6. GDN Decode (3.4%, 2.8ms)
- 30× gdn_packed_decode_kernel at 92μs each
- 30 linear attention layers, already using corex fused kernel
- Limited room for improvement

## CUDA Kernel Detail (Top 20)

| # | Kernel | Time | % | Calls | Avg | Category |
|---|--------|------|---|-------|-----|----------|
| 1 | fenceWaitE | 10.9ms | 15.2% | 972 | 11μs | NCCL |
| 2 | general_gemm (large) | 7.7ms | 10.8% | 160 | 48μs | GEMM |
| 3 | fenceOps | 5.1ms | 7.1% | 486 | 11μs | NCCL |
| 4 | Memcpy DtoD | 4.5ms | 6.2% | 496 | 9μs | COPY |
| 5 | gdn_packed_decode | 2.8ms | 3.8% | 30 | 92μs | GDN |
| 6 | kernelCopy | 2.9ms | 4.0% | 243 | 12μs | COPY |
| 7 | elementWise (legacy) | 2.8ms | 3.9% | 243 | 12μs | ELEM |
| 8 | general_gemm (small) | 2.3ms | 3.3% | 40 | 59μs | GEMM |
| 9 | direct_w13_kernel | 1.4ms | 2.0% | 40 | 35μs | MoE |
| 10 | fused_add_rms_norm | 1.4ms | 1.9% | 80 | 17μs | NORM |
| 11 | topk_gating_softmax | 1.2ms | 1.6% | 40 | 29μs | MoE |
| 12 | direct_w2_reduce | 1.1ms | 1.5% | 40 | 27μs | MoE |
| 13 | reduce_kernel | 1.0ms | 1.3% | 30 | 34μs | GDN |
| 14 | silu elementwise | 0.8ms | 1.1% | 40 | 19μs | ACT |
| 15 | silu_and_mul | 0.8ms | 1.0% | 40 | 19μs | ACT |

## Optimization Priority (by estimated impact)

| Priority | Target | Current | Est. saving | Approach |
|----------|--------|---------|-------------|----------|
| **P0** | Fused GEMM+all-reduce | 20.3ms NCCL | 8-10ms (10-12%) | Wire `ix_full_bridge_fused_ar.so` into attention output + MoE output projections |
| **P1** | Eliminate dtype casts | 5.9ms casts | 3-4ms (4-5%) | Keep tensors in fp16 through NCCL; remove .float()/.to(dtype) round-trips |
| **P2** | Reduce DtoD memcpy | 7.4ms copy | 2-3ms (3%) | Make tensors contiguous in-place; avoid reshape that triggers copy |
| **P3** | Eliminate aten::item | 3.7ms sync | 2-3ms (3%) | Replace .item() with GPU-side conditionals where possible |
| P4 | Fuse elementwise ops | 8.9ms | 1-2ms (1-2%) | Custom fused kernels for add+mul chains |
