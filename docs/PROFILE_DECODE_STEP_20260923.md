# Decode Step Profile — 2026-09-23

## Environment
- Model: Qwen3.6-35B-A3B (MoE, 64 layers: 40 MoE + 24 linear attention)
- Hardware: 4× Iluvatar BI-V100 (32GB HBM each), TP=4
- Config: `BI100_MOE_COREX_DIRECT_ROUTED=1`, `BI100_MOE_COREX_TOPK_SOFTMAX=1`
- Current throughput: **p10=13.2 p50=13.3 avg=13.3 tok/s**

## Profile (torch.profiler, 1 decode step at step 16, T=1)

```
Self CPU time total: 103.137ms
Self CUDA time total: 54.889ms
```

### Top-25 CUDA Kernels

| # | Kernel | CUDA time | % | Calls | Avg/call | Category |
|---|--------|-----------|---|-------|----------|----------|
| 1 | fenceWaitE (NCCL barrier) | 10.877ms | 19.8% | 972 | 11μs | **COMM** |
| 2 | general_gemm (large) | 7.677ms | 14.0% | 160 | 48μs | GEMM |
| 3 | fenceOps (NCCL sync) | 5.108ms | 9.3% | 486 | 11μs | **COMM** |
| 4 | Memcpy DtoD | 4.464ms | 8.1% | 496 | 9μs | **COPY** |
| 5 | gdn_packed_decode | 2.998ms | 5.5% | 30 | 100μs | GDN |
| 6 | kernelCopy | 2.860ms | 5.2% | 243 | 12μs | **COPY** |
| 7 | elementWise (misc) | 2.801ms | 5.1% | 243 | 12μs | **COPY** |
| 8 | aten::copy_ | 2.409ms | 4.4% | 192 | 13μs | **COPY** |
| 9 | general_gemm (small) | 2.372ms | 4.3% | 40 | 59μs | GEMM |
| 10 | cudaLaunchKernel | 2.075ms | 3.8% | 3072 | 0.7μs | OVERHEAD |
| 11 | direct_w13_kernel | 1.533ms | 2.8% | 40 | 38μs | MoE |
| 12 | aten::mul | 1.353ms | 2.5% | 93 | 15μs | ELEM |
| 13 | fused_add_rms_norm | 1.342ms | 2.4% | 80 | 17μs | NORM |
| 14 | topk_gating_softmax | 1.176ms | 2.1% | 40 | 29μs | MoE |
| 15 | direct_w2_reduce_kernel | 1.067ms | 1.9% | 40 | 27μs | MoE |
| 16 | aten::mean | 1.008ms | 1.8% | 30 | 34μs | GDN |
| 17 | reduce_kernel | 1.008ms | 1.8% | 30 | 34μs | GDN |
| 18 | aten::add | 830μs | 1.5% | 71 | 12μs | ELEM |
| 19 | silu kernel | 778μs | 1.4% | 40 | 19μs | MoE |
| 20 | silu_and_mul | 765μs | 1.4% | 40 | 19μs | MoE |

### Category Summary

| Category | CUDA time | % of total | Description |
|----------|-----------|------------|-------------|
| **COMM (NCCL)** | 15.985ms | **29.1%** | TP=4 all-reduce fences |
| **COPY (memcpy+convert)** | 12.534ms | **22.8%** | DtoD memcpy, dtype casts, tensor copies |
| GEMM (non-MoE) | 10.049ms | 18.3% | Attention q/k/v/o projections |
| **MoE (fused)** | 5.319ms | 9.7% | topk + w13 + act + w2_reduce |
| GDN | 5.014ms | 9.1% | Linear attention decode |
| NORM | 1.342ms | 2.4% | fused_add_rms_norm |
| ELEM | 2.183ms | 4.0% | add, mul, elementwise |
| OVERHEAD | 2.075ms | 3.8% | cudaLaunchKernel host-side |

### Bottleneck Analysis

1. **NCCL communication (29%)** is the #1 bottleneck. Each TP=4 all-reduce
   triggers fence+sync pairs. 972+486=1458 fence ops per decode step.
   Optimization: fuse all-reduce with preceding GEMM (ix_full_bridge_fused_ar.so
   exists in prebuilt but is not wired into the hot path).

2. **Tensor copy/conversion (23%)** is #2. 496 DtoD memcpy + 243 kernelCopy +
   192 aten::copy_ = 931 copy ops per step. Root causes likely include:
   - dtype casts (fp32↔fp16) around NCCL all-reduce
   - non-contiguous tensor reshapes triggering implicit copies
   - KV cache slot_mapping conversions

3. **MoE is already optimized (9.7%)**. direct_routed.w13 (1.5ms) +
   w2_reduce (1.1ms) + topk (1.2ms) + activation (1.5ms) = 5.3ms.
   Further MoE optimization has <1% total impact potential.

4. **GDN decode (9.1%)** — gdn_packed_decode at 100μs/call is the heaviest
   single kernel. 30 calls = 30 linear attention layers.

### Next Optimization Targets (by estimated impact)

| Target | Est. saving | Approach |
|--------|-------------|----------|
| Fused all-reduce | 5-10ms (10-18%) | Wire ix_full_bridge_fused_ar into attention output projection |
| Reduce dtype casts | 3-5ms (5-9%) | Eliminate fp32↔fp16 round-trips around NCCL |
| Reduce DtoD copies | 2-3ms (4-5%) | Make tensors contiguous in-place, avoid reshape copies |
