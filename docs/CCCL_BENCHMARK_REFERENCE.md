# CCCL benchmark reference data — extracted from 27 tuning headers

> Auto-extracted from `cccl_upstream/cub/cub/device/dispatch/tuning/*.cuh`
> 199 benchmark annotations across 27 files, 286 template specializations
> This is the data NVIDIA spent millions of GPU-hours generating on A100/H100/B200.
> muh needs equivalent data for BI-V100.

## Summary

| Algorithm | CCCL lines | muh lines | muh/CCCL | Benchmarks | Specializations | Priority |
|-----------|-----------|-----------|----------|------------|-----------------|----------|
| radix_sort | 2382 | 223 | 9% | 70 | 0 | P0 — sampling hot path |
| select_if | 2730 | 460 | 17% | 0* | 37 | P0 — top-p filtering |
| scan_by_key | 2009 | 146 | 7% | 30 | 58 | P1 — softmax denominator |
| reduce_by_key | 1736 | 172 | 10% | 32 | 63 | P1 — score aggregation |
| unique_by_key | 1540 | 167 | 11% | 29 | 58 | P1 — KV cache dedup |
| scan | 1526 | 371 | 24% | 16 | 12 | P0 — prefix sum hot path |
| three_way_partition | 789 | 100 | 13% | 0 | 0 | P1 — token classification |
| rle_non_trivial_runs | 692 | 69 | 10% | 8 | 15 | P1 — attention mask |
| segmented_sort | 641 | 190 | 30% | 0 | 0 | P1 — per-seq token ranking |
| rle_encode | 627 | 64 | 10% | 4 | 15 | P1 — mask compression |
| transform | 550 | 186 | 34% | 0 | 0 | P1 — RMSNorm/SiLU/RoPE |
| reduce | 479 | 298 | 62% | 6 | 2 | P0 — attention score reduce |
| histogram | 364 | 77 | 21% | 4 | 3 | P1 — repetition penalty |
| topk | 122 | 114 | 93% | 0 | 0 | P0 — sampling core |

*select_if has 37 muh benchmark annotations from the 3-dimension restore fix

## Decode hot path — benchmark reference values

### reduce (Output TPS × 16.796 = 83% weight)

CCCL SM100 benchmarks (the target we need to match or beat on BI-V100):

```
# float32, offset=4, accum=4:
ipt_16.tpb_512.ipv_2  1.061295  1.000000  1.065478  1.167139   geo=1.072

# float64, offset=4, accum=8:
ipt_16.tpb_640.ipv_1  1.017834  1.000000  1.015835  1.057092   geo=1.023

# int64, offset=4, accum=8:
ipt_15.tpb_512.ipv_2  1.019887  1.000000  1.017636  1.058036   geo=1.024

# int64, offset=8, accum=8:
ipt_15.tpb_512.ipv_1  1.019414  1.000000  1.017218  1.057143   geo=1.023

# Deterministic float32 (SM90):
ipt_13.tpb_224        1.107188  1.009709  1.097114  1.316820   geo=1.127

# Deterministic float64 (SM86):
ipt_11.tpb_128        1.232089  1.002124  1.245336  1.582279   geo=1.250
```

Current muh BI-V100 values (theoretical, NOT benchmarked):
- float32: items=24, threads=512, vec=2  (CCCL SM100: items=16, threads=512, vec=2)
- float64: items=16, threads=384, vec=2  (CCCL SM100: items=16, threads=640, vec=1)
- det float32: items=32, threads=384     (CCCL SM90: items=13, threads=224)
- det float64: items=16, threads=384     (CCCL SM86: items=11, threads=128)

**Critical gap**: muh items are 1.5-2.5× CCCL SM100 values. Rationale was "16 SMs need
larger tiles to compensate for fewer CTAs." This MUST be validated on real hardware.
If register pressure causes occupancy drop, the 2.5× items advantage evaporates.

### scan (softmax denominator, TTFT impact)

CCCL SM100 benchmarks with full delay tuning:

```
# int8, offset=4:
ipt_18.tpb_512.ns_768.dcid_7.l2w_820.trp_1.ld_0   1.189  1.006  1.173  1.305   geo=1.163

# int16, offset=4:
ipt_13.tpb_512.ns_1384.dcid_7.l2w_720.trp_1.ld_0  1.128  1.003  1.120  1.308   geo=1.135

# float32, offset=4:
ipt_22.tpb_384.ns_1904.dcid_6.l2w_830.trp_1.ld_0  1.148  0.997  1.140  1.463   geo=1.182

# float32, offset=8:
ipt_19.tpb_416.ns_956.dcid_7.l2w_550.trp_1.ld_1   1.146  0.994  1.137  1.456   geo=1.178

# float64, offset=4:
ipt_23.tpb_416.ns_772.dcid_5.l2w_710.trp_1.ld_0   1.089  1.016  1.086  1.265   geo=1.111

# float64, offset=8:
ipt_22.tpb_320.ns_328.dcid_2.l2w_965.trp_1.ld_0   1.080  1.000  1.076  1.249   geo=1.100

# SM90 int128:
tpb_576.ipt_21.ns_860.l2w_630                      (no speedup data in comment)
```

Key tuning dimensions absent from muh:
- `dcid` (delay constructor ID): 8 variants (0-7), each a different backoff strategy
- `l2w` (L2 write latency in ns): BI-V100 L2=6MB vs SM100 L2=50MB — needs recalibration
- `ns` (delay in nanoseconds): range 64-2044ns across all scan benchmarks
- `trp` (transpose): 0=DIRECT, 1=WARP_TRANSPOSE
- `ld` (load modifier): 0=LOAD_DEFAULT, 1=LOAD_CA/LOAD_LDG

### radix_sort (70 benchmarks — most data-rich algorithm)

Top-performing SM100 configurations:

```
# Large key (8B), offset=4:
ipt_14.tpb_320  1.256  1.000  1.228  1.487   geo=1.231

# Small key (1B), offset=4:
ipt_20.tpb_512  1.013  0.968  1.016  1.048   geo=1.011

# Medium key (4B), offset=4:
ipt_21.tpb_512  1.003  0.995  1.004  1.019   geo=1.005
```

Qwen3.6 sampling: vocab_size=152064, logits are float32 (4B keys).
Bits per pass: sizeof(float32)=4 → bits_per_pass=11 → ⌈32/11⌉=3 passes.
Each pass: 2^11=2048 histogram bins × sizeof(int)=4 = 8KB SMEM for histogram.
Total sort SMEM ≈ 8KB + threads×items×sizeof(float32) staging.

## Delay algorithms reference (for lookback-based algorithms)

| dcid | Algorithm | Description |
|------|-----------|-------------|
| 0 | no_delay | No delay between lookback iterations |
| 1 | fixed_delay | Fixed ns delay |
| 2 | exponential_backoff | Double delay each retry |
| 3 | exponential_backoff_jitter | Backoff + random jitter |
| 4 | exponential_backoff_jitter_window | Backoff + jitter + window |
| 5 | exponential_backon_jitter_window | Increase delay (backon) + jitter + window |
| 6 | exponential_backon_jitter | Increase delay + jitter |
| 7 | exponential_backon | Increase delay monotonically |

BI-V100 implications:
- L2 cache 6MB (SM100: 50MB) → tile_state fits in L2 for fewer concurrent CTAs
- 16 SMs → max 32 concurrent tiles → lower contention → shorter delays likely optimal
- bandwidth_per_SM=56GB/s (SM100: 54GB/s) → similar per-SM behavior
- Recommended starting point: dcid=7 (exponential_backon) with ns×0.5, l2w×0.6 scaling
