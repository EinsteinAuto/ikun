# muh vs CCCL SM100: Type Specialization Parity Analysis

Generated: 2026-07-31

## Summary

| Algorithm | CCCL SM100 branches | muh BI-V100 branches | Status |
|-----------|--------------------:|---------------------:|--------|
| reduce | 4+2 det = 6 | 4+2 det+1 default = 7 | ✓ PARITY+ |
| scan (lookback) | 7 | 7 (after 35ef79c5) | ✓ PARITY |
| scan (lookahead) | 6 | 6 | ✓ PARITY |
| topk | 1 (dynamic by key_size) | 1 (dynamic by key_size) | ✓ PARITY |
| transform | 1 (dynamic by elem_size) | 1 (dynamic by elem_size) | ✓ PARITY |
| batch_memcpy | 1 (uniform) | 1 (uniform) | ✓ PARITY |
| for | 1 (uniform) | 1 (uniform) | ✓ PARITY |

## Detailed Breakdown

### reduce (tuning_reduce.cuh)

CCCL SM100 specializes by `(accum_type × offset_size)`:
- `int64 + o4`: ipt=15, tpb=512, ipv=2
- `int64 + o8`: ipt=15, tpb=512, ipv=1
- `float32 + o4`: ipt=16, tpb=512, ipv=2
- `float64 + o4`: ipt=16, tpb=640, ipv=1

muh BI-V100 maps these with SMEM-derived corrections:
- `bi100_float32_plus_o4`: tpb=512, ipt=16, ipv=2 (direct match)
- `bi100_float64_plus_o4`: tpb=512, ipt=12, ipv=1 (SM100 ipt=16 → SMEM overflow at 8B, reduced)
- `bi100_int64_plus_o4`: tpb=384, ipt=16, ipv=2 (SM100 tpb=512 → SMEM overflow, reduced threads)
- `bi100_int64_plus_o8`: tpb=384, ipt=16, ipv=1 (same, vec=1 for 8B offset)
- `bi100_det_float32`: tpb=224, ipt=13 (deterministic path, RAKING)
- `bi100_det_float64`: tpb=128, ipt=11 (deterministic path, RAKING)
- `bi100_default`: tpb=256, ipt=16, ipv=4 (fallback)

### scan (tuning_scan.cuh)

CCCL SM100 lookback specializes by `(input_value_size × offset_size)`:
```
offset=4: 1B→(512,18) 2B→(512,13) 4B→(384,22) 8B→(416,23)
offset=8: 1B→(384,14) [2B=skip] 4B→(416,19) 8B→(320,22)
```

muh BI-V100 after commit 35ef79c5:
```
offset=4: 1B→(512,18) 2B→(512,13) 4B→(384,22) 8B→(416,14*)
offset=8: 1B→(384,14) 4B→(416,19) 8B→(320,19*)
```
*items reduced to fit 49152B SMEM

All delay parameters halved (ns×0.5, l2w×0.6) to account for
BI-V100 L2=6MB vs SM100 L2=50MB.

### SMEM Constraint Validation

Every muh bi100_* struct satisfies: `nominal_tile = tpb × ipt × 4 ≤ 49152`

| Struct | tpb | ipt | nominal_tile | Status |
|--------|----:|----:|-------------:|--------|
| bi100_lookback_1B_o4 | 512 | 18 | 36864 | ✓ |
| bi100_lookback_2B_o4 | 512 | 13 | 26624 | ✓ |
| bi100_lookback_4B_o4 | 384 | 22 | 33792 | ✓ |
| bi100_lookback_4B_o8 | 416 | 19 | 31616 | ✓ |
| bi100_lookback_8B_o4 | 416 | 14 | 23296 | ✓ |
| bi100_lookback_8B_o8 | 320 | 19 | 24320 | ✓ |
| bi100_lookback_1B_o8 | 384 | 14 | 21504 | ✓ |
| bi100_float32_plus_o4 | 512 | 16 | 32768 | ✓ |
| bi100_float64_plus_o4 | 512 | 12 | 24576 | ✓ |
| bi100_int64_plus_o4 | 384 | 16 | 24576 | ✓ |
| bi100_int64_plus_o8 | 384 | 16 | 24576 | ✓ |

## Non-Hot-Path Algorithms (20 missing)

These 20 CCCL algorithms have muh/schema/*.yaml but no tuning header.
They are NOT on the vllm inference hot path for Qwen3.6 decode.
If any competition test case triggers them, they will use CCCL defaults
which may cause SMEM overflow on BI-V100 for large types.

Priority to add (by SMEM overflow risk):
1. `radix_sort` (89KB tuning, 161 type dispatches) — HIGH risk
2. `reduce_by_key` (72KB, 134 dispatches) — HIGH risk
3. `select_if` (107KB, 2729 lines) — MEDIUM risk
4. `scan_by_key` (88KB) — MEDIUM risk
5. `unique_by_key` (61KB) — LOW risk
6. Others: LOW risk (small tile sizes, unlikely SMEM overflow)

## SMEM Overflow Detection (from muh/dispatch.py)

Running `python3 muh_kernel_map.py` against all 6 tuning headers
detected 5 lookahead structs with incorrect SMEM estimates:

| Struct | SMEM calc | Limit | Status |
|--------|----------:|------:|--------|
| bi100_lookahead_1B | 162,816 | 49,152 | ✗ OVERFLOW |
| bi100_lookahead_2B | 97,280 | 49,152 | ✗ OVERFLOW |
| bi100_lookahead_4B | 80,896 | 49,152 | ✗ OVERFLOW |
| bi100_lookahead_4B_float | 89,088 | 49,152 | ✗ OVERFLOW |
| bi100_lookahead_8B | 89,088 | 49,152 | ✗ OVERFLOW |

**Root cause**: Lookahead SMEM ≠ `threads × items × elem_bytes`.
The lookahead pipeline uses multi-stage buffering where SMEM =
`(reduce_squad + scan_store_squad) × items × accum_size × stages`.
The simple tile formula overestimates by including lookahead items
that live in registers, not SMEM.

**Impact**: These are currently non-functional on BI-V100 anyway
(lookahead requires SM90+ warpspeed pipeline support). The dispatch
correctly falls back to lookback algorithm. But the values in the
structs are misleading — they should either be corrected or removed.

**Action**: Issue #27 (scan benchmark) TC-04 covers this:
"lookahead 可行性评估 — 测试 ScanAlgorithm::lookahead 是否能在 BI-V100 上编译运行"
