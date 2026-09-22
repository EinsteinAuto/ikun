# CCCL Reduce Architecture Notes

> Source: `dispatch_reduce.cuh`, `kernel_reduce.cuh`, `agent_reduce.cuh`, `tuning_reduce.cuh`, `util_arch.cuh`
> Read: 2026-08-04 by Claude from CCCL upstream in project_6/cccl_upstream/

## Key Architecture

### Two-pass dispatch (dispatch_reduce.cuh)

```
num_items <= single_tile.threads * single_tile.items
  → SingleTile: one CTA, one kernel launch
  → DeviceReduceSingleTileKernel(d_in, d_out, num_items, ...)

num_items > single_tile threshold
  → Pass 1: DeviceReduceKernel — N CTAs each reduce their share → d_block_reductions[N]
  → Pass 2: DeviceReduceSingleTileKernel — 1 CTA reduces d_block_reductions[N] → d_out
```

Grid size for Pass 1: `max_blocks = sm_occupancy * sm_count * subscription_factor(5)`
For BI-V100: `2 * 16 * 5 = 160 blocks` max.
Each block processes `ceil(num_items / 160)` elements.

### Tile consumption (agent_reduce.cuh)

**Critical: tile data is in registers, NOT SMEM.**

```cpp
AccumT items[ITEMS_PER_THREAD];  // <-- register array, per-thread
// ... load from global memory ...
thread_aggregate = ThreadReduce(items, reduction_op);  // per-thread reduction

// Only SMEM used:
BlockReduce(temp_storage.reduce).Reduce(thread_aggregate, reduction_op);
```

`TempStorage` = `BlockReduce::TempStorage` ≈ threads * sizeof(AccumT) bytes.
NOT threads * items * sizeof(AccumT).

### Vectorized loads

```cpp
ATTEMPT_VECTORIZATION = (vec_size > 1) && (ITEMS_PER_THREAD % vec_size == 0)
    && is_pointer<InputIteratorT>
    && (is_primitive<InputT> || is_trivially_relocatable<InputT>)
    && sizeof(InputT) <= 8;
```

For fp32 scores: vec_size=2 → loads 8 bytes (2 floats) per instruction.
For fp16 KV cache: vec_size=4 → loads 8 bytes (4 halfs) per instruction.

### scale_mem_bound vs scale_reg_bound (util_arch.cuh)

Two scaling functions with different constraints:

**scale_mem_bound** (memory-bound algorithms: reduce, transform):
- items = clamp(nominal * 4 / type_size, 1, nominal * 2)  ← allows 2x expansion
- threads = min(nominal, round_up(48KB / (type_size * items), 32))

**scale_reg_bound** (register-bound algorithms: scan with complex state):
- items = max(1, nominal * 4 / max(4, type_size))  ← no expansion past nominal
- threads = min(nominal, ceil_div(48KB / (type_size * items), 32) * 32)

Key difference: scale_reg_bound uses `max(4, type_size)` preventing items from exceeding nominal for small types, and uses `ceil_div` instead of `round_up` for thread count. Both use 48KB as the cap, but this limits REGISTER PRESSURE (spill to local memory), not actual SMEM usage.

## Impact on muh tuning

### Our SMEM model was wrong for reduce

`test_smem_safety.py` and `check_smem()` in `muh_kernel_map.py` compute
`tile_bytes = threads * items * type_size` and check against 49152.

This is the scale_mem_bound cap, NOT the actual SMEM usage. The actual SMEM
for reduce is approximately `threads * max(sizeof(AccumT), 4)` bytes — about
2-8 KB, not 32-49 KB.

CCCL's SM100 float64 tuning uses `threads=640, items=16` → scale_mem_bound
"tile" = 640*16*8 = 81920 > 49152. But this doesn't overflow SMEM — it only
means scale_mem_bound will cap threads down. The actual kernel SMEM usage
with threads=640 is only ~5120 bytes.

### Our float64/int64 tuning may be too conservative

We use threads=384 items=16 for float64, capped by scale_mem_bound. CCCL
uses threads=640 items=16 on SM100. The question is whether BI-V100's register
file (255 regs/thread) can hold 16 float64 items without spilling.

16 * 8 = 128 bytes = 32 registers per thread for tile data alone.
With overhead (thread_aggregate, loop variables, etc.), ~40 registers/thread.
255 max registers → no spill risk. threads=640 may be safe on BI-V100.

**TODO**: Benchmark threads=640 items=16 for float64 on BI-V100.

### paged_attn.py forces V1

Line 99: `use_v1 = True` overrides V1/V2 heuristic. V2 is completely disabled.
For 100K token sequences, V1 makes one CTA iterate over all KV blocks — bad
for latency. V2 would partition the work and reduce across partitions, which
is exactly CCCL's two-pass pattern.

**TODO**: Re-enable V2 for max_seq_len > 8192. Use muh's partition_size tuning.

### _PARTITION_SIZE = 512 is hardcoded

Not controlled by muh. Should be tunable: larger partition = fewer blocks =
less overhead but more work per block. Optimal value depends on SM count.
For 16 SMs: partition_size=1024 may be better (fewer partitions to reduce).

---

## CCCL Scan Architecture (dispatch_scan.cuh)

> Added: 2026-08-04

### Two algorithm paths

**Lookback** (all GPUs including BI-V100):
- Each CTA processes one tile, uses `ScanTileState` in global memory for inter-CTA communication
- Lookback delay policy controls how aggressively CTAs poll predecessors
- SMEM: static only (`__shared__`), passed as `0` dynamic SMEM
- BI-V100 optimal: `no_delay` (dcid=0) because 16 SMs → ~32 CTAs → tile_status fits in 6MB L2

**Lookahead** (SM100+ only, PTX ISA >= 860):
- Pipeline-based with `__pipeline_memcpy_async` and bulk copy
- Uses dynamic SMEM with auto-selected `num_stages`
- **Not available on BI-V100** — requires NVIDIA PTX ISA 860+ instructions
- All lookahead structs in our tuning_scan.cuh can remain empty shells

### ScanTileState allocation

Scan requires `d_temp_storage` for tile status descriptors:
```
tile_size = threads * items
num_tiles = ceil(num_items / tile_size)
temp_bytes = tile_state.AllocationSize(num_tiles)
```

For BI-V100 with 100K tokens and tile_size=384*22=8448:
num_tiles = ceil(100000/8448) = 12 tiles → negligible temp storage.

### Grid size for scan

Lookback scan launches `num_tiles` blocks (one per tile), NOT `sm_count * subscription_factor`.
This is different from reduce, which uses `GridEvenShare`.
For scan, every CTA processes exactly one tile and communicates with neighbors.

With 12 tiles on 16 SMs: all tiles fit in one wave, zero lookback contention.
This is why `no_delay` works on BI-V100 — the entire scan completes in a single wave.

### Lookahead num_stages optimization (SM100 only)

CCCL dynamically selects pipeline depth:
```cpp
max_stages = ceil(num_items / (sm_count * tile_size)) + 1
while (smem_for_stages(num_stages+1) <= max_dynamic_smem) num_stages++
```

For BI-V100 this is irrelevant (no pipeline support), but the formula shows
NVIDIA's strategy: match pipeline depth to problem size / SM count ratio.

---

## CCCL Scan Agent Architecture (agent_scan.cuh)

> Added: 2026-08-04

### Critical difference from reduce: scan DOES use SMEM for tile data

```cpp
union _TempStorage {
    BlockLoadT::TempStorage load;       // SMEM for WARP_TRANSPOSE load
    BlockStoreT::TempStorage store;     // SMEM for WARP_TRANSPOSE store
    struct {
        TilePrefixCallbackOpT::TempStorage prefix;  // lookback state
        BlockScanT::TempStorage scan;                // block scan
    } scan_storage;
};
```

This is a **union** — load, store, and scan share the same SMEM, used
in phases separated by `__syncthreads()`. Actual SMEM = max of three.

For `BLOCK_LOAD_WARP_TRANSPOSE`:
  load_smem ≈ threads * items * sizeof(InputT)

For `BlockScan`:
  scan_smem ≈ threads * sizeof(AccumT) + prefix_callback

The dominant term is load/store: threads * items * type_size.

**Conclusion: our SMEM constraint `threads * items * type_size ≤ 48KB`
is CORRECT for scan but WRONG (overly conservative) for reduce.**

### Tile processing flow

```
1. BlockLoad(SMEM).Load(d_in + offset, items[ITEMS_PER_THREAD])
2. __syncthreads()
3. BlockScan(SMEM).Scan(items, ..., prefix_op)  // lookback here
4. __syncthreads()
5. BlockStore(SMEM).Store(d_out + offset, items)
```

Each CTA processes exactly one tile (tile_idx = start_tile + blockIdx.x).
Inter-CTA communication happens in step 3 via TilePrefixCallbackOp,
which reads predecessor tile states from global memory (the lookback).

### Lookback protocol (TilePrefixCallbackOp)

For tile k, the callback:
1. Sets own tile state to PARTIAL with local aggregate
2. Looks back at tiles k-1, k-2, ... until finding an INCLUSIVE prefix
3. Combines found prefix with local aggregate → own INCLUSIVE prefix
4. Sets own tile state to INCLUSIVE

The LookbackDelayPolicy controls how aggressively step 2 polls:
- no_delay: spin immediately (best when few CTAs, e.g., BI-V100 16 SMs)
- exponential_backon: exponentially increase delay between polls
  (best when many CTAs compete for L2 coherence, e.g., SM100 148 SMs)

### Impact on muh tuning

For reduce: items_per_thread can be larger because SMEM only stores
~threads*4 bytes for BlockReduce. The 48KB cap prevents register spill.

For scan: items_per_thread is genuinely SMEM-limited because
BlockLoad/BlockStore use threads*items*type_size bytes of SMEM.

This means:
- tuning_reduce.cuh: consider increasing items beyond scale_mem_bound cap
  for better ILP, especially for small types (fp16, int8)
- tuning_scan.cuh: current values are correctly SMEM-bounded, don't increase

---

## CCCL Lookback Delay Protocol (single_pass_scan_operators.cuh)

> Added: 2026-08-04

### delay() has a GridThreshold gate — renders delay_ns IRRELEVANT on BI-V100

```cpp
template <int Delay, unsigned int GridThreshold = 500>
void delay() {
    if (Delay > 0) {
        if (gridDim.x < GridThreshold)  // <-- THIS IS THE KEY
            __threadfence_block();       // small grid: just fence
        else
            __nanosleep(Delay);          // large grid: actual sleep
    }
}
```

GridThreshold defaults to 500. BI-V100 scan with 100K fp32 elements:
  tile_size = 384 * 22 = 8448
  num_tiles = ceil(100000/8448) = 12 blocks
  12 << 500 → ALL delay calls reduce to __threadfence_block()

This means: on BI-V100, the entire delay infrastructure (ns, dcid, l2w)
is a no-op. no_delay, fixed_delay(1904), exponential_backon_jitter(1904,830)
ALL execute the same __threadfence_block(). 

### Why our benchmark showed no_delay as "best"

Not because no_delay is a better strategy, but because ALL strategies
produce identical machine code on a 12-block grid. The ~3% speedup
difference between dcid=0 and dcid=6 in bench_bi100.py is noise.

### Impact on tuning_scan.cuh

All scan delay parameters (delay_ns, delay_l2w, delay algorithm) can be
simplified to no_delay for BI-V100. The heuristic scaling (ns×0.5, l2w×0.6)
was both wrong AND irrelevant — the values don't matter because they're
never used as nanosleep arguments.

The only scan tuning parameters that matter on BI-V100 are:
  - threads_per_block (affects SMEM usage and occupancy)
  - items_per_thread (affects SMEM usage and ILP)
  - load_algorithm (WARP_TRANSPOSE vs DIRECT)
  - scan_algorithm (RAKING vs WARP_SCANS)
  - load_modifier (DEFAULT vs LDG)

### summary_statistics.cu → paged_attention V2 compound reduce

The Welford parallel merge in summary_statistics.cu is structurally 
identical to paged_attention V2's cross-partition reduce:

| summary_statistics | paged_attention V2 |
|---|---|
| summary_stats_data{n,min,max,mean,M2,M3,M4} | partition_result{max_logit, exp_sum, output_partial} |
| unary_op: x → {n=1, mean=x, M2=0, ...} | per-partition attention: Q@K^T → softmax → V·weights |
| binary_op: Welford parallel merge | online softmax merge: rescale by exp(old_max - new_max) |
| thrust::transform_reduce | DeviceReduce pass 2 |

The compound accumulator size for V2 is sizeof(float)*3 = 12 bytes.
This affects tuning: scale_mem_bound(512, 16, 12) → different items/threads
than a simple float32 reduce.

---

## CCCL Transform Architecture (tuning_transform.cuh)

> Added: 2026-08-04

### 4 algorithms, only 2 available on BI-V100

| Algorithm | Requirement | BI-V100 |
|---|---|---|
| prefetch | universal | ✓ available |
| vectorized | contiguous + trivially_relocatable + power-of-2 size | ✓ available |
| ldgsts | SM80+ cp.async (NVIDIA-specific PTX) | ✗ |
| ublkcp | SM90+ bulk copy (NVIDIA-specific PTX) | ✗ |

### cc_to_min_bytes_in_flight — the correct value for BI-V100

CCCL's hardcoded mapping:
  B200  (SM100, 54 GB/s/SM): 64KB
  H200  (SM90,  25 GB/s/SM): 48KB
  A100  (SM80,  19 GB/s/SM): 16KB
  V100  (SM70,  11 GB/s/SM): 12KB

BI-V100 (56 GB/s/SM) is closest to B200. Our 64KB is aggressive but
bench_bi100 confirms bif=8 (64KB) dominates bif=0 (32KB). So 64KB stands.

However: bytes_in_flight only affects the PREFETCH algorithm path.
For vllm's RMSNorm/SiLU/RoPE (contiguous fp16 arrays), the VECTORIZED
path is selected instead, where bytes_in_flight is ignored and
items_per_thread is set directly.

### vectorized policy selection for BI-V100

CCCL's tuned_vectorized_policy for fallback (cc < 8.0):
  TransformVectorizedPolicy{256, 8, 4}  // 256 threads, 8 items, vec=4

For RMSNorm with fp16 (store_size=2):
  items_per_thread=8, vec_size=4 → 8 elements/thread, 4 per vector load
  tile = 256 * 8 = 2048 elements per CTA
  With 16 SMs × 2 occupancy = 32 CTAs → 65536 elements/wave

Qwen3.6 hidden_size=3584 → RMSNorm processes 3584 elements.
3584 / 2048 = 2 tiles → fits in one wave on BI-V100. Good.

### Impact on our tuning_transform.cuh

Our bi100_bytes_in_flight=64KB is correct for prefetch but irrelevant
for vectorized. We should also set vectorized policy parameters directly:
  threads=256, items=8, vec=4 (CCCL default for older arch)
  OR test threads=128 with items=16 (A100 triad tuning) for higher ILP.

Benchmark result (bif=8, alg=1, pref=2, tpb=256, unrl=1, vsp2=1):
  alg=1 = vectorized (confirmed — prefetch would be alg=0)
  tpb=256 = matches CCCL default
  vsp2=1 → vec_size parameter (powers of 2, so vsp2=1 means vec_size=2)
  speedups: 1.203199 1.058919 1.019168 (fp16, 1M/16M/64M)

---

## GridEvenShare Work Distribution (grid_even_share.cuh)

> Added: 2026-08-04

### Two strategies: RAKE vs STRIP_MINE

**RAKE** (scan uses this): consecutive tiles per block
  block k gets tiles [k*avg .. k*avg + avg-1]
  block_stride = TILE_ITEMS (contiguous, no gaps)

**STRIP_MINE** (reduce uses this): interleaved tiles
  block k gets tiles k, k+grid_size, k+2*grid_size, ...
  block_stride = grid_size * TILE_ITEMS (strided)

### Concrete numbers for BI-V100 attention score reduce

tile_items = 512 * 24 = 12288 (bi100_plus_float32_o4)
max_grid = 2 * 16 * 5 = 160 (occupancy * SMs * subscription)

| seq_len | total_tiles | grid_size | tiles/block | waves |
|---------|-------------|-----------|-------------|-------|
| 1K      | 1           | 1         | 1           | 1     |
| 8K      | 1           | 1         | 1           | 1     |
| 32K     | 3           | 3         | 1           | 1     |
| 100K    | 9           | 9         | 1           | 1     |
| 1M      | 82          | 82        | 1           | 3     |

Even at 100K tokens, only 9 CTAs are needed → everything fits in one
wave on 16 SMs. This means:

1. Reduce tuning (items/threads) matters less than expected — there 
   are so few tiles that the per-tile overhead dominates, not throughput.

2. The V1/V2 choice in paged_attn.py matters MORE — V1 doesn't use
   GridEvenShare at all, it's a single CTA iterating sequentially.
   V2's partition-based approach enables parallel reduction.

3. For short sequences (≤8K, 1 tile), SingleTile path triggers:
   just 1 CTA, 1 kernel launch, no temp storage.

### The "big shares" distribution

GridEvenShare handles uneven tile counts:
  avg_tiles_per_block = total_tiles / grid_size
  big_shares = total_tiles % grid_size (blocks that get +1 tile)

For 100K tokens with 9 tiles and 9 blocks: avg=1, big_shares=0.
All blocks equal. No imbalance.

For 1M tokens with 82 tiles and 82 blocks: avg=1, big_shares=0.
Still perfectly balanced at 1 tile/block.

Only when max_grid_size limits grid_size do we get imbalance:
e.g., 200 tiles with max_grid=160 → avg=1, big_shares=40 (40 blocks
get 2 tiles, 120 blocks get 1 tile).
