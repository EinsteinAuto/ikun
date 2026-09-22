# Paged Attention Kernel Architecture for BI-V100

## Derived from CCCL Algorithm Patterns

This document designs a complete paged attention kernel from first principles,
using CCCL's algorithm implementations as the algorithmic foundation.
Every module maps to a proven CCCL pattern.

---

## 1. Problem Definition

Paged attention computes, for each query token in a decode step:

    output[h, d] = softmax(Q[h] · K[t]^T / √d) · V[t]

where K and V are stored in a **paged block table** (non-contiguous physical memory).

**Qwen3.6 parameters:**
- head_dim (d) = 256
- num_heads (H) = 24
- num_kv_heads (kv_h) = 4, GQA ratio = 6
- seq_len (T) = up to 100,000
- block_size = 16 tokens per physical block
- SMEM per block = 48KB

**The challenge:** K/V are scattered across physical blocks.
A naive implementation does 6,250 random memory accesses for 100K tokens.

---

## 2. Algorithm Decomposition (Three Levels from CCCL)

### Level 1: Warp Reduce (from `warp_reduce_shfl.cuh`)

**CCCL pattern:** `shfl.sync.down` butterfly reduction in log2(32) = 5 steps.
Each step: `output = reduction_op(input, ShuffleDown(input, 1 << step))`.

**In attention:** Within one warp (32 threads), each thread holds QK^T scores
for a subset of KV tokens. Warp reduce computes:
- `max_score = warp_reduce(scores, max_op)` — for softmax numerical stability
- `exp_sum = warp_reduce(exp(scores - max_score), plus_op)` — softmax denominator
- `weighted_v = warp_reduce(exp(scores - max_score) * V[t], plus_op)` — numerator

This is a **compound reduction** — the same pattern as CCCL's `summary_statistics.cu`
where (count, mean, M2) are reduced together with a custom binary op.

**Our compound type:**
```
struct attention_partial {
    float max_score;      // running max of QK^T
    float exp_sum;        // sum of exp(score - max_score)
    float weighted_v[D];  // sum of exp(score - max_score) * V
};
```

**Binary op** (from `summary_statistics.cu`):
```
attention_partial combine(attention_partial a, attention_partial b) {
    float new_max = max(a.max_score, b.max_score);
    float scale_a = exp(a.max_score - new_max);
    float scale_b = exp(b.max_score - new_max);
    return {
        new_max,
        scale_a * a.exp_sum + scale_b * b.exp_sum,
        scale_a * a.weighted_v + scale_b * b.weighted_v  // element-wise
    };
}
```

This is exactly the online softmax from Flash Attention.
It's also exactly CCCL's binary reduction op pattern.

### Level 2: Block Reduce (from `block_reduce_warp_reductions.cuh`)

**CCCL pattern:** Each warp produces a `warp_aggregate`. Lane 0 of each warp
writes it to `SMEM warp_aggregates[warp_id]`. Then thread 0 serially reduces
across warps:
```
for (warp_idx = 1; warp_idx < warps; ++warp_idx)
    aggregate = reduction_op(aggregate, warp_aggregates[warp_idx]);
```

**In attention:** One thread block processes one partition of the KV sequence
(e.g., PARTITION_SIZE = 512 tokens). Multiple warps within the block each handle
a chunk of these 512 tokens.

- Warp 0: tokens 0..63 (BLOCK_N=64 at a time, or 32 for head_dim=256)
- Warp 1: tokens 64..127
- ...
- Warp W-1: tokens (W-1)*64..511

Each warp produces an `attention_partial`. Block reduce merges them:
```
__shared__ attention_partial warp_partials[NUM_WARPS];
warp_partials[warp_id] = my_warp_result;
__syncthreads();
if (threadIdx.x == 0) {
    attention_partial block_result = warp_partials[0];
    for (int w = 1; w < NUM_WARPS; w++)
        block_result = combine(block_result, warp_partials[w]);
    // Write block_result to global: tmp_output, exp_sums, max_logits
}
```

**SMEM layout for attention_partial at head_dim=256:**
- max_score: 4 bytes
- exp_sum: 4 bytes  
- weighted_v[256]: 256 × 4 = 1024 bytes
- Total per warp: 1032 bytes
- For 4 warps: 4128 bytes (fits easily in 48KB)

### Level 3: Cross-Partition Coordination (from `agent_scan.cuh` + decoupled lookback)

**CCCL pattern:** `TilePrefixCallbackOp` implements decoupled lookback.
Each tile block:
1. Computes its local aggregate
2. Publishes local aggregate to global `tile_state` (PARTIAL status)
3. Warp 0 looks back through predecessor tiles:
   - If predecessor has INCLUSIVE status → directly use its prefix
   - If predecessor has PARTIAL status → accumulate and keep looking back
4. Once prefix is resolved, update own status to INCLUSIVE

**In attention (V2):** Each partition block has its `attention_partial`.
The cross-partition reduction is simpler than scan because attention
partitions are **commutative** — we don't need prefix sums, just a
global reduce.

But the coordination pattern is the same:
1. Each partition block writes its (max_logit, exp_sum, partial_output) to
   global memory: `tmp_output[seq, head, partition, :]`
2. A separate reduction kernel (or the last partition block) reads all
   partitions and does the final combine.

**Simplification over CCCL's lookback:** Since attention partitions are
independent (no prefix dependency), we don't need the lookback polling loop.
Each partition can run fully independently. The reduction is a simple
parallel reduce over `num_partitions` compound values.

For 100K tokens / 512 partition_size = ~200 partitions.
200 `attention_partial` values × (4 + 4 + 256×4) = 200 × 1032 = ~200KB.
One block can reduce all 200 in registers + SMEM.

---

## 3. Paged K/V Gather (from `block_load.cuh` + `cache_modified_input_iterator.cuh`)

**CCCL pattern:** `BlockLoadWarpTranspose` loads contiguous global memory
into a striped register layout that enables coalesced access. Each thread
loads `ITEMS_PER_THREAD` elements, and the warp transposes them so each
thread gets its tile of the data.

**In paged attention:** K/V are not contiguous — they're indexed through
`block_tables[seq, logical_block] → physical_block`.
- Key cache: `[num_blocks, kv_heads, head_dim/x, block_size, x]`
  where x = 16/sizeof(dtype) is the packing factor
- Value cache: `[num_blocks, kv_heads, head_dim, block_size]`

The gather pattern (from `prefix_prefill.py`, which works on BI-V100):
```
# For BLOCK_N tokens starting at position start_n:
token_ids = start_n + tl.arange(0, BLOCK_N)
logical_blocks = token_ids // block_size
within_block = token_ids % block_size
physical_blocks = tl.load(block_tables + seq * stride + logical_blocks * stride)

# K gather: compute 2D offset array [HEAD_DIM, BLOCK_N]
off_k = (physical_blocks[None, :] * stride_kc_b +
         kv_head * stride_kc_h +
         (offs_d[:, None] // x) * stride_kc_dx +
         within_block[None, :] * stride_kc_bs +
         (offs_d[:, None] % x) * stride_kc_x)
k = tl.load(key_cache + off_k, mask=valid_mask)
```

This is an **indirect gather** — the physical block ID comes from a table lookup.
CCCL's `CacheModifiedInputIterator` handles the cache hint part, but the
indirect indexing is our addition.

**Memory access pattern:**
- block_tables lookup: 1 global read per BLOCK_N tokens (amortized)
- K gather: BLOCK_N × HEAD_DIM / x global reads (scattered by physical block)
- V gather: BLOCK_N × HEAD_DIM global reads (similar scatter)

For BLOCK_N=32, HEAD_DIM=256, x=8: 32 × 32 = 1024 reads for K per iteration.
At 16 bytes per read (128-bit): 16KB per K load.
V is similar. Total per iteration: ~32KB — fits in L2 (6MB on BI-V100).

---

## 4. GQA (Grouped Query Attention) Handling

**The insight:** 6 query heads share 1 KV head. Loading KV once and
computing 6 sets of QK^T scores is 6x more compute-efficient than
loading KV 6 times.

**CCCL analogy:** This is like `BlockReduce` where we have 6 different
reduction operations on the same input data. CCCL doesn't have this exact
pattern, but the principle is: share data loads, parallelize computation.

**Implementation:**
- Each thread block handles one `(seq, kv_head, partition)` triple
- Within the block, 6 query heads are processed simultaneously
- Q vectors: 6 × HEAD_DIM = 6 × 256 = 1536 values in registers (per thread
  this is 1536/32 = 48 registers — feasible)
- K/V: loaded once for the kv_head, broadcast across all 6 query heads
- Scores: 6 × BLOCK_N values per iteration
- Weighted V: 6 × HEAD_DIM per thread's accumulator

This reduces K/V cache reads by 6x (the GQA ratio).

Grid: `(num_seqs, num_kv_heads, num_partitions)` = `(1, 4, 200)` = 800 blocks
instead of `(1, 24, 200)` = 4800 blocks.

Each block does 6x more compute but reads KV only once.

---

## 5. SMEM Budget

For one block processing BLOCK_N=32 KV tokens across 6 query heads:

| Item | Size | Notes |
|------|------|-------|
| K tile [HEAD_DIM, BLOCK_N] | 32×256×2 = 16KB | fp16, loaded from paged cache |
| V tile [BLOCK_N, HEAD_DIM] | 32×256×2 = 16KB | fp16, loaded from paged cache |
| Warp partials [4 warps × attention_partial] | 4×(4+4+256×4) = 4.1KB | For block-level reduce |
| Q vectors [6 × HEAD_DIM] | 6×256×4 = 6KB | In registers ideally, SMEM if spills |
| **Total** | **42.1KB** | **≤ 48KB ✓** |

Tight but feasible. If Q stays in registers (likely with 4 warps × 32 threads
= 128 threads, each handling 6×256/128 = 12 Q values), total SMEM is 36.1KB.

---

## 6. Kernel Launch Configuration

**Phase 1: Partitioned Attention**
- Grid: `(num_seqs, num_kv_heads, num_partitions)`
- Block: `(NUM_WARPS × 32)` = 128 threads (4 warps)
- Each block processes:
  - PARTITION_SIZE = 512 KV tokens
  - 6 query heads (GQA broadcast)
  - Produces 6 × (max_logit, exp_sum, partial_output[256])

**Phase 2: Cross-Partition Reduction**
- Grid: `(num_seqs, num_kv_heads)` 
- Block: 128 threads
- Each block reduces ~200 partitions × 6 query heads
- Uses `combine()` op (same as CCCL `BlockReduce` but with `attention_partial`)

**Phase 1 iterations per block:**
- PARTITION_SIZE / BLOCK_N = 512 / 32 = 16 iterations
- Each iteration: load K[32, 256] + V[32, 256], compute 6×32 scores, update 6 accumulators

---

## 7. Implementation Mapping

| Module | CCCL Source | Our Implementation |
|--------|------------|-------------------|
| Warp-level QK^T + softmax | `warp_reduce_shfl.cuh` | Triton: `tl.sum()` within warp-sized groups |
| Block-level partition reduce | `block_reduce_warp_reductions.cuh` | Triton: shared memory + `tl.reduce()` |
| Cross-partition combine | `agent_scan.cuh` (simplified, no lookback) | Separate reduction kernel |
| Paged K/V gather | `block_load.cuh` + indirect indexing | `prefix_prefill.py` pattern adapted |
| Online softmax | `summary_statistics.cu` binary op | `combine(attention_partial, attention_partial)` |
| GQA broadcast | (no exact CCCL analog) | Multiple Q per KV load |

---

## 8. Why This Design Beats Python V2

Current Python V2 (3 bmm launches + Python overhead):
- gather all KV → permute → contiguous → bmm → reshape → softmax → bmm → reduce
- **Python-CUDA boundary crossed 10+ times per decode step**
- **Full KV tensor materialized in GPU memory** (200MB-2.4GB depending on GQA)

This kernel (2 GPU launches, zero Python-CUDA crossings during compute):
- Phase 1: single kernel, K/V loaded tile-by-tile from paged cache (never materialized)
- Phase 2: single kernel, reduces 200 partitions in SMEM
- **KV cache stays in paged format** — no gather/permute/contiguous overhead
- **GQA broadcast within kernel** — KV loaded once for 6 heads

Expected improvement over Python V2: **10-100x** (eliminating Python overhead
and memory allocation dominates at decode batch_size=1).

Expected improvement over no V2 (V1 only for seq ≤ 8192): **enables long-context
decode** which V1 cannot do due to SMEM overflow at 48KB.

---

## 9. Implementation Priority

1. **Triton implementation** — if Triton works on BI-V100 with BLOCK=32, head_dim=256:
   Use the `prefix_prefill.py` paged gather pattern, add the compound reduction.
   This is the fastest path to a working kernel.

2. **Compiled CUDA kernel** — if `/usr/local/corex/` has a compiler (ixcc):
   Write the kernel in CUDA using the CCCL patterns directly.
   `warp_reduce_shfl` → `__shfl_down_sync` PTX
   `block_reduce` → SMEM warp_aggregates pattern
   Compile with `torch.utils.cpp_extension.load()` at Docker build time.

3. **Python V2** (current) — fallback if neither Triton nor CUDA works:
   Already written, tested, has GQA broadcast optimization.
   This is the floor, not the ceiling.
