"""
paged_attention_v2_pytorch.py — BI-V100 PagedAttention V2 (CCCL-informed)
===========================================================================

Fills the `raise NotImplementedError()` hole in vllm/_custom_ops.py.

Algorithm: Partitioned attention with log-sum-exp reduction.
Architecture informed by CCCL patterns:
  - summary_statistics.cu: fuse multiple statistics in a single reduction pass
  - warp_reduce_shfl.cuh: accumulate (max, sum, weighted_output) as one compound type
  - block_reduce_warp_reductions.cuh: reduce across partitions via shared accumulators

Key optimization: Batched partition attention via reshaped 3D bmm.
  Instead of looping over P partitions with P × torch.bmm calls,
  reshape KV into [H, P*part_len, d] and Q into [H, 1, d], then
  slice scores into [H, P, part_len] for partition-wise softmax.
  This gives ONE bmm launch for all partitions.

  For seq_len=100K, PARTITION_SIZE=512:
    Before: 195 × bmm([H,1,d] @ [H,d,512]) = 195 kernel launches
    After:  1 × bmm([H,1,d] @ [H,d,100K]) + reshape = 1 kernel launch

  The partition-wise softmax is then a reshape + per-chunk operation:
    scores: [H, 100K] → [H, P, 512] → max/exp/sum per partition

Phase 2 reduction (cross-partition combine) follows CCCL's summary_statistics
binary_op pattern: combine (max_a, sum_a, out_a) with (max_b, sum_b, out_b)
using the numerically stable log-sum-exp rescaling.
"""

import torch
from typing import Optional

_PARTITION_SIZE = 1024  # CCCL dispatch_scan.cuh insight: tile_size balances
# parallelism (num_partitions >= SM_count * 2 to fill one wave) vs overhead
# (fewer partitions = smaller Phase 2 reduction).
# BI-V100: 16 SMs, max ~32 concurrent CTAs.
# For 100K tokens: 1024 → 98 partitions (3 waves), 512 → 195 (6 waves).
# 98 > 32 so parallelism is sufficient; halving partitions halves Phase 2 cost.

# CCCL dispatch_reduce.cuh GridEvenShare formula (line ~180):
#   max_blocks = sm_occupancy * sm_count * subscription_factor
#   subscription_factor = 5 (default in cub/util_device.cuh)
# For BI-V100: sm_count=16, sm_occupancy ~= 2 (limited by registers/SMEM)
#   → max_blocks = 2 * 16 * 5 = 160
# If seq_len=100K with PARTITION_SIZE=1024 → 98 partitions < 160 → fine.
# Threshold for V1→V2 handoff: when single-tile can't hold all tokens.
#   CCCL single_tile threshold = threads * items_per_thread
#   = 512 * 24 = 12288 tokens → V1 handles ≤12288, V2 handles >12288.
# This aligns with BI-V100 paged_attn.py _PARTITION_SIZE=512:
#   V2 triggers when seq_len > 512 * (max_blocks_per_seq_for_v1).
_BI100_SM_COUNT = 16
_BI100_SM_OCCUPANCY = 2  # conservative: 2 CTAs per SM
_BI100_SUBSCRIPTION_FACTOR = 5  # CCCL default
_BI100_MAX_GRID = _BI100_SM_OCCUPANCY * _BI100_SM_COUNT * _BI100_SUBSCRIPTION_FACTOR  # 160


def paged_attention_v2_pytorch(
    output: torch.Tensor,          # [num_seqs, num_heads, head_size]
    exp_sums: torch.Tensor,        # [num_seqs, num_heads, max_num_partitions]
    max_logits: torch.Tensor,      # [num_seqs, num_heads, max_num_partitions]
    tmp_output: torch.Tensor,      # [num_seqs, num_heads, max_num_partitions, head_size]
    query: torch.Tensor,           # [num_seqs, num_heads, head_size]
    key_cache: torch.Tensor,       # [num_blocks, num_kv_heads, head_size/x, block_size, x]
    value_cache: torch.Tensor,     # [num_blocks, num_kv_heads, head_size, block_size]
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,    # [num_seqs, max_blocks_per_seq]
    seq_lens: torch.Tensor,        # [num_seqs]
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> None:
    num_seqs, num_heads, head_size = query.shape
    gqa_ratio = num_heads // num_kv_heads
    max_num_partitions = tmp_output.shape[2]

    # Initialize unused slots
    max_logits.fill_(float('-inf'))
    exp_sums.zero_()
    tmp_output.zero_()

    # CCCL kernel_reduce.cuh SingleTile fast path (line ~270):
    #   if (num_items <= threads_per_block * items_per_thread)
    #     → InvokeSingleTile() — one CTA, no temp buffer, no Phase 2
    # PyTorch translation: if seq_len fits in one partition, skip Phase 2 entirely.
    # This avoids the partition/reshape/bmm overhead for short decode sequences.
    # Qwen3.6 typical decode: seq_len grows from 1 to 100K over generation.
    # Early tokens (seq_len < 1024) hit this fast path every step.
    _SINGLE_TILE_THRESHOLD = _PARTITION_SIZE  # sequences this short skip partitioning

    for seq_idx in range(num_seqs):
        seq_len = int(seq_lens[seq_idx].item())
        if seq_len == 0:
            output[seq_idx].zero_()
            continue

        num_blocks_seq = (seq_len + block_size - 1) // block_size
        num_partitions = (seq_len + _PARTITION_SIZE - 1) // _PARTITION_SIZE

        # ─── CCCL SingleTile fast path ───────────────────────────
        # From kernel_reduce.cuh: when everything fits in one tile,
        # do a single-pass attention without partition overhead.
        # agent_reduce.cuh ConsumeRange → BlockReduce → done.
        if num_partitions == 1:
            blk_ids = block_tables[seq_idx, :num_blocks_seq]
            q = query[seq_idx].float()  # [H, d]

            # Gather KV (same as below but no partition reshape)
            k_gathered = key_cache[blk_ids]
            k_flat = (k_gathered
                      .permute(0, 3, 1, 2, 4)
                      .reshape(-1, num_kv_heads, head_size))[:seq_len]
            v_flat = (value_cache[blk_ids]
                      .permute(0, 3, 1, 2)
                      .reshape(-1, num_kv_heads, head_size))[:seq_len]

            if k_scale != 1.0:
                k_flat = k_flat.float().mul_(k_scale)
            if v_scale != 1.0:
                v_flat = v_flat.float().mul_(v_scale)

            if gqa_ratio > 1:
                k_kv = k_flat.permute(1, 2, 0).float().contiguous()
                v_kv = v_flat.permute(1, 0, 2).float().contiguous()
                q_grouped = q.view(num_kv_heads, gqa_ratio, 1, head_size)
                scores = torch.matmul(q_grouped, k_kv.unsqueeze(1)).squeeze(2)
                scores = scores.reshape(num_heads, seq_len) * scale
            else:
                k_t = k_flat.permute(1, 2, 0).float().contiguous()
                scores = torch.bmm(q.unsqueeze(1), k_t).squeeze(1) * scale

            if alibi_slopes is not None:
                positions = torch.arange(seq_len, device=query.device, dtype=torch.float32)
                scores = scores + alibi_slopes.unsqueeze(1) * positions.unsqueeze(0)

            # Direct softmax + V weighted sum — no partition overhead
            weights = torch.softmax(scores, dim=-1)  # [H, seq_len]
            if gqa_ratio > 1:
                w_grouped = weights.view(num_kv_heads, gqa_ratio, 1, seq_len)
                result = torch.matmul(w_grouped, v_kv.unsqueeze(1)).squeeze(2)
                output[seq_idx] = result.reshape(num_heads, head_size).to(output.dtype)
            else:
                v_perm = v_flat.permute(1, 0, 2).float().contiguous()
                result = torch.bmm(weights.unsqueeze(1), v_perm).squeeze(1)
                output[seq_idx] = result.to(output.dtype)

            # Store dummy partition values for compatibility
            max_logits[seq_idx, :, 0] = scores.max(dim=-1).values
            exp_sums[seq_idx, :, 0] = weights.sum(dim=-1)
            tmp_output[seq_idx, :, 0, :] = output[seq_idx].float()
            continue
        # ─── End SingleTile fast path ────────────────────────────

        # =============================================================
        # Batched KV gather: ONE index_select, ONE reshape
        # Pattern: avoid per-block Python loop (CCCL does this via
        # block-cooperative load, we do it via batched indexing)
        # =============================================================
        blk_ids = block_tables[seq_idx, :num_blocks_seq]

        # Key: [nblk, kv_h, d/x, blk_sz, x] → [nblk*blk_sz, kv_h, d]
        k_gathered = key_cache[blk_ids]
        k_flat = (k_gathered
                  .permute(0, 3, 1, 2, 4)
                  .reshape(-1, num_kv_heads, head_size))[:seq_len]

        # Value: [nblk, kv_h, d, blk_sz] → [nblk*blk_sz, kv_h, d]
        v_flat = (value_cache[blk_ids]
                  .permute(0, 3, 1, 2)
                  .reshape(-1, num_kv_heads, head_size))[:seq_len]

        if k_scale != 1.0:
            k_flat = k_flat.float().mul_(k_scale)
        if v_scale != 1.0:
            v_flat = v_flat.float().mul_(v_scale)

        # =============================================================
        # GQA broadcast: avoid materializing the expanded KV tensor
        #
        # Qwen3.6: H=24, kv_h=4, gqa_ratio=6, head_dim=256
        # Old: expand kv_h→H then contiguous → allocates seq_len×H×d (1.2GB at 100K)
        # New: reshape Q as [kv_h, gqa, 1, d], K as [kv_h, 1, d, seq_len]
        #      → bmm with broadcasting → [kv_h, gqa, 1, seq_len]
        #      → reshape to [H, seq_len]
        # Saves: gqa_ratio × memory (6x for Qwen3.6 = 1GB per decode step)
        # =============================================================
        q = query[seq_idx].float()  # [H, d]

        if gqa_ratio > 1:
            # K: [seq_len, kv_h, d] → [kv_h, d, seq_len] (no GQA expansion)
            k_kv = k_flat.permute(1, 2, 0).float().contiguous()  # [kv_h, d, seq_len]
            v_kv = v_flat.permute(1, 0, 2).float().contiguous()  # [kv_h, seq_len, d]

            # Q: [H, d] → [kv_h, gqa, 1, d]
            q_grouped = q.view(num_kv_heads, gqa_ratio, 1, head_size)

            # Scores: [kv_h, gqa, 1, d] @ [kv_h, 1, d, seq_len] → [kv_h, gqa, 1, seq_len]
            scores_all = torch.matmul(q_grouped, k_kv.unsqueeze(1)).squeeze(2)  # [kv_h, gqa, seq_len]
            scores_all = scores_all.reshape(num_heads, seq_len) * scale  # [H, seq_len]
        else:
            k_t = k_flat.permute(1, 2, 0).float().contiguous()  # [H, d, seq_len]
            scores_all = torch.bmm(q.unsqueeze(1), k_t).squeeze(1) * scale  # [H, seq_len]

        # Alibi bias (if needed)
        if alibi_slopes is not None:
            positions = torch.arange(seq_len, device=query.device, dtype=torch.float32)
            scores_all = scores_all + alibi_slopes.unsqueeze(1) * positions.unsqueeze(0)

        # Pad to exact multiple of _PARTITION_SIZE for clean reshape
        padded_len = num_partitions * _PARTITION_SIZE
        if padded_len > seq_len:
            pad_size = padded_len - seq_len
            scores_padded = torch.full(
                (num_heads, padded_len), float('-inf'),
                dtype=scores_all.dtype, device=scores_all.device)
            scores_padded[:, :seq_len] = scores_all
        else:
            scores_padded = scores_all

        # Reshape: [H, padded_len] → [H, P, part_sz]
        scores_parts = scores_padded.view(num_heads, num_partitions, _PARTITION_SIZE)

        # Per-partition online softmax (vectorized over H and P simultaneously)
        # Pattern from CCCL summary_statistics: compute (max, sum) in one pass
        part_max = scores_parts.max(dim=-1).values          # [H, P]
        scores_exp = torch.exp(scores_parts - part_max.unsqueeze(-1))  # [H, P, part_sz]
        part_sum = scores_exp.sum(dim=-1)                    # [H, P]

        # Weighted values per partition: need V reshaped the same way
        # V: [seq_len, H, d] → pad → [padded_len, H, d] → [H, P, part_sz, d]
        if gqa_ratio > 1:
            v_perm = v_kv  # already [kv_h, seq_len, d], no GQA expansion needed
            # Will handle GQA in the bmm below via broadcast
        else:
            v_perm = v_flat.permute(1, 0, 2).float().contiguous()  # [H, seq_len, d]
        # Weighted V sum per partition
        # NOTE: v_perm shape differs by GQA mode:
        #   GQA: v_perm = v_kv = [kv_h, seq_len, d]
        #   No GQA: v_perm = [H, seq_len, d]
        # scores_exp: [H, P, part_sz] → [kv_h, gqa, P, part_sz]
        # v_perm: [kv_h, seq_len, d] → [kv_h, P, part_sz, d]
        if gqa_ratio > 1:
            se_grouped = scores_exp.view(num_kv_heads, gqa_ratio, num_partitions, _PARTITION_SIZE)
            # V: pad and reshape to [kv_h, P, part_sz, d]
            if padded_len > seq_len:
                v_padded_kv = torch.zeros(
                    (num_kv_heads, padded_len, head_size),
                    dtype=v_kv.dtype, device=v_kv.device)
                v_padded_kv[:, :seq_len, :] = v_kv
            else:
                v_padded_kv = v_kv
            v_parts_kv = v_padded_kv.view(num_kv_heads, num_partitions, _PARTITION_SIZE, head_size)
            # Broadcast: [kv_h, gqa, P, 1, part_sz] @ [kv_h, 1, P, part_sz, d]
            #          → [kv_h, gqa, P, 1, d]
            part_out_grouped = torch.matmul(
                se_grouped.unsqueeze(3),       # [kv_h, gqa, P, 1, part_sz]
                v_parts_kv.unsqueeze(1)        # [kv_h, 1, P, part_sz, d]
            ).squeeze(3)                        # [kv_h, gqa, P, d]
            part_out = part_out_grouped.reshape(num_heads, num_partitions, head_size)
        else:
            # Non-GQA: v_perm is [H, seq_len, d], pad and reshape normally
            if padded_len > seq_len:
                v_padded = torch.zeros(
                    (num_heads, padded_len, head_size),
                    dtype=v_perm.dtype, device=v_perm.device)
                v_padded[:, :seq_len, :] = v_perm
            else:
                v_padded = v_perm
            v_parts = v_padded.view(num_heads, num_partitions, _PARTITION_SIZE, head_size)
            HP = num_heads * num_partitions
            scores_exp_flat = scores_exp.reshape(HP, 1, _PARTITION_SIZE)
            v_parts_flat = v_parts.reshape(HP, _PARTITION_SIZE, head_size)
            part_out_flat = torch.bmm(scores_exp_flat, v_parts_flat)  # [HP, 1, d]
            part_out = part_out_flat.view(num_heads, num_partitions, head_size)  # [H, P, d]

        # Store partition results
        max_logits[seq_idx, :, :num_partitions] = part_max
        exp_sums[seq_idx, :, :num_partitions] = part_sum
        tmp_output[seq_idx, :, :num_partitions, :] = part_out.to(tmp_output.dtype)

        # =============================================================
        # Phase 2: Cross-partition reduction (CCCL binary_op pattern)
        #
        # CCCL kernel_reduce.cuh insight: when grid_size fits in a single
        # tile (num_partitions <= threads * items_per_thread), the reduce
        # uses SingleTile path — one CTA, no temp buffer, no pass 2 kernel.
        #
        # For BI-V100 with 98 partitions (100K tokens / 1024 partition_size):
        #   SingleTile threshold = 512 * 24 = 12288 >> 98 → always SingleTile
        #   This means Phase 2 is never the bottleneck.
        #
        # CCCL single_pass_scan_operators.cuh insight: delay() has a
        # GridThreshold=500 gate. BI-V100 scan grids are always < 500 blocks,
        # so ALL delay strategies (no_delay, fixed_delay, exponential_backon)
        # collapse to __threadfence_block(). Delay tuning is irrelevant here.
        #
        # Phase 2 follows summary_statistics.cu binary_op: combine
        # (max_a, sum_a, out_a) ⊕ (max_b, sum_b, out_b) via log-sum-exp.
        # Fully vectorized — no loop over partitions.
        # =============================================================
        pm = max_logits[seq_idx, :, :num_partitions]     # [H, P]
        ps = exp_sums[seq_idx, :, :num_partitions]       # [H, P]
        po = tmp_output[seq_idx, :, :num_partitions, :]  # [H, P, d]

        global_max = pm.max(dim=-1).values               # [H]
        rescale = torch.exp(pm - global_max.unsqueeze(-1)) * ps  # [H, P]
        total = rescale.sum(dim=-1, keepdim=True)         # [H, 1]

        # CCCL norm.cu principle: fuse transform with reduce to minimize traversals.
        # Instead of: weights = rescale/total; final = bmm(weights, po)
        # Do: final = bmm(rescale, po) / total
        # Saves one element-wise division kernel launch (rescale/total → H*P elements).
        # The division moves to the output (H*d elements, typically smaller than H*P).
        # [H, 1, P] @ [H, P, d] → [H, 1, d] → [H, d]
        final = torch.bmm(rescale.unsqueeze(1), po.float()).squeeze(1) / total  # [H, d]
        output[seq_idx] = final.to(output.dtype)
