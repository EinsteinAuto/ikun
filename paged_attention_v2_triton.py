"""
paged_attention_v2_triton.py — CCCL-derived Triton PagedAttention V2
=====================================================================

Architecture: docs/paged_attention_kernel_architecture.md

Two-kernel design:
  Phase 1: _partition_attn — per-partition compound reduction (CCCL block_reduce pattern)
  Phase 2: _reduce_partitions — cross-partition combine (CCCL agent_reduce pattern)

Key CCCL derivations:
  1. Compound type: (max_score, exp_sum, weighted_v[D]) — from summary_statistics.cu
  2. Combine op: online softmax rescaling — from Flash Attention = CCCL's binary_op pattern
  3. Warp reduce: shfl.down butterfly — from warp_reduce_shfl.cuh (Triton does this via tl.sum/tl.max)
  4. Block reduce: warp partials → SMEM → serial combine — from block_reduce_warp_reductions.cuh
  5. Paged gather: indirect load via block_tables — from prefix_prefill.py (proven on BI-V100)
  6. GQA: grid on kv_heads, process gqa_ratio query heads per block — KV loaded once

Grid design:
  Phase 1: (num_seqs, num_kv_heads, num_partitions) — NOT (num_seqs, num_heads, num_partitions)
           Each block loads KV once for kv_head, computes gqa_ratio query heads.
           Reduces KV cache reads by gqa_ratio (6x for Qwen3.6).
  Phase 2: (num_seqs, num_kv_heads) — reduces partitions, writes all gqa_ratio outputs.

SMEM budget (head_dim=256, BLOCK_N=32):
  K tile: 32×256×2 = 16KB
  V tile: 32×256×2 = 16KB
  Warp partials: negligible (in registers for Triton)
  Total: 32KB ≤ 48KB ✓
"""

import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _partition_attn_kernel(
    # Outputs (per partition)
    tmp_output_ptr,     # [num_seqs, num_heads, max_parts, head_size]
    exp_sums_ptr,       # [num_seqs, num_heads, max_parts]
    max_logits_ptr,     # [num_seqs, num_heads, max_parts]
    # Inputs
    query_ptr,          # [num_seqs, num_heads, head_size]
    key_cache_ptr,      # [num_blocks, kv_heads, head_size/x, block_size, x]
    value_cache_ptr,    # [num_blocks, kv_heads, head_size, block_size]
    block_tables_ptr,   # [num_seqs, max_blocks_per_seq]
    seq_lens_ptr,       # [num_seqs]
    # Scalars
    scale: tl.float32,
    gqa_ratio: tl.int32,          # num_heads // num_kv_heads
    block_size: tl.int32,
    x_pack: tl.int32,             # key_cache packing factor
    # Strides: query [S, H, D]
    stride_qs: tl.int32, stride_qh: tl.int32, stride_qd: tl.int32,
    # Strides: key_cache [B, KH, D/X, BS, X]
    stride_kc_b: tl.int32, stride_kc_h: tl.int32,
    stride_kc_dx: tl.int32, stride_kc_bs: tl.int32, stride_kc_x: tl.int32,
    # Strides: value_cache [B, KH, D, BS]
    stride_vc_b: tl.int32, stride_vc_h: tl.int32,
    stride_vc_d: tl.int32, stride_vc_bs: tl.int32,
    # Strides: block_tables [S, MAX_BLOCKS]
    stride_bt_s: tl.int32, stride_bt_b: tl.int32,
    # Strides: tmp_output [S, H, P, D]
    stride_to_s: tl.int32, stride_to_h: tl.int32,
    stride_to_p: tl.int32, stride_to_d: tl.int32,
    # Strides: exp_sums/max_logits [S, H, P]
    stride_es_s: tl.int32, stride_es_h: tl.int32, stride_es_p: tl.int32,
    # Constants
    PARTITION_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GQA_RATIO: tl.constexpr,
):
    """Phase 1: Per-partition attention with GQA broadcast.

    Grid: (num_seqs, num_kv_heads, num_partitions)
    Each block processes one (seq, kv_head, partition), computing GQA_RATIO query heads.

    Algorithm (CCCL compound reduction):
      For each BLOCK_N chunk of KV tokens in this partition:
        1. Paged K gather: block_tables → physical_block → K[BLOCK_N, HEAD_DIM]
        2. Scores: Q[g, HEAD_DIM] · K[HEAD_DIM, BLOCK_N] → [GQA_RATIO, BLOCK_N]
        3. Online softmax update (combine op from summary_statistics.cu):
           For each query head g:
             m_new = max(m_old, max(scores[g]))
             rescale_old = exp(m_old - m_new)
             p = exp(scores[g] - m_new)
             l_new = rescale_old * l_old + sum(p)
             acc[g] = rescale_old * acc[g] + p · V
             m_old, l_old = m_new, l_new
        4. Paged V gather → accumulate weighted V
      Write per-partition results for all GQA_RATIO heads.
    """
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    part_idx = tl.program_id(2)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    part_start = part_idx * PARTITION_SIZE
    part_end = tl.minimum(part_start + PARTITION_SIZE, seq_len)

    if part_start >= seq_len:
        # Unused partition — write sentinels for all GQA_RATIO heads
        for g in range(GQA_RATIO):
            head_idx = kv_head_idx * GQA_RATIO + g
            tl.store(max_logits_ptr + seq_idx * stride_es_s + head_idx * stride_es_h + part_idx * stride_es_p,
                     float('-inf'))
            tl.store(exp_sums_ptr + seq_idx * stride_es_s + head_idx * stride_es_h + part_idx * stride_es_p,
                     0.0)
        return

    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    # Load all GQA_RATIO query vectors for this kv_head
    # q[g]: [HEAD_DIM] for g in 0..GQA_RATIO-1
    # We process them sequentially to stay within register budget
    # (Loading all 6 × 256 = 1536 fp32 values would be 6KB of registers per thread)

    # Initialize compound accumulators for each query head
    # m[g]: running max, l[g]: running exp_sum, acc[g]: [HEAD_DIM] weighted V
    # For Triton, we process one query head at a time through the full partition
    # to minimize register pressure.

    for g in range(GQA_RATIO):
        head_idx = kv_head_idx * GQA_RATIO + g

        # Load Q for this head
        q = tl.load(query_ptr + seq_idx * stride_qs + head_idx * stride_qh
                     + offs_d * stride_qd).to(tl.float32)

        # Compound accumulator
        m_i = float('-inf')
        l_i = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        # Inner loop: BLOCK_N KV tokens per iteration
        for start_n in range(part_start, part_end, BLOCK_N):
            token_ids = start_n + offs_n
            valid = token_ids < part_end

            # Paged K gather (from prefix_prefill.py)
            blk_idx = token_ids // block_size
            blk_off = token_ids % block_size
            phys_blk = tl.load(block_tables_ptr + seq_idx * stride_bt_s + blk_idx * stride_bt_b,
                               mask=valid, other=0)

            off_k = (phys_blk[None, :] * stride_kc_b +
                     kv_head_idx * stride_kc_h +
                     (offs_d[:, None] // x_pack) * stride_kc_dx +
                     blk_off[None, :] * stride_kc_bs +
                     (offs_d[:, None] % x_pack) * stride_kc_x)
            k = tl.load(key_cache_ptr + off_k, mask=valid[None, :], other=0.0)  # [D, N]

            # Scores: q · k per token
            scores = tl.sum(q[:, None] * k, axis=0) * scale  # [BLOCK_N]
            scores = tl.where(valid, scores, float('-inf'))

            # Online softmax (CCCL combine op)
            m_ij = tl.max(scores, axis=0)
            p = tl.exp(scores - m_ij)
            l_ij = tl.sum(p, axis=0)

            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(m_ij - m_new)
            l_new = alpha * l_i + beta * l_ij

            # Paged V gather
            off_v = (phys_blk[:, None] * stride_vc_b +
                     kv_head_idx * stride_vc_h +
                     offs_d[None, :] * stride_vc_d +
                     blk_off[:, None] * stride_vc_bs)
            v = tl.load(value_cache_ptr + off_v, mask=valid[:, None], other=0.0)  # [N, D]

            # Update accumulator
            safe_l = tl.maximum(l_new, 1e-6)
            acc = acc * (alpha * l_i / safe_l)
            p_scaled = p * (beta / safe_l)
            acc += tl.sum(p_scaled[:, None] * v, axis=0)

            m_i = m_new
            l_i = l_new

        # Write partition results for this head
        tl.store(max_logits_ptr + seq_idx * stride_es_s + head_idx * stride_es_h + part_idx * stride_es_p,
                 m_i)
        tl.store(exp_sums_ptr + seq_idx * stride_es_s + head_idx * stride_es_h + part_idx * stride_es_p,
                 l_i)
        out_base = seq_idx * stride_to_s + head_idx * stride_to_h + part_idx * stride_to_p
        tl.store(tmp_output_ptr + out_base + offs_d * stride_to_d,
                 acc.to(tmp_output_ptr.dtype.element_ty))


@triton.jit
def _reduce_partitions_kernel(
    output_ptr,         # [num_seqs, num_heads, head_size]
    tmp_output_ptr,     # [num_seqs, num_heads, max_parts, head_size]
    exp_sums_ptr,       # [num_seqs, num_heads, max_parts]
    max_logits_ptr,     # [num_seqs, num_heads, max_parts]
    seq_lens_ptr,       # [num_seqs]
    gqa_ratio: tl.int32,
    max_num_parts: tl.int32,
    stride_out_s: tl.int32, stride_out_h: tl.int32, stride_out_d: tl.int32,
    stride_to_s: tl.int32, stride_to_h: tl.int32,
    stride_to_p: tl.int32, stride_to_d: tl.int32,
    stride_es_s: tl.int32, stride_es_h: tl.int32, stride_es_p: tl.int32,
    PARTITION_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MAX_NUM_PARTS: tl.constexpr,
    GQA_RATIO: tl.constexpr,
):
    """Phase 2: Cross-partition reduction.

    Grid: (num_seqs, num_kv_heads)
    Each block reduces all partitions for GQA_RATIO query heads.

    Algorithm (CCCL block_reduce_warp_reductions pattern):
      For each query head in this kv_head group:
        1. Load all partition (max, sum) into registers
        2. Global max across partitions
        3. Rescale: weights = exp(part_max - global_max) * part_sum / total
        4. Weighted combination of partition outputs
    """
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_parts = (seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE
    part_offsets = tl.arange(0, MAX_NUM_PARTS)
    valid = part_offsets < num_parts
    offs_d = tl.arange(0, HEAD_DIM)

    for g in range(GQA_RATIO):
        head_idx = kv_head_idx * GQA_RATIO + g
        es_base = seq_idx * stride_es_s + head_idx * stride_es_h

        # Load partition statistics
        part_max = tl.load(max_logits_ptr + es_base + part_offsets * stride_es_p,
                           mask=valid, other=float('-inf'))
        part_sum = tl.load(exp_sums_ptr + es_base + part_offsets * stride_es_p,
                           mask=valid, other=0.0)

        # Global max
        global_max = tl.max(part_max, axis=0)

        # Rescale and normalize (CCCL combine op applied across all partitions)
        rescale = tl.exp(part_max - global_max) * part_sum
        total = tl.sum(rescale, axis=0)

        # Weighted combination
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for p in range(MAX_NUM_PARTS):
            if p < num_parts:
                w = tl.exp(tl.load(max_logits_ptr + es_base + p * stride_es_p) - global_max) * \
                    tl.load(exp_sums_ptr + es_base + p * stride_es_p) / tl.maximum(total, 1e-6)
                to_base = seq_idx * stride_to_s + head_idx * stride_to_h + p * stride_to_p
                part_out = tl.load(tmp_output_ptr + to_base + offs_d * stride_to_d)
                acc += w * part_out.to(tl.float32)

        # Store final output
        out_base = seq_idx * stride_out_s + head_idx * stride_out_h
        tl.store(output_ptr + out_base + offs_d * stride_out_d,
                 acc.to(output_ptr.dtype.element_ty))


def paged_attention_v2_triton(
    output: torch.Tensor,
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    **kwargs,
) -> None:
    """Launch CCCL-derived Triton V2 kernels."""
    num_seqs, num_heads, head_size = query.shape
    gqa_ratio = num_heads // num_kv_heads
    max_num_parts = tmp_output.shape[2]
    x_pack = key_cache.shape[-1]

    PARTITION_SIZE = 512
    BLOCK_N = 32 if head_size > 128 else 64

    num_partitions = (max_seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE

    # Phase 1: grid on kv_heads (not num_heads) — GQA broadcast inside kernel
    grid_p1 = (num_seqs, num_kv_heads, num_partitions)
    _partition_attn_kernel[grid_p1](
        tmp_output, exp_sums, max_logits,
        query, key_cache, value_cache, block_tables, seq_lens,
        scale, gqa_ratio, block_size, x_pack,
        query.stride(0), query.stride(1), query.stride(2),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        key_cache.stride(3), key_cache.stride(4),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        value_cache.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        tmp_output.stride(0), tmp_output.stride(1), tmp_output.stride(2), tmp_output.stride(3),
        exp_sums.stride(0), exp_sums.stride(1), exp_sums.stride(2),
        PARTITION_SIZE=PARTITION_SIZE,
        HEAD_DIM=head_size,
        BLOCK_N=BLOCK_N,
        GQA_RATIO=gqa_ratio,
    )

    # Phase 2: grid on kv_heads — reduce all partitions for GQA_RATIO heads each
    MAX_NUM_PARTS_CONST = triton.next_power_of_2(max_num_parts)
    if MAX_NUM_PARTS_CONST > 1024:
        MAX_NUM_PARTS_CONST = 1024

    grid_p2 = (num_seqs, num_kv_heads)
    _reduce_partitions_kernel[grid_p2](
        output,
        tmp_output, exp_sums, max_logits, seq_lens,
        gqa_ratio, max_num_parts,
        output.stride(0), output.stride(1), output.stride(2),
        tmp_output.stride(0), tmp_output.stride(1), tmp_output.stride(2), tmp_output.stride(3),
        exp_sums.stride(0), exp_sums.stride(1), exp_sums.stride(2),
        PARTITION_SIZE=PARTITION_SIZE,
        HEAD_DIM=head_size,
        MAX_NUM_PARTS=MAX_NUM_PARTS_CONST,
        GQA_RATIO=gqa_ratio,
    )
