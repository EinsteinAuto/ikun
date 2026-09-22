"""
corex_fa2_dispatch.py — FlashAttention2 three-mode dispatch for BI-V100

Upstream ref: xllm/core/kernels/ilu/attention.cpp
Bridge ref:   ix_full_bridge_v2.cpp → ixformer::infer::ixinfer_flash_attn_unpad_with_block_tables
                                    → ixformer::infer::xllm_paged_attention

Three modes:
  1. Packed prefill (flash_attn_varlen via ixformer)
  2. Paged decode short context (xllm_paged_attention v1, ctx ≤ 32K)
  3. Paged decode long context (ixinfer_flash_attn_unpad_with_block_tables, ctx > 32K)

Replaces: paged_attn.py _forward_prefix_pytorch (Python Q-tiling fallback)
"""

import logging
import torch
from typing import Optional

logger = logging.getLogger("corex_fa2")

_logged_modes = set()


def _log_once(mode: str, msg: str):
    if mode not in _logged_modes:
        logger.info(msg)
        _logged_modes.add(mode)


# =====================================================================
# Mode 1: Packed prefill — flash_attn_varlen_func
# =====================================================================

def prefill_flash_attn(
    query: torch.Tensor,       # (total_q, num_heads, head_dim)
    key: torch.Tensor,         # (total_k, num_kv_heads, head_dim)
    value: torch.Tensor,       # (total_k, num_kv_heads, head_dim)
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    causal: bool = True,
) -> torch.Tensor:
    """Prefill via ixformer flash_attn_varlen_func."""
    _log_once("prefill", f"Using CoreX FA2 packed prefill: "
              f"Hq={query.shape[1]} D={query.shape[2]}")

    # Try ixformer.contrib first (newer images)
    try:
        from ixformer.contrib.flash_attn import flash_attn_varlen_func
        out = flash_attn_varlen_func(
            query, key, value,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            softmax_scale=scale,
            causal=causal,
        )
        return out
    except (ImportError, AttributeError):
        pass

    # Try ixformer.functions
    try:
        from ixformer.functions import flash_attn_varlen_func
        out = flash_attn_varlen_func(
            query, key, value,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            softmax_scale=scale,
            causal=causal,
        )
        return out
    except (ImportError, AttributeError):
        pass

    raise RuntimeError("prefill_flash_attn: no ixformer flash_attn available")


# =====================================================================
# Mode 2: Paged decode short context — xllm_paged_attention (v1)
# =====================================================================

def decode_paged_v1(
    query: torch.Tensor,       # (num_tokens, num_heads, head_dim)
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    num_kv_heads: int,
    scale: float,
    max_context_len: int,
) -> torch.Tensor:
    """Decode via paged attention v1 (ixformer)."""
    _log_once("decode_v1", f"Using CoreX paged decode v1: "
              f"Hq={query.shape[1]} Hkv={num_kv_heads} D={query.shape[2]}")

    out = torch.empty_like(query)

    # Try ix_full_bridge_v2
    try:
        from ex_engine.python.ix_ops_dispatch import paged_attention_v1
        paged_attention_v1(
            out, query, key_cache, value_cache,
            num_kv_heads, scale, block_tables, context_lens,
            block_size, max_context_len)
        return out
    except (ImportError, RuntimeError):
        pass

    # Direct ixformer path
    try:
        import ixformer.functions as ixf_F
        ixf_F.vllm_single_query_cached_kv_attention(
            out, query, key_cache, value_cache,
            num_kv_heads, scale, block_tables, context_lens,
            block_size, max_context_len, None)
        return out
    except (ImportError, AttributeError):
        pass

    raise RuntimeError("decode_paged_v1: no C++ implementation available")


# =====================================================================
# Mode 3: Paged decode long context — ixinfer_flash_attn_unpad
# =====================================================================

def decode_flash_paged(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor,
    max_seq_q: int,
    max_seq_k: int,
    scale: float,
) -> torch.Tensor:
    """Decode via flash attention with block tables (long context)."""
    _log_once("decode_flash", f"Using CoreX flash paged decode: "
              f"max_k={max_seq_k}")

    out = torch.empty_like(query)

    # Try ix_full_bridge_v2
    try:
        from ex_engine.python.ix_ops_dispatch import flash_attn_with_block_tables
        return flash_attn_with_block_tables(
            query, key_cache, value_cache,
            block_tables, cu_seq_q, cu_seq_k,
            max_seq_q, max_seq_k, scale)
    except (ImportError, RuntimeError):
        pass

    # Direct ixformer
    try:
        import ixformer.functions as ixf_F
        lse = None
        return ixf_F.ixinfer_flash_attn_unpad_with_block_tables(
            query, key_cache, value_cache, out,
            block_tables, cu_seq_q, cu_seq_k,
            max_seq_q, max_seq_k,
            True, -1, -1, scale, 0.0, False, None, None, lse)
    except (ImportError, AttributeError):
        pass

    raise RuntimeError("decode_flash_paged: no C++ implementation available")


# =====================================================================
# Unified dispatch — auto-select mode based on attn_metadata
# =====================================================================

# Threshold: use flash paged decode for context > 32K tokens
V1_V2_THRESHOLD = 32768


def dispatch_attention(
    query: torch.Tensor,
    key_or_cache,
    value_or_cache,
    attn_metadata,
    num_kv_heads: int,
    scale: float,
    block_size: int = 16,
    **kwargs,
) -> torch.Tensor:
    """
    Unified attention dispatch.

    Checks attn_metadata to determine:
      - prefill → flash_attn_varlen_func
      - decode short → xllm_paged_attention (v1)
      - decode long  → ixinfer_flash_attn_unpad_with_block_tables
    """
    is_prefill = getattr(attn_metadata, 'num_prefill_tokens', 0) > 0

    if is_prefill:
        return prefill_flash_attn(
            query, key_or_cache, value_or_cache,
            attn_metadata.query_start_loc,
            attn_metadata.seq_start_loc,
            attn_metadata.max_prefill_seq_len,
            attn_metadata.max_prefill_seq_len,
            scale, causal=True)
    else:
        # Decode path
        context_lens = attn_metadata.seq_lens_tensor
        max_ctx = int(context_lens.max().item()) if context_lens.numel() > 0 else 0

        if max_ctx > V1_V2_THRESHOLD:
            # Long context: flash paged decode
            batch = query.shape[0]
            cu_seq_q = torch.arange(batch + 1, dtype=torch.int32,
                                    device=query.device)
            cu_seq_k = torch.zeros(batch + 1, dtype=torch.int32,
                                   device=query.device)
            cu_seq_k[1:] = context_lens.cumsum(0).to(torch.int32)
            return decode_flash_paged(
                query, key_or_cache, value_or_cache,
                attn_metadata.block_tables,
                cu_seq_q, cu_seq_k, 1, max_ctx, scale)
        else:
            # Short context: paged v1
            return decode_paged_v1(
                query, key_or_cache, value_or_cache,
                attn_metadata.block_tables, context_lens,
                block_size, num_kv_heads, scale, max_ctx)
