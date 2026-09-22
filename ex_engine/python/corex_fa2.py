"""
corex_fa2.py — FlashAttention2 dispatch for BI-V100

Comp 168 log shows THREE dispatch paths:
  corex_fa2.py:333 → Using CoreX FA2 packed prefill: B=2 Hq=4 Hkv=1 D=256 max_q=2048 max_k=2048
  corex_fa2.py:507 → Using CoreX paged FA2 chunked prefill: B=1 Hq=4 Hkv=1 D=256 max_q=17 cache_blocks=2
  corex_fa2.py:225 → Using CoreX paged decode: B=1 Hq=4 Hkv=1 D=256 max_k=45455 partition=256

Dispatch priority (from upstream xllm ILU):
  Tier 0: ix_bridge → ixformer::infer C++ functions (via ix_full_bridge.cpp)
  Tier 1: ixformer.contrib.vllm_flash_attn Python wrappers (in base image)
  Tier 2: ixformer.functions.vllm_single_query_cached_kv_attention (V1 paged)
"""

import logging
import torch
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# ix_bridge (C++ bridge — Tier 0)
# -----------------------------------------------------------------------
_bridge = None
_bridge_available = False

def _ensure_bridge():
    global _bridge, _bridge_available
    if _bridge is not None:
        return _bridge_available
    try:
        from ex_engine.python import ix_bridge
        if ix_bridge.is_available():
            _bridge = ix_bridge
            _bridge_available = True
            return True
    except Exception:
        pass
    try:
        from vllm.model_executor.models.ex_engine.python import ix_bridge
        if ix_bridge.is_available():
            _bridge = ix_bridge
            _bridge_available = True
            return True
    except Exception:
        pass
    return False

# -----------------------------------------------------------------------
# ixformer Python-level backends (Tier 1/2)
# -----------------------------------------------------------------------
_flash_varlen_func = None
_flash_kvcache_func = None
_paged_attn_v1 = None
_ix_available = False

try:
    from ixformer.contrib.vllm_flash_attn import (
        flash_attn_varlen_func as _flash_varlen_func,
    )
    _ix_available = True
except ImportError:
    pass

try:
    from ixformer.contrib.vllm_flash_attn import (
        flash_attn_with_kvcache as _flash_kvcache_func,
    )
except ImportError:
    pass

try:
    import ixformer.functions as ixf_F
    _paged_attn_v1 = ixf_F.vllm_single_query_cached_kv_attention
except (ImportError, AttributeError):
    pass

# -----------------------------------------------------------------------
# Logging state
# -----------------------------------------------------------------------
_logged_packed_prefill = False
_logged_paged_chunked = False
_logged_paged_decode = False


# =========================================================================
# Mode 1: Packed Prefill (no KV cache, fresh sequences)
# =========================================================================
def fa2_packed_prefill(
    query, key, value, cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    softmax_scale=None, causal=True, window_size=(-1, -1),
):
    global _logged_packed_prefill
    batch_size = cu_seqlens_q.shape[0] - 1
    num_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    head_dim = query.shape[2]
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5

    if not _logged_packed_prefill:
        logger.info(
            "Using CoreX FA2 packed prefill: B=%d Hq=%d Hkv=%d D=%d "
            "max_q=%d max_k=%d",
            batch_size, num_heads, num_kv_heads, head_dim,
            max_seqlen_q, max_seqlen_k)
        _logged_packed_prefill = True

    # Tier 0: ix_bridge
    if _ensure_bridge():
        try:
            output = torch.empty_like(query)
            block_tables = torch.empty(0, dtype=torch.int32, device=query.device)
            _bridge.flash_attn_prefill(
                query, key, value, output, block_tables,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, softmax_scale, causal,
                window_size[0], window_size[1])
            return output
        except Exception as e:
            logger.debug("ix_bridge prefill failed: %s", e)

    # Tier 1: ixformer Python
    if _flash_varlen_func is not None:
        return _flash_varlen_func(
            q=query, k=key, v=value,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale, causal=causal,
            window_size=window_size)

    raise RuntimeError("CoreX FA2 packed prefill: no backend available")


# =========================================================================
# Mode 2: Paged Decode (single token per sequence, KV in block cache)
# =========================================================================
def fa2_paged_decode(
    query, key_cache, value_cache, block_tables, cache_seqlens,
    softmax_scale=None, head_mapping=None,
    block_size=16, max_seq_len=0, alibi_slopes=None,
):
    global _logged_paged_decode
    batch_size = query.shape[0]
    num_heads = query.shape[2] if query.dim() == 4 else query.shape[1]
    head_dim = query.shape[-1]
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5
    if max_seq_len == 0:
        max_seq_len = int(cache_seqlens.max().item())

    if not _logged_paged_decode:
        num_kv_heads = key_cache.shape[1] if key_cache.dim() >= 3 else num_heads
        logger.info(
            "Using CoreX paged decode: B=%d Hq=%d Hkv=%d D=%d "
            "max_k=%d partition=256",
            batch_size, num_heads, num_kv_heads, head_dim, max_seq_len)
        _logged_paged_decode = True

    # Tier 0: ix_bridge → ixformer::infer::xllm_paged_attention
    if _ensure_bridge():
        try:
            q_in = query.squeeze(1) if query.dim() == 4 else query
            output = torch.empty_like(q_in)
            num_kv_heads = key_cache.shape[1] if key_cache.dim() >= 3 else num_heads
            _bridge.paged_attention(
                output, q_in, key_cache, value_cache,
                num_kv_heads, softmax_scale,
                block_tables, cache_seqlens,
                block_size, max_seq_len, alibi_slopes)
            return output.unsqueeze(1) if query.dim() == 4 else output
        except Exception as e:
            logger.debug("ix_bridge paged_attention failed: %s", e)

    # Tier 2: ixf_F.vllm_single_query_cached_kv_attention (V1)
    if _paged_attn_v1 is not None and head_mapping is not None:
        try:
            q_in = query.squeeze(1) if query.dim() == 4 else query
            output = torch.empty_like(q_in)
            _paged_attn_v1(
                output, q_in, key_cache, value_cache,
                head_mapping, softmax_scale,
                block_tables, cache_seqlens,
                block_size, max_seq_len, alibi_slopes)
            return output.unsqueeze(1) if query.dim() == 4 else output
        except Exception as e:
            logger.debug("V1 paged attention failed: %s", e)

    # Tier 1: flash_attn_with_kvcache
    if _flash_kvcache_func is not None:
        try:
            return _flash_kvcache_func(
                q=query, k_cache=key_cache, v_cache=value_cache,
                cache_seqlens=cache_seqlens, softmax_scale=softmax_scale,
                causal=True, block_table=block_tables)
        except Exception as e:
            logger.debug("flash_attn_with_kvcache failed: %s", e)

    raise RuntimeError("CoreX FA2 paged decode: no backend available")


# =========================================================================
# Mode 3: Paged Chunked Prefill
# =========================================================================
def fa2_paged_chunked_prefill(
    query, key, value, key_cache, value_cache,
    cu_seqlens_q, max_seqlen_q, block_tables, cache_seqlens,
    softmax_scale=None, causal=True, window_size=(-1, -1), block_size=16,
):
    global _logged_paged_chunked
    batch_size = cu_seqlens_q.shape[0] - 1
    num_heads = query.shape[1]
    num_kv_heads = key.shape[1] if key is not None else num_heads
    head_dim = query.shape[2]
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5

    max_cache_blocks = 0
    if block_tables is not None and block_tables.numel() > 0:
        max_cache_blocks = (block_tables >= 0).sum(dim=-1).max().item()

    if not _logged_paged_chunked:
        logger.info(
            "Using CoreX paged FA2 chunked prefill: B=%d Hq=%d Hkv=%d D=%d "
            "max_q=%d cache_blocks=%d",
            batch_size, num_heads, num_kv_heads, head_dim,
            max_seqlen_q, max_cache_blocks)
        _logged_paged_chunked = True

    # Use varlen for chunked prefill
    if _flash_varlen_func is not None:
        try:
            return _flash_varlen_func(
                q=query, k=key, v=value,
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_q,
                softmax_scale=softmax_scale, causal=causal,
                window_size=window_size)
        except Exception as e:
            logger.debug("FA2 chunked prefill via varlen failed: %s", e)

    raise RuntimeError("CoreX FA2 chunked prefill: no backend available")


# =========================================================================
# Unified dispatch
# =========================================================================
class CoreXFA2:
    def __init__(self, num_heads, num_kv_heads, head_dim):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.available = _ix_available or _ensure_bridge()

    @property
    def is_available(self):
        return self.available

    def packed_prefill(self, query, key, value, cu_seqlens_q, cu_seqlens_k,
                       max_seqlen_q, max_seqlen_k, **kwargs):
        return fa2_packed_prefill(
            query, key, value, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale=self.scale, **kwargs)

    def paged_decode(self, query, key_cache, value_cache, block_tables,
                     cache_seqlens, **kwargs):
        return fa2_paged_decode(
            query, key_cache, value_cache, block_tables, cache_seqlens,
            softmax_scale=self.scale, **kwargs)

    def chunked_prefill(self, query, key, value, key_cache, value_cache,
                        cu_seqlens_q, max_seqlen_q, block_tables,
                        cache_seqlens, **kwargs):
        return fa2_paged_chunked_prefill(
            query, key, value, key_cache, value_cache,
            cu_seqlens_q, max_seqlen_q, block_tables, cache_seqlens,
            softmax_scale=self.scale, **kwargs)
