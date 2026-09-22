import ixformer._C as ops
import torch

__all__ = [
    "store_kv_cache",
    "ref_store_kv_cache",
]


def ref_store_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_batch_idx: torch.Tensor,
    cache_seqlens: torch.Tensor,
):
    """
    Args:
        k:                  (batch_size, seqlen_new, head_num, head_dim)            torch.float16, torch.bfloat16
        v:                  (batch_size, seqlen_new, head_num, head_dim)            torch.float16, torch.bfloat16
        k_cache:            (batch_size_cache, seqlen_cache, head_num, head_dim)    torch.float16, torch.bfloat16
        v_cache:            (batch_size_cache, seqlen_cache, head_num, head_dim)    torch.float16, torch.bfloat16
        cache_batch_idx:    (batch_size,)                                           torch.int32
                            The indices used to index into the KV cache.
        cache_seqlens:      (batch_size,)                                           torch.int32
                            The sequence lengths of the KV cache.
    Returns:
        None
    """
    # 等价实现
    # concatenate k with k_cache, starting at the indices specified by cache_seqlens.
    seqlen_new = k.size(1)
    for kv_batch_idx, cache_kv_batch_idx in enumerate(cache_batch_idx):
        cache_len = cache_seqlens[kv_batch_idx]
        cache_start_idx = cache_len
        cache_end_idx = cache_len + seqlen_new

        k_cache[cache_kv_batch_idx, cache_start_idx:cache_end_idx] = k[kv_batch_idx]
        v_cache[cache_kv_batch_idx, cache_start_idx:cache_end_idx] = v[kv_batch_idx]


def store_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_batch_idx: torch.Tensor,
    cache_seqlens: torch.Tensor,
):
    """
    Currently, only head_dim%2==0 is supported.
    Args:
        k:                  (batch_size, seqlen_new, head_num, head_dim)            torch.float16, torch.bfloat16
        v:                  (batch_size, seqlen_new, head_num, head_dim)            torch.float16, torch.bfloat16
        k_cache:            (batch_size_cache, seqlen_cache, head_num, head_dim)    torch.float16, torch.bfloat16
        v_cache:            (batch_size_cache, seqlen_cache, head_num, head_dim)    torch.float16, torch.bfloat16
        cache_batch_idx:    (batch_size,)                                           torch.int32
                             The indices used to index into the KV cache.
        cache_seqlens:      (batch_size,)                                           torch.int32
                             The sequence lengths of the KV cache
    Returns:
        None
    """
    head_dim = k.size(-1)
    assert head_dim % 2 == 0, "Currently, only head_dim%2==0 is supported."

    ops.infer.store_kv_cache(
        k,
        v,
        k_cache,
        v_cache,
        cache_batch_idx,
        cache_seqlens,
    )
