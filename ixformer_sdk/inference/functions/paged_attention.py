from typing import Union

import ixformer._C as ops
import torch

__all__ = [
    "paged_attention",
    "paged_attention_flashinfer",
    "paged_attention_cache_appended",
]
# paged_attention_cache_append


def paged_attention_cache_appended(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_format: str = "HND",  # STD NHD HND
    key_cache_scales: torch.Tensor = None,
    value_cache_scales: torch.Tensor = None,
):
    if isinstance(key, torch.Tensor):
        ops.infer.paged_attention_cache_appended(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            key.stride(0),
            value.stride(0),
            key_cache.stride(0),
            value_cache.stride(0),
            kv_cache_format,
            key_cache_scales,
            value_cache_scales,
        )
    else:
        raise NotImplementedError()


def paged_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: torch.Tensor = None,
    use_sqrt_alibi: bool = False,
    key_cache_scales: torch.Tensor = None,
    value_cache_scales: torch.Tensor = None,
    kv_cache_format: str = "HND",
    algo: int = -1,
):
    """
    kv_cache_format
        STD : k/v format as same as vllm
        NHD : k/v format is [block_size, num_kv_heads, head_dim] in one page
        HND : k/v format is [num_kv_heads, block_size, head_dim] in one page
    algo
        -1 : auto chooes algorithm according to kv_cache_format
        0  : use the first algorithm
        1  : use the second algorithm
    """
    if isinstance(query, torch.Tensor):
        ops.infer.paged_attention(
            output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            seq_lens,
            block_size,
            max_seq_len,
            use_sqrt_alibi,
            alibi_slopes,
            key_cache_scales,
            value_cache_scales,
            kv_cache_format,
            algo,
        )
    else:
        raise NotImplementedError()


def paged_attention_flashinfer(
    output: torch.Tensor,
    query: torch.Tensor,
    paged_kv_data,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    scale: float,
    max_seq_len: int = -1,
    use_sqrt_alibi: bool = False,
    alibi_slopes: torch.Tensor = None,
    kv_cache_format: str = "HND",
    # key_cache_scales: torch.Tensor = None,
    # value_cache_scales: torch.Tensor = None,
):
    """
    out / query : [num_seqs, num_qo_heads, head_size]
    paged_kv_data
        Tensor:
            NHD [max_num_pages, 2, page_size, num_kv_heads, head_size]
            HND [max_num_pages, 2, num_kv_heads, page_size, head_size]
        tuple(k_data, v_data)
            NHD [max_num_pages, page_size, num_kv_heads, head_size]
            HND [max_num_pages, num_kv_heads, page_size, head_size]
    paged_kv_indptr int32 : [num_seqs + 1]
    paged_kv_indices int32 : [max_num_pages]
    paged_kv_last_page_len int32 : [num_seqs]
    """
    if isinstance(paged_kv_data, tuple):
        k_data, v_data = paged_kv_data
        pack_kv_data = (None, k_data, v_data)
    else:
        pack_kv_data = (paged_kv_data, None, None)

    if isinstance(query, torch.Tensor):
        ops.infer.paged_attention_flashinfer(
            output,
            query,
            *pack_kv_data,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            scale,
            max_seq_len,
            use_sqrt_alibi,
            alibi_slopes,
            kv_cache_format,
        )
    else:
        raise NotImplementedError()
