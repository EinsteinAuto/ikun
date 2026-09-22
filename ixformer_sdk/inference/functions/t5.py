from typing import Union

import ixformer._C as ops
import torch

__all__ = [
    "t5_split_qkv",
    "t5_split_qkv_update_kv_cache",
    "ref_t5_split_qkv_update_kv_cache",
    "ref_t5_split_qkv",
]


def reshape_query(query, head_num, head_dim):
    batch_size, seq_len, _ = query.shape
    query = query.view(batch_size, seq_len, head_num, head_dim)
    query = query.transpose(1, 2)
    return query


def ref_t5_split_qkv(qkv: "torch.Tensor", head_num: int, head_dim: int):
    assert qkv.size(-1) == head_dim * head_num * 3
    batch_size, seq_len, _ = qkv.shape
    q, k, v = torch.chunk(qkv, 3, dim=-1)
    q = reshape_query(q, head_num, head_dim)
    k = reshape_query(k, head_num, head_dim)
    v = reshape_query(v, head_num, head_dim)
    return q, k, v


def t5_split_qkv(qkv: "torch.Tensor", head_num: int, head_dim: int):
    
    """
    Args:
        qkv:                (batch_size, seq_len, head_dim * head_num * 3)        torch.half, torch.bfloat16
        head_num:                                                                 int
        head_dim:                                                                 int
    Returns:
        q:                  (batch_size, head_num, seq_len, head_dim)             torch.half, torch.bfloat16
        k:                  (batch_size, head_num, seq_len, head_dim)             torch.half, torch.bfloat16  
        v:                  (batch_size, head_num, seq_len, head_dim)             torch.half, torch.bfloat16                  
    """
    batch_size, seq_len, _ = qkv.shape
    q = qkv.new_empty([batch_size, head_num, seq_len, head_dim])
    k = qkv.new_empty([batch_size, head_num, seq_len, head_dim])
    v = qkv.new_empty([batch_size, head_num, seq_len, head_dim])
    ops.infer.t5_split_qkv(qkv, q, k, v, head_num, head_dim)
    return q, k, v


def ref_t5_split_qkv_update_kv_cache(
    qkv: "torch.Tensor",
    past_key: "torch.Tensor",
    past_value: "torch.Tensor",
    head_num: int,
    head_dim: int,
):
    assert qkv.size(-1) == head_dim * head_num * 3
    batch_size, seq_len, _ = qkv.shape
    q, k, v = torch.chunk(qkv, 3, dim=-1)
    q = reshape_query(q, head_num, head_dim)
    k = reshape_query(k, head_num, head_dim)
    v = reshape_query(v, head_num, head_dim)
    k = torch.cat([past_key, k], dim=2)
    v = torch.cat([past_value, v], dim=2)
    return q, k, v


def t5_split_qkv_update_kv_cache(
    qkv: "torch.Tensor",
    past_key: "torch.Tensor",
    past_value: "torch.Tensor",
    head_num: int,
    head_dim: int,
):
    
    """
    Args:
        qkv:                (batch_size, 1 , head_dim * head_num * 3)               torch.half, torch.bfloat16
        past_key:           (batch_size, head_num, seq_len - 1, head_dim)           torch.half, torch.bfloat16
        past_value:         (batch_size, head_num, seq_len - 1, head_dim)           torch.half, torch.bfloat16
        head_num:                                                                   int
        head_dim:                                                                   int
    Returns:
        q:                  (batch_size, head_num, 1, head_dim)                     torch.half, torch.bfloat16
        k:                  (batch_size, head_num, seq_len, head_dim)               torch.half, torch.bfloat16  
        v:                  (batch_size, head_num, seq_len, head_dim)               torch.half, torch.bfloat16                  
    """
    batch_size, _, past_seq_len, _ = list(past_key.shape)
    seq_len = past_seq_len + 1
    q = qkv.new_empty([batch_size, head_num, 1, head_dim])
    k = qkv.new_empty([batch_size, head_num, seq_len, head_dim])
    v = qkv.new_empty([batch_size, head_num, seq_len, head_dim])
    ops.infer.t5_split_qkv_update_kv_cache(
        qkv, past_key, past_value, q, k, v, head_num, head_dim
    )
    return q, k, v
