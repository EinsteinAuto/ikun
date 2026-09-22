import math
from typing import List, Union

import torch

from .flash_attn import ixinfer_flash_attn_pad

__all__ = ["scaled_dot_product_attention", "ref_scaled_dot_product_attention"]


def ref_scaled_dot_product_attention(
    query: "torch.Tensor",
    key: "torch.Tensor",
    value: "torch.Tensor",
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
):
    assert len(query.shape) >= 3
    assert len(key.shape) >= 3
    assert len(value.shape) >= 3
    batch_size = query.shape[0]
    L = query.shape[-2]
    S = key.shape[-2]

    if is_causal and attn_mask is not None:
        raise RuntimeError()

    if attn_mask is None and is_causal is False:
        attn_mask = torch.ones([batch_size, 1, 1, S]).bool().to(query.device)

    if is_causal:
        attn_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(query.device)
    if attn_mask.dtype == torch.bool:
        attn_mask = (
            torch.zeros_like(attn_mask).to(query.dtype).masked_fill(~attn_mask, -10000)
        )
    # attn_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0) if is_causal else attn_mask
    # attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf')) if attn_mask.dtype==torch.bool else attn_mask
    attn_weight = torch.softmax(
        (query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask, dim=-1
    )
    attn_weight = torch.dropout(attn_weight, dropout_p, True)
    return attn_weight.to(query.dtype) @ value


def scaled_dot_product_attention(
    query: "torch.Tensor",
    key: "torch.Tensor",
    value: "torch.Tensor",
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
):
    # 1. pytorch版本 query,key的head_dim相等， value可以不相等。但是我们实现的版本需要相等
    # 2. pytorch版本的 L, S可以不相等， 但是我们必须相等，并且为64的倍数
    # 3. pytorch支持 加上 mask
    # 4. mask: 0 表示padding[与后端相反，所以注意如果用传进来的，取反转int；如果是自己生成，则直接生成后端的mask]
    
    """
    Args:
        query:              (N, ..., L, E)        torch.float16, torch.bfloat16
        key:                (N, ..., S, E)        torch.float16, torch.bfloat16
        value:              (N, ..., S, E)        torch.float16, torch.bfloat16
        attn_mask:          (N, ..., L, S)        bool, torch.float32
        dropout_p:                                float32
                Dropout probability; if greater than 0.0, dropout is applied  
        is_causal:                                bool
                If true, assumes causal attention masking and errors if both attn_mask and is_causal are set.
    Returns:
        Tensor:            (N, ..., L, E)         torch.float16, torch.bfloat16
    """
    
    assert len(query.shape) >= 4, "len(query.shape) <4"
    assert len(key.shape) >= 4, "len(key.shape) <4"
    assert len(value.shape) >= 4, "len(value.shape) <4"
    query_shape = list(query.shape)
    key_shape = list(key.shape)
    value_shape = list(value.shape)

    batch_size = query_shape[0]
    q_seq_len = query_shape[-2]
    kv_seq_len = key_shape[-2]

    if len(query_shape) > 4:
        query.view(batch_size, -1, q_seq_len, query_shape[-1])
    if len(key_shape) > 4:
        key.view(batch_size, -1, kv_seq_len, key_shape[-1])
    if len(value_shape) > 4:
        value.view(batch_size, -1, kv_seq_len, value_shape[-1])

    training = query.requires_grad
    atten_scale = 1.0 / (query.size(-1) ** 0.5)

    if is_causal and attn_mask is not None:  # 两者必须有有一个
        raise RuntimeError()

    head_num = query.size(1)
    # 注意，training,inference都支持mask广播；
    if training:
        raise NotImplementedError("not support training!")

    else:  # inference支持mask广播；
        if attn_mask is None and is_causal is False:  # mask 全0
            attn_mask = None
            # attn_mask = (
            #     torch.zeros([batch_size, 1, 1, kv_seq_len]).int().to(query.device)
            # )  # 底层代码1代表mask，0代表保留
        elif is_causal:  # mask 下三角是0，上三角是1
            attn_mask = (
                torch.ones([batch_size, 1, q_seq_len, kv_seq_len], dtype=torch.int)
                .triu(diagonal=1)
                .to(query.device)
            )  # 上三角是1，下三角和对角线是0
        elif attn_mask is not None:  # 外部传进来的，取反，转int
            assert attn_mask.dtype == torch.bool or attn_mask.dtype == torch.float
            assert attn_mask.dim() == 4  # 必须是4维
            if attn_mask.dtype == torch.bool:
                attn_mask = (~attn_mask).int()  # 取非操作，然后转int
        return ixinfer_flash_attn_pad(query, key, value, attn_mask, None, atten_scale)
