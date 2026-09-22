import math
from typing import Literal, Optional, Union

import ixformer._C as ops
import ixformer._C._functions as CF
import torch

from ixformer.core import config

from .linear import linear
from .paged_attention import paged_attention as paged_attention_ixformer_impl

__all__ = [
    "ref_lmdeploy_paged_attention",
    "lmdeploy_paged_attention",
]

weak_ref_tensor = ops.infer.weak_ref_tensor


def ref_lmdeploy_paged_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: torch.Tensor,
    scale: float,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    max_context_len: int,
    alibi_slopes: torch.Tensor = None,
    softcap: float = 0.0,
    window_left: int = -1,
    window_right: int = -1,
    use_sqrt_alibi: bool = False,
    quant_type: int = 0,
    is_bbhh: bool = False,
):
    assert window_right in [-1, 0]

    if is_bbhh:
        key_cache = key_cache.permute(0, 2, 1, 3).contiguous()
        value_cache = value_cache.permute(0, 2, 1, 3).contiguous()

    def get_alibi_mask(num_heads, seqlen, device, dtype):
        x = torch.arange(0, seqlen, device=device, dtype=torch.float32).view(-1, 1)
        y = torch.tensor(seqlen - 1, device=device, dtype=torch.float32).view(1, -1)
        offsets = -(y - x).view(1, 1, seqlen)
        return offsets

    def ref_masked_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = query * scale
        dtype = query.dtype
        device = query.device
        query = query.to(torch.float32)
        key = key.to(torch.float32)
        value = value.to(torch.float32)
        attn = torch.einsum("qhd,khd->hqk", query, key)
        if attn_mask is not None:
            attn_mask = attn_mask
            attn = attn + attn_mask
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("hqk,khd->qhd", attn, value)
        out = out.to(device).to(dtype)
        return out

    head_size = query.shape[-1]
    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[1]
    num_input_tokens = query.shape[0]

    num_q_per_kv = num_query_heads // num_kv_heads
    slopes = (
        alibi_slopes.view(num_query_heads, 1, 1)
        if alibi_slopes is not None
        else alibi_slopes
    )

    for i in range(num_input_tokens):
        q = query[i].unsqueeze(0)
        block_table = block_tables[i]
        context_len = int(context_lens[i])

        keys = []
        values = []
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, block_offset, :]
            keys.append(k)

            v = value_cache[block_number, :, block_offset, :]
            values.append(v)
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)
        if num_q_per_kv > 1:
            keys = torch.repeat_interleave(keys, num_q_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_q_per_kv, dim=1)
        if alibi_slopes is not None:
            offsets = get_alibi_mask(
                num_query_heads, context_len, output.device, output.dtype
            )
            mask = offsets * slopes
            mask = mask.to(output.dtype)
            if window_left != -1:
                index = torch.ones_like(mask, dtype=torch.int32, device=mask.device)
                index[:, :, (context_len - 1 - window_left) :] = 0
                index = index.bool()
                mask.masked_fill_(index, float("-inf"))
        else:
            if window_left != -1:
                mask = torch.zeros([1, 1, context_len], dtype=q.dtype, device=q.device)
                index = torch.ones_like(mask, dtype=torch.int32, device=mask.device)
                index[:, :, (context_len - 1 - window_left) :] = 0
                index = index.bool()
                mask.masked_fill_(index, float("-inf"))
            else:
                mask = None

        out = ref_masked_attention(
            q,
            keys,
            values,
            scale,
            mask,
        )
        out = out.view(num_query_heads, head_size)
        if softcap != 0.0:
            out = softcap * torch.tanh(out / softcap)
        output[i].copy_(out, non_blocking=True)

    return output


def lmdeploy_paged_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: torch.Tensor,
    scale: float,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    max_context_len: int,
    alibi_slopes: torch.Tensor = None,
    softcap: float = 0.0,
    causal: bool = True,
    window_left: int = -1,
    window_right: int = -1,
    use_cuda_graph: bool = False,
    use_sqrt_alibi: bool = False,
    quant_type: int = 0,
    is_bbhh: bool = False,
):
    """
    is_bbhh = False  key_cache, value_cache: [num_blocks, block_size, num_kv_heads, head_size]
    is_bbhh = True  key_cache, value_cache: [num_blocks, num_kv_heads, block_size, head_size]

    is_bbhh = False
     Arguments:
        query:              [torch.half, torch.bfloat16]  [num_tokens, num_heads, head_size]
        key_cache:          [torch.half, torch.bfloat16]  [num_blocks, num_kv_heads, block_size, head_size]
        value_cache:        [torch.half, torch.bfloat16]  [num_blocks, num_kv_heads, block_size, head_size]
        num_kv_heads:       int
        scale:              float
        block_tables:       [torch.int64]                 [num_tokens, max_num_blocks_per_seq]
        context_lens:       [torch.int32]                 [num_tokens]
        block_size:         int
        max_context_len:    int
        alibi_slopes:       [torch.float32]               [num_heads]
        softcap:            float
        causal:             bool
        window_left:        int
        window_right:       int
        use_sqrt_alibi:     bool: False
    Return:
        output:             [torch.half, torch.bfloat16]  [num_tokens, num_heads, head_size]
    """

    ops.infer.lmdeploy_paged_attention(
        output,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes,
        causal,
        window_left,
        window_right,
        softcap,
        use_cuda_graph,
        use_sqrt_alibi,
        is_bbhh,
        quant_type,
    )
    return output

    # lmdeploy_paged_attention = lmdeploy_paged_attention_ixinfer
