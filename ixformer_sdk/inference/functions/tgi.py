import math
from typing import List, Optional

import ixformer._C as ops
import torch

__all__ = [
    "tgi_apply_rotary_emb_torch",
    "tgi_apply_rotary",
    "tgi_gather_prefill_logprobs",
    "ref_paged_attention_v1",
    "ref_paged_attention_v3",
    "get_alibi_slopes",
    "paged_attention_v1",
    "reshape_and_cache_v1",
    "paged_attention_v7",
    "reshape_and_cache",
    "paged_attention_v3",
    "ref_reshape_and_cache_v3",
    "reshape_and_cache_v3",
]


def get_alibi_slopes(total_num_heads: int) -> torch.Tensor:
    closest_power_of_2 = 2 ** math.floor(math.log2(total_num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        dtype=torch.float32,
    )
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != total_num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            dtype=torch.float32,
        )
        num_remaining_heads = min(
            closest_power_of_2, total_num_heads - closest_power_of_2
        )
        extra_powers = torch.arange(
            start=1, end=1 + 2 * num_remaining_heads, step=2, dtype=torch.int32
        )
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
    return slopes


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
    query = query.to(torch.float32).cpu()
    key = key.to(torch.float32).cpu()
    value = value.to(torch.float32).cpu()
    attn = torch.einsum("qhd,khd->hqk", query, key)
    if attn_mask is not None:
        attn_mask = attn_mask.cpu()
        attn = attn + attn_mask
    attn = torch.softmax(attn, dim=-1)
    out = torch.einsum("hqk,khd->qhd", attn, value)
    out = out.to(device).to(dtype)
    return out


def ref_paged_attention_v1(
    output: torch.Tensor,
    query: torch.Tensor,
    num_q_per_kv: int,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    use_alibi: bool,
) -> None:
    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]

    num_input_tokens = query.shape[0]
    device = output.device
    slopes = (
        get_alibi_slopes(num_query_heads)
        .to(device)
        .to(torch.float32)
        .view(num_query_heads, 1, 1)
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

            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape(num_kv_heads, head_size)
            keys.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values.append(v)
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)
        if num_q_per_kv > 1:
            keys = torch.repeat_interleave(keys, num_q_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_q_per_kv, dim=1)
        scale = 1.0 / (head_size**0.5)
        if use_alibi:
            offsets = get_alibi_mask(
                num_query_heads, context_len, output.device, output.dtype
            )
            mask = offsets * slopes
            mask = mask.to(output.dtype)
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
        output[i].copy_(out, non_blocking=True)


def ref_paged_attention_v3(
    output: torch.Tensor,
    query: torch.Tensor,
    num_q_per_kv: int,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    use_alibi: bool,
) -> None:
    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[1]
    head_size = query.shape[2]
    block_size = value_cache.shape[2] * 4

    num_input_tokens = query.shape[0]
    device = output.device
    slopes = (
        get_alibi_slopes(num_query_heads)
        .to(device)
        .to(torch.float32)
        .view(num_query_heads, 1, 1)
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

            k = key_cache[block_number, :, block_offset // 4, :, block_offset % 4, :]
            k = k.reshape(num_kv_heads, head_size)
            keys.append(k)

            v = value_cache[block_number, :, block_offset // 4, :, block_offset % 4, :]
            v = v.reshape(num_kv_heads, head_size)
            values.append(v)
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)
        if num_q_per_kv > 1:
            keys = torch.repeat_interleave(keys, num_q_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_q_per_kv, dim=1)
        scale = 1.0 / (head_size**0.5)
        if use_alibi:
            offsets = get_alibi_mask(
                num_query_heads, context_len, output.device, output.dtype
            )
            mask = offsets * slopes
            mask = mask.to(output.dtype)
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
        output[i].copy_(out, non_blocking=True)


def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        seq_len, head_nums, _ = x.shape
        return torch.stack((-x2, x1), dim=-1).reshape(seq_len, head_nums, -1)


def tgi_apply_rotary_emb_torch(
    x: "torch.Tensor",
    cos: "torch.Tensor",
    sin: "torch.Tensor",
    interleaved: bool = False,
):
    """
    x: (seqlen, num_heads, headdim)
    cos, sin: (seqlen, 1, rotary_dim / 2)
    interleaved: bool. 在interleaved的实现中,对奇偶维度旋转需要将维度两两交错，实现较为复杂。
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    assert cos.shape == sin.shape
    if cos.dim() == 2:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    if interleaved:
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
    else:
        cos = cos.repeat(1, 1, 2)
        sin = sin.repeat(1, 1, 2)
    return torch.cat(
        [
            x[..., :ro_dim].float() * cos.float()
            + rotate_half(x[..., :ro_dim].float(), interleaved) * sin.float(),
            x[..., ro_dim:].float(),
        ],
        dim=-1,
    ).to(x.dtype)


def tgi_apply_rotary(
    querys: List[torch.Tensor],
    cos: "torch.Tensor",
    sin: "torch.Tensor",
    outs: List[torch.Tensor] = None,
    is_neox_style: bool = True,
):
    """
    Args:
        querys:             [(num_tokens, num_heads, head_size)]      List[torch.half],List[torch.float],List[torch.bfloat16]
        cos:                (max_position, 1, head_size //2)          torch.half, torch.float, torch.bfloat16
        sin:                (max_position, 1, head_size //2)          torch.half, torch.float, torch.bfloat16
        is_neox_style:                                                bool
                        判断是否使用Neox,默认为True,即不使用interleaved
        outs:               [(num_tokens, num_heads, head_size)]      List[torch.half],List[torch.float],List[torch.bfloat16]
    Returns:
        outs:               [(num_tokens, num_heads, head_size)]      List[torch.half],List[torch.float],List[torch.bfloat16]
    """

    assert sin.shape == cos.shape
    rotary_dim = cos.shape[-1]
    return_type = False
    if len(querys) == 1:
        query = querys[0]
        query_dim = query.shape[-1]
        query1 = query[..., :rotary_dim]
        query2 = query[..., rotary_dim : 2 * rotary_dim]
    elif len(querys) == 2:
        return_type = True
        query1 = querys[0]
        query2 = querys[1]
        assert query1.shape == query2.shape
        query_dim = query1.shape[-1] * 2
    else:
        raise ValueError(
            f"Invalid number for querys: {len(querys)}. " "Expected number 1, or 2."
        )

    assert rotary_dim * 2 <= query_dim
    if outs is None:
        query_shape = query1.shape
        out = torch.empty(*(query_shape[:-1] + [rotary_dim * 2]))
        out1 = out[..., :rotary_dim]
        out2 = out[..., rotary_dim : 2 * rotary_dim]
    else:
        assert len(querys) == len(outs)
        for query, out in zip(querys, outs):
            assert query.shape == out.shape
        if len(outs) == 1:
            out = outs[0]
            out1 = out[..., :rotary_dim]
            out2 = out[..., rotary_dim : 2 * rotary_dim]
        else:
            out1 = outs[0]
            out2 = outs[1]

    if cos.dim() == 2:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    ops.infer.tgi_rotary_embedding_neox(
        query1, query2, cos, sin, out1, out2, is_neox_style
    )

    if return_type:
        return out1, out2
    else:
        return torch.cat([out1, out2], dim=-1)


def tgi_gather_prefill_logprobs(
    logits: "torch.Tensor",
    prefill_tokens_indices: "torch.Tensor",
    output: "torch.Tensor" = None,
):
    """
    Args:
        logits:                     (num_tokens, vocab_size)                    torch.half, torch.bfloat16
        prefill_tokens_indices:     (tokens_indices)                            torch.int
        output:                     (tokens_indices, 1)                         torch.half, torch.bfloat16
    Returns:
        output:                     (tokens_indices, 1)                         torch.half, torch.bfloat16
    """
    if output is None:
        output = logits.new_empty(prefill_tokens_indices.shape)
    ops.infer.tgi_gather_prefill_logprobs(logits, prefill_tokens_indices, output)
    return output


def paged_attention_v1(
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
    use_sqrt_alibi: bool = False,
):
    """
    Args:
        output:                         (num_tokens, num_heads, head_size)                                      torch.half, torch.bfloat16
        query:                          (num_tokens, num_heads, head_size)                                      torch.half, torch.bfloat16
        key_cache:                      (num_blocks, num_kv_heads, block_size, head_size)                       torch.half, torch.bfloat16
        value_cache:                    (num_blocks, num_kv_heads, block_size, head_size)                       torch.half, torch.bfloat16
        num_kv_heads:                                                                                           int
        scale:                                                                                                  float   
        block_tables:                   (num_tokens, max_num_blocks_per_seq)                                    torch.int64
        context_lens_cpu:               (num_tokens)                                                            torch.int32
        context_lens:                   (num_tokens)                                                            torch.int32      
        block_size:                                                                                             int
        max_context_len:                                                                                        int
        alibi_slopes:                   (num_heads)                                                             torch.float32
        use_sqrt_alibi:                                                                                         bool                        
    Returns:
        output:                         (num_tokens, num_heads, head_size)                                      torch.half, torch.bfloat16
    """
    ops.infer.tgi_single_query_cached_kv_attention(
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
        query.stride(0),
        use_sqrt_alibi,
        alibi_slopes,
    )


def paged_attention_v3(
    output: "torch.Tensor",
    query: "torch.Tensor",
    key_cache: "torch.Tensor",
    value_cache: "torch.Tensor",
    head_mapping: "torch.Tensor",
    scale: float,
    block_tables: "torch.Tensor",
    context_lens: "torch.Tensor",
    block_size: int,
    max_context_len: int,
    alibi_slopes: "torch.Tensor" = None,
    use_sqrt_alibi: bool = False,
): 
    """
    Args:
        output:                         (num_tokens, num_heads, head_size)                                      torch.half, torch.bfloat16
        query:                          (num_tokens, num_heads, head_size)                                      torch.half, torch.bfloat16
        key_cache:                      (num_blocks, num_kv_heads, block_size, head_size)                       torch.half, torch.bfloat16
        value_cache:                    (num_blocks, num_kv_heads, block_size, head_size)                       torch.half, torch.bfloat16
        num_kv_heads:                                                                                           int
        scale:                                                                                                  float   
        block_tables:                   (num_tokens, max_num_blocks_per_seq)                                    torch.int64
        context_lens_cpu:               (num_tokens)                                                            torch.int32
        context_lens:                   (num_tokens)                                                            torch.int32      
        block_size:                                                                                             int
        max_context_len:                                                                                        int
        alibi_slopes:                   (num_heads)                                                             torch.float32
        use_sqrt_alibi:                                                                                         bool                        
    Returns:
        output:                         (num_tokens, num_heads, head_size)                                      torch.half, torch.bfloat16
    """   
    ops.infer.single_query_cached_kv_attention_v3(
        output,
        query,
        key_cache,
        value_cache,
        head_mapping,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        query.stride(0),
        use_sqrt_alibi,
        alibi_slopes,
    )



def paged_attention_v7(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    max_context_len: int,
    alibi_slopes: torch.Tensor = None,
    use_sqrt_alibi: bool = False,
):
    
    """
    Args:
        output:                         (num_tokens, num_heads, head_size)                                      torch.half, torch.bfloat16
        query:                          (num_tokens, num_heads, head_size)                                      torch.half, torch.bfloat16
        key_cache:                      (num_blocks, num_kv_heads, block_size, head_size)                       torch.half, torch.bfloat16
        value_cache:                    (num_blocks, num_kv_heads, block_size, head_size)                       torch.half, torch.bfloat16
        num_kv_heads:                                                                                           int
        scale:                                                                                                  float   
        block_tables:                   (num_tokens, max_num_blocks_per_seq)                                    torch.int64
        context_lens:                   (num_tokens)                                                            torch.int32      
        block_size:                                                                                             int
        max_context_len:                                                                                        int
        alibi_slopes:                   (num_heads)                                                             torch.float32
        use_sqrt_alibi:                                                                                         bool                        
    Returns:
        output:                         (num_tokens, num_heads, head_size)                                      torch.half, torch.bfloat16
    """
    num_blocks = key_cache.size(0)
    head_size = query.size(-1)
    key_cache = key_cache.view(num_blocks, num_kv_heads, block_size, head_size)
    value_cache = value_cache.view(num_blocks, num_kv_heads, block_size, head_size)
    ops.infer.vllm_paged_attention(
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
        True,
        -1,
        -1,
        0.0,
        False,
        use_sqrt_alibi,
    )
    return output

def reshape_and_cache_v1(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """
    Args:
        key:                        (num_tokens, num_heads, head_size)                                                                                  torch.half, torch.float, torch.bfloat16
        value:                      (num_tokens, num_heads, head_size)                                                                                  torch.half, torch.float, torch.bfloat16
        key_cache:                  (num_blocks, num_heads, head_size//8, block_size, 8) or (num_blocks, num_heads, head_size//4, block_size, 4)        torch.half, torch.float, torch.bfloat16
                    if dtype=torch.half or torch.bfloat16,key_cache shape: (num_blocks, num_heads, head_size//8, block_size, 8)
                    if dtype=torch.float,key_cache shape: (num_blocks, num_heads, head_size//4, block_size, 4)
        value_cache:                (num_blocks, num_heads, head_size//8, block_size, 8) or (num_blocks, num_heads, head_size//4, block_size, 4)        torch.half, torch.float, torch.bfloat16
                    if dtype=torch.half or torch.bfloat16,value_cache shape: (num_blocks, num_heads, head_size//8, block_size, 8)
                    if dtype=torch.float,value_cache shape: (num_blocks, num_heads, head_size//4, block_size, 4)
        slot_mapping:               (num_tokens)                                                                                                        torch.long
    Returns:
        None, 对key_cache,value_cache进行in place 操作
    """
    ops.infer.vllm_cache_ops_reshape_and_cache_v4(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        key.stride(0),
        value.stride(0),
    )


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """
    Args:
        key:                        (num_tokens, num_heads, head_size)                  torch.half, torch.bfloat16
        value:                      (num_tokens, num_heads, head_size)                  torch.half, torch.bfloat16
        key_cache:                  (num_blocks, num_heads, block_size, head_size)      torch.half, torch.bfloat16
        value_cache:                (num_blocks, num_heads, block_size, head_size)      torch.half, torch.bfloat16
        slot_mapping:               (num_tokens)                                        torch.long
    Returns:
        None, 对key_cache,value_cache进行in place 操作
    """
    num_tokens, num_kv_heads, head_size = key.shape
    num_blocks = key_cache.size(0)
    key_cache = key_cache.view(num_blocks, num_kv_heads, -1, head_size)
    value_cache = value_cache.view(num_blocks, num_kv_heads, -1, head_size)
    ops.infer.vllm_cache_ops_reshape_and_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        key.stride(0),
        value.stride(0),
    )


def ref_reshape_and_cache_v3(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    num_tokens,
    num_heads,
    head_size,
    block_size,
):
    reshaped_key = key.view(num_tokens, num_heads, head_size // 32, 32)
    reshaped_value = value.reshape(num_tokens, num_heads, head_size // 32, 32)
    for i in range(num_tokens):

        block_idx = torch.div(slot_mapping[i], block_size, rounding_mode="floor")
        block_offset = slot_mapping[i] % block_size

        key_cache[
            block_idx, :, block_offset // 4, :, block_offset % 4, :
        ] = reshaped_key[i]
        value_cache[
            block_idx, :, block_offset // 4, :, block_offset % 4, :
        ] = reshaped_value[i]


def reshape_and_cache_v3(
    key: "torch.Tensor",
    value: "torch.Tensor",
    key_cache: "torch.Tensor",
    value_cache: "torch.Tensor",
    slot_mapping: "torch.Tensor",
):
    """
    Args:
        key:                        (num_tokens, num_heads, head_size)                                      torch.half, torch.bfloat16
        value:                      (num_tokens, num_heads, head_size)                                      torch.half, torch.bfloat16
        key_cache:                  (num_blocks, num_heads, block_size // 4, head_size // 32, 4, 32)        torch.half, torch.bfloat16
                    目前block_size 只支持16,head_size 只支持64,128,256
        value_cache:                (num_blocks, num_heads, block_size // 4, head_size // 32, 4, 32)        torch.half, torch.bfloat16
        slot_mapping:               (num_tokens)                                                            torch.int
    Returns:
        None, 对key_cache,value_cache进行in place 操作
    """
    if key.dim() != 3 or key.shape != value.shape or key.size(-1) not in [64, 128, 256]:
        raise NotImplementedError(
            "reshape_and_cache_v3 only support key.dim()==3 and key.shape== value.shape and head_size must be 64, 128 , 256!"
        )
    ops.infer.cache_ops_reshape_and_cache_v3(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        key.stride(0),
        value.stride(0),
    )
