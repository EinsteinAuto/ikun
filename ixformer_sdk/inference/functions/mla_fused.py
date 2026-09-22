from typing import Optional

import ixformer._C as ops
import torch

__all__ = [
    # 0.6.3
    "ref_minicpm3_fused_rope",
    "ref_minicpm3_fused_copy_kv",
    "minicpm3_fused_rope",
    "minicpm3_fused_copy_kv",
    # 0.6.6
    "ref_mla_rope_phi",
    "mla_rope_phi",
    "ref_mla_rope",
    "mla_rope",
    "ref_mla_copy_kv",
    "mla_copy_kv",
]


def _rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


# vllm 0.6.3
def ref_minicpm3_fused_rope(
    positions: torch.Tensor,
    long_prompt_offset: torch.Tensor,
    long_short_cos_sin_cache: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    out_query: Optional[torch.Tensor] = None,
    out_key: Optional[torch.Tensor] = None,
):
    idx = torch.add(positions, long_prompt_offset)
    cos_sin = torch.index_select(long_short_cos_sin_cache, 0, idx)

    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.repeat(1, 2).unsqueeze(-2)
    sin = sin.repeat(1, 2).unsqueeze(-2)

    out_query = query * cos + _rotate_neox(query) * sin
    out_key = key * cos + _rotate_neox(key) * sin

    return out_query, out_key


def minicpm3_fused_rope(
    positions: torch.Tensor,
    long_prompt_offset: torch.Tensor,
    long_short_cos_sin_cache: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    out_query: Optional[torch.Tensor] = None,
    out_key: Optional[torch.Tensor] = None,
):
    """
    Args:
        positions:                  (num_tokens,)                                torch.int64
        long_prompt_offset:         (num_tokens,)                                torch.int64
        long_short_cos_sin_cache:   (max_length, head_dim)                       torch.float16, torch.bfloat16
        query:                      (num_tokens, num_q_heads, head_dim)          same as long_short_cos_sin_cache
        key:                        (num_tokens, num_kv_heads, head_dim)         same as long_short_cos_sin_cache
        out_query:                  same as query
        out_key:                    same as key
    Returns:
        out_query:                  same as query
        out_key:                    same as key
    """

    if out_query is None:
        out_query = torch.empty_like(query)
    if out_key is None:
        out_key = torch.empty_like(key)

    ops.infer.minicpm3_fused_rope(
        positions,
        long_prompt_offset,
        long_short_cos_sin_cache,
        query,
        key,
        out_query,
        out_key,
    )
    return out_query, out_key


def ref_minicpm3_fused_copy_kv(
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    v: torch.Tensor,
    new_k: Optional[torch.Tensor] = None,
    new_v: Optional[torch.Tensor] = None,
):
    num_tokens, num_heads, k_head_dim = k_nope.shape
    head_dim = k_pe.shape[-1] + k_head_dim
    v_head_dim = v.shape[-1]

    if new_k is None:
        new_k = k_nope.new_empty([num_tokens, num_heads, head_dim])
    if new_v is None:
        new_v = k_nope.new_empty([num_tokens, num_heads, head_dim])

    new_k[:, :, :k_head_dim] = k_nope
    new_k[:, :, k_head_dim:] = k_pe
    new_v[:, :, :v_head_dim] = v
    new_v[:, :, v_head_dim:] = 0

    return new_k.view(num_tokens, -1), new_v.view(num_tokens, -1)


def minicpm3_fused_copy_kv(
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    v: torch.Tensor,
    new_k: Optional[torch.Tensor] = None,
    new_v: Optional[torch.Tensor] = None,
):
    """
    Args:
        k_nope:       (num_tokens, num_heads, k_head_dim)        torch.float16, torch.bfloat16
        k_pe:         (num_tokens, 1, head_dim - k_head_dim)     same as k_nope
        v:            (num_tokens, num_heads, v_head_dim)        same as k_nope
        new_k:        (num_tokens, num_heads, head_dim)          same as k_nope
        new_v:        (num_tokens, num_heads, head_dim)          same as k_nope
    Returns:
        new_k:        (num_tokens, num_heads, head_dim)          same as k_nope
        new_v:        (num_tokens, num_heads, head_dim)          same as k_nope
    """

    num_tokens, num_heads, k_head_dim = k_nope.shape
    head_dim = k_pe.shape[-1] + k_head_dim

    if new_k is None:
        new_k = k_nope.new_empty([num_tokens, num_heads * head_dim])
    if new_v is None:
        new_v = k_nope.new_empty([num_tokens, num_heads * head_dim])

    ops.infer.minicpm3_fused_copy_kv(k_nope, k_pe, v, new_k, new_v)

    return new_k, new_v


# vllm 0.6.6
def ref_mla_rope_phi(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    long_short_cos_sin_cache: torch.Tensor,
    k: int,
    offsets: Optional[torch.Tensor] = None,
):
    long_prompt_offset = (
        torch.any(positions > k).float() * torch.full_like(positions, k)
    ).long()
    idx = (
        torch.add(positions, long_prompt_offset)
        if long_prompt_offset is not None
        else positions
    )

    idx = torch.add(idx, offsets) if offsets is not None else idx
    cos_sin = torch.index_select(long_short_cos_sin_cache, 0, idx)

    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.repeat(1, 2).unsqueeze(-2)
    sin = sin.repeat(1, 2).unsqueeze(-2)

    query = query * cos + _rotate_neox(query) * sin
    key = key * cos + _rotate_neox(key) * sin

    return query, key


def mla_rope_phi(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    key_out: torch.Tensor,
    long_short_cos_sin_cache: torch.Tensor,
    long_offset: torch.Tensor,
    k: int,
    offsets: Optional[torch.Tensor] = None,
):
    """
    Args:
        positions:                  (num_tokens,)                                torch.int64
        query:                      (num_tokens, num_q_heads, head_dim)          same as long_short_cos_sin_cache
        key:                        (num_tokens, 1, head_dim)                    same as long_short_cos_sin_cache
        key_out:                    (num_tokens, num_q_heads, head_dim)          same as long_short_cos_sin_cache
        long_short_cos_sin_cache:   (max_length, head_dim)                       same as long_short_cos_sin_cache
        long_offset:                (1,)                                         torch.bool
        k:                          int
        offsets:                    (num_tokens,)
    Returns:
        query:
        key_out:
    """

    ops.infer.mla_rope_phi(
        positions,
        query,
        key,
        key_out,
        long_short_cos_sin_cache,
        long_offset,
        k,
        offsets,
    )
    return query, key_out


def ref_mla_rope(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
    rotary_dim: int = None,
    is_neox_style: bool = False,
):
    """PyTorch-native implementation equivalent to forward()."""
    head_size = query.size(-1)
    rotary_dim = rotary_dim or head_size
    query_rot = query[..., :rotary_dim]
    key_rot = key[..., :rotary_dim]
    if rotary_dim < head_size:
        query_pass = query[..., rotary_dim:]
        key_pass = key[..., rotary_dim:]

    cos_sin = cos_sin_cache[
        torch.add(positions, offsets) if offsets is not None else positions
    ]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if is_neox_style:
        cos = cos.repeat(1, 1, 2).unsqueeze(-2)
        sin = sin.repeat(1, 1, 2).unsqueeze(-2)
    else:
        cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
        sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

    rotate_fn = _rotate_neox if is_neox_style else _rotate_gptj
    query_rot = query_rot * cos + rotate_fn(query_rot) * sin
    key_rot = key_rot * cos + rotate_fn(key_rot) * sin

    if rotary_dim < head_size:
        query = torch.cat((query_rot, query_pass), dim=-1)
        key = torch.cat((key_rot, key_pass), dim=-1)
    else:
        query = query_rot
        key = key_rot
    return query, key


def mla_rope(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    key_out: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
    is_neox_style: bool = False,
):
    """
    Args:
        positions:                  (num_tokens,)                                torch.int64
        query:                      (num_tokens, num_q_heads, head_dim)          torch.half torch.bfloat torch.float
        key:                        (num_tokens, 1, head_dim)                    same as query
        key_out:                    (num_tokens, num_q_heads, head_dim)          same as query
        cos_sin_cache:              (max_length, head_dim)                       same as query
        offsets:                    (num_tokens,)                                same as query
        is_neox_style:              bool
    Returns:
        query:
        key_out:
    """

    ops.infer.mla_rope(
        positions,
        query,
        key,
        key_out,
        cos_sin_cache,
        is_neox_style,
        offsets,
    )
    return query, key_out


def ref_mla_copy_kv(key_pe, key_nope, value_nope):
    shape = key_nope.shape[:-1] + (key_pe.shape[-1] + key_nope.shape[-1],)
    key = torch.empty(shape, device=key_nope.device, dtype=key_nope.dtype)
    value = torch.empty_like(key)

    key[..., : key_nope.size(-1)] = key_nope
    key[..., key_nope.size(-1) :] = key_pe
    value[..., : value_nope.size(-1)] = value_nope
    value[..., value_nope.size(-1) :] = 0.0
    return key, value


def mla_copy_kv(key_nope, value_nope, key, value):
    """
    Args:
        key_nope:                   (num_tokens, num_heads, k_nope_dim)          torch.float16, torch.bfloat16, torch.float
        value_nope:                 (num_tokens, num_heads, v_head_dim)          same as key_nope
        key:                        (num_tokens, num_heads, head_dim)            same as key_nope
        value:                      (num_tokens, num_heads, head_dim)            same as key_nope
    Returns:
        key:
        value:
    """

    ops.infer.mla_copy_kv(key_nope, value_nope, key, value)
    return key, value
