import math
from typing import Optional, Union

import ixformer._C as ops
import ixformer._C._functions as CF
import torch

from ixformer.core import config

from .linear import linear
from .paged_attention import paged_attention as paged_attention_ixformer_impl

__all__ = [
    "ref_vllm_paged_attention",
    "vllm_paged_attention",
    "ref_vllm_paged_attention_mla",
    "vllm_paged_attention_mla",
    "vllm_paged_attention_mla_fused",
    "ref_vllm_paged_attention_mla_int8",
    "vllm_paged_attention_mla_int8",
    "ref_vllm_paged_attention_v5",
    "vllm_paged_attention_v5",
    "ref_vllm_paged_attention_v4",
    "vllm_paged_attention_v4",
    "ref_vllm_reshape_and_cache_v4",
    "vllm_reshape_and_cache_v4",
    "ref_vllm_reshape_and_cache",
    "vllm_reshape_and_cache",
    "vllm_cache_ops_reshape_and_cache",
    "ref_reshape_and_cache_flash",
    "reshape_and_cache_flash",
    "ref_vllm_rotary_embedding",
    "vllm_rotary_embedding",
    "ref_vllm_rotary_embedding_phi",
    "vllm_rotary_embedding_phi",
    "ref_vllm_batched_rotary_embedding",
    "vllm_batched_rotary_embedding",
    "ref_vllm_copy_blocks",
    "vllm_copy_blocks",
    "ref_vllm_swap_blocks",
    "vllm_swap_blocks",
    "vllm_gather_cache",
    "vllm_gather_cache_int8",
    "ref_vllm_gather_cache_int8",
    "ref_vllm_gather_cache",
    "ref_vllm_concat_and_cache_mla",
    "vllm_concat_and_cache_mla",
    "ref_vllm_concat_and_cache_mla_int8",
    "vllm_concat_and_cache_mla_int8",
    "vllm_llama_mlp",
    "gptq_gemm",
    "vllm_gptq_shuffle",
    "vllm_moe_topk_softmax",
    "vllm_moe_align_block_size",
    "ref_vllm_invoke_fused_moe_kernel",
    "vllm_invoke_fused_moe_kernel",
    "advance_step_flashattn",
    "weak_ref_tensor",
    # customized ops
    "vllm_rotary_embedding_with_key_layer_norm",
    "ref_vllm_rotary_embedding_with_key_layer_norm",
]

weak_ref_tensor = ops.infer.weak_ref_tensor


def ref_vllm_paged_attention(
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
):
    assert window_right in [-1, 0]

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


def vllm_paged_attention_ixinfer(
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
):
    """
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
        causal,
        window_left,
        window_right,
        softcap,
        use_cuda_graph,
        use_sqrt_alibi,
    )
    return output


def vllm_paged_attention_ixformer(
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
    use_sqrt_alibi: bool = False,
    need_view: bool = True,
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

    if need_view:
        num_blocks = key_cache.size(0)
        head_size = query.size(-1)
        key_cache = key_cache.view(num_blocks, num_kv_heads, block_size, head_size)
        value_cache = value_cache.view(num_blocks, num_kv_heads, block_size, head_size)
    paged_attention_ixformer_impl(
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
        use_sqrt_alibi,
    )
    return output


def ref_vllm_paged_attention_mla(
    output: torch.Tensor,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: float,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    max_context_len: int,
):
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
        attn = torch.einsum("qhd,khd->hqk", query, key)
        if attn_mask is not None:
            attn_mask = attn_mask
            attn = attn + attn_mask
        attn = attn.to(torch.float)
        attn = torch.softmax(attn, dim=-1)
        value = value.to(torch.float)
        out = torch.einsum("hqk,khd->qhd", attn, value)
        out = out.to(device).to(dtype)
        return out

    num_heads = query.shape[-2]
    kv_lora_rank = output.shape[-1]
    block_size = kv_cache.shape[1]
    num_input_tokens = query.shape[0]

    for i in range(num_input_tokens):
        q = query[i].unsqueeze(0)
        block_table = block_tables[i]
        context_len = int(context_lens[i])

        keys = []
        values = []
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = kv_cache[block_number, block_offset, :]
            keys.append(k)

            v = kv_cache[block_number, block_offset, :kv_lora_rank]
            values.append(v)
        keys = torch.stack(keys, dim=0).unsqueeze(-2).repeat(1, num_heads, 1)
        values = torch.stack(values, dim=0).unsqueeze(-2).repeat(1, num_heads, 1)
        mask = None

        out = ref_masked_attention(
            q,
            keys,
            values,
            scale,
            mask,
        )
        out = out.view(num_heads, kv_lora_rank)
        output[i].copy_(out, non_blocking=True)

    return output


def ref_vllm_paged_attention_mla_int8(
    output: torch.Tensor,
    query: torch.Tensor,
    query_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_cache_scale: torch.Tensor,
    scale: float,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    max_context_len: int,
):
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
        attn = torch.einsum("qhd,khd->hqk", query, key)
        if attn_mask is not None:
            attn_mask = attn_mask
            attn = attn + attn_mask
        attn = attn.to(torch.float)
        attn = torch.softmax(attn, dim=-1)
        value = value.to(torch.float)
        out = torch.einsum("hqk,khd->qhd", attn, value)
        out = out.to(device).to(dtype)
        return out

    # dequant q
    num_heads = query.shape[-2]
    kv_lora_rank = output.shape[-1]
    block_size = kv_cache.shape[1]
    num_input_tokens = query.shape[0]
    query = query * query_scale.unsqueeze(-1)

    for i in range(num_input_tokens):
        q = query[i].unsqueeze(0)
        block_table = block_tables[i]
        context_len = int(context_lens[i])

        keys = []
        values = []
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = kv_cache[block_number, block_offset, :kv_lora_rank]
            k_scale = kv_cache_scale[block_number, block_offset, 0]
            k_pe = kv_cache[block_number, block_offset, kv_lora_rank:]
            k_pe_scale = kv_cache_scale[block_number, block_offset, 1]
            k = k * k_scale
            v = k
            k_pe = k_pe * k_pe_scale
            k = torch.cat((k, k_pe), dim=-1)
            keys.append(k)
            values.append(v)
        keys = torch.stack(keys, dim=0).unsqueeze(-2).repeat(1, num_heads, 1)
        values = torch.stack(values, dim=0).unsqueeze(-2).repeat(1, num_heads, 1)
        mask = None

        out = ref_masked_attention(
            q,
            keys,
            values,
            scale,
            mask,
        )
        out = out.view(num_heads, kv_lora_rank)
        output[i].copy_(out, non_blocking=True)

    return output


def vllm_paged_attention_mla(
    output: torch.Tensor,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: float,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    max_context_len: int,
    use_cuda_graph: bool = False,
):
    """
    Args:
        output:                         (num_tokens, num_heads, kv_lora_rank)                                   torch.half, torch.bfloat16
        query:                          (num_tokens, num_heads, kv_lora_rank+qk_rope_head_dim)                  torch.half, torch.bfloat16
        kv_cache:                       (num_blocks, block_size, kv_lora_rank+qk_rope_head_dim)                 torch.half, torch.bfloat16
        scale:                                                                                                  float
        block_tables:                   (num_tokens, max_num_blocks_per_seq)                                    torch.int64
        context_lens:                   (num_tokens)                                                            torch.int32
        max_context_len:                                                                                        int
        use_cuda_graph:                                                                                         bool
    Returns:
        output:                         (num_tokens, num_heads, kv_lora_rank)                                   torch.half, torch.bfloat16
    """
    ops.infer.vllm_paged_attention_mla(
        output,
        query,
        kv_cache,
        scale,
        block_tables,
        context_lens,
        max_context_len,
        use_cuda_graph,
    )
    return output


def vllm_paged_attention_mla_int8(
    output: torch.Tensor,
    query: torch.Tensor,
    query_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_cache_scale: torch.Tensor,
    scale: float,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    max_context_len: int,
    use_cuda_graph: bool = False,
):
    """
    Args:
        output:                         (num_tokens, num_heads, kv_lora_rank)                                   torch.half, torch.bfloat16
        query:                          (num_tokens, num_heads, kv_lora_rank+qk_rope_head_dim)                  torch.int8
        query_scale:                    (num_tokens, num_heads)                                                 torch.float
        kv_cache:                       (num_blocks, block_size, kv_lora_rank+qk_rope_head_dim)                 torch.half, torch.bfloat16
        kv_cache_scale:                 (num_blocks, block_size, 2)                                             torch.float
        scale:                                                                                                  float
        block_tables:                   (num_tokens, max_num_blocks_per_seq)                                    torch.int64
        context_lens:                   (num_tokens)                                                            torch.int32
        max_context_len:                                                                                        int
        use_cuda_graph:                                                                                         bool
    Returns:
        output:                         (num_tokens, num_heads, kv_lora_rank)                                   torch.half, torch.bfloat16
    """
    ops.infer.vllm_paged_attention_mla_int8(
        output,
        query,
        query_scale,
        kv_cache,
        kv_cache_scale,
        scale,
        block_tables,
        context_lens,
        max_context_len,
        use_cuda_graph,
    )
    return output


def vllm_paged_attention_mla_fused(
    output: torch.Tensor,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: float,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    max_context_len: int,
    k_c_normed: torch.Tensor = None,
    k_pe: torch.Tensor = None,
    use_cuda_graph: bool = False,
):
    """
    Args:
        q_nope:                         (num_tokens, num_heads, kv_lora_rank)                                   torch.half, torch.bfloat16
        q_pe:                           (num_tokens, num_heads, qk_rope_head_dim)                               torch.half, torch.bfloat16
        kv_cache:                       (num_blocks, block_size, kv_lora_rank+qk_rope_head_dim)                 torch.half, torch.bfloat16
        scale:                                                                                                  float
        block_tables:                   (num_tokens, max_num_blocks_per_seq)                                    torch.int64
        context_lens:                   (num_tokens)                                                            torch.int32
        max_context_len:                                                                                        int
        k_c_normed:                     (num_tokens, kv_lora_rank)                                              torch.half, torch.bfloat16
        k_pe:                           (num_tokens, qk_rope_head_dim)                                          torch.half, torch.bfloat16
        use_cuda_graph:                                                                                         bool
    Returns:
        output:                         (num_tokens, num_heads, kv_lora_rank)                                   torch.half, torch.bfloat16
    """
    ops.infer.vllm_paged_attention_mla_fused(
        output,
        q_nope,
        q_pe,
        kv_cache,
        scale,
        block_tables,
        context_lens,
        max_context_len,
        k_c_normed,
        k_pe,
        use_cuda_graph,
    )
    return output


def ref_vllm_paged_attention_v5(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: torch.Tensor,
    scale: float,
    block_tables: torch.Tensor,
    context_lens_cpu: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    max_context_len: int,
    alibi_slopes: torch.Tensor = None,
):
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

    return output


def vllm_paged_attention_v5(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: torch.Tensor,
    scale: float,
    block_tables: torch.Tensor,
    context_lens_cpu: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    max_context_len: int,
    alibi_slopes: torch.Tensor = None,
    use_sqrt_alibi: bool = False,
    need_view: bool = True,
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
    if need_view:
        num_blocks = key_cache.size(0)
        head_size = query.size(-1)
        key_cache = key_cache.view(num_blocks, num_kv_heads, block_size, head_size)
        value_cache = value_cache.view(num_blocks, num_kv_heads, block_size, head_size)
    ops.infer.vllm_paged_attention_v5(
        output,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        context_lens_cpu,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes,
        use_sqrt_alibi,
    )
    return output


def ref_vllm_paged_attention_v4(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: torch.Tensor,
    scale: float,
    block_tables: torch.Tensor,
    context_lens_cpu: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    max_context_len: int,
    alibi_slopes: torch.Tensor = None,
    use_sqrt_alibi: bool = False,
):
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

    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]
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
        if alibi_slopes is not None:
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

    return output


def vllm_paged_attention_v4(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: torch.Tensor,
    scale: float,
    block_tables: torch.Tensor,
    context_lens_cpu: torch.Tensor,
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
    ops.infer.vllm_paged_attention_v4(
        output,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        context_lens_cpu,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes,
        use_sqrt_alibi,
    )
    return output


def ref_vllm_rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool = True,
):
    def _rotate_neox(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x = torch.stack((-x2, x1), dim=-1)
        return x.flatten(-2)

    query_shape = query.shape
    key_shape = key.shape
    B = query.shape[0]
    query = query.view(B, -1, head_size)
    key = key.view(B, -1, head_size)

    cos_sin = cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if is_neox_style:
        cos = cos.repeat(1, 1, 2).unsqueeze(-2)
        sin = sin.repeat(1, 1, 2).unsqueeze(-2)
    else:
        cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
        sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

    rotate_fn = _rotate_neox if is_neox_style else _rotate_gptj
    query_rot = query * cos + rotate_fn(query) * sin
    key_rot = key * cos + rotate_fn(key) * sin

    query = query_rot.flatten(-2).view(query_shape)
    key = key_rot.flatten(-2).view(key_shape)
    return query, key


def vllm_rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool = True,
):

    """
    Args:
        positions:                      (num_tokens)                                        torch.long
        query:                          (num_tokens, num_heads * head_size)                 torch.half, torch.float, torch.bfloat16
        key:                            (num_tokens, num_heads * head_size)                 torch.half, torch.float, torch.bfloat16
        head_size:                                                                          int
        cos_sin_cache:                  (max_position, head_size)                           torch.half, torch.float, torch.bfloat16
        is_neox_style:                                                                      bool
    Returns:
        None. 对query, key 做in place 操作
    """
    ops.infer.vllm_rotary_embedding(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox_style,
    )


def ref_vllm_rotary_embedding_phi(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    long_offset: torch.Tensor,
    k: int,
    offsets: torch.Tensor = None,
):
    def _rotate_neox(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    query_shape = query.shape
    key_shape = key.shape
    B = query.shape[0]
    query = query.view(B, -1, head_size)
    key = key.view(B, -1, head_size)

    if long_offset is None:
        long_offset = (
            torch.any(positions > k).float() * torch.full_like(positions, k)
        ).long()
    idx = torch.add(positions, long_offset) if long_offset is not None else positions
    idx = torch.add(idx, offsets) if offsets is not None else idx
    cos_sin = torch.index_select(cos_sin_cache, 0, idx)

    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.repeat(1, 2).unsqueeze(-2)
    sin = sin.repeat(1, 2).unsqueeze(-2)

    query = query * cos + _rotate_neox(query) * sin
    key = key * cos + _rotate_neox(key) * sin

    query = query.flatten(-2).view(query_shape)
    key = key.flatten(-2).view(key_shape)

    return query, key


def vllm_rotary_embedding_phi(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    long_offset: torch.Tensor,
    k: int,
    offsets: torch.Tensor = None,
):
    """
    Args:
        positions:                      (num_tokens)                                        torch.long
        query:                          (num_tokens, num_heads * head_size)                 torch.half, torch.float, torch.bfloat16
        key:                            (num_tokens, num_heads * head_size)                 torch.half, torch.float, torch.bfloat16
        cos_sin_cache:                  (max_position, head_size)                           torch.half, torch.float, torch.bfloat16
        head_size:                                                                          int
        cos_sin_cache:                  (max_position, head_size)                           torch.half, torch.float, torch.bfloat16
        long_offset:                    (1,)                                                torch.bool
        k:                              int
        offsets:                        (num_tokens)                                        torch.half, torch.float, torch.bfloat16
    Returns:
        None. 对query, key 做in place 操作
    """
    ops.infer.vllm_rotary_embedding_phi(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        long_offset,
        k,
        offsets,
    )


def ref_vllm_rotary_embedding_with_key_layer_norm(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    key_out: torch.Tensor = None,
    eps: float = 1e-5,
    is_neox_style: bool = True,
):
    B = key.shape[0]
    query_size = query.size()
    query, key = ref_vllm_rotary_embedding(
        positions,
        query.view(B, -1),
        key.view(B, -1),
        head_size,
        cos_sin_cache,
        is_neox_style,
    )
    query = query.view(query_size)
    key = key.view(B, -1, head_size)

    norm_key = torch.nn.functional.layer_norm(
        key,
        [
            head_size,
        ],
        weight,
        bias,
        eps,
    )
    if key_out is not None:
        key_out.copy_(norm_key)
    else:
        key_out = norm_key
    return query, key_out


def vllm_rotary_embedding_with_key_layer_norm(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    key_out: torch.Tensor = None,
    eps: float = 1e-5,
    is_neox_style: bool = True,
):
    """
    Args:
        positions:                      (num_tokens)                                                                                torch.int64
        query:                          (num_tokens, num_heads * head_size) or (num_tokens, num_heads, head_size)                   torch.half, torch.float, torch.bfloat16
        key:                            (num_tokens, num_kv_heads * head_size) or (num_tokens, num_kv_heads, head_size)             torch.half, torch.float, torch.bfloat16
        weight:                         (head_size)                                                                                 torch.half, torch.float, torch.bfloat16
        bias:                           (head_size)                                                                                 torch.half, torch.float, torch.bfloat16
        head_size:                                                                                                                  int
        cos_sin_cache:                  (max_position, rot_dim)                                                                     torch.half, torch.float, torch.bfloat16
        key_out:                        (num_tokens, num_kv_heads * head_size) or (num_tokens, num_kv_heads, head_size)             torch.half, torch.float, torch.bfloat16
        eps:                                                                                                                        float
        is_neox_style:                                                                                                              bool
    Returns:
        query:                          (num_tokens, num_heads * head_size) or (num_tokens, num_heads, head_size)                   torch.half, torch.float, torch.bfloat16
        key_out:                        (num_tokens, num_kv_heads * head_size) or (num_tokens, num_kv_heads, head_size)             torch.half, torch.float, torch.bfloat16
    """
    ops.infer.vllm_rotary_embedding_with_key_layer_norm(
        positions,
        query,
        key,
        weight,
        bias,
        head_size,
        cos_sin_cache,
        key_out,
        eps,
        is_neox_style,
    )
    key_out = key if key_out is None else key_out
    return query, key_out


def ref_vllm_batched_rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox_style,
    rotary_dim: int,
    offsets: torch.Tensor,
):
    def _rotate_neox(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x = torch.stack((-x2, x1), dim=-1)
        return x.flatten(-2)

    query = query.view(*query.shape[:-1], -1, head_size)
    key = key.view(*key.shape[:-1], -1, head_size)

    query_rot = query[..., :rotary_dim]
    key_rot = key[..., :rotary_dim]
    if rotary_dim < head_size:
        query_pass = query[..., rotary_dim:]
        key_pass = key[..., rotary_dim:]

    cos_sin = cos_sin_cache[torch.add(positions, offsets)]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if is_neox_style:
        # NOTE(woosuk): Here we assume that the positions tensor has the
        # shape [batch_size, seq_len].
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
    query = query.flatten(-2)
    key = key.flatten(-2)
    return query, key


def vllm_batched_rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool,
    rotary_dim: int,
    offsets: torch.Tensor,
):

    """
    Args:
        positions:                      (num_tokens)                                                torch.long
        query:                          (num_tokens, num_heads * head_size)                         torch.half, torch.float, torch.bfloat16
        key:                            (num_tokens, num_heads * head_size)                         torch.half, torch.float, torch.bfloat16
        head_size:                                                                                  int
        cos_sin_cache:                  (max_position, head_size)                                   torch.half, torch.float, torch.bfloat16
        is_neox_style:                                                                              bool
        rotary_dim:                                                                                 int
        offsets:                        (positions, head_size)                                      torch.int64
    Returns:
        None. 对query, key 做in place 操作
    """
    ops.infer.vllm_batched_rotary_embedding(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox_style,
        rotary_dim,
        offsets,
    )


def ref_vllm_reshape_and_cache_v4(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    num_tokens, num_heads, head_size = key.shape
    x = 16 // torch.tensor([], dtype=key.dtype).element_size()
    block_size = key_cache.size(3)

    reshaped_key = key.reshape(num_tokens, num_heads, head_size // x, x)
    for i in range(num_tokens):
        block_idx = torch.div(slot_mapping[i], block_size, rounding_mode="floor")
        block_offset = slot_mapping[i] % block_size
        key_cache[block_idx, :, :, block_offset, :] = reshaped_key[i]
        value_cache[block_idx, :, :, block_offset] = value[i]


def vllm_reshape_and_cache_v4(
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
    ops.infer.vllm_reshape_and_cache_v4(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        key.stride(0),
        value.stride(0),
    )


def vllm_cache_ops_reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):

    """
    Args:
        key:                        (num_tokens, num_heads, head_size)                    torch.half, torch.bfloat16
        value:                      (num_tokens, num_heads, head_size)                    torch.half, torch.bfloat16
        key_cache:                  (num_blocks, num_heads, block_size, head_size)        torch.half, torch.bfloat16
        value_cache:                (num_blocks, num_heads, block_size, head_size)        torch.half, torch.bfloat16
        slot_mapping:               (num_tokens)                                          torch.long
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


def ref_vllm_reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    num_tokens, _, _ = key.shape
    block_size = key_cache.size(2)
    v_dim = value.shape[-1]

    for i in range(num_tokens):
        block_idx = torch.div(slot_mapping[i], block_size, rounding_mode="floor")
        block_offset = slot_mapping[i] % block_size
        key_cache[block_idx, :, block_offset, :] = key[i]
        value_cache[block_idx, :, block_offset, :v_dim] = value[i]


def vllm_reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):

    """
    Args:
        key:                        (num_tokens, num_heads, head_size)                    torch.half, torch.float, torch.bfloat16
        value:                      (num_tokens, num_heads, head_size)                    torch.half, torch.float, torch.bfloat16
        key_cache:                  (num_blocks, num_heads, block_size, head_size)        torch.half, torch.float, torch.bfloat16
        value_cache:                (num_blocks, num_heads, block_size, head_size)        torch.half, torch.float, torch.bfloat16
        slot_mapping:               (num_tokens)                                          torch.long
    Returns:
        None, 对key_cache,value_cache进行in place 操作
    """
    ops.infer.vllm_reshape_and_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        key.stride(0),
        value.stride(0),
    )


def ref_reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
):
    num_tokens, _, _ = key.shape
    block_size = key_cache.size(2)

    for i in range(num_tokens):
        block_idx = torch.div(slot_mapping[i], block_size, rounding_mode="floor")
        block_offset = slot_mapping[i] % block_size
        key_cache[block_idx, :, block_offset, :] = key[i]
        value_cache[block_idx, :, block_offset, :] = value[i]


def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
) -> None:

    """
    Args:
        key:                        (num_tokens, num_heads, head_size)                    torch.half, torch.float, torch.bfloat16
        value:                      (num_tokens, num_heads, head_size)                    torch.half, torch.float, torch.bfloat16
        key_cache:                  (num_blocks, num_heads, block_size, head_size)        torch.half, torch.float, torch.bfloat16
        value_cache:                (num_blocks, num_heads, block_size, head_size)        torch.half, torch.float, torch.bfloat16
        slot_mapping:               (num_tokens)                                          torch.long
        kv_cache_dtype:                                                                   str
        k_scale:                                                                          float
        v_scale:                                                                          float
    Returns:
        None, 对key_cache,value_cache进行in place 操作
    """
    assert k_scale == 1 and v_scale == 1
    assert kv_cache_dtype == "auto"

    ops.infer.vllm_reshape_and_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        key.stride(0),
        value.stride(0),
    )


def ref_vllm_copy_blocks(
    key_caches,
    value_caches,
    block_mapping,
):
    for k, v in zip(key_caches, value_caches):
        src = block_mapping[:, 0]
        dst = block_mapping[:, 1]
        k[dst] = k[src]
        v[dst] = v[src]


def vllm_copy_blocks(
    key_caches,
    value_caches,
    block_mapping,
):

    """
    Args:
        key_caches:                 [(num_blocks, num_heads, block_size, head_size)]        List[torch.half],List[torch.float],List[torch.bfloat16]
        value_caches:               [(num_blocks, num_heads, block_size, head_size)]        List[torch.half],List[torch.float],List[torch.bfloat16]
        block_mapping:              (num_tokens, 2)                                         torch.int64
    Returns:
        None, 对key_caches,value_caches进行in place 操作
    """
    ops.infer.vllm_copy_blocks(
        key_caches,
        value_caches,
        block_mapping,
    )


def ref_vllm_swap_blocks(src, dst, mapping):
    for item in mapping:
        src_idx = item[0]
        dst_idx = item[1]
        dst[dst_idx] = src[src_idx].to(dst.device)


def vllm_swap_blocks(src: "torch.Tensor", dst: "torch.Tensor", mapping: "torch.Tensor"):

    """
    Args:
        src:                [(num_blocks, num_kv_heads, block_size, head_size)]        List[torch.half],List[torch.float],List[torch.bfloat16]
        dst:                [(num_blocks, num_kv_heads, block_size, head_size)]        List[torch.half],List[torch.float],List[torch.bfloat16]
        mapping:            (num_tokens, 2)                                            torch.int64
    Returns:
        None, 对dst进行in place 操作
    """
    ops.infer.vllm_swap_blocks(src, dst, mapping)


def ref_vllm_concat_and_cache_mla(
    kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
):
    num_tokens, kv_lora_rank = kv_c.shape
    _, _, pe_dim = k_pe.shape
    _, block_size, rope_dim = kv_cache.shape
    assert kv_lora_rank + pe_dim == rope_dim

    for i in range(num_tokens):
        block_idx = torch.div(slot_mapping[i], block_size, rounding_mode="floor")
        block_offset = slot_mapping[i] % block_size
        kv_cache[block_idx, block_offset, :kv_lora_rank] = kv_c[i]
        kv_cache[block_idx, block_offset, kv_lora_rank:] = k_pe[i, 0]


def ref_vllm_gather_cache(
    src_cache: torch.Tensor,  # [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    dst: torch.Tensor,  # [TOT_TOKENS, ENTRIES...]
    block_table: torch.Tensor,  # [BATCH, BLOCK_INDICES]
    cu_seq_lens: torch.Tensor,  # [BATCH+1]
    batch_size: int,
    seq_starts: torch.Tensor = None,
):
    # 验证输入张量的设备一致性
    assert src_cache.device == dst.device == block_table.device == cu_seq_lens.device
    if seq_starts is not None:
        assert seq_starts.device == src_cache.device

    # 获取基本参数
    block_size = src_cache.size(1)
    entry_size = src_cache.flatten(2, -1).size(2)

    # 处理每个批次
    for bid in range(batch_size):
        seq_start = cu_seq_lens[bid]
        seq_end = cu_seq_lens[bid + 1]
        seq_len = seq_end - seq_start

        # 计算需要的块数
        tot_blocks = math.ceil(seq_len / block_size)

        # 获取当前批次的块表
        if seq_starts is not None:
            offset = seq_starts[bid] // block_size
            batch_block_table = block_table[bid, offset : offset + tot_blocks]
        else:
            batch_block_table = block_table[bid, :tot_blocks]

        # 准备目标位置
        dst_seq = dst[seq_start:seq_end]

        # 处理完整块
        full_blocks = seq_len // block_size
        if full_blocks > 0:
            # 获取所有完整块的源数据 [full_blocks, block_size, entry_size]
            src_blocks = src_cache[batch_block_table[:full_blocks]]
            # 展平并复制到目标位置
            dst_seq[: full_blocks * block_size].copy_(src_blocks.flatten(0, 1))

        # 处理部分块
        partial_size = seq_len % block_size
        if partial_size > 0:
            last_block = src_cache[batch_block_table[full_blocks], :partial_size]
            dst_seq[full_blocks * block_size :].copy_(last_block)


def vllm_gather_cache(
    src_cache: torch.Tensor,  # [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    dst: torch.Tensor,  # [TOT_TOKENS, ENTRIES...]
    block_table: torch.Tensor,  # [BATCH, BLOCK_INDICES]
    cu_seq_lens: torch.Tensor,  # [BATCH+1]
    batch_size: int,
    seq_starts: torch.Tensor = None,
):
    """
    Args:
        src_cache:              [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]                           torch.float16, torch.bfloat16                                                                    int
        dst:                    [TOT_TOKENS, ENTRIES...]                                       torch.float16, torch.bfloat16
        block_table:            [BATCH, BLOCK_INDICES]                                         torch.int
        cu_seq_lens:            [BATCH+1]                                                      torch.int
        batch_size:                                                                            int
        seq_starts:             [BATCH] or None                                                torch.int
    """
    ops.infer.vllm_gather_cache(
        src_cache, dst, block_table, cu_seq_lens, batch_size, seq_starts
    )


def ref_vllm_gather_cache_int8(
    src_cache: torch.Tensor,  # [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    src_cache_scale: torch.Tensor,  # [NUM_BLOCKS, BLOCK_SIZE, 2]
    kv_lora_rank: int,
    dst: torch.Tensor,  # [TOT_TOKENS, ENTRIES...]
    block_table: torch.Tensor,  # [BATCH, BLOCK_INDICES]
    cu_seq_lens: torch.Tensor,  # [BATCH+1]
    batch_size: int,
    seq_starts: torch.Tensor = None,
):
    # 验证输入张量的设备一致性
    assert (
        src_cache.device
        == src_cache_scale.device
        == dst.device
        == block_table.device
        == cu_seq_lens.device
    )
    if seq_starts is not None:
        assert seq_starts.device == src_cache.device

    # 获取基本参数
    block_size = src_cache.size(1)

    # 处理每个批次
    for bid in range(batch_size):
        seq_start = cu_seq_lens[bid]
        seq_end = cu_seq_lens[bid + 1]
        seq_len = seq_end - seq_start

        # 计算需要的块数
        tot_blocks = math.ceil(seq_len / block_size)

        # 获取当前批次的块表
        if seq_starts is not None:
            offset = seq_starts[bid] // block_size
            batch_block_table = block_table[bid, offset : offset + tot_blocks]
        else:
            batch_block_table = block_table[bid, :tot_blocks]

        # 准备目标位置
        dst_seq = dst[seq_start:seq_end]

        # 处理完整块
        full_blocks = seq_len // block_size
        if full_blocks > 0:
            # 获取所有完整块的源数据 [full_blocks, block_size, entry_size]
            src_cache_blocks = src_cache[batch_block_table[:full_blocks]]
            src_scale_blocks = src_cache_scale[batch_block_table[:full_blocks]]
            src_k_cache_blocks = src_cache_blocks[
                ..., :kv_lora_rank
            ] * src_scale_blocks[..., 0].unsqueeze(-1)
            src_k_pe_blocks = src_cache_blocks[..., kv_lora_rank:] * src_scale_blocks[
                ..., 1
            ].unsqueeze(-1)
            src_blocks = torch.cat((src_k_cache_blocks, src_k_pe_blocks), dim=-1).to(
                dst.dtype
            )
            # 展平并复制到目标位置
            dst_seq[: full_blocks * block_size].copy_(src_blocks.flatten(0, 1))

        # 处理部分块
        partial_size = seq_len % block_size
        if partial_size > 0:
            last_block = src_cache[batch_block_table[full_blocks], :partial_size]
            last_src_scale_blocks = src_cache_scale[
                batch_block_table[full_blocks], :partial_size
            ]
            last_src_k_cache_blocks = last_block[
                ..., :kv_lora_rank
            ] * last_src_scale_blocks[..., 0].unsqueeze(-1)
            last_src_k_pe_blocks = last_block[
                ..., kv_lora_rank:
            ] * last_src_scale_blocks[..., 1].unsqueeze(-1)
            last_block = torch.cat(
                (last_src_k_cache_blocks, last_src_k_pe_blocks), dim=-1
            ).to(dst.dtype)
            dst_seq[full_blocks * block_size :].copy_(last_block)


def vllm_gather_cache_int8(
    src_cache: torch.Tensor,  # [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    src_cache_scale: torch.Tensor,  # [NUM_BLOCKS, BLOCK_SIZE, 2]
    kv_lora_rank: int,
    dst: torch.Tensor,  # [TOT_TOKENS, ENTRIES...]
    block_table: torch.Tensor,  # [BATCH, BLOCK_INDICES]
    cu_seq_lens: torch.Tensor,  # [BATCH+1]
    batch_size: int,
    seq_starts: torch.Tensor = None,
):
    """
    Args:
        src_cache:              [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]                           torch.int8
        src_cache_scale:        [NUM_BLOCKS, BLOCK_SIZE, 2]                                    torch.float32
        kv_lora_rank:                                                                          int
        dst:                    [TOT_TOKENS, ENTRIES...]                                       torch.float16, torch.bfloat16
        block_table:            [BATCH, BLOCK_INDICES]                                         torch.int
        cu_seq_lens:            [BATCH+1]                                                      torch.int
        batch_size:                                                                            int
        seq_starts:             [BATCH] or None                                                torch.int
    """
    ops.infer.vllm_gather_cache_int8(
        src_cache,
        src_cache_scale,
        kv_lora_rank,
        dst,
        block_table,
        cu_seq_lens,
        batch_size,
        seq_starts,
    )


def ref_vllm_concat_and_cache_mla_int8(
    kv_c_int8: torch.Tensor,
    kv_c_scale: torch.Tensor,
    k_pe_int8: torch.Tensor,
    k_pe_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_cache_scale: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    scale: torch.Tensor,
) -> None:

    num_tokens, kv_lora_rank = kv_c_int8.shape
    _, block_size, _ = kv_cache_scale.shape

    for i in range(num_tokens):
        block_idx = torch.div(slot_mapping[i], block_size, rounding_mode="floor")
        block_offset = slot_mapping[i] % block_size
        kv_cache[block_idx, block_offset, :kv_lora_rank] = kv_c_int8[i]
        kv_cache[block_idx, block_offset, kv_lora_rank:] = k_pe_int8[i][0]
        kv_cache_scale[block_idx, block_offset, 0] = kv_c_scale[i]
        kv_cache_scale[block_idx, block_offset, 1] = k_pe_scale[i][0]


def vllm_concat_and_cache_mla(
    kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
):
    ops.infer.vllm_concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping)


def vllm_concat_and_cache_mla_int8(
    kv_c_int8,
    kv_c_scale,
    k_pe_int8,
    k_pe_scale,
    kv_cache,
    kv_cache_scale,
    slot_mapping,
    kv_cache_dtype,
    scale,
):
    """
    Args:
        kv_c_int8:         [num_tokens, kv_lora_rank]                           torch.int8
        kv_c_scale:        [num_tokens]                                         torch.float32
        k_pe_int8:         [num_tokens, n, pe_dim]                              torch.int8
        k_pe_scale:        [num_tokens, n]                                      torch.float32
        kv_cache:          [num_blocks, block_size, (kv_lora_rank + pe_dim)]    torch.int8
        kv_cache_scale:    [num_blocks, block_size, 2]                          torch.float32
        slot_mapping:      [num_tokens]                                         torch.long
    """

    ops.infer.vllm_concat_and_cache_mla_int8(
        kv_c_int8,
        kv_c_scale,
        k_pe_int8,
        k_pe_scale,
        kv_cache,
        kv_cache_scale,
        slot_mapping,
    )


class vllm_llama_mlp(CF.VllmLlamaMlp):
    def __init__(
        self,
        gate_up_proj_weight: "torch.Tensor",
        down_proj_weight: "torch.Tensor",
        hidden_size: int,
        intermediate_size: int,
        tp: int,
    ) -> None:
        gate_up_proj_weight = gate_up_proj_weight
        down_proj_weight = down_proj_weight
        hidden_size = hidden_size
        intermediate_size = intermediate_size
        tp = tp
        super().__init__(
            gate_up_proj_weight,
            down_proj_weight,
            hidden_size,
            intermediate_size,
            tp,
        )

    def __call__(self, x: "torch.Tensor", group=None):
        x1 = x
        if group is None:
            super().forward(x1, x1)
        else:
            from ixformer.distributed._distributed import _check_group

            group = _check_group(group)
            super().forward(x1, x1, group)
        return x


def gptq_gemm(
    input: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    use_exllama: bool,
    weight_bits: int,
) -> torch.Tensor:
    """
    use_exllama
        - True  uesExllama
        - False GeneralGptq
    g_idx.is_empty()
        - True don't use g_idx
        - False use g_idx
    1. use_exllama == False && use g_idx
        [General gptq] desc_act == True && parallel in k dimension && group_size != -1
    2. use_exllama == True && use g_idx (g_idx has been argsort)
        [Exllama with g_idx] desc_act == True && parallel in n dimension && group_size != -1
    3. use_exllama == True && don't use g_idx
        [Exllama] desc_act == False || desc_act == True && group_size == -1

    Args:
        input:              (m, k)                              torch.float16, torch.bfloat16
        qweight:            (k // (32 / bits), n)               torch.int32
        qzeros:             (k / group_size, n / (32 / bits))   torch.int32
        scales:             (k // group_size, n)                torch.float16, torch.bfloat16
        g_idx:              (k)                                 torch.int32
        use_exllama:                                            bool
                            wheather use exllama
        weight_bits:                                            int
                            quant bits of weight
    Returns:
        output:             (m, n)                              torch.float16, torch.bfloat16
    """
    bs = input.shape[0]
    group_size = input.shape[1] // scales.shape[0]

    # condition : without gidx or group_size == -1
    ixinfer_gemm_supported = (
        weight_bits == 4
        and (g_idx is None or g_idx.numel() == 0)
        and (scales.shape[0] == 1 or group_size in [32, 128])
    )
    if use_exllama:
        if bs <= 8 or ixinfer_gemm_supported:
            output = ops.infer.quantized_linear(
                input,
                qweight,
                scales,
                "gptq-ex",
                weight_bits,
                qzeros,
                None,
                group_size,
                g_idx,
                "unknown",
            )
        else:
            # GPTQ GEMM TODO
            o_dtype_str = "fp16" if input.dtype == torch.half else "bf16"
            deq_w = ops.infer.quantized_weight_dequant(
                qweight,
                scales,
                "gptq-ex",
                o_dtype_str,
                weight_bits,
                qzeros,
                group_size,
                g_idx,
            )

            output = linear(input, deq_w.transpose(0, 1).contiguous())
    else:
        if bs <= 8:
            output = ops.infer.quantized_linear(
                input,
                qweight,
                scales,
                "gptq",
                weight_bits,
                qzeros,
                None,
                group_size,
                g_idx,
                "unknown",
            )
        else:
            # GPTQ GEMM TODO
            o_dtype_str = "fp16" if input.dtype == torch.half else "bf16"
            deq_w = ops.infer.quantized_weight_dequant(
                qweight,
                scales,
                "gptq",
                o_dtype_str,
                weight_bits,
                qzeros,
                group_size,
                g_idx,
            )
            output = linear(input, deq_w.transpose(0, 1).contiguous())
    return output


def vllm_gptq_shuffle(qweights, g_idx, weight_bits):
    ops.infer.vllm_gptq_shuffle(qweights, g_idx, weight_bits)


def vllm_moe_topk_softmax(
    topk_weights: "torch.Tensor",
    topk_ids: "torch.Tensor",
    token_expert_indicies: "torch.Tensor",
    gating_output: "torch.Tensor",
):

    """
    Args:
        topk_weights:           (num_tokens,topk)           torch.float
        topk_ids:               (num_tokens,topk)           torch.int
        token_expert_indicies:  (num_tokens,topk)           torch.int
        gating_output:          (num_tokens,num_experts)    torch.float
    Returns:
        None, 对topk_weights,topk_ids进行in place 操作
    """
    assert isinstance(topk_weights, torch.Tensor)
    assert gating_output.dtype == torch.float32
    ops.infer.moe_topk_softmax(
        topk_weights, topk_ids, token_expert_indicies, gating_output, False
    )


def vllm_moe_align_block_size(
    topk_ids: "torch.Tensor",
    num_experts: int,
    block_size: int,
    sorted_ids: "torch.Tensor",
    expert_ids: "torch.Tensor",
    num_tokens_post_pad: "torch.Tensor",
):
    """
    Args:
        topk_ids:               (num_tokens,topk)                                       torch.int
        num_experts:                                                                    int
        block_size:                                                                     int
        sorted_ids:             (topk_ids.numel() + num_experts * (block_size - 1))     torch.int
        expert_ids:             (topk_ids.numel() + num_experts)                        torch.int
        num_tokens_post_pad:    (1)                                                     torch.int
    Returns:
        None
    """

    ops.infer.moe_align_block_size(
        topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad
    )


def ref_vllm_invoke_fused_moe_kernel(
    A: "torch.Tensor",
    B: "torch.Tensor",
    C: "torch.Tensor",
    topk_weight: "torch.Tensor",
    topk_ids: "torch.Tensor",
    sorted_token_ids: "torch.Tensor",
    expert_ids: "torch.Tensor",
    num_tokens_post_padded: "torch.Tensor",
    mul_routed_weight: bool,
    top_k: int,
    block_size_m: int,
    persistent: bool = False,
    w_scale: torch.Tensor = None,
    a_scale: torch.Tensor = None,
):

    expert_num, N, K = B.shape
    M, topk = C.shape[:2]

    clone_A = A.clone()
    clone_B = B.clone()
    if clone_A.shape[0] == M:
        clone_A = clone_A.view(M, -1, K).repeat(1, topk, 1).reshape(-1, K)
    topk_ids = topk_ids.view(-1)

    if A.dtype == torch.int8:
        use_scale = True
        clone_A = clone_A.to(torch.float32)
        clone_B = clone_B.to(torch.float32)
        tmp = torch.zeros(M * topk, N, dtype=torch.float32, device=C.device)
    else:
        use_scale = False
        tmp = torch.zeros(M * topk, N, dtype=C.dtype, device=C.device)

    for i in range(expert_num):  # expert_num
        mask = topk_ids == i
        if mask.sum():
            tmp[mask] = clone_A[mask] @ clone_B[i].transpose(0, 1)
            if use_scale:
                tmp[mask] = tmp[mask] * w_scale[i].view(1, N)

    if mul_routed_weight:
        tmp = tmp * topk_weight.view(-1, 1)

    if use_scale:
        tmp = tmp.view(M, topk, N)
        tmp = tmp * a_scale.view(M, -1, 1)

    C[:] = tmp.to(C.dtype).view(M, topk, N)
    return C


def vllm_invoke_fused_moe_kernel(
    A: "torch.Tensor",
    B: "torch.Tensor",
    C: "torch.Tensor",
    topk_weight: "torch.Tensor",
    topk_ids: "torch.Tensor",
    sorted_token_ids: "torch.Tensor",
    expert_ids: "torch.Tensor",
    num_tokens_post_padded: "torch.Tensor",
    mul_routed_weight: bool,
    top_k: int,
    block_size_m: int,
    persistent: bool = False,
    w_scale: torch.Tensor = None,
    a_scale: torch.Tensor = None,
):

    """
    Args:
        A:                  (bs*seq, K) / (bs*seq*top_k, K)                       torch.float16, torch.bfloat16
        B:                  (num_experts, N, K)                                   torch.float16, torch.bfloat16
        C:                  (bs*seq, top_k, N)                                    torch.half,torch.bfloat16
        topk_weight:        (bs*seq, topk)                                        torch.float32
        topk_ids:           (bs*seq, topk)                                        torch.int32
        sorted_token_ids:   (topk_ids.numel() + num_experts * (block_size - 1))   torch.int32
        expert_ids:         (topk_ids.numel() + num_experts)                      torch.int32
        num_tokens_post_pad:(1)                                                   torch.int32
        mul_routed_weight:                                                        bool
        top_k:                                                                    int
        block_size_m:                                                             int
    Returns:
        C:                  (bs*seq, top_k, N)                                    torch.half,torch.bfloat16
    """
    ops.infer.invoke_fused_moe_kernel(
        A,
        B,
        C,
        topk_weight,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        top_k,
        block_size_m,
        persistent,
        w_scale,
        a_scale.view(-1) * topk_weight.view(-1)
        if a_scale is not None and mul_routed_weight
        else a_scale,
    )


def advance_step_flashattn(
    num_seqs: int,
    num_queries: int,
    block_size: int,
    input_tokens: "torch.Tensor",
    sampled_token_ids: "torch.Tensor",
    input_positions: "torch.Tensor",
    seq_lens: "torch.Tensor",
    slot_mapping: "torch.Tensor",
    block_tables: "torch.Tensor",
):
    ops.infer.vllm_advance_step_flashattn(
        num_seqs,
        num_queries,
        block_size,
        input_tokens,
        sampled_token_ids,
        input_positions,
        seq_lens,
        slot_mapping,
        block_tables,
    )


if config.IXFORMER_PAGED_ATTENTION_ALGO == "ixformer":
    print("set IXFORMER_PAGED_ATTENTION_ALGO: ixformer")
    vllm_paged_attention = vllm_paged_attention_ixformer
else:
    vllm_paged_attention = vllm_paged_attention_ixinfer
