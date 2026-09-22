import math
from typing import List, Optional, Union

import ixformer._C as ops
import torch
from ixformer.inference.functions import vllm_paged_attention

from ixformer.core import config

__all__ = [
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
    "ref_flash_attn_varlen_func",
    "ref_flash_attn_with_kvcache",
    "flash_attn_with_cache_batch_idx",
    "flash_attn_decode_with_cache_batch_idx",
    "ref_flash_attn_with_cache_batch_idx",
    "flash_attn_prefill_with_cache_batch_idx",
    "merge_attn_states",
    "ref_merge_attn_states",
]


def ixinfer_flash_attn_unpad_wrapper(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    causal,
    window_left,
    window_right,
    softmax_scale,
    softcap,
    sqrt_alibi,
    alibi_slopes,
):
    ops.infer.ixinfer_flash_attn_unpad(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal,
        False,  # need_lse =False
        softmax_scale,
        sqrt_alibi,
        alibi_slopes,
    )


if config.IXFORMER_UNPAD_ATTENTION_ALGO == "ixinfer":
    ixinfer_flash_attn_unpad_op = ops.infer.ixinfer_flash_attn_unpad_new
else:
    ixinfer_flash_attn_unpad_op = ixinfer_flash_attn_unpad_wrapper


# https://github.com/vllm-project/flash-attention/blob/v2.6.2/vllm_flash_attn/flash_attn_interface.py
def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    sqrt_alibi=False,
    return_softmax_lse=False,
    *,
    out=None,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """

    assert (
        deterministic is False
    ), "For the inference model, we don't need this parameter."
    assert (
        return_attn_probs is False
    ), "For the inference model, we don't need this parameter."
    assert dropout_p == 0, "For the inference model, we don't need this parameter."

    if out is None:
        out = torch.empty_like(q)

    if softmax_scale is None:
        softmax_scale = 1.0 / (q.size(-1) ** 0.5)

    num_tokens, head_num, head_dim = q.shape
    lse = (
        torch.empty([head_num, num_tokens], device=q.device, dtype=torch.float32)
        if return_softmax_lse
        else None
    )

    if block_table is None:
        ixinfer_flash_attn_unpad_op(
            q,
            k,
            v,
            out,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal,
            window_size[0],
            window_size[1],
            softmax_scale,
            softcap,
            sqrt_alibi,
            alibi_slopes,
            lse,
        )
    else:
        ops.infer.ixinfer_flash_attn_unpad_with_block_tables(
            q,
            k,
            v,
            out,
            block_table,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal,
            window_size[0],
            window_size[1],
            softmax_scale,
            softcap,
            sqrt_alibi,
            alibi_slopes,
            lse,
        )
    if return_softmax_lse:
        return out, lse
    return out


# https://github.com/vllm-project/flash-attention/blob/v2.6.1/vllm_flash_attn/flash_attn_interface.py#L1175
def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # 0.0 means deactivated
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    return_softmax_lse=False,
    max_context_len: int = None,
    use_cuda_graph: bool = False,
    use_sqrt_alibi: bool = False,
    *,
    out=None,
):
    """
    If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
    k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
    the previous step, and update them with the new keys/values from the current step, and do
    attention with the updated cache, all in 1 kernel.

    If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
    For example, the KV cache could be pre-allocated with the max sequence length, and you can use
    cache_seqlens to keep track of the current sequence lengths of each sequence in the batch.

    Also apply rotary embedding if rotary_cos and rotary_sin are passed in. The key @k will be
    rotated by rotary_cos and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If causal or local (i.e., window_size != (-1, -1)), the query @q will be rotated by rotary_cos
    and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If not causal and not local, the query @q will be rotated by rotary_cos and rotary_sin at
    indices cache_seqlens only (i.e. we consider all tokens in @q to be at position cache_seqlens).

    See tests/test_flash_attn.py::test_flash_attn_kvcache for examples of how to use this function.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Note: Does not support backward pass.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, nheads_k, page_block_size, headdim) if there's a block_table (i.e. paged KV cache)
            page_block_size must be a multiple of 256.
        v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, nheads_k, page_block_size, headdim) if there's a block_table (i.e. paged KV cache)
        k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate
            k with k_cache, starting at the indices specified by cache_seqlens.
        v [optional]: (batch_size, seqlen_new, nheads_k, headdim). Similar to k.
        rotary_cos [optional]: (seqlen_ro, rotary_dim / 2). If not None, we apply rotary embedding
            to k and q. Only applicable if k and v are passed in. rotary_dim must be divisible by 16.
        rotary_sin [optional]: (seqlen_ro, rotary_dim / 2). Similar to rotary_cos.
        cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
            KV cache.
        block_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
        cache_batch_idx: (batch_size,), dtype torch.int32. The indices used to index into the KV cache.
            If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
            If the indices are not distinct, and k and v are provided, the values updated in the cache
                 might come from any of the duplicate indices.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        rotary_interleaved: bool. Only applicable if rotary_cos and rotary_sin are passed in.
            If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False,
            rotary embedding will combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1
            (i.e. GPT-NeoX style).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
           If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic
           to automatically determine the number of splits.
           Don't change this unless you know what you are doing.
        return_softmax_lse: bool. Whether to return the logsumexp of the attention scores.

    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    assert k is None, "Updated *inplace* with the new key/values not supported."
    assert v is None, "Updated *inplace* with the new key/values not supported."
    assert rotary_cos is None and rotary_sin is None and cache_batch_idx is None
    assert num_splits == 0
    assert return_softmax_lse is False
    assert rotary_interleaved is True

    assert (
        max_context_len is not None
    ), "flash_attn_with_kvcache needs to pass in the parameter 'max_context_len'."

    if out is None:
        output = torch.empty_like(q)
    else:
        output = out
    output_shape = list(output.shape)

    # For the official interface, the data layout is as follows：
    # q: (batch_size, seqlen, nheads, headdim)
    # k_cache, v_cache (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table

    # However, we adopts another data layout：
    # q: (num_tokens, nheads, headdim)
    # k_cache, v_cache (num_blocks, nheads_k, 16, headdim) if there's a block_table
    batch_size, seqlen, nheads, headdim = q.shape

    # We assume shape is [num_blocks, nheads_k, page_block_size, headdim]
    num_blocks, nheads_k, page_block_size, headdim = k_cache.shape

    assert page_block_size == 16
    q = q.view(batch_size * seqlen, nheads, headdim)
    output = output.view(batch_size * seqlen, nheads, headdim)

    vllm_paged_attention(
        output,
        q,
        k_cache,
        v_cache,
        nheads_k,
        softmax_scale,
        block_table,
        cache_seqlens,
        16,
        max_context_len,
        alibi_slopes,
        softcap,
        True,
        window_size[0],
        window_size[1],
        use_cuda_graph,
        use_sqrt_alibi,
    )

    return output.view(*output_shape)


def flash_attn_prefill_with_cache_batch_idx(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cache_batch_idx: torch.Tensor,
    max_context_len: int,
    softmax_scale: Optional[float] = None,
    causal: Optional[bool] = False,
    window_size: Optional[tuple] = (-1, -1),  # -1 means infinite context window
    softcap: Optional[float] = 0.0,  # 0.0 means deactivated
    alibi_slopes: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
):
    if output is None:
        output = torch.empty_like(q)

    if softmax_scale is None:
        head_dim = k_cache.shape[-1]
        softmax_scale = 1 / head_dim**0.5

    ops.infer.flash_attn_with_cache_batch_idx(
        q,
        k_cache,
        v_cache,
        output,
        cache_seqlens,
        cache_batch_idx,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        alibi_slopes,
    )

    return output


def flash_attn_decode_with_cache_batch_idx(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cache_batch_idx: torch.Tensor,
    max_context_len: int,
    softmax_scale: float,
    causal: bool,
    window_size=(-1, -1),
    softcap: float = 0,
    alibi_slopes: torch.Tensor = None,
    output: torch.Tensor = None,
):
    if output is None:
        output = torch.empty_like(q)
    assert q.shape[1] == 1
    # q = q.view(q.shape[0], -1)
    ops.infer.flash_attn_decode_with_cache_batch_idx(
        q,
        k_cache,
        v_cache,
        output,
        cache_seqlens,
        cache_batch_idx,
        max_context_len,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        alibi_slopes,
    )
    return output


def flash_attn_with_cache_batch_idx(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cache_batch_idx: torch.Tensor,
    max_context_len: int,
    softmax_scale: Optional[float] = None,
    causal: Optional[bool] = False,
    window_size: Optional[tuple] = (-1, -1),  # -1 means infinite context window
    softcap: Optional[float] = 0.0,  # 0.0 means deactivated
    alibi_slopes: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
):
    """
    Args:
        q:                 (batch_size, seqlen, nheads, headdim)                   torch.float16, torch.bfloat16
        k_cache:           (batch_size_cache, nheads_k, seqlen_cache, headdim)     torch.float16, torch.bfloat16
        v_cache:           (batch_size_cache, nheads_k, seqlen_cache, headdim)     torch.float16, torch.bfloat16
        cache_seqlens:     (batch_size,)                                           torch.int32
        cache_batch_idx:   (batch_size,)                                           torch.int32
        max_context_len:                                                           int
        softmax_scale:                                                             float
        causal:                                                                    bool
        window_size:                                                               tuple                            not implemented yet.
        softcap:                                                                   float                            not implemented yet.
        alibi_slopes:       (nheads,)                                              torch.float32                    causal must be true when alibi is not None
        output:             (batch_size, seqlen, nheads, headdim)                  torch.float16, torch.bfloat16
    Returns:
        output:             (batch_size, seqlen, nheads, headdim)                  torch.float16, torch.bfloat16
    """
    assert len(q.shape) == 4
    if (
        q.shape[1] == 1 and window_size[0] == -1 and window_size[1] == -1
    ):  # remove window size check when kernel supported.
        return flash_attn_decode_with_cache_batch_idx(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=cache_batch_idx,
            max_context_len=max_context_len,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            output=output,
        )
    else:
        return flash_attn_prefill_with_cache_batch_idx(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=cache_batch_idx,
            max_context_len=max_context_len,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            output=output,
        )


def get_alibi_mask(num_heads, seqlen, device, dtype, sqrt_alibi=False):
    offsets = torch.arange(seqlen)
    offsets = offsets[None, :] - offsets[:, None]
    offsets = offsets.to(device)
    if sqrt_alibi:  # sqrt distance for alibi bias
        offsets = torch.sqrt(torch.abs(offsets)) * torch.sign(offsets)

    return offsets


def get_alibi_mask_decode(num_heads, seqlen, device, dtype, sqrt_alibi=False):
    x = torch.arange(0, seqlen, device=device, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(seqlen - 1, device=device, dtype=torch.float32).view(1, -1)
    if sqrt_alibi:
        offsets = -torch.sqrt((y - x)).view(1, 1, seqlen)
    else:
        offsets = -(y - x).view(1, 1, seqlen)
    return offsets


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    device=None,
):
    row_idx = torch.arange(seqlen_q, device=device, dtype=torch.long).view(-1, 1)
    mask = (
        torch.arange(seqlen_k, device=device, dtype=torch.long)
        .view(1, -1)
        .repeat(seqlen_q, 1)
    )
    # [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]]
    return ((row_idx + seqlen_k - seqlen_q - window_size[0]) <= mask) & (
        (row_idx + seqlen_k - seqlen_q + window_size[1]) >= mask
    )


def compute_softmax_lse(x):
    # 为了数值稳定性，先减去最大值
    input_tensor = x
    if x.shape[-1] == 0:
        lse = torch.full(x.shape[:-1], float("inf"), device=x.device)
        softmax_out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        return softmax_out, lse.view(x.shape[:-1])
    max_values = torch.max(input_tensor, dim=-1, keepdim=True)[0]
    input_tensor = input_tensor - max_values

    # 计算以 2 为底的指数
    log2_e = math.log2(math.e)
    exp2_tensor = torch.exp2(input_tensor * log2_e)

    # 计算指数和
    exp2_sum = torch.sum(exp2_tensor, dim=-1, keepdim=True)

    # 计算 softmax
    softmax_output = exp2_tensor / exp2_sum

    lse = max_values * log2_e + torch.log2(exp2_sum)
    return softmax_output, lse.view(x.shape[:-1])


def ref_flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    sqrt_alibi=False,
    return_softmax_lse=False,
    *,
    out=None,
) -> torch.Tensor:
    num_seqs = len(cu_seqlens_q) - 1
    num_tokens, num_query_heads, head_size = q.shape
    if block_table is None:
        _, num_kv_heads, _ = k.shape
    else:
        _, num_kv_heads, _, _ = k.shape

    num_query_heads = q.shape[1]
    slopes = (
        alibi_slopes.view(num_query_heads, 1, 1)
        if alibi_slopes is not None
        else alibi_slopes
    )

    if softmax_scale is None:
        softmax_scale = 1.0 / (q.size(-1) ** 0.5)

    outputs: List[torch.Tensor] = []

    # head_num, num_tokens
    lse = torch.empty([q.shape[1], q.shape[0]], dtype=torch.float32, device=q.device)

    for i in range(num_seqs):
        query_start_idx = cu_seqlens_q[i]
        query_end_idx = cu_seqlens_q[i + 1]
        kv_start_idx = cu_seqlens_k[i]
        kv_end_idx = cu_seqlens_k[i + 1]

        query_len = query_end_idx - query_start_idx
        kv_len = kv_end_idx - kv_start_idx

        sq = q[query_start_idx:query_end_idx]

        if block_table is None:
            sk = k[kv_start_idx:kv_end_idx]
            sv = v[kv_start_idx:kv_end_idx]
        else:
            table = block_table[i]
            ks = []
            vs = []
            need_blocks = (kv_len + 15) // 16
            for index in range(need_blocks):
                offset = (
                    16
                    if index != (need_blocks - 1)
                    else min(16, 16 if kv_len % 16 == 0 else kv_len % 16)
                )
                ks.append(k[table[index], :, :offset, :].permute(1, 0, 2))
                vs.append(v[table[index], :, :offset, :].permute(1, 0, 2))
            sk = torch.cat(ks, dim=0)
            sv = torch.cat(vs, dim=0)
            assert sk.shape[0] == kv_len
            assert sv.shape[0] == kv_len

        if num_query_heads != num_kv_heads:
            assert (
                num_query_heads > num_kv_heads and num_query_heads % num_kv_heads == 0
            )
            sk = torch.repeat_interleave(sk, num_query_heads // num_kv_heads, dim=1)
            sv = torch.repeat_interleave(sv, num_query_heads // num_kv_heads, dim=1)
        attn = torch.einsum("qhd,khd->hqk", sq, sk * softmax_scale).float()
        # 0 -> mask out 1 -> calculation
        mask = torch.ones(query_len, kv_len, device=q.device)
        shift = (
            kv_len - query_len
        )  # flash attention use bottom-right as default, so we do not use shift = 0
        if causal:
            mask = torch.tril(mask, diagonal=shift).bool()
        else:
            mask = mask.bool()

        if window_size[0] != -1 and window_size[1] != -1:
            # [left, right]
            win_mask = construct_local_mask(
                query_len,
                kv_len,
                window_size=(
                    window_size[0],
                    window_size[1],
                ),  # -1 means infinite window size
                device=mask.device,
            )
            mask = win_mask & mask
        elif window_size[1] != -1:
            # [-1, right]
            right = window_size[1]
            win_mask = torch.tril(mask, diagonal=shift + right).bool()
            mask = win_mask & mask
        elif window_size[0] != -1:
            # [left, -1]
            left = window_size[0]
            win_mask = torch.triu(mask, diagonal=shift - left).bool()
            mask = win_mask & mask

        # 1 -> 0, 0 -> 1 to mask out
        zero_index = ~(mask.sum(dim=-1).bool())
        mask = ~mask
        mask = mask.float() * -10000

        if alibi_slopes is not None:
            offsets = get_alibi_mask(
                num_query_heads, kv_len, q.device, q.dtype, sqrt_alibi
            )
            alibi_mask = offsets * slopes
            alibi_mask = alibi_mask.to(attn.dtype)
            alibi_mask = alibi_mask[:, -query_len:]

        attn = attn + mask.to(attn.dtype).to(attn.device)  # num_heads, kv_len, head_dim

        if alibi_slopes is not None:
            attn = attn + alibi_mask

        # attn.masked_fill_(mask, float("-inf"))
        # attn = torch.softmax(attn, dim=-1).to(sv.dtype)
        attn, tmp_lse = compute_softmax_lse(attn)
        attn = attn.to(sv.dtype)

        sout = torch.einsum("hqk,khd->qhd", attn, sv)
        if softcap != 0:
            sout = softcap * torch.tanh(sout / softcap)
        sout[zero_index] = 0.0
        outputs.append(sout)

        lse[:, query_start_idx:query_end_idx] = tmp_lse

    outputs = torch.cat(outputs, dim=0)

    if out is not None:
        out.copy_(outputs)
    else:
        out = outputs
    if return_softmax_lse:
        return out, lse
    return out


def ref_flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # 0.0 means deactivated
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    return_softmax_lse=False,
    max_context_len: int = None,
    use_sqrt_alibi: bool = False,
    *,
    out=None,
) -> torch.Tensor:
    assert k is None
    assert v is None
    assert rotary_cos is None
    assert rotary_sin is None
    assert cache_batch_idx is None
    assert causal is True
    assert rotary_interleaved
    assert num_splits == 0
    assert not return_softmax_lse

    head_size = q.size(-1)

    num_seqs = cache_seqlens.size(0)
    block_tables = block_table.cpu().numpy()

    _, num_kv_heads, block_size, head_size = k_cache.shape

    assert block_size == 16

    num_query_heads = q.shape[-2]
    q_shape = q.shape
    q = q.view(-1, num_query_heads, head_size)

    slopes = (
        alibi_slopes.view(num_query_heads, 1, 1)
        if alibi_slopes is not None
        else alibi_slopes
    )

    outputs: List[torch.Tensor] = []

    for i in range(num_seqs):
        kv_len = cache_seqlens[i].item()

        sq = q[i : i + 1]

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        sk = k_cache[block_indices].permute(
            0, 2, 1, 3
        )  # -> num_blocks, block_size, num_head, head_size
        sk = sk.reshape(-1, num_kv_heads, head_size)
        sk = sk[:kv_len]
        sv = v_cache[block_indices].permute(0, 2, 1, 3)
        sv = sv.reshape(-1, num_kv_heads, head_size)
        sv = sv[:kv_len]

        if num_query_heads != num_kv_heads:
            sk = torch.repeat_interleave(sk, num_query_heads // num_kv_heads, dim=1)
            sv = torch.repeat_interleave(sv, num_query_heads // num_kv_heads, dim=1)

        attn = torch.einsum("qhd,khd->hqk", sq, sk * softmax_scale).float()
        empty_mask = torch.ones(1, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len).bool().to(q.device)

        if window_size != (-1, -1):
            # sliding_window_mask = torch.triu(empty_mask,
            #                                  diagonal=kv_len -
            #                                  (query_len + sliding_window) +
            #                                  1).bool().logical_not()
            sliding_window_mask = construct_local_mask(
                1,
                kv_len,
                window_size=(
                    window_size[0],
                    window_size[1],
                ),  # -1 means infinite window size
                device=mask.device,
            )
            mask |= sliding_window_mask
        mask = mask.float() * -1000

        if alibi_slopes is not None:
            offsets = get_alibi_mask_decode(
                num_query_heads, kv_len, q.device, q.dtype, use_sqrt_alibi
            )
            alibi_mask = offsets * slopes
            alibi_mask = alibi_mask.to(attn.dtype)

        attn = attn + mask.to(attn.dtype)  # num_heads, kv_len, head_dim

        if alibi_slopes is not None:
            attn = attn + alibi_mask

        # attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(sv.dtype)
        sout = torch.einsum("hqk,khd->qhd", attn, sv)
        if softcap != 0:
            sout = softcap * torch.tanh(sout / softcap)
        outputs.append(sout)

    outputs = torch.cat(outputs, dim=0)

    if out is not None:
        out_shape = out.shape
        out = out.view(*outputs.shape)
        out.copy_(outputs)
    else:
        out_shape = q_shape
        out = outputs

    return out.view(*out_shape)


def ref_flash_attn_with_cache_batch_idx(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cache_batch_idx: torch.Tensor,
    max_context_len: int,
    softmax_scale: Optional[float] = None,
    causal: Optional[bool] = False,
    window_size: Optional[tuple] = (-1, -1),  # -1 means infinite context window
    softcap: Optional[float] = 0.0,  # 0.0 means deactivated
    alibi_slopes: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
):
    basic_output = torch.empty_like(q)
    cache_batch_idx_cpu = cache_batch_idx.cpu()
    cache_seqlens_cpu = cache_seqlens.cpu()

    for i, batch_idx in enumerate(cache_batch_idx_cpu):
        cur_q = q[i]
        cur_k = k_cache[batch_idx, :, : cache_seqlens_cpu[i], :]
        cur_v = v_cache[batch_idx, :, : cache_seqlens_cpu[i], :]

        cur_q = cur_q.transpose(0, 1)
        # cur_k = cur_k.transpose(0, 1)
        # cur_v = cur_v.transpose(0, 1)

        num_q_heads = cur_q.shape[0]
        num_kv_heads = cur_k.shape[0]

        assert num_q_heads >= num_kv_heads and num_q_heads % num_kv_heads == 0
        kv_repeat = num_q_heads // num_kv_heads
        if kv_repeat > 1:
            cur_k = cur_k.repeat_interleave(kv_repeat, dim=0)
            cur_v = cur_v.repeat_interleave(kv_repeat, dim=0)
        attn = torch.matmul(cur_q, cur_k.transpose(-1, -2)).float() * softmax_scale

        query_len = cur_q.shape[1]
        kv_len = cur_k.shape[1]

        mask = None
        if causal:
            mask = torch.ones(query_len, kv_len, device=q.device)
            shift = kv_len - query_len
            mask = torch.tril(mask, diagonal=shift).bool()

            if window_size[0] != -1 and window_size[1] != -1:
                # [left, right]
                win_mask = construct_local_mask(
                    query_len,
                    kv_len,
                    window_size=(
                        window_size[0],
                        window_size[1],
                    ),  # -1 means infinite window size
                    device=mask.device,
                )
                mask = win_mask & mask
            elif window_size[1] != -1:
                # [-1, right]
                right = window_size[1]
                win_mask = torch.tril(mask, diagonal=shift + right).bool()
                mask = win_mask & mask
            elif window_size[0] != -1:
                # [left, -1]
                left = window_size[0]
                win_mask = torch.triu(mask, diagonal=shift - left).bool()
                mask = win_mask & mask

        if mask is not None:
            attn.masked_fill_(~mask, float("-inf"))

        if alibi_slopes is not None:
            slopes = alibi_slopes.view(num_q_heads, 1, 1)
            offsets = get_alibi_mask_decode(num_q_heads, kv_len, q.device, q.dtype)
            alibi_mask = offsets * slopes
            alibi_mask = alibi_mask.to(attn.dtype)
            attn = attn + alibi_mask

        attn = torch.softmax(attn, dim=-1).to(q.dtype)
        if mask is not None:
            attn.masked_fill_(~mask, 0)

        out = torch.matmul(attn, cur_v)
        basic_output[i, :, :, :] = out.transpose(0, 1)
    if output is None:
        output = basic_output
    else:
        output.copy_(basic_output)
    return output


def ref_merge_attn_states(
    out_1, lse_1, out_2, lse_2, output=None, return_lse: Optional[bool] = False
):
    lse_1 = torch.where(
        lse_1 == float("inf"), torch.full_like(lse_1, -float("inf")), lse_1
    )
    lse_2 = torch.where(
        lse_2 == float("inf"), torch.full_like(lse_2, -float("inf")), lse_2
    )
    num_heads, seq_len = lse_1.shape

    lse_2 = lse_2.transpose(0, 1).view(seq_len, num_heads, 1)
    lse_1 = lse_1.transpose(0, 1).view(seq_len, num_heads, 1)

    s_max = torch.maximum(lse_1, lse_2)

    d = torch.exp2(lse_1 - s_max) + torch.exp2(lse_2 - s_max)
    v_merged = out_1 * torch.exp2(lse_1 - s_max) + out_2 * torch.exp2(lse_2 - s_max)
    v_merged = v_merged / d
    v_merged = v_merged.to(out_1.dtype)
    if output is None:
        output = v_merged
    else:
        output.copy_(v_merged)
    if return_lse:
        output_lse = s_max + torch.log2(d)
        output_lse = output_lse.squeeze(-1).transpose(0, 1).contiguous()
        return output, output_lse
    else:
        return output


def merge_attn_states(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output: torch.Tensor = None,
    return_lse: Optional[bool] = False,
):
    """
    Args:
        prefix_output:     (seq_len, head_num, head_dim)        torch.float16, torch.bfloat16
        prefix_lse:        (head_num, seq_len)                  torch.float32
        suffix_output:     (seq_len, head_num, head_dim)        torch.float16, torch.bfloat16
        suffix_lse:        (head_num, seq_len)                  torch.float32
    Returns:
        output:            (seq_len, head_num, head_dim)        torch.float16, torch.bfloat16
        output_lse:        (head_num, seq_len)                  torch.float32
    """
    if output is None:
        output = torch.empty_like(prefix_output)
    output_lse = torch.empty_like(prefix_lse) if return_lse else None

    ops.infer.merge_attn_states(
        prefix_output, prefix_lse, suffix_output, suffix_lse, output, output_lse
    )
    if output_lse is not None:
        return output, output_lse
    else:
        return output
