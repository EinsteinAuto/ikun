import math
from typing import List, Union

import ixformer._C as ops
import torch

from .flash_attn import ixinfer_flash_attn_unpad

__all__ = [
    "flash_attn_varlen_func",
    "ref_flash_attn_varlen_func",
    "flash_attn_func",
    "ref_flash_attn_func",
]


def ref_flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    causal: bool = False,
    return_attn_probs: bool = False,
):
    if return_attn_probs:
        raise NotImplementedError("return_attn_probs not supported!")
    out = torch.zeros_like(q)
    unpad_causal_torch(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        torch.float16,
        softmax_scale,
        causal,
    )
    return out


def unpad_causal_torch(
    q,
    k,
    v,
    output,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seq_len_q,
    max_seq_len_kv,
    dtype,
    atten_scale,
    is_causal=True,
):

    head_num = q.size(1)
    head_num_kv = k.size(1)
    head_dim = q.size(2)

    assert head_num % head_num_kv == 0
    if atten_scale == None:
        atten_scale = 1.0 / (q.size(-1) ** 0.5)
    # tokens,head_num,head_dim
    if head_num != head_num_kv:
        # k = k.repeat(1, head_num//head_num_kv, 1)#[0,1,2,0,1,2,0,1,2,0,1,2]
        # v = v.repeat(1, head_num//head_num_kv, 1)

        k = repeat_kv(k, head_num // head_num_kv)  # [0,0,0,0,1,1,1,1,2,2,2,2] GROUP
        v = repeat_kv(v, head_num // head_num_kv)

    batch_size = cu_seqlens_q.size(0) - 1

    for i in range(batch_size):
        q_start_index = cu_seqlens_q[i]
        q_end_index = cu_seqlens_q[i + 1]
        cur_q_len = q_end_index - q_start_index
        # 1*seq_len,head_num,head_dim
        cur_q = q[q_start_index:q_end_index]

        k_start_index = cu_seqlens_k[i]
        k_end_index = cu_seqlens_k[i + 1]
        cur_k_len = k_end_index - k_start_index

        cur_k = k[k_start_index:k_end_index]
        cur_v = v[k_start_index:k_end_index]

        # mask = torch.tril(torch.ones([cur_q_len, cur_k_len], dtype=torch.bool)).cuda()
        # mask = mask.unsqueeze(0).unsqueeze(0)
        if is_causal:
            # Create attention mask.
            attn_mask = torch.triu(
                torch.ones(cur_q_len, cur_k_len, dtype=dtype), diagonal=1
            )
            attn_mask = attn_mask * torch.finfo(dtype).min
            attn_mask = attn_mask.to(dtype=dtype, device="cuda")
        else:
            attn_mask = None

        ref_output = ref_masked_attention(
            cur_q,
            cur_k,
            cur_v,
            atten_scale,
            attn_mask=attn_mask,
        )
        output[q_start_index:q_end_index].copy_(ref_output)


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask=None,
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


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    causal: bool = False,
    return_attn_probs: bool = False,
    out: torch.Tensor = None,
):
    """
    Args:
        q:              (total_q, nheads, headdim)          torch.float16, torch.bfloat16
            where total_q = total number of query tokens in the batch. 
        k:              (total_k, nheads_k, headdim)        torch.float16, torch.bfloat16
            where total_k = total number of key tokens in the batch.
        v:              (total_k, nheads_k, headdim)        torch.float16, torch.bfloat16   
        cu_seqlens_q:   (batch_size + 1)                    torch.int32
            The cumulative sequence lengths of the sequences in the batch, used to index into q.
        cu_seqlens_k:   (batch_size + 1)                    torch.int32
            The cumulative sequence lengths of the sequences in the batch, used to index into kv.
        max_seqlen_q:                                       int
            Maximum query sequence length in the batch.
        max_seqlen_k:                                       int
            Maximum key sequence length in the batch.
        dropout_p:                                          float
            Dropout probability. dropout_p should be set to 0.0 during evaluation
        softmax_scale:                                      float
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(headdim).
        causal:                                             bool
            Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        return_attn_probs:                                  bool
            Whether to return the attention probabilities. This option is for testing only. The returned probabilities are not guaranteed to be correct (they might not have the right scaling).
        out:            (total, nheads, headdim)            torch.float16, torch.bfloat16
    Returns:
        out:            (total, nheads, headdim)            torch.float16, torch.bfloat16
    """

 
    assert len(q.shape) == 3, "q.shape != [total_q, nheads, head_dim]"
    assert len(k.shape) == 3, "k.shape != [total_k, nheads_k, head_dim]"
    assert len(v.shape) == 3, "v.shape != [total_k, nheads_k, head_dim]"
    assert len(cu_seqlens_q.shape) == 1, "cu_seqlens_q.shape != [batch_size+1]"
    assert len(cu_seqlens_k.shape) == 1, "cu_seqlens_k.shape != [batch_size+1]"

    if return_attn_probs:
        raise NotImplementedError("return_attn_probs not supported!")
    atten_scale = softmax_scale
    training = q.requires_grad
    nheads = q.size(1)
    nheads_k = k.size(1)
    if training:
        raise NotImplementedError("not support training!")
    else:  # 推理支持group query attention
        assert nheads % nheads_k == 0
        return ixinfer_flash_attn_unpad(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal,
            atten_scale,
            out=out,
        )


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    if len(x.shape) == 4:
        batch, seq_len, n_kv_heads, head_dim = x.shape
    elif len(x.shape) == 3:
        tokens, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    if len(x.shape) == 4:
        return (
            x[:, :, :, None, :]
            .expand(batch, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch, seq_len, n_kv_heads * n_rep, head_dim)
        )
    elif len(x.shape) == 3:
        return (
            x[:, :, None, :]
            .expand(tokens, n_kv_heads, n_rep, head_dim)
            .reshape(tokens, n_kv_heads * n_rep, head_dim)
        )


def mha(q, k, v, atten_scale, is_causal):
    q = q.permute(0, 2, 1, 3).contiguous()  # batch num_head seq_len head_dim
    k = k.permute(0, 2, 1, 3).contiguous()
    v = v.permute(0, 2, 1, 3).contiguous()

    # 2. q*kt softmax
    scores_qk = torch.matmul(q.float(), k.float().transpose(-2, -1)) * atten_scale
    q_seq_len = q.size(2)
    kv_seq_len = k.size(2)
    if is_causal:
        # Create attention mask.
        attn_mask = torch.triu(
            torch.ones(q_seq_len, kv_seq_len, dtype=torch.int), diagonal=1
        )
        attn_mask = attn_mask.to(dtype=torch.int, device="cuda")
    else:
        attn_mask = None
    # softmax
    # print(scores_qk.shape,attn_mask.shape)
    if attn_mask is not None:
        # print(scores_qk.shape,attn_mask.shape)
        scores_qk = scores_qk + attn_mask * (-100000)
    scores_qk = torch.nn.functional.softmax(scores_qk, dim=-1)
    # 3.  x = qk_scores * v
    scores_v = torch.matmul(scores_qk, v.float())
    scores_v = scores_v.half()
    return scores_v.permute(0, 2, 1, 3).contiguous()


def ref_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    causal: bool = False,
    return_attn_probs: bool = False,
):
    if return_attn_probs:
        raise NotImplementedError("return_attn_probs not supported!")
    head_num = q.size(2)
    head_num_kv = k.size(2)
    if head_num != head_num_kv:
        k = repeat_kv(k, head_num // head_num_kv)  # [0,0,0,0,1,1,1,1,2,2,2,2] GROUP
        v = repeat_kv(v, head_num // head_num_kv)
    output_pt = mha(q, k, v, softmax_scale, causal)
    return output_pt


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    causal: bool = False,
    return_attn_probs: bool = False,
):
    """
    Args:
        q:              (batch_size, seqlen, nheads, headdim)           torch.float16, torch.bfloat16
        k:              (batch_size, seqlen, nheads_k, headdim)         torch.float16, torch.bfloat16
        v:              (batch_size, seqlen, nheads_k, headdim)         torch.float16, torch.bfloat16
        dropout_p:                                                      float
            Dropout probability. dropout_p should be set to 0.0 during evaluation
        softmax_scale:                                                  float
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(headdim).
        causal:                                             bool
            Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        return_attn_probs:                                  bool
            Whether to return the attention probabilities. This option is for testing only. The returned probabilities are not guaranteed to be correct (they might not have the right scaling).
    Returns:
        Tensor:         (total, nheads, headdim)                        torch.float16, torch.bfloat16
    """
    
    if return_attn_probs:
        raise NotImplementedError("return_attn_probs not supported!")
    atten_scale = softmax_scale
    training = q.requires_grad

    q_dim = q.dim()
    assert q_dim == 4

    batch_size, max_seqlen_q, nheads, head_dim = q.shape
    _, max_seqlen_k, nheads_k, head_dim_k = k.shape
    assert head_dim == head_dim_k
    if training:
        raise NotImplementedError("not support training!")
    else:  # 推理支持group query attention
        assert nheads % nheads_k == 0

        q = q.view(batch_size * max_seqlen_q, nheads, head_dim)
        k = k.view(batch_size * max_seqlen_k, nheads_k, head_dim)
        v = v.view(batch_size * max_seqlen_k, nheads_k, head_dim)

        cu_seqlens_q = torch.ones([batch_size + 1]) * max_seqlen_q
        cu_seqlens_q[0] = 0
        cu_seqlens_k = torch.ones([batch_size + 1]) * max_seqlen_k
        cu_seqlens_k[0] = 0
        cu_seqlens_q = cu_seqlens_q.cuda().int()
        cu_seqlens_k = cu_seqlens_k.cuda().int()
        cu_seqlens_q = torch.cumsum(cu_seqlens_q, dim=0).int()
        cu_seqlens_k = torch.cumsum(cu_seqlens_k, dim=0).int()
        output = ixinfer_flash_attn_unpad(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal,
            atten_scale,
            sqrt_alibi=False,
            alibi_slopes=None,
        )
        return output.view(batch_size, max_seqlen_q, nheads, head_dim)
