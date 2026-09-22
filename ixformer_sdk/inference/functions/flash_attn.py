import math
from typing import List, Union

import ixformer._C as ops
import torch
from torch.autograd.function import Function, FunctionCtx

__all__ = [
    "ixinfer_flash_attn_unpad",
    "ixinfer_flash_attn_pad",
    "ref_ixinfer_flash_attn_pad",
]


def ixinfer_flash_attn_unpad(
    # total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    q: "torch.Tensor",
    # total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    k: "torch.Tensor",
    # total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    v: "torch.Tensor",
    # total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
    cu_seqlens_q: "torch.Tensor",  # b+1
    cu_seqlens_k: "torch.Tensor",  # b+1
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    atten_scale: float = None,
    sqrt_alibi: bool = False,
    # total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
    alibi_slopes: "torch.Tensor" = None,
    out: "torch.Tenosr" = None,
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
        atten_scale:                                        float
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(headdim).
        is_causal:                                          bool
            Whether to apply causal attention mask (e.g., for auto-regressive modeling).
         sqrt_alibi:                                        bool
            Whether to apply abilimode
        out:            (total, nheads, headdim)            torch.float16, torch.bfloat16
    Returns:
        out:            (total, nheads, headdim)            torch.float16, torch.bfloat16
            if not q.size(-1) % 32 == 0: out shape is (total_q, nheads, q.size(-1) + (32 - q.size(-1) % 32))
    """
    if atten_scale is None:
        atten_scale = 1.0 / (q.size(-1) ** 0.5)

    # 判断是否pad
    cur_head = q.size(-1)
    cur_head32 = cur_head
    if not cur_head % 32 == 0:
        cur_head32 = cur_head + (32 - cur_head % 32)
        q_infer = torch.nn.functional.pad(q, [0, cur_head32 - cur_head, 0, 0], value=0)
        k_infer = torch.nn.functional.pad(k, [0, cur_head32 - cur_head, 0, 0], value=0)
        v_infer = torch.nn.functional.pad(v, [0, cur_head32 - cur_head, 0, 0], value=0)
    else:
        q_infer = q
        k_infer = k
        v_infer = v

    if out is None:
        out = torch.empty_like(q_infer)
    # ixinfer 新接口版
    ops.infer.ixinfer_flash_attn_unpad(
        q_infer,
        k_infer,
        v_infer,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        is_causal,
        False,  # need_lse =False
        atten_scale,
        sqrt_alibi,
        alibi_slopes,
    )
    if not cur_head % 32 == 0:
        out = out[:, :, :cur_head]
    return out


def ref_ixinfer_flash_attn_pad(
    # [ batch  num_heads seq_q head_size]
    q: torch.Tensor,
    # [ batch  num_heads_k max_seq_kv head_size]
    k: torch.Tensor,
    # [ batch  num_heads_k max_seq_kv head_size]
    v: torch.Tensor,
    # [ batch  num_heads seq_q seq_kv] seq_kv<=max_seq_kv
    mask: torch.Tensor,
    # [ batch  num_heads seq_q head_size]
    atten_scale: float = None,
    kv_seq_start: int = None,
    kv_seq_end: int = None,
):
    head_dim = q.size(-1)
    k_effective = k[:, :, kv_seq_start:kv_seq_end, :]
    v_effective = v[:, :, kv_seq_start:kv_seq_end, :]
    # 2. q*kt softmax
    scores_qk = (
        torch.matmul(q.float(), k_effective.float().transpose(-2, -1)) * atten_scale
    )
    # softmax
    # print(scores_qk.shape,mask.shape)
    if mask is not None:
        if mask.dtype == torch.int32:
            scores_qk = scores_qk + mask * (-100000)
        elif mask.dtype == torch.float32:
            scores_qk = scores_qk + mask
        else:
            print(
                f"mask dtype is not surported {mask.dtype},now surport int32 and float32"
            )
    scores_qk = torch.nn.functional.softmax(scores_qk, dim=-1)
    # 3.  x = qk_scores * v
    scores_v = torch.matmul(scores_qk, v_effective.float())
    return scores_v.half()


def ixinfer_flash_attn_pad(
    # [ batch  num_heads seq_q head_size]
    q: torch.Tensor,
    # [ batch  num_heads_k max_seq_kv head_size]
    k: torch.Tensor,
    # [ batch  num_heads_k max_seq_kv head_size]
    v: torch.Tensor,
    # [ batch  num_heads seq_q seq_kv] seq_kv<=max_seq_kv
    mask: torch.Tensor,
    # [ batch  num_heads seq_q head_size]
    atten_scale: float = None,
    kv_seq_start: int = None,
    kv_seq_end: int = None,
):
    """
    Args:
        q:              (batch_size, num_head, seq_len_q, head_dim)                     torch.float16, torch.bfloat16
        k:              (batch_size, num_head_kv, seq_len_kv, head_dim)                 torch.float16, torch.bfloat16
        v:              (batch_size, num_head_kv, seq_len_kv, head_dim)                 torch.float16, torch.bfloat16   
        mask:           (batch_size, num_head, seq_len_q, kv_seq_start:kv_seq_end)      torch.int32, torch.int64, torch.float32
        atten_scale:                                                                    float
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(headdim).
        kv_seq_start:                                                                   int
            kv sequence start index used for computation in the batch
        kv_seq_end:                                                                     int
            kv sequence end index used for computation in the batch.
    Returns:
        out:            (batch_size, num_head, seq_len_q, head_dim)                     torch.float16, torch.bfloat16
    """

    # 判断是否pad
    cur_head = q.size(-1)
    cur_head32 = cur_head
    if not cur_head % 32 == 0:
        cur_head32 = cur_head + (32 - cur_head % 32)
        q_infer = torch.nn.functional.pad(q, [0, cur_head32 - cur_head, 0, 0], value=0)
        k_infer = torch.nn.functional.pad(k, [0, cur_head32 - cur_head, 0, 0], value=0)
        v_infer = torch.nn.functional.pad(v, [0, cur_head32 - cur_head, 0, 0], value=0)
    else:
        q_infer = q
        k_infer = k
        v_infer = v

    if atten_scale is None:
        atten_scale = 1.0 / (q.size(-1) ** 0.5)
    if kv_seq_start is None or kv_seq_end is None:
        kv_seq_start = 0
        kv_seq_end = k.size(-2)  # kv seq len
    elif kv_seq_start < 0 or kv_seq_end > k.size(-2) or kv_seq_start >= kv_seq_end:
        raise NotImplementedError(
            "must kv_seq_start<0 or kv_seq_end>k.size(-2) or kv_seq_start>=kv_seq_end!"
        )
    out_shape = list(q_infer.shape)
    out = torch.empty(out_shape, dtype=q.dtype, device=q.device)
    if mask is not None:
        ops.infer.ixinfer_flash_attn_pad_fwd(
            q_infer, k_infer, v_infer, mask, out, atten_scale, kv_seq_start, kv_seq_end
        )
    else:
        ops.infer.ixinfer_flash_attn_pad_fwd_nomask(
            q_infer, k_infer, v_infer, out, atten_scale, kv_seq_start, kv_seq_end
        )
    if not cur_head % 32 == 0:
        out = out[:, :, :, :cur_head]
    return out
