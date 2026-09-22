from typing import List, Tuple, Union

import ixformer._C as ops
import torch
from torch.autograd.function import Function

# adding by xuelu.peng 20240417
# from https://github.com/NVIDIA/apex/blob/master/apex/transformer/functional/fused_rope.py#L59
__all__ = ["fused_apply_rotary_pos_emb", "fused_apply_split_rotary_pos_emb", "fused_apply_rotary_pos_emb_cache"]


class FusedRoPEFunc(Function):
    """
    Fused RoPE function

    This implementation assumes the input tensor to be in `sbhd` format and the RoPE tensor to be
    of shape (s, 1, 1, d). It accepts arbitrary memory layouts to avoid the expensive
    `.contiguous()` calls, thus it may not achieve the best memory access pattern.
    """

    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,
        transpose_output_memory: bool = False,
    ) -> torch.Tensor:
        # assert transpose_output_memory == False
        output = ops.train.fused_rope_forward(t, freqs, transpose_output_memory)
        ctx.save_for_backward(freqs)
        ctx.transpose_output_memory = transpose_output_memory

        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:

        (freqs,) = ctx.saved_tensors
        grad_input = ops.train.fused_rope_backward(
            grad_output, freqs, ctx.transpose_output_memory
        )
        return grad_input, None, None


class FusedFluxRoPEFunc(Function):
    """
    Fused FluxRoPE function

    This implementation assumes the input tensor to be in `bshd` format and the RoPE tensor to be
    of shape (s, d), and output shape is the same as input shape.
    """

    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        imp_mode : int = 1
    ) -> torch.Tensor:
        # assert transpose_output_memory == False
        output = ops.train.fused_rope_forward_cached(t, cos, sin, imp_mode)
        ctx.save_for_backward(cos, sin)
        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        (cos, sin) = ctx.saved_tensors
        grad_input = ops.train.fused_rope_backward_cached(
            grad_output, cos, sin
        )
        return grad_input, None, None, None
        


def fused_apply_rotary_pos_emb(
    t: torch.Tensor,
    freqs: torch.Tensor,
    transpose_output_memory: bool = False,
) -> torch.Tensor:
    """Apply rotary positional embedding to input tensor T in `sbhd` format, where
    s: sequence length
    b: batch size
    h: head num
    d: dim of each head

    Args:
        t (Tensor): Input tensor T is of shape [s, b, h, d], dtype : torch.float32, torch.half
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [s, 1, 1, d] and
        `float` dtype
        transpose_output_memory (bool): Default to False. Whether to transpose the 's' and 'b'
        dimension of the output's underlying memory format. This is very helpful when you want to
        get a contiguous tensor after calling `output.transpose(0, 1)`.

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    return FusedRoPEFunc.apply(t, freqs, transpose_output_memory)


def fused_apply_rotary_pos_emb_cache(
    t: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    imp_mode: int = 1,
) -> torch.Tensor:
    """Apply rotary positional embedding to input tensor T in `bshd` format, where
    s: sequence length
    b: batch size
    h: head num
    d: dim of each head

    Args:
        t (Tensor): Input tensor T is of shape [b, s, h, d], dtype : torch.float32, torch.half, torch.bfloat16
        cos/sin (Tensor): Rotary Positional embedding tensor freq is of shape [s, d] and
        `float` dtype
        imp_mode (bool): Default to 1. 1 for flux/cogvideox/hunyuan-dit, img_mode=0 for Stable Audio. For now, only img_mode = 1 is supported.

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    return FusedFluxRoPEFunc.apply(t, cos, sin, imp_mode)


class FusedSplitRoPEFunc(torch.autograd.Function):
    """
    Fused Split and RoPE function

    This implementation assumes the input tensor to be in `sbh3d` format and the RoPE tensor to be
    of shape (s, 1, 1, d). It accepts arbitrary memory layouts to avoid the expensive
    `.contiguous()` calls, thus it may not achieve the best memory access pattern.

    input:    mix_q_k_v [s,b,hn_kv,h/hn_kv+2,d]
    output:   output_q, output_k, output_v [s,b,h,d]
    """
    @staticmethod
    def forward(
        ctx,
        mixed_q_k_v: torch.Tensor,
        freqs: torch.Tensor,
        transpose_output_memory: bool = False,
    ) -> torch.Tensor: 
        assert transpose_output_memory == False, "do not support transpose_output now"
        assert mixed_q_k_v.is_contiguous() == True, "mixed_q_k_v should be contiguous in FusedSplitRoPEFunc." 

        s, b, hn_kv, repplus2, d = mixed_q_k_v.size()
        num_key_value_groups = repplus2-2

        q, k, v = torch.split(mixed_q_k_v, (num_key_value_groups,1,1), dim=3)

        ctx.hn_kv = hn_kv
        output_q, output_k, output_v = torch.empty_like(q).view(s,b,-1,d),torch.empty_like(q).view(s,b,-1,d),torch.empty_like(q).view(s,b,-1,d)

        ops.train.fused_split_rope_forward(
            q, k, v, freqs, output_q, output_k, output_v, transpose_output_memory, hn_kv, num_key_value_groups
        )

        ctx.save_for_backward(freqs)
        ctx.transpose_output_memory = transpose_output_memory

        return output_q, output_k, output_v

    @staticmethod
    def backward(
        ctx, grad_o_q: torch.Tensor, grad_o_k: torch.Tensor, grad_o_v: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        # grad_o_q: [s,b,h,d]
        s,b,h,d = grad_o_q.size()

        hn_kv = ctx.hn_kv

        mixed_shape = (s, b, hn_kv,(h//hn_kv+2), d)

        if hn_kv == h:
            grad_mixed_q_k_v = torch.empty(mixed_shape, dtype=grad_o_q.dtype, device=grad_o_q.device,memory_format=torch.contiguous_format)  # torch.empty效率比torch.zeros高
        else:
            grad_mixed_q_k_v = torch.zeros(mixed_shape, dtype=grad_o_q.dtype, device=grad_o_q.device)  # 支持 gqa 的情况，kernel内需要进行累加，需要把qkv的梯度置零
        grad_q, grad_k, grad_v = torch.split(grad_mixed_q_k_v.view(s,b,hn_kv,-1,d), (h//hn_kv,1,1), dim=3)

        (freqs,) = ctx.saved_tensors
        ops.train.fused_split_rope_backward(
            grad_o_q, grad_o_k, grad_o_v, freqs, grad_q, grad_k, grad_v, ctx.transpose_output_memory
        )

        return grad_mixed_q_k_v, None, None

def fused_apply_split_rotary_pos_emb(
    mixed_q_k_v: torch.Tensor,
    freqs: torch.Tensor,
    transpose_output_memory: bool = False,
) -> torch.Tensor:
    """ Split mixed_q_k_v and apply rotary positional embedding to q and k in `sbhd` format, where
    s: sequence length
    b: batch size
    h: head num
    d: dim of each head
    hn_kv: num head of key and value

    Args:
        mixed_q_k_v (Tensor): Input tensor T is of shape  [s,b,hn_kv,h/hn_kv+2,d]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [s, 1, 1, d] and
        `float` dtype
        transpose_output_memory (bool): Default to False. Whether to transpose the 's' and 'b'
        dimension of the output's underlying memory format. This is very helpful when you want to
        get a contiguous tensor after calling `output.transpose(0, 1)`.

    Returns:
        Tensors: The input tensors after split and applying RoPE
    """
    return FusedSplitRoPEFunc.apply(mixed_q_k_v, freqs, transpose_output_memory)
