from typing import List, Tuple, Union

import ixformer._C as ops
import torch

# adding by xuelu.peng 20240417
# from https://github.com/NVIDIA/apex/blob/master/apex/transformer/functional/fused_rope.py#L59
__all__ = ["fused_apply_rotary_pos_emb", "ref_fused_apply_rotary_pos_emb"]

# Copied from Megatron-Core for testing.
# https://github.com/NVIDIA/Megatron-LM/blob/5f2877d85cb26e47ce6dcdae4b80adf376abf4e8/megatron/core/models/common/embeddings/rotary_pos_embedding.py#L139
def apply_rotary_pos_emb(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embedding to input tensor T.

    check https://kexue.fm/archives/8265 for detailed formulas

    Arguments:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    rot_dim = freqs.shape[-1]

    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    t = (t * cos_) + (_rotate_half(t) * sin_)
    return torch.cat((t, t_pass), dim=-1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Change sign so the last dimension becomes [-odd, +even]

    Arguments:
        x (Tensor): Input tensor

    Returns:
        Tensor: Tensor rotated half
    """

    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def ref_fused_apply_rotary_pos_emb(
    t: torch.Tensor, freqs: torch.Tensor, transpose_output_memory: bool = False
):
    output_unfused = apply_rotary_pos_emb(t, freqs)
    return output_unfused


def fused_apply_rotary_pos_emb(
    t: torch.Tensor,
    freqs: torch.Tensor,
    transpose_output_memory: bool = False,
) -> torch.Tensor:

    """
    Args:
        t:                          (sequence length,batch size,head num,head_dim)              torch.float16, torch.bfloat16, torch.float32
        freqs:                      (sequence length,1 ,1, head_dim)                            torch.float32
        transpose_output_memory:                                                                bool
                                Default to False. Whether to transpose the 's' and 'b' dimension of the output's underlying memory format. This is very helpful when you want to get a contiguous tensor after calling `output.transpose(0, 1)`.

    Returns:
        Tensor:                     (sequence length,batch size,head num,head_dim)              torch.float16, torch.bfloat16, torch.float32
    """    
    output = ops.train.fused_rope_forward(t, freqs, transpose_output_memory)
    return output
