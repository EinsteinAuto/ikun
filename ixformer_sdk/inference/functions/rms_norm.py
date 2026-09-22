from typing import Union

import ixformer._C as ops
import torch
from torch.nn import init

__all__ = ["ref_rms_norm", "rms_norm", "ref_residual_rms_norm", "residual_rms_norm"]


def ref_rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    output: torch.Tensor = None,
):
    dtype = input.dtype
    input = input.float()
    weight = weight.float()
    rms_out = input * torch.rsqrt(input.pow(2).mean(-1, keepdim=True) + eps)
    rms_out = rms_out * weight
    rms_out = rms_out.to(dtype)
    if output is not None:
        output.copy_(rms_out)
    else:
        output = rms_out
    return output


def rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    output: torch.Tensor = None,
):
    """
    This function is deprecated, please use residual_rms_norm.
    Args:
        input:              (..., hidden_size)    torch.float16, torch.bfloat16, torch.float32
        weight:             (hidden_size)         torch.float16, torch.bfloat16, torch.float32
        eps:                                      float32
    Returns:
        output:             (..., hidden_size)    torch.float16, torch.bfloat16, torch.float32
    """
    if output is None:
        output = torch.empty_like(input)

    ops.infer.rms_norm(input, weight, output, None, eps)

    return output


def ref_residual_rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    residual_alpha: float = 1.0,
    residual: torch.Tensor = None,
    residual_bias: torch.Tensor = None,
    output: torch.Tensor = None,
    residual_output: torch.Tensor = None,
    is_post: bool = False,
):
    dtype = input.dtype

    if residual_bias is not None:
        input = input + residual_bias

    if residual is not None:
        residual_output = torch.add(
            input, residual * residual_alpha, out=residual_output
        )
        input = input.float() + residual.float() * residual_alpha
    else:
        input = input.float()
    weight = weight.float()
    rms_out = input * torch.rsqrt(input.pow(2).mean(-1, keepdim=True) + eps)
    rms_out = rms_out * weight
    rms_out = rms_out.to(dtype)
    if output is not None:
        output.copy_(rms_out)
    else:
        output = rms_out

    if is_post and residual_output is not None:
        residual_output = output

    return output, residual_output


def residual_rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    residual_alpha: float = 1.0,
    residual: torch.Tensor = None,
    residual_bias: torch.Tensor = None,
    output: torch.Tensor = None,
    residual_output: torch.Tensor = None,
    is_post: bool = False,
):
    """
    Args:
        input:              (..., hidden_size)    torch.float16, torch.bfloat16, torch.float32
        weight:             (hidden_size)         torch.float16, torch.bfloat16, torch.float32
        eps:                                      float32
        residual_alpha:                           float32
        residual:           (..., hidden_size)    torch.float16, torch.bfloat16, torch.float32
        residual_bias:      (hidden_size)         torch.float16, torch.bfloat16, torch.float32
        is_post:                                  bool
    Returns:
        output:             (..., hidden_size)    torch.float16, torch.bfloat16, torch.float32 If set to None, an inplace operation will be performed on input.
        residual_output:    (..., hidden_size)    torch.float16, torch.bfloat16, torch.float32 If set to None, an inplace operation will be performed on residual.
    """
    if residual is None:
        if output is None:
            output = torch.empty_like(input)

        ops.infer.rms_norm(input, weight, output, residual_bias, eps)
    else:
        ops.infer.residual_rms_norm(
            input,
            residual,
            weight,
            output,
            residual_output,
            residual_bias,
            residual_alpha,
            eps,
            is_post,
        )
        residual_output = residual_output if residual_output is not None else residual
        output = output if output is not None else input

    return output, residual_output
