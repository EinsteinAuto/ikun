from typing import Union

import ixformer._C as ops
import torch

__all__ = ["residual_bias", "ref_residual_bias"]


def ref_residual_bias(
    input: torch.Tensor,
    residual: torch.Tensor,
    bias: torch.Tensor = None,
    alpha: float = 1,
):
    if bias is not None:
        output = residual.float() * alpha + input.float() + bias.float()
    else:
        output = residual.float() * alpha + input.float()

    return output.to(residual.dtype)


def residual_bias(
    input: torch.Tensor,
    residual: torch.Tensor,
    bias: torch.Tensor = None,
    alpha: float = 1,
    output: torch.Tensor = None
):
    """
    Args:
        input:              [batch_count, seq_len, hidden_size] or [batch_tokens, hidden_size]      torch.half
        residual:           [batch_count, seq_len, hidden_size] or [batch_tokens, hidden_size]      torch.half
        bias:               [hidden_size]                                                           torch.half
        alpha: float
    Returns:
        output:             [batch_count, seq_len, hidden_size] or [batch_tokens, hidden_size]      torch.half
    """
    if output is None:
        output = torch.empty_like(input)
    if alpha is None:
        alpha = 1
    if bias is not None:
        ops.train.add_residual_bias_forward(input, residual, bias, alpha, output)
    else:
        ops.train.add_residual_bias_forward(input, residual, alpha, output)
    return output
