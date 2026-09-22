from typing import Union

import ixformer._C as ops
import torch
from torch.autograd.function import Function, FunctionCtx

import ixformer

__all__ = ["residual_bias"]


class ResidualBiasFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        residual: torch.Tensor,
        bias: torch.Tensor,
        output: torch.Tensor,
        alpha=1,
    ):
        if output is None:
            output = torch.empty_like(input)
        if alpha is None:
            alpha = 1
        if bias is not None:
            ops.train.add_residual_bias_forward(input, residual, bias, alpha, output)
        else:
            ops.train.add_residual_bias_forward(input, residual, alpha, output)
        ctx.has_bias = bias is not None
        ctx.alpha = alpha
        return output

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output):
        grad_input = torch.empty_like(grad_output)
        grad_residual = torch.empty_like(grad_output)
        if ctx.has_bias:
            grad_bias = torch.empty(
                [grad_output.size(-1)],
                dtype=grad_output.dtype,
                device=grad_output.device,
            )
            ops.train.add_residual_bias_backward(
                grad_output,
                grad_input,
                grad_residual,
                grad_bias,
                ctx.alpha,
            )
            return (grad_input, grad_residual, grad_bias, None, None)
        else:
            ops.train.add_residual_bias_backward(
                grad_output,
                grad_input,
                grad_residual,
                ctx.alpha,
            )
            return (grad_input, grad_residual, None, None, None)


def residual_bias(
    input: torch.Tensor,
    residual: torch.Tensor,
    bias: torch.Tensor = None,
    output: torch.Tensor = None,
    alpha=1,
):
    """
    等价实现：
        input = residual.float() * alpha + input.float() + bias.float()

    参数说明:
    Args:
        input: shape:[batch_count, seq_len, hidden_size],dtype:[torch.half]
        residual: shape:[batch_count, seq_len, hidden_size],dtype:[torch.half]
        bias: shape:[hidden_size],dtype:[torch.half]
        alpha: float
    return:
        output: shape:[batch_count, seq_len, hidden_size],dtype:[torch.half]
    """
    return ResidualBiasFunction.apply(input, residual, bias, output, alpha)
