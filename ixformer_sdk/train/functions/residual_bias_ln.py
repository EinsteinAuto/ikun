from typing import Union

import ixformer._C as ops
import torch
from torch.autograd.function import Function, FunctionCtx

__all__ = ["residual_bias_ln"]


class ResidualBiasLnFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        residual: torch.Tensor,
        bias: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        output: torch.Tensor,
        alpha=1,
        is_post_ln=True,
    ):
        norm_size = ln_weight.size(-1)

        mean_size = input.numel() // norm_size
        input_hat = torch.empty_like(input)
        rstd = torch.empty([mean_size], dtype=input.dtype, device=input.device)
        if bias is not None:
            ops.train.add_residual_bias_ln_training_forward(
                input,
                residual,
                bias,
                ln_weight,
                ln_bias,
                alpha,
                is_post_ln,
                output,
                input_hat,
                rstd,
            )
        else:
            ops.train.add_residual_bias_ln_training_forward(
                input,
                residual,
                ln_weight,
                ln_bias,
                alpha,
                is_post_ln,
                output,
                input_hat,
                rstd,
            )
        ctx.norm_size = norm_size
        ctx.has_bias = bias is not None
        ctx.alpha = alpha
        ctx.save_for_backward(input_hat, rstd, ln_weight)

        return output

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output):
        input_hat, rstd_data, ln_weight = ctx.saved_tensors

        grad_input = torch.empty_like(input_hat)
        grad_residual = torch.empty_like(input_hat)
        grad_ln_weight = torch.empty_like(ln_weight)
        grad_ln_bias = torch.empty_like(ln_weight)

        if ctx.has_bias:
            grad_bias = torch.empty_like(ln_weight)
            ops.train.add_residual_bias_ln_backward(
                input_hat,
                rstd_data,
                ln_weight,
                grad_output,
                grad_ln_weight,
                grad_ln_bias,
                grad_input,
                grad_residual,
                grad_bias,
                ctx.alpha,
            )
            return (
                grad_input,
                grad_residual,
                grad_bias,
                grad_ln_weight,
                grad_ln_bias,
                None,
                None,
                None,
                None,
            )
        else:
            ops.train.add_residual_bias_ln_backward(
                input_hat,
                rstd_data,
                ln_weight,
                grad_output,
                grad_ln_weight,
                grad_ln_bias,
                grad_input,
                grad_residual,
                ctx.alpha,
            )
            return (
                grad_input,
                grad_residual,
                None,
                grad_ln_weight,
                grad_ln_bias,
                None,
                None,
                None,
                None,
            )


def residual_bias_ln(
    input: torch.Tensor,
    residual: torch.Tensor,
    bias: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    alpha=1,
    is_post_ln=True,
    output: torch.Tensor = None,
    training: bool = False,
):
    """
    等价实现：
        input = residual.float() * alpha + input.float() + bias.float()
        output = torch.nn.functional.layer_norm(
            input, [input.shape[-1]], ln_weight.float(), ln_bias.float(), eps=1e-5)

    参数说明:
    Args:
        input: shape:[batch_count * seq_len, hidden_size],dtype:[torch.half]
        residual: shape:[batch_count * seq_len, hidden_size],dtype:[torch.half]
        bias: shape:[hidden_size],dtype:[torch.half]
        ln_weight:shape:[hidden_size],,dtype:[torch.half]
        ln_bias:shape:[hidden_size],,dtype:[torch.half]
        alpha: float
        is_post_ln: bool, 是否应用layernorm 后处理
    return:
        output: shape:[batch_count * seq_len, hidden_size],dtype:[torch.half]
    """
    if alpha is None:
        alpha = 1
    if not is_post_ln:
        raise NotImplementedError()
    if ln_weight is None or ln_bias is None:
        raise NotImplementedError()

    if output is None:
        output = torch.empty_like(input)
    if not training:
        if bias is not None:
            ops.infer.add_residual_bias_ln_forward(
                input, residual, bias, ln_weight, ln_bias, alpha, is_post_ln, output
            )
        else:
            ops.infer.add_residual_bias_ln_forward(
                input, residual, ln_weight, ln_bias, alpha, is_post_ln, output
            )
        return output
    else:
        return ResidualBiasLnFunction.apply(
            input, residual, bias, ln_weight, ln_bias, output, alpha, is_post_ln
        )
