from typing import List, Tuple, Union

import ixformer._C as ops
import torch
from torch.autograd.function import Function, FunctionCtx

__all__ = ["layernorm"]


class LayerNormFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        output: torch.Tensor,
        normalized_shape=None,
        training: bool = False,
    ):

        if ln_weight is None or ln_bias is None:
            raise NotImplementedError()
        # normalized_shape 需要是list或者tuple，并且不能为空
        if normalized_shape == None:
            norm_size = ln_weight.size(-1)
        else:
            norm_size = 1
            if isinstance(normalized_shape, int):
                norm_size = normalized_shape
                normalized_shape = [normalized_shape]

            elif (
                isinstance(normalized_shape, list)
                or isinstance(normalized_shape, tuple)
            ) and len(normalized_shape) >= 1:
                for i in normalized_shape:
                    norm_size = i * norm_size
            else:
                raise f"layer_norm(): argument 'normalized_shape' (position 2) must be tuple of ints, not {type(normalized_shape)}"
            if norm_size != ln_weight.size(-1):
                raise f"layer_norm(): argument 'norm_size'  must == ln_weight.size(-1)"
        if output is None:
            output = torch.empty_like(input)
        if training:
            mean_size = input.numel() // norm_size

            input_hat = torch.empty_like(input)
            rstd = torch.empty([mean_size], dtype=input.dtype, device=input.device)
            ops.train.layernorm_training_forward(
                input, ln_weight, ln_bias, output, input_hat, rstd
            )
            ctx.norm_size = norm_size
            ctx.save_for_backward(input_hat, rstd, ln_weight)
        else:
            ops.train.layernorm_forward(input, ln_weight, ln_bias, output)
        return output

    @staticmethod
    # def backward(ctx: FunctionCtx, grad_output, dh, dr):
    def backward(ctx: FunctionCtx, grad_output):
        input_hat, rstd, ln_weight = ctx.saved_tensors

        grad_input = torch.empty_like(input_hat)
        grad_weight = torch.empty_like(ln_weight)
        grad_bias = torch.empty_like(ln_weight)
        ops.train.layernorm_weightbias_backward(
            input_hat, grad_output, grad_weight, grad_bias
        )
        ops.train.layernorm_input_backward(
            input_hat, rstd, grad_output, ln_weight, grad_input
        )
        return grad_input, grad_weight, grad_bias, None, None, None


def layernorm(
    input: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    normalized_shape=None,
    output: torch.Tensor = None,
    training: bool = False,
):
    """
    等价实现：
        torch.nn.functional.layer_norm( input, normalized_shape, ln_weight, ln_bias, eps=0.000001)
    Arguments:
        input: (batch_count * seq_len, hidden_size), dtype:[torch.half]
        ln_weight: (hidden_size), dtype:[torch.half]
        ln_bias:(hidden_size),dtype:[torch.half]
        normalized_shape: list[int], [hidden_size]
    Return:
        output: (batch_count * seq_len, hidden_size), dtype:[torch.half]

    """
    return LayerNormFunction.apply(
        input, ln_weight, ln_bias, output, normalized_shape, training
    )
