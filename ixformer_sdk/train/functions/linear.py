import os
from typing import Union

import ixformer._C as ops
import torch
from torch.autograd.function import Function, FunctionCtx

__all__ = ["linear"]


class LinearFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
        output: torch.Tensor = None,
    ):
        if bias is not None:
            if output is None:
                output = ops.train.linear_forward(input, weight, bias)
            else:
                ops.train.linear_forward_(input, weight, bias, output)
        else:
            if output is None:
                output = ops.train.linear_forward(input, weight)
            else:
                ops.train.linear_forward_(input, weight, output)

        ctx.has_bias = bias is not None
        ctx.save_for_backward(input, weight)

        return output

    @staticmethod
    def backward(ctx: FunctionCtx, dy: torch.Tensor):
        x, w = ctx.saved_tensors

        dx = ops.train.linear_backward_dx(w, dy, x.shape)

        dw = ops.train.linear_backward_dw(x, dy, w.shape)

        if ctx.has_bias:
            reduce_dims = list(range(dy.ndim - 1))
            db = torch.sum(dy, reduce_dims)
            return dx, dw, db, None
        else:
            return dx, dw, None, None


def gemv_conditions(input, weight, bias, gemv_max_batch):
    # gemv 使用的条件 input:[m,k] weight:[n,k]
    # 1. m<=gemv_max_batch
    # 2. k%2==0 n%2==0
    # 3. bias is None
    input = input.view(-1, input.shape[-1])
    weight = weight.view(-1, weight.shape[-1])
    m = input.shape[0]
    k = input.shape[1]
    n = weight.shape[0]
    if bias is None and m <= gemv_max_batch and k % 2 == 0 and n % 2 == 0:
        return True
    return False


def linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    output: torch.Tensor = None,
    use_gemv: bool = True,
    gemv_max_batch=1,
):
    """
     Arguments:
        input    :  [...,k]      dtype: [torch.half, torch.bfloat16]
        weights :   [n,k]        dtype: [torch.half, torch.bfloat16]
        use_gemv: bool           是否使用gemv
            gemv 使用的条件 input:[m,k] weight:[n,k]
                1. m<=gemv_max_batch
                2. k%2==0 n%2==0
                3. bias is None
        gemv_max_batch: int       用于是否满足gemv使用条件的判断
    Return:
        output  :   [...,n]      dtype: [torch.half, torch.bfloat16]

    """
    return LinearFunction.apply(input, weight, bias, output)
