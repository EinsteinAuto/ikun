from typing import Union

import ixformer._C as ops
import torch
from torch.autograd.function import Function, FunctionCtx

__all__ = ["swiglu"]


class SwigluFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output_shape = list(input.shape)
        output_shape[-1] = output_shape[-1] // 2
        output = input.new_empty(output_shape)
        ops.train.swiglu_training_forward(input, output)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = torch.empty_like(input)
        ops.train.swiglu_training_backward(input, grad_output, grad_input)
        return grad_input


def swiglu(input):
    """
    等价实现：
    def ref_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x1, x2 = x.chunk(chunks=2, dim=-1)
        res = torch.nn.functional.silu(x1) * x2
        return res.to(dtype)


    参数说明:
    Args:
        input: dtype:torch.float, torch.half, torch.bfloat16
    return:
        output: dtype:torch.float, torch.half, torch.bfloat16
    """
    return SwigluFunction.apply(input)
