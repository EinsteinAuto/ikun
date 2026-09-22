from typing import Union

import ixformer._C as ops
import torch
from torch.autograd.function import Function, FunctionCtx

__all__ = ["geglu"]


class GegluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output_shape = list(input.shape)
        output_shape[-1] = output_shape[-1] // 2
        output = input.new_empty(output_shape)
        ops.train.geglu_training_forward(input, output)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = torch.empty_like(input)
        ops.train.geglu_training_backward(input, grad_output, grad_input)
        return grad_input


def geglu(input: "torch.Tensor"):
    """
    等价实现：
    def ref_gelu_and_mul(x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x1, x2 = x.chunk(chunks=2, dim=-1)
        res = NNF.gelu(x2) * x1
        return res.to(dtype)

    Args:
        input: dtype:[torch.float, torch.half, torch.bfloat16]
    Returns:
        output: (....,input.shape[-1] //2), dtype:[torch.float, torch.half, torch.bfloat16]
    """
    return GegluFunction.apply(input)
