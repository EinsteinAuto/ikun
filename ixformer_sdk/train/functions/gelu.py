from typing import List, Union

import ixformer._C as ops
import torch
from torch.autograd.function import Function, FunctionCtx

__all__ = [
    "gelu",
]


class GeluFunction(Function):
    @staticmethod
    def forward(
        ctx, input: torch.Tensor, in_place: bool = False, training: bool = False
    ):
        if training:
            if in_place:
                ctx.save_for_backward(input.clone())
            else:
                ctx.save_for_backward(input)
        if in_place:
            return ops.train.gelu_forward(input, input)
        else:
            return ops.train.gelu_forward(input)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_outputs):
        input = ctx.saved_tensors[0]
        grad_input = ops.train.gelu_backward(input, grad_outputs)
        return grad_input, None, None


def gelu(
    input: torch.Tensor, in_place: bool = False, training: bool = False
) -> torch.Tensor:
    """
    等价实现：
    torch.nn.functional.gelu

    Args:
        input: dtype:[torch.float, torch.half, torch.bfloat16]
        in place: bool. Whether to operate directly on the original input data.
    Returns:
        output: dtype:[torch.float, torch.half, torch.bfloat16]
    """
    return GeluFunction.apply(input, in_place, training)
