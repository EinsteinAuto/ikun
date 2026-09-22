from typing import Union

import ixformer._C as ops
import torch
from torch.autograd.function import Function, FunctionCtx

__all__ = ["softmax", "ref_softmax"]


def ref_softmax(input: torch.Tensor, dim: int = None, _stacklevel: int = 3, dtype=None):
    out = torch.nn.functional.softmax(
        input, dim=dim, _stacklevel=_stacklevel, dtype=dtype
    )
    return out


def softmax(input: torch.Tensor, dim=None, _stacklevel=3, dtype=None):
    
    """
    Args:
        input:              (...)                                           torch.float16
        dim:                                                                int
            要进行softmax的维度,目前只支持最后一维， dim==-1 or dim == input.dim()-1
        _stacklevel:                                                        int
            这个参数只是为了与pytorch中对齐。 stacklevel is used in python to indicate warning mechanism how far up the stack it has to go to find the line that called the function which issued the warning.
        dtype:                                                              torch.float16                        
    Returns:
        Tensor:             (...)                                           torch.float16
    """    
    output = torch.empty_like(input)
    ops.infer.softmax(input, output, dim)
    output = output.to(dtype)
    return output
