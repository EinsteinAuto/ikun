import os
from typing import Union

import ixformer._C as ops
import torch

__all__ = ["i8w8o32", "ref_i8w8o32"]


def ref_i8w8o32(input: torch.Tensor, weight: torch.Tensor):
    output = torch.nn.functional.linear(input.float(), weight.float()).int()
    return output


def i8w8o32(input: torch.Tensor, weight: torch.Tensor):
    
    """
    Args:
        input:              (bs, ic)              torch.int8
        weight:             (oc, ic)              torch.int8
    Returns:
        Tensor:             (bs, oc))             torch.int32
    """
    if not torch.is_tensor(input):
        raise RuntimeError("Not impl.")
    output_shape = list(input.shape)
    output_shape[-1] = weight.size(0)
    output = torch.empty(output_shape, dtype=torch.int32, device=input.device)
    ic_dim = input.size(-1)
    input = input.view(-1, ic_dim)
    ops.infer.linear_i8w8o32(input.view(-1, ic_dim), weight, output)
    return output
