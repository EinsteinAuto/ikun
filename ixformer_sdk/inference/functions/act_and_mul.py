from typing import List, Union

import ixformer._C as ops
import torch
import torch.nn.functional as NNF

__all__ = ["ref_silu_and_mul", "ref_gelu_and_mul", "ref_gelu_tanh_and_mul",
           "silu_and_mul", "gelu_and_mul", "gelu_tanh_and_mul"]


def ref_silu_and_mul(input: "torch.Tensor") -> torch.Tensor:
    x1, x2 = input.chunk(chunks=2, dim=-1)
    res = NNF.silu(x1) * x2
    return res


def ref_gelu_and_mul(input: "torch.Tensor", gate_first=True) -> torch.Tensor:
    x1, x2 = input.chunk(chunks=2, dim=-1)
    if gate_first:
        res = NNF.gelu(x1) * x2
    else:
        res = NNF.gelu(x2) * x1
    return res


def ref_gelu_tanh_and_mul(input: "torch.Tensor") -> torch.Tensor:
    x1, x2 = input.chunk(chunks=2, dim=-1)
    res = NNF.gelu(x1) * x2
    return res


def silu_and_mul(input: torch.Tensor, output: torch.Tensor = None):
    
    """
    Args:
        input:              (..., 2*hidden_size)    torch.float16, torch.bfloat16, torch.float32
        output:             (..., hidden_size)      torch.float16, torch.bfloat16, torch.float32
        
    Returns:
        output:             (..., hidden_size)      torch.float16, torch.bfloat16, torch.float32
    """
    if output is None:
        output_shape = list(input.shape)
        output_shape[-1] = output_shape[-1] // 2
        output = input.new_empty(output_shape)
    
    ops.infer.silu_and_mul(input, output)
    
    return output


def gelu_and_mul(input: "torch.Tensor", output: torch.Tensor = None, gate_first=True):
    
    """
    Args:
        input:              (..., 2*hidden_size)    torch.float16, torch.bfloat16, torch.float32
        output:             (..., hidden_size)      torch.float16, torch.bfloat16, torch.float32
        gate_first:                                 bool
    Returns:
        output:             (..., hidden_size)      torch.float16, torch.bfloat16, torch.float32
    """
    if output is None:
        output_shape = list(input.shape)
        output_shape[-1] = output_shape[-1] // 2
        output = input.new_empty(output_shape)

    ops.infer.gelu_and_mul(input, output, gate_first)

    return output


def gelu_tanh_and_mul(input: torch.Tensor, output: torch.Tensor = None):
    
    """
    Args:
        input:              (..., 2*hidden_size)    torch.float16, torch.bfloat16, torch.float32
        output:             (..., hidden_size)      torch.float16, torch.bfloat16, torch.float32
    Returns:
        output:             (..., hidden_size)      torch.float16, torch.bfloat16, torch.float32
    """
    if output is None:
        output_shape = list(input.shape)
        output_shape[-1] = output_shape[-1] // 2
        output = input.new_empty(output_shape)

    ops.infer.gelu_tanh_and_mul(input, output)

    return output