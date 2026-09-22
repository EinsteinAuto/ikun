import os
from typing import Union

import ixformer._C as ops
import torch

__all__ = ["linear", "ref_linear", "mixed_type_linear", "ref_mixed_type_linear"]


def ref_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    act_type=-1,
):
    output = torch.nn.functional.linear(input, weight, bias)
    if act_type == -1:
        act_fn = torch.nn.Identity()
    elif act_type == 3:
        act_fn = torch.nn.GELU()
    elif act_type == 4:
        act_fn = torch.nn.ReLU()
    elif act_type == 12:
        act_fn = torch.nn.SiLU()
    else:
        raise KeyError("act_type not supported")
    output = act_fn(output)
    return output


def gemv_conditions(input, weight, bias, gemv_max_batch):
    # gemv 使用的条件 input:[m,k] weight:[n,k]
    # 1. m<=gemv_max_batch
    # 2. k%32==0 n%2==0
    # 3. bias is None
    input = input.view(-1, input.shape[-1])
    weight = weight.view(-1, weight.shape[-1])
    m = input.shape[0]
    k = input.shape[1]
    n = weight.shape[0]
    if bias is None and m <= gemv_max_batch and k % 32 == 0 and n % 2 == 0:
        return True
    return False


def linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    output: torch.Tensor = None,
    persistent: bool = False,
    act_type : int = -1,
):
    """
    Args:
        input:              (...,k)                 torch.float16, torch.bfloat16
        weight:             (n, k)                  torch.float16, torch.bfloat16
        bias:               (n)                     torch.float16, torch.bfloat16
        output:             (...,n)                 torch.float16, torch.bfloat16
        persistent:                                 bool
                    是否限制 Gemm Kernel 的 Block 数量
    Returns:
        output:             (...,n)                 torch.float16, torch.bfloat16
    """
    if not input.is_contiguous():
        input = input.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    use_gemv = True
    gemv_max_batch = 1
    disable_infer_gemm_ex = os.getenv("DISABLE_INFER_GEMM_EX", "0")
    use_gemv = (
        use_gemv
        and gemv_conditions(input, weight, bias, gemv_max_batch)
        and disable_infer_gemm_ex != "1"
    )

    if output is None:
        output_shape = list(input.shape)
        output_shape[-1] = weight.shape[0]
        output = input.new_empty(output_shape)

    if not use_gemv:
        output = ops.infer.linear(input, weight, act_type, bias, output, persistent)
    else:
        output = ops.infer.linear_ex(input, weight, bias, output)
    return output


def ref_mixed_type_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    output: torch.Tensor = None,
    persistent=False,  # TODO: support persistent
):
    input = input.to(weight.dtype)
    if bias:
        bias = bias.to(weight.dtype)
    output = torch.nn.functional.linear(input, weight, bias)
    return output


def mixed_type_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    output: torch.Tensor = None,
    persistent=False,  # TODO: support persistent
):
    """
    Args:
        input:      (...,k)     torch.half, torch.bfloat16
        weight:     (m, k)      torch.float32
        bias:                   not supported
        output:     (...,m)     torch.float32
        persistent:             bool 
    Returns:
        output:     (...,m)     torch.float32
    """
    output = ops.infer.mixed_type_linear(input, weight, bias, output)
    return output
