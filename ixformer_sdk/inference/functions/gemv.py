import os
from typing import Union

import ixformer._C as ops
import torch

__all__ = ["gemv", "ref_gemv"]


def ref_gemv(x: torch.Tensor, A: torch.Tensor, gemv_max_batch: int = 1):
    output = torch.nn.functional.linear(x, A)
    return output


def gemv_conditions(input, weight, gemv_max_batch):
    # gemv 使用的条件 input:[m,k] weight:[n,k]
    # 1. m<=gemv_max_batch
    # 2. k%2==0 n%2==0
    # 3. bias is None
    input = input.view(-1, input.shape[-1])
    weight = weight.view(-1, weight.shape[-1])
    m = input.shape[0]
    k = input.shape[1]
    n = weight.shape[0]
    if m <= gemv_max_batch and k % 2 == 0 and n % 2 == 0:
        return True
    return False


def gemv(x: torch.Tensor, A: torch.Tensor, gemv_max_batch: int = 1):
    
    """
    Args:
        x:                  (..., k)                torch.float16, torch.bfloat16
        A:                  (n,k)                   torch.float16, torch.bfloat16, torch.float32
        gemv_max_batch:                             int
                        用于是否满足gemv使用条件的判断,目前只支持到1   
    Returns:
        Tensor:             (..., n)                torch.float16, torch.bfloat16
    """
    disable_infer_gemm_ex = os.getenv("DISABLE_INFER_GEMM_EX", "0")
    use_gemv = gemv_conditions(x, A, gemv_max_batch) and disable_infer_gemm_ex != "1"
    assert use_gemv == True
    output_shape = list(x.shape)
    output_shape[-1] = A.shape[0]
    output = x.new_empty(output_shape)
    output = ops.infer.linear_ex(x, A, None, output)
    return output
