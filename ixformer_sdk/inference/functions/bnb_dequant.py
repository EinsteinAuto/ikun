from typing import List, Union

import ixformer._C as ops
import torch
from torch.autograd.function import Function, FunctionCtx

__all__ = [
    "bnb_dequant",
    "ref_bnb_dequant",
]


def ref_bnb_dequant(
    qA: torch.Tensor,
    SA: torch.Tensor,
    training: bool = False,
    scale: float = 127.0,
    dequant_type: int = 0,
):
    A = torch.empty(qA.shape, dtype = SA.dtype, device = SA.device)
    if dequant_type == 0:
        for i in range(qA.size(0)): 
            A[i:] = qA[i:] * (SA[i].to(torch.float) / scale).to(SA.dtype)
    else:
        for i in range(qA.size(1)):
            A[:,i] =  qA[:,i] * (SA[i].to(torch.float) / scale).to(SA.dtype)
            
    return A


def bnb_dequant(
    qA: torch.Tensor,
    SA: torch.Tensor,
    training: bool = False,
    scale: float = 127.0,
    dequant_type: int = 0,
) -> torch.Tensor:
    
    """
    Args:
        qA:                  (row, col)            torch.int8
                dequant input
        SA:                  (row) or (col)        torch.half
                scale vector
        training:                                  bool
        scale:                                     float
        dequnt_type:                               int
                0 : every row shared a scale, SA shape : [row]
                1 : every col shared a scale, SA shape : [col]
    Returns:
        Tensor:              (row, col)            torch.half
                dequant output    
        
    """
    return ops.infer.bnb_dequant(qA, SA, scale, dequant_type)
