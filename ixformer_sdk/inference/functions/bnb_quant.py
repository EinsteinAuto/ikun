from typing import List, Union

import ixformer._C as ops
import torch

__all__ = ["bnb_quant", "ref_bnb_quant"]


# A : input   shape : [row, col]
# SA : scale vector
# quant_type
#   0 : every row shared a scale, SA shape : [row]
#   1 : every col shared a scale, SA shape : [col]
def ref_bnb_quant(
    A: torch.Tensor,
    SA: torch.Tensor,
    training: bool = False,
    scale: float = 127.0,
    quant_type: int = 0,
):
    qA = torch.empty(A.shape, device = SA.device)
    if quant_type == 0:
        for i in range(A.size(0)):
            qA[i:] = torch.round(A[i:] * (scale / SA[i].to(torch.float)))
    else:
        for i in range(A.size(1)):
            qA[:,i] =  torch.round(A[:,i] * (scale / SA[i].to(torch.float)))

    qA_clamped = torch.clamp(qA, min=-128, max=127)
    qA = qA_clamped.to(torch.int8)
    return qA


def bnb_quant(
    A: torch.Tensor,
    SA: torch.Tensor,
    training: bool = False,
    scale: float = 127.0,
    quant_type: int = 0,
) -> torch.Tensor:
    
    """
    Args:
        A:                   (row, col)            torch.half
                quant input
        SA:                  (row) or (col)        torch.half
                scale vector
        training:                                  bool
        scale:                                     float
        qunt_type:                                 int
                0 : every row shared a scale, SA shape : [row]
                1 : every col shared a scale, SA shape : [col]
    Returns:
        Tensor:              (row, col)            torch.int8
                quant output    
        
    """
    return ops.infer.bnb_quant(A, SA, scale, quant_type)
