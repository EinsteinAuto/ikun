from typing import List, Union

import ixformer._C as ops
import torch

__all__ = ["bnb_qgemm", "ref_bnb_qgemm"]


# qA : quant input              shape : [bs, in_feature]
# qW : quant weight             shape : [out_feature, in_feature]
# SA : scale vector of qA       shape : [bs]
# SW : scale vector of qW       shape : [out_feature]


def ref_bnb_qgemm(
    qA: torch.Tensor,
    qW: torch.Tensor,
    SA: torch.Tensor,
    SW: torch.Tensor,
    training: bool = False,
    scaleA: float = 127.0,
    scaleW: float = 127.0,
):
    y = torch.nn.functional.linear(qA.to(torch.float), qW.to(torch.float))
    out = torch.empty(y.shape, dtype = SA.dtype, device = SA.device)
    for i in range(qA.size(0)):
        for j in range(qW.size(0)): 
            out[i][j] = y[i][j] * (SA[i].to(torch.float) / scaleA) * (SW[j].to(torch.float) / scaleW)
    return out.to(SA.dtype)


def bnb_qgemm(
    qA: torch.Tensor,
    qW: torch.Tensor,
    SA: torch.Tensor,
    SW: torch.Tensor,
    training: bool = False,
    scaleA: float = 127.0,
    scaleW: float = 127.0,
) -> torch.Tensor:
    
    """
    Args:
        qA:                 (bs, in_feature)                torch.int8
        qW:                 (out_feature, in_feature)       torch.int8
        SA:                 (bs)                            torch.half
            scale vector of qA 
        SA:                 (out_feature)                   torch.half
            scale vector of qW 
        training:                                           bool
        scaleA:                                             float
        scaleW:                                             float
    Returns:
        Tensor:             (bs, out_feature)               torch.half
    """
    return ops.infer.bnb_qgemm(qA, qW, SA, SW, scaleA, scaleW)
