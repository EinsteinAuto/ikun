from typing import List, Union

import ixformer._C as ops
import torch

__all__ = ["bnb_rowcol_absmax", "ref_bnb_rowcol_absmax"]


# input  : input   shape : [row, col]
# threshold : abs of element exceeds threshold will be ignored
# type
#   0 : row absmax
def ref_bnb_rowcol_absmax(
    input: torch.Tensor,
    training: bool = False,
    threshold: float = 0.0,
    type: int = 0,
):
    input = input.float()
    if threshold ==0.0:
        threshold = float('inf')
    mask = (torch.abs(input) < threshold)
    masked_input = mask * input
    masked_input = masked_input.half()
    if type == 0:
        out = torch.amax(torch.abs(masked_input), dim=1)
        
    else:
        out = torch.amax(torch.abs(masked_input), dim=0)
    return out


def bnb_rowcol_absmax(
    input: torch.Tensor,
    training: bool = False,
    threshold: float = 0.0,
    type: int = 0,
) -> torch.Tensor:
    
    """
    Args:
        input:              (row, col)        torch.half
                目前col值必须满足col%2==0
        training:                             bool
        threshold:                            float
                abs of element exceeds threshold will be ignored
        type:                                  int
                row absmax, 目前只支持type=0
    Returns:
        Tensor:             (row)              torch.half
    """
    return ops.infer.bnb_rowcol_absmax(input, threshold, type)
