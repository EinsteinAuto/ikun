from typing import List, Union

import ixformer._C as ops
import torch
from torch.autograd.function import Function, FunctionCtx

__all__ = ["bnb_mm_dequant"]


# A : quant input              shape : [row, col]  shape : torch.int
def bnb_mm_dequant(
    A: torch.Tensor,
    quant_state: tuple,
    row_stats: torch.Tensor,
    col_stats: torch.Tensor,
    bias: torch.Tensor = None,
    add_bias: bool = False,
    training: bool = False,
) -> torch.Tensor:
    
    """
    Args:
        A:                  (row, col)            torch.int8
        quant_state:                              tuple
        row_stats:          (row)                 torch.float
        col_stats:          (col)                 torch.float
        bias:               (col)                 torch.half
        add_bias:                                 bool
        training:                                 bool
    Returns:
        Tensor:             (row, col)            torch.half
        
    """
    
    assert A.dtype == torch.int
    if bias is not None:
        add_bias = True
        print("bias.dtype:", bias.dtype)
        assert bias.dtype == torch.half
    else:
        bias = A
    out_shape = quant_state[0]
    if len(out_shape) == 3:
        out_shape = (out_shape[0] * out_shape[1], out_shape[2])
    out = torch.full(size=out_shape, fill_value=0, dtype=torch.half, device=A.device)
    new_row_stats = torch.full(
        size=(out_shape[0],), fill_value=0, dtype=torch.float, device=A.device
    )
    new_col_stats = torch.full(
        size=(out_shape[1],), fill_value=0, dtype=torch.float, device=A.device
    )

    assert (
        new_row_stats.shape[0] == row_stats.shape[0]
    ), f"{new_row_stats.shape} vs {row_stats.shape}"
    assert (
        new_col_stats.shape[0] == col_stats.shape[0]
    ), f"{new_col_stats.shape} vs {col_stats.shape}"
    numRows = out_shape[0]
    numCols = out_shape[1]
    ops.infer.bnb_mm_dequant(
        A,
        row_stats,
        col_stats,
        out,
        new_row_stats,
        new_col_stats,
        numRows,
        numCols,
        add_bias,
        bias,
    )
    return out
