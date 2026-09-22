from typing import List, Union

import ixformer._C as ops
from torch.autograd.function import Function, FunctionCtx

__all__ = ["bnb_double_quant"]

import ctypes as ct

import torch
from torch import Tensor


def get_ptr(A):
    if A is None:
        return None
    else:
        return ct.c_void_p(A.data.data_ptr())


class COOSparseTensor:
    def __init__(self, rows, cols, nnz, rowidx, colidx, values):
        assert rowidx.dtype == torch.int
        assert colidx.dtype == torch.int
        assert values.dtype == torch.half
        assert values.numel() == nnz
        assert rowidx.numel() == nnz
        assert colidx.numel() == nnz

        self.rows = rows
        self.cols = cols
        self.nnz = nnz
        self.rowidx = rowidx
        self.colidx = colidx
        self.values = values


def coo_zeros(rows, cols, nnz, device, dtype=torch.half):
    rowidx = torch.full(size=(nnz,), fill_value=0, dtype=torch.int, device=device)

    colidx = torch.full((nnz,), fill_value=0, dtype=torch.int, device=device)
    values = torch.full((nnz,), fill_value=0, dtype=dtype, device=device)
    return COOSparseTensor(rows, cols, nnz, rowidx, colidx, values)


def get_colrow_absmax(
    A, row_stats=None, col_stats=None, nnz_block_ptr=None, threshold=0.0
):
    cols = A.shape[-1]
    if len(A.shape) == 3:
        rows = A.shape[0] * A.shape[1]
    else:
        rows = A.shape[0]

    col_tiles = (cols + 255) // 256
    tiled_rows = ((rows + 15) // 16) * 16
    if row_stats is None:
        row_stats = torch.full(
            size=(rows,), fill_value=-50000.0, dtype=torch.float, device=A.device
        )
    if col_stats is None:
        col_stats = torch.full(
            size=(cols,), fill_value=-50000.0, dtype=torch.float, device=A.device
        )

    # if nnz_block_ptr is None and threshold > 0.0:
    nnz_block_ptr = torch.full(
        size=(tiled_rows * col_tiles + 1,),
        fill_value=0,
        dtype=torch.int,
        device=A.device,
    )

    ops.infer.bnb_getColRowStats(
        A, row_stats, col_stats, nnz_block_ptr, threshold, rows, cols
    )

    return row_stats, col_stats, nnz_block_ptr


# A : quant input              shape : [row, col] shape:torch.half
def bnb_double_quant(
    A: torch.Tensor, training: bool = False, threshold: float = 0.0
) -> torch.Tensor:
    
    """
    Args:
        A:                  (row, col)            torch.float16
                quant input
        training:                                 bool
        threshold:                                float
                abs of element exceeds threshold will be ignored
    Returns:
        out_row:            (row, col)            torch.int8
        out_col:            (row, col)            torch.int8
        row_stats           (row)                 torch.float
        col_stats           (col)                 torch.float
        coo_tensor
    """

    assert A.dtype == torch.half

    cols = A.shape[-1]
    if len(A.shape) == 3:
        rows = A.shape[0] * A.shape[1]
    else:
        rows = A.shape[0]

    row_stats, col_stats, nnz_row_ptr = get_colrow_absmax(A, threshold=threshold)

    out_col = torch.full(size=A.shape, fill_value=0, dtype=torch.int8, device=A.device)
    out_row = torch.full(size=A.shape, fill_value=0, dtype=torch.int8, device=A.device)

    coo_tensor = None
    if threshold > 0.0:
        nnz = nnz_row_ptr.cpu().numpy()[-1]
        if nnz > 0:
            coo_tensor = coo_zeros(A.shape[0], A.shape[1], nnz, A.device)

            ops.infer.bnb_doubleRowColQuant(
                A,
                row_stats,
                col_stats,
                out_col,
                out_row,
                coo_tensor.rowidx,
                coo_tensor.colidx,
                coo_tensor.values,
                nnz_row_ptr,
                threshold,
                rows,
                cols,
            )
            val, idx = torch.sort(torch.Tensor(coo_tensor.rowidx.cpu().numpy()))
            coo_tensor.rowidx = val
            coo_tensor.colidx = torch.Tensor(coo_tensor.colidx.cpu().numpy())[idx].to(
                torch.int32
            )
            coo_tensor.values = torch.Tensor(coo_tensor.values.cpu().numpy())[idx].to(
                torch.half
            )
            # coo_tensor.colidx = coo_tensor.colidx[idx]
            # coo_tensor.values = coo_tensor.values[idx]
        else:
            ops.infer.bnb_doubleRowColQuant(
                A,
                row_stats,
                col_stats,
                out_col,
                out_row,
                out_row,
                out_row,
                out_row,
                out_row,
                0.0,
                rows,
                cols,
            )
    else:
        ops.infer.bnb_doubleRowColQuant(
            A,
            row_stats,
            col_stats,
            out_col,
            out_row,
            out_row,
            out_row,
            out_row,
            out_row,
            threshold,
            rows,
            cols,
        )

    return out_row, out_col, row_stats, col_stats, coo_tensor
