from typing import Union

import ixformer._C as ops
import torch

__all__ = ["solve", "ref_slove"]


def ref_slove(
    A: torch.Tensor, B: torch.Tensor, *, left: bool = True, out: torch.Tensor = None
):
    out = torch.linalg.solve(A, B, left=left)
    return out


def solve(
    A: torch.Tensor, B: torch.Tensor, *, left: bool = True, out: torch.Tensor = None
):
    """
    Args:
        A:                  (..., n, n)                                                         torch.float
        B:                  (..., n) or (..., n, k) or (n,...) or (n, k) or (n)                 torch.float
        left:                                                                                   bool
                whether to solve the system AX=B or XA=B. Default: True, 目前只支持left =True
        out:                (..., n, k)                                                         torch.float
    Returns:
        out:                (..., n, k)                                                         torch.float
    """

    n = A.shape[-1]
    batch_count = A.numel() // (n * n)
    if B.dim() == 1:
        k = 1
    elif B.dim() == 2:
        if A.dim() > 2 and B.shape == (batch_count, n):
            k = 1
        else:
            nid = 0 if left else 1
            k = B.shape[nid ^ 1]
    else:
        k = B.size(B.dim() - 1 if left else B.dim() - 2)

    if n <= 64 and k <= 64 and left:
        return ops.infer.solve(A, B, left)
    else:
        device = A.device
        cpu_A = A.cpu()
        cpu_B = B.cpu()
        cpu_res = ref_slove(A=cpu_A, B=cpu_B, left=left)
        return cpu_res.to(device)
