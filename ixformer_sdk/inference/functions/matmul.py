import ixformer._C as ops
import torch

__all__ = ["matmul", "ref_matmul"]


def ref_matmul(input, other, *, transa, transb, alpha):
    if transa:
        dims = list(range(input.ndim))
        dims[-1], dims[-2] = dims[-2], dims[-1]
        input = input.permute(*dims).contiguous()

    if transb:
        dims = list(range(other.ndim))
        dims[-1], dims[-2] = dims[-2], dims[-1]
        other = other.permute(*dims).contiguous()

    return alpha * torch.matmul(input, other)


def matmul(
    input: torch.Tensor,
    other: torch.Tensor,
    *,
    transa: bool = False,
    transb: bool = False,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Args:
        input:              (...,m,k) or (...,k,m)                                      torch.half
                当transa为False  shape : [...,m,k], 当transa为True  shape :  [...,k,m]
        other:              (...,k,n) or (...,n,k)                                      torch.half
                当transa为False  shape : [...,m,k], 当transa为True  shape :  [...,k,m]
        transa:                                                                         bool
        transb:                                                                         bool
        alpha:                                                                          float
    Returns:
        Tensor:             (..., m, n)                                                 torch.half
    """
    if not input.is_contiguous():
        input = input.contiguous()

    if not other.is_contiguous():
        if not other.transpose(-2, -1).is_contiguous():
            other = other.contiguous()

    return ops.train.matmul(
        input, other, transa=transa, transb=transb, alpha=alpha, beta=0.0
    )
