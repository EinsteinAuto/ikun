import ixformer._C as ops
import torch
from torch.autograd.function import Function, FunctionCtx

__all__ = ["matmul", "MatmulFunction"]


class MatmulFunction(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        input: torch.Tensor,
        other: torch.Tensor,
        out: torch.Tensor = None,
        transa: bool = False,
        transb: bool = False,
        alpha: float = 1.0,
        beta: float = 0.0,
    ):
        ctx.save_for_backward(input, other)
        ctx.params = (transa, transb, alpha, beta)
        if out is None:
            return ops.train.matmul(
                input, other, transa=transa, transb=transb, alpha=alpha, beta=beta
            )
        else:
            return ops.train.matmul(
                input,
                other,
                out=out,
                transa=transa,
                transb=transb,
                alpha=alpha,
                beta=beta,
            )

    @staticmethod
    def backward(ctx: FunctionCtx, dy):
        input, other = ctx.saved_tensors
        transa, transb, alpha, beta = ctx.params

        if beta in [1, None]:
            raise RuntimeError("Backward don't support beta == 1.0f")

        if not transa and not transb:
            dx = matmul(dy, other, transb=True, alpha=alpha)
            do = matmul(input, dy, transa=True, alpha=alpha)
            return dx, do, None, None, None, None, None

        if transa and not transb:
            dx = matmul(other, dy, transb=True, alpha=alpha)
            do = matmul(input, dy, alpha=alpha)
            return dx, do, None, None, None, None, None

        if not transa and transb:
            dx = matmul(dy, other, alpha=alpha)
            do = matmul(dy, input, transa=True, alpha=alpha)
            return dx, do, None, None, None, None, None

        if transa and transb:
            dx = matmul(other, dy, transa=True, transb=True, alpha=alpha)
            do = matmul(dy, input, transa=True, transb=True, alpha=alpha)
            return dx, do, None, None, None, None, None


def matmul(
    input: torch.Tensor,
    other: torch.Tensor,
    *,
    out: torch.Tensor = None,
    transa: bool = False,
    transb: bool = False,
    alpha: float = 1.0,
    beta: float = 0.0
) -> torch.Tensor:
    """
    等价实现:
    def pt_matmul(a, b, transa, transb, alpha):
        if transa:
            dims = list(range(a.ndim))
            dims[-1], dims[-2] = dims[-2], dims[-1]
            a = a.permute(*dims).contiguous()

        if transb:
            dims = list(range(b.ndim))
            dims[-1], dims[-2] = dims[-2], dims[-1]
            b = b.permute(*dims).contiguous()

        return alpha * torch.matmul(a, b)
     Arguments:
        input:
            当transa为False  shape : [...,m,k]       dtype: torch.half
            当transa为True  shape :  [...,k,m]       dtype: torch.half
        other:
            当transb为False  shape : [...,k,n]       dtype: torch.half
            当transb为True  shape :  [...,n,k]       dtype: torch.half
    Return:
        output:   [...m,n]                         dtype: [torch.half]

    """
    if not input.is_contiguous():
        input = input.contiguous()

    if not other.is_contiguous():
        if not other.transpose(-2, -1).is_contiguous():
            other = other.contiguous()

    return MatmulFunction.apply(input, other, out, transa, transb, alpha, beta)
