import ixformer.inference.functions as ops
import torch


def fused_add_rmsnorm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
):
    r"""Fused add root mean square normalization.

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    residual: torch.Tensor
        Residual tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    """
    return ops.residual_rms_norm(
        input=input,
        residual=residual,
        weight=weight,
        eps=eps,
    )


def gemma_fused_add_rmsnorm():
    pass


def gemma_rmsnorm():
    pass


def rmsnorm(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    r"""Root mean square normalization.

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.

    Returns
    -------
    output: torch.Tensor
        Normalized tensor, shape (batch_size, hidden_size).
    """

    return ops.rms_norm(
        input=input,
        weight=weight,
        eps=eps,
    )
