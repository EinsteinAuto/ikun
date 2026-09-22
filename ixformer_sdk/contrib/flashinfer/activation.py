import ixformer.inference.functions as ops
import torch


def gelu_and_mul():
    pass


def gelu_tanh_and_mul():
    pass


def silu_and_mul(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    r"""Fused SiLU and Mul operation.

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (..., 2 * hidden_size).

    out: Optional[torch.Tensor]
        The the output tensor, if specified, the kernel will update this tensor inplace.

    Returns
    -------
    output: torch.Tensor
        Output tensor, shape (..., hidden_size).
    """
    return ops.silu_and_mul(input=input, output=out)
