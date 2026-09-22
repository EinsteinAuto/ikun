import ixformer._C as ops
import torch

__all__ = [
    "ref_add",
    "add",
]


def ref_add(input: torch.Tensor, other: torch.Tensor, out: torch.Tensor = None):
    return torch.add(input, other, out=out)


def add(input: torch.Tensor, other: torch.Tensor, out: torch.Tensor = None):
    """
    out = input + other
    Support elementwise addition, but broadcasting is not supported yet.
    Note: The dtype of input and other needs to be the same.
    Args:
        input:              (...)        torch.float32, torch.float16, torch.bfloat16
        other:              (...)        same as input
        out:                (...)        same as input
    Returns:
        out:                (...)        same as input
    """
    if input.dtype not in [torch.float16, torch.float32, torch.bfloat16]:
        return torch.add(input, other, out=out)
    if not input.is_contiguous() or not other.is_contiguous():
        return torch.add(input, other, out=out)
    if out is not None and not out.is_contiguous():
        return torch.add(input, other, out=out)

    if input.dtype != other.dtype:
        return torch.add(input, other, out=out)
    if out is not None and out.dtype != input.dtype:
        return torch.add(input, other, out=out)

    assert input.shape == other.shape, (f"broadcasting is not supported yet."
        "input is {input.shape}, other is {other.shape}")
    
    if out is None:
        out = torch.empty_like(input)

    ops.infer.add(input, other, out)

    return out
