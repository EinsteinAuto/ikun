from typing import List, Tuple, Union

import ixformer._C as ops
import torch

__all__ = [
    "layer_norm",
    "ref_layer_norm",
    "residual_layer_norm",
    "ref_residual_layer_norm",
    "ref_residual_layer_norm_bias_alpha",
    "residual_layer_norm_bias_alpha",
    "ref_layer_norm_2sb_fused",
    "layer_norm_2sb_fused",
]


def ref_layer_norm(
    input: torch.Tensor,
    normalized_shape: List[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    output: torch.Tensor = None,
):

    if weight is None or bias is None or weight.dim() > 1 or bias.dim() > 1:
        raise NotImplementedError(
            "layer_norm only support weight.dim() ==1 and bias.dim()==1!"
        )
    if normalized_shape == None:
        norm_size = weight.size(-1)
        normalized_shape = [norm_size]
    else:
        if (
            isinstance(normalized_shape, list) or isinstance(normalized_shape, tuple)
        ) and len(normalized_shape) == 1:
            norm_size = normalized_shape[0]
        else:
            raise ValueError(
                f"layer_norm(): argument 'normalized_shape' (position 2) must be tuple of ints and length of tuple is equal to 1, not {type(normalized_shape)}"
            )
    if norm_size != weight.size(-1):
        raise ValueError(f"layer_norm(): argument 'norm_size'  must == weight.size(-1)")

    norm_out = torch.nn.functional.layer_norm(
        input, normalized_shape, weight, bias, eps=eps
    )
    if output is not None:
        assert output.shape == norm_out.shape
        output.copy_(norm_out)
    else:
        output = norm_out

    return output


def layer_norm(
    input: torch.Tensor,
    normalized_shape: List[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    output: torch.Tensor = None,
):
    """
    This function is deprecated, please use residual_layer_norm.
    等价实现：
        torch.nn.functional.layer_norm( input, normalized_shape, weight, bias, eps=0.000001)
    Args:
        input:              (..., hidden_size)    torch.float16, torch.bfloat16, torch.float32
        normalized_shape:                         list[int]
        weight:             (hidden_size)         torch.float16, torch.bfloat16, torch.float32
        bias:               (hidden_size)         torch.float16, torch.bfloat16, torch.float32
        eps:                                      float32
    Returns:
        output:             (..., hidden_size)    torch.float16, torch.bfloat16, torch.float32
    """
    if weight is None or bias is None or weight.dim() > 1 or bias.dim() > 1:
        raise NotImplementedError(
            "layer_norm only support weight.dim() ==1 and bias.dim()==1!"
        )
    if normalized_shape == None:
        norm_size = weight.size(-1)
        normalized_shape = [norm_size]
    else:
        if (
            isinstance(normalized_shape, list) or isinstance(normalized_shape, tuple)
        ) and len(normalized_shape) == 1:
            norm_size = normalized_shape[0]
        else:
            raise ValueError(
                f"layer_norm(): argument 'normalized_shape' (position 2) must be tuple of ints and length of tuple is equal to 1, not {type(normalized_shape)}"
            )
    if norm_size != weight.size(-1):
        raise ValueError(f"layer_norm(): argument 'norm_size'  must == weight.size(-1)")
    if output is None:
        output = torch.empty_like(input)
    ops.infer.layer_norm(input, weight, bias, None, output, eps)
    return output


def ref_residual_layer_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor = None,
    residual_bias: torch.Tensor = None,
    eps: float = 1e-5,
    output: torch.Tensor = None,
    residual_output: torch.Tensor = None,
):
    normalized_shape = [weight.size(-1)]

    if residual_bias is not None:
        input = input + residual_bias

    if residual is not None:
        residual_output = torch.add(input, residual, out=residual_output)
        input = residual_output

    norm_out = torch.nn.functional.layer_norm(
        input, normalized_shape, weight, bias, eps=eps
    )

    if output is None:
        output = norm_out
    else:
        output.copy_(norm_out)

    return output, residual_output


def residual_layer_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor = None,
    residual_bias: torch.Tensor = None,
    eps: float = 1e-5,
    output: torch.Tensor = None,
    residual_output: torch.Tensor = None,
):
    """
    Args:
        input:              (..., hidden_size)    torch.float16, torch.bfloat16, torch.float32
        weight:             (hidden_size)         torch.float16, torch.bfloat16, torch.float32
        bias:               (hidden_size)         torch.float16, torch.bfloat16, torch.float32
        residual:           (..., hidden_size)    torch.float16, torch.bfloat16, torch.float32
        residual_bias:      (hidden_size)         torch.float16, torch.bfloat16, torch.float32
        eps:                                      float32
    Returns:
        output:             (..., hidden_size)    torch.float16, torch.bfloat16, torch.float32 If set to None, an inplace operation will be performed on input.
        residual_output:    (..., hidden_size)    torch.float16, torch.bfloat16, torch.float32 If set to None, an inplace operation will be performed on residual.
    """
    if residual is None:
        if output is None:
            output = torch.empty(input.shape, device=input.device, dtype=input.dtype)
        ops.infer.layer_norm(input, weight, bias, residual_bias, output, eps)
    else:
        ops.infer.residual_layer_norm(
            input,
            residual,
            weight,
            bias,
            residual_bias,
            output,
            residual_output,
            1.0,
            eps,
            False,
        )
        residual_output = residual_output if residual_output is not None else residual
        output = output if output is not None else input

    return output, residual_output


def ref_residual_layer_norm_bias_alpha(
    input: torch.Tensor,
    normalized_shape: List[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    residual_bias: torch.Tensor = None,
    alpha: float = 1.0,
    eps: float = 1e-5,
    is_post_ln=False,
):
    if (
        weight is None
        or bias is None
        or residual is None
        or weight.dim() > 1
        or bias.dim() > 1
    ):
        raise NotImplementedError(
            "residual_layer_norm only support weight.dim() ==1 and bias.dim()==1!"
        )
    if normalized_shape == None:
        norm_size = weight.size(-1)
        normalized_shape = [norm_size]
    else:
        if (
            isinstance(normalized_shape, list) or isinstance(normalized_shape, tuple)
        ) and len(normalized_shape) == 1:
            norm_size = normalized_shape[0]
        else:
            raise ValueError(
                f"residual_layer_norm(): argument 'normalized_shape' (position 2) must be tuple of ints and length of tuple is equal to 1, not {type(normalized_shape)}"
            )

    if norm_size != weight.size(-1):
        raise ValueError(
            f"residual_layer_norm(): argument 'norm_size'  must == weight.size(-1)"
        )
    dtype = input.dtype
    if residual_bias is None:
        x = input.float() + residual.float() * alpha
    else:
        x = input.float() + residual.float() * alpha + residual_bias.float()

    y = torch.nn.functional.layer_norm(
        x.to(dtype), normalized_shape, weight, bias, eps=eps
    )

    if is_post_ln:
        return y, y
    else:
        return y, x.to(dtype)


def residual_layer_norm_bias_alpha(
    input: torch.Tensor,
    normalized_shape: List[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    residual_bias: torch.Tensor = None,
    alpha: float = 1.0,
    eps: float = 1e-5,
    is_post_ln=False,
):
    """
    等价实现：
        residual = input + residual.float() * alpha + residual_bias
        output = torch.nn.functional.layer_norm(
            residual, normalized_shape, weight, bias, eps=eps
        )
        residual = output if is_post_ln else residual

    Args:
        input:              (..., hidden_size)    torch.float16, torch.bfloat16, torch.float32
        normalized_shape                          list[int]
        weight:             (hidden_size)         torch.float16, torch.bfloat16, torch.float32
        bias:               (hidden_size)         torch.float16, torch.bfloat16, torch.float32
        residual:           (..., hidden_size)    torch.float16, torch.bfloat16, torch.float32
        residual_bias:      (hidden_size)         torch.float16, torch.bfloat16, torch.float32
        alpha:                                    float32
        eps:                                      float32
        is_post_ln:                               bool
    Returns:
        input:              (..., hidden_size)    torch.float16, torch.bfloat16, torch.float32
        residual:           (..., hidden_size)    torch.float16, torch.bfloat16, torch.float32
                            Inplace operation will be performed on residual and input.
    """
    if (
        weight is None
        or bias is None
        or residual is None
        or weight.dim() > 1
        or bias.dim() > 1
    ):
        raise NotImplementedError(
            "residual_layer_norm only support weight.dim() ==1 and bias.dim()==1!"
        )
    if normalized_shape == None:
        norm_size = weight.size(-1)
        normalized_shape = [norm_size]
    else:
        if (
            isinstance(normalized_shape, list) or isinstance(normalized_shape, tuple)
        ) and len(normalized_shape) == 1:
            norm_size = normalized_shape[0]
        else:
            raise ValueError(
                f"residual_layer_norm(): argument 'normalized_shape' (position 2) must be tuple of ints and length of tuple is equal to 1, not {type(normalized_shape)}"
            )

    if norm_size != weight.size(-1):
        raise ValueError(
            f"residual_layer_norm(): argument 'norm_size'  must == weight.size(-1)"
        )

    ops.infer.residual_layer_norm(
        input, residual, weight, bias, residual_bias, None, None, alpha, eps, is_post_ln
    )

    return input, residual


def ref_layer_norm_2sb_fused(
    input: torch.Tensor,
    normalized_shape: List[int],
    weight1: torch.Tensor,
    bias1: torch.Tensor,
    weight2: torch.Tensor,
    bias2: torch.Tensor,
    eps: float = 1e-5,
):
    assert input.shape[-1] <= 16384
    if not (input.dtype == torch.float16 or input.dtype == torch.bfloat16):
        raise NotImplementedError(
            "layer_norm_2sb() only support data format of float16 or bfloat16 now!"
        )
    if (
        weight1 is None
        or bias1 is None
        or weight2 is None
        or bias2 is None
        or weight1.dim() > 1
        or bias1.dim() > 1
        or weight2.dim() > 1
        or bias2.dim() > 1
    ):
        raise NotImplementedError(
            "layer_norm_2sb only support weight1.dim() ==1, bias1.dim()==1, weight2.dim() ==1 and  bias2.dim()==1 !"
        )
    if normalized_shape == None:
        norm_size = weight1.size(-1)
        normalized_shape = [norm_size]
    else:
        if (
            isinstance(normalized_shape, list) or isinstance(normalized_shape, tuple)
        ) and len(normalized_shape) == 1:
            norm_size = normalized_shape[0]
        else:
            raise ValueError(
                f"layer_norm_2sb(): argument 'normalized_shape' (position 2) must be tuple of ints and length of tuple is equal to 1, not {type(normalized_shape)}"
            )

    if norm_size != weight1.size(-1) or norm_size != weight2.size(-1):
        raise ValueError(
            f"layer_norm_2sb(): argument 'norm_size'  must == weight.size(-1)"
        )

    output1 = torch.nn.functional.layer_norm(
        input, normalized_shape, weight1, bias1, eps=eps
    )

    output2 = torch.nn.functional.layer_norm(
        input, normalized_shape, weight2, bias2, eps=eps
    )

    return output1, output2


def layer_norm_2sb_fused(
    input: torch.Tensor,
    normalized_shape: List[int],
    weight1: torch.Tensor,
    bias1: torch.Tensor,
    weight2: torch.Tensor,
    bias2: torch.Tensor,
    eps: float = 1e-5,
):
    """
    等价实现：
        output1 = torch.nn.functional.layer_norm(
            input, normalized_shape, weight1, bias1, eps=eps
        )
        output2 = torch.nn.functional.layer_norm(
            input, normalized_shape, weight2, bias2, eps=eps
        )

    Args:
        input:              (..., hidden_size)    torch.float16, torch.bfloat16
        normalized_shape                          list[int]
        weight1:            (hidden_size)         torch.float16, torch.bfloat16
        bias1:              (hidden_size)         torch.float16, torch.bfloat16
        weight2:            (hidden_size)         torch.float16, torch.bfloat16
        bias2:              (hidden_size)         torch.float16, torch.bfloat16
        eps:                                      float32
    Returns:
        output1:            (..., hidden_size)    torch.float16, torch.bfloat16
        output2:            (..., hidden_size)    torch.float16, torch.bfloat16
    """
    assert input.shape[-1] <= 16384
    if not (input.dtype == torch.float16 or input.dtype == torch.bfloat16):
        raise NotImplementedError(
            "layer_norm_2sb() only support data format of float16 or bfloat16 now!"
        )
    if (
        weight1 is None
        or bias1 is None
        or weight2 is None
        or bias2 is None
        or weight1.dim() > 1
        or bias1.dim() > 1
        or weight2.dim() > 1
        or bias2.dim() > 1
    ):
        raise NotImplementedError(
            "layer_norm_2sb only support weight1.dim() ==1, bias1.dim()==1, weight2.dim() ==1 and  bias2.dim()==1 !"
        )
    if normalized_shape == None:
        norm_size = weight1.size(-1)
        normalized_shape = [norm_size]
    else:
        if (
            isinstance(normalized_shape, list) or isinstance(normalized_shape, tuple)
        ) and len(normalized_shape) == 1:
            norm_size = normalized_shape[0]
        else:
            raise ValueError(
                f"layer_norm_2sb(): argument 'normalized_shape' (position 2) must be tuple of ints and length of tuple is equal to 1, not {type(normalized_shape)}"
            )

    if norm_size != weight1.size(-1) or norm_size != weight2.size(-1):
        raise ValueError(
            f"layer_norm_2sb(): argument 'norm_size'  must == weight.size(-1)"
        )
    output1 = torch.empty_like(input)
    output2 = torch.empty_like(input)
    ops.infer.layer_norm_2sb(
        input, weight1, bias1, weight2, bias2, eps, output1, output2
    )

    return output1, output2
