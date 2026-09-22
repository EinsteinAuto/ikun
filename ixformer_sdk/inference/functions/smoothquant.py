import ixformer._C as ops
import torch
import torch.nn.functional as NNF

__all__ = [
    "ref_dynamic_scaled_quant_dynamic_int8",
    "dynamic_scaled_quant_dynamic_int8",
    "dynamic_scaled_quant_smoothquant",
    "ref_silu_and_mul_smoothquant",
    "silu_and_mul_smoothquant",
    "ref_residual_rms_norm_dynamic_int8",
    "residual_rms_norm_dynamic_int8",
    "ref_residual_layer_norm_dynamic_int8",
    "residual_layer_norm_dynamic_int8",
    "ref_layer_norm_2sb_smoothquant",
    "layer_norm_2sb_smoothquant",
    "ref_residual_layer_norm_2sb_smoothquant",
    "residual_layer_norm_2sb_smoothquant",
]


def ref_dynamic_scaled_quant_dynamic_int8(
    input: torch.Tensor,
    smooth_scales: torch.Tensor = None,
    i8_output: torch.Tensor = None,
    output_scales: torch.Tensor = None,
):
    if i8_output is None:
        i8_output = torch.empty(input.shape, dtype=torch.int8, device=input.device)
    if output_scales is None:
        output_scales = torch.empty(
            input.shape[:-1], dtype=torch.float32, device=input.device
        )

    scales_shape = input.shape[:-1]
    output = input.float()
    if smooth_scales is not None:
        output *= smooth_scales.view(1, -1)
    amax_, _ = torch.max(torch.abs(output), dim=-1, keepdim=True)
    scales = amax_ / 127.0
    output = output / scales
    output = torch.clamp(torch.round(output), -127, 127).to(torch.int8)

    if i8_output is not None:
        i8_output.copy_(output)
        output = i8_output
    if output_scales is not None:
        output_scales.view(-1).copy_(scales.view(-1))
        scales = output_scales

    return output, scales.view(scales_shape)


def dynamic_scaled_quant_dynamic_int8(
    input: torch.Tensor,
    smooth_scales: torch.Tensor = None,
    i8_output: torch.Tensor = None,
    output_scales: torch.Tensor = None,
):
    """
    Args:
        input:           (..., k)   torch.float16,torch.bfloat16
        smooth_scales:   (k)        torch.float16,torch.bfloat16
                         if smooth_scales is None, api is dynamic-per-token quantization.
    Returns:
        i8_output:       (..., k)   torch.int8
        output_scales:   (...)      torch.float32
    """
    if i8_output is None:
        i8_output = torch.empty(input.shape, dtype=torch.int8, device=input.device)
    if output_scales is None:
        output_scales = torch.empty(
            input.shape[:-1], dtype=torch.float32, device=input.device
        )
    hidden_size = input.shape[-1]

    if smooth_scales is None:
        ops.infer.scaled_int8_quant(i8_output, input, output_scales, 1)
        return i8_output, output_scales

    ops.infer.dynamic_scaled_quant_smoothquant(
        input.view(-1, hidden_size),
        smooth_scales,
        i8_output.view(-1, hidden_size),
        output_scales,
    )

    return i8_output, output_scales


# For backward compatibility
dynamic_scaled_quant_smoothquant = dynamic_scaled_quant_dynamic_int8


def ref_silu_and_mul_smoothquant(
    input, smooth_scales, i8_output=None, output_scales=None
):
    x1, x2 = input.chunk(chunks=2, dim=-1)
    x = NNF.silu(x1) * x2

    return ref_dynamic_scaled_quant_dynamic_int8(
        x, smooth_scales, i8_output, output_scales
    )


def silu_and_mul_smoothquant(input, smooth_scales, i8_output=None, output_scales=None):
    """
    Args:
        input:           (..., 2*k)   torch.float16,torch.bfloat16
        smooth_scales:   (k)          torch.float16,torch.bfloat16
                         if smooth_scales is None, api is dynamic-per-token quantization.
    Returns:
        i8_output:       (..., k)   torch.int8
        output_scales:   (...)      torch.float32
    """
    if i8_output is None:
        output_shape = input.shape[:-1] + (input.shape[-1] // 2,)
        i8_output = torch.empty(output_shape, dtype=torch.int8, device=input.device)
    if output_scales is None:
        output_scales = torch.empty(
            input.shape[:-1], dtype=torch.float32, device=input.device
        )

    ops.infer.silu_and_mul_smoothquant(i8_output, input, smooth_scales, output_scales)

    return i8_output, output_scales


def ref_residual_rms_norm_dynamic_int8(
    input: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor = None,
    residual_bias: torch.Tensor = None,
    eps: float = 1e-5,
    smooth_scales: torch.Tensor = None,
    output: torch.Tensor = None,
    residual_output: torch.Tensor = None,
    output_scales: torch.Tensor = None,
    is_post: bool = False,
):
    dtype = input.dtype
    if output is None:
        output = torch.empty(input.shape, dtype=torch.int8, device=input.device)
    if output_scales is None:
        output_scales = torch.empty(
            input.shape[:-1], dtype=torch.float32, device=input.device
        )
    if residual_bias is not None:
        input = input + residual_bias

    if residual is not None:
        residual_output = torch.add(input, residual, out=residual_output)
        input = residual_output
    input = input.float()
    weight = weight.float()
    rms_output = input * torch.rsqrt(input.pow(2).mean(-1, keepdim=True) + eps)
    rms_output = (rms_output * weight).to(dtype)

    if residual is not None and is_post:
        residual_output.copy_(rms_output)
    output, output_scales = ref_dynamic_scaled_quant_dynamic_int8(
        rms_output, smooth_scales, output, output_scales.view(-1)
    )

    return output, residual_output, output_scales


def residual_rms_norm_dynamic_int8(
    input: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor = None,
    residual_bias: torch.Tensor = None,
    eps: float = 1e-5,
    smooth_scales: torch.Tensor = None,
    output: torch.Tensor = None,
    residual_output: torch.Tensor = None,
    output_scales: torch.Tensor = None,
    is_post: bool = False,
):
    """
    Args:
        input:              (..., hidden_size)    torch.float16, torch.bfloat16, torch.float32
        weight:             (hidden_size)         torch.float16, torch.bfloat16, torch.float32
        residual:           (..., hidden_size)    torch.float16, torch.bfloat16, torch.float32
        residual_bias:      (hidden_size)         torch.float16, torch.bfloat16, torch.float32
        eps:                                      float32
        smooth_scales:      (hidden_size)         torch.float16, torch.bfloat16, torch.float32
        is_post:                                  bool
    Returns:
        output:             (..., hidden_size)    torch.float16, torch.bfloat16, torch.float32
        residual_output:    (..., hidden_size)    torch.float16, torch.bfloat16, torch.float32 If set to None, an inplace operation will be performed on residual.
        output_scales:      (...)                 torch.float16, torch.bfloat16, torch.float32
    """
    if output is None:
        output = torch.empty(input.shape, dtype=torch.int8, device=input.device)
    if output_scales is None:
        output_scales = torch.empty(
            input.shape[:-1], dtype=torch.float32, device=input.device
        )

    if residual is None:
        ops.infer.rmsnorm_dynamic_int8(
            input, weight, output, output_scales, smooth_scales, residual_bias, eps
        )
    else:
        ops.infer.residual_rmsnorm_dynamic_int8(
            input,
            residual,
            weight,
            output,
            output_scales,
            smooth_scales,
            residual_output,
            residual_bias,
            eps,
            is_post,
        )
        residual_output = residual if residual_output is None else residual_output

    return output, residual_output, output_scales


def ref_residual_layer_norm_dynamic_int8(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor = None,
    residual_bias: torch.Tensor = None,
    eps: float = 1e-5,
    smooth_scales: torch.Tensor = None,
    output: torch.Tensor = None,
    residual_output: torch.Tensor = None,
    output_scales: torch.Tensor = None,
):
    normalized_shape = [weight.size(-1)]

    if output is None:
        output = torch.empty(input.shape, dtype=torch.int8, device=input.device)
    if output_scales is None:
        output_scales = torch.empty(
            input.shape[:-1], dtype=torch.float32, device=input.device
        )

    if residual_bias is not None:
        input = input + residual_bias

    if residual is not None:
        residual_output = torch.add(input, residual, out=residual_output)
        input = residual_output

    norm_output = torch.nn.functional.layer_norm(
        input, normalized_shape, weight, bias, eps=eps
    )

    output, output_scales = ref_dynamic_scaled_quant_dynamic_int8(
        norm_output, smooth_scales, output, output_scales.view(-1)
    )

    return output, residual_output, output_scales


def residual_layer_norm_dynamic_int8(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor = None,
    residual_bias: torch.Tensor = None,
    eps: float = 1e-5,
    smooth_scales: torch.Tensor = None,
    output: torch.Tensor = None,
    residual_output: torch.Tensor = None,
    output_scales: torch.Tensor = None,
):
    """
    Args:
        input:              (..., hidden_size)    torch.float16, torch.bfloat16, torch.float32
        weight:             (hidden_size)         torch.float16, torch.bfloat16, torch.float32
        bias:               (hidden_size)         torch.float16, torch.bfloat16, torch.float32
        residual:           (..., hidden_size)    torch.float16, torch.bfloat16, torch.float32
        residual_bias:      (hidden_size)         torch.float16, torch.bfloat16, torch.float32
        eps:                                      float32
        smooth_scales:      (hidden_size)         torch.float16, torch.bfloat16, torch.float32
    Returns:
        output:             (..., hidden_size)    torch.float16, torch.bfloat16, torch.float32
        residual_output:    (..., hidden_size)    torch.float16, torch.bfloat16, torch.float32 If set to None, an inplace operation will be performed on residual.
        output_scales:      (...)                 torch.float16, torch.bfloat16, torch.float32
    """
    if output is None:
        output = torch.empty(input.shape, dtype=torch.int8, device=input.device)
    if output_scales is None:
        output_scales = torch.empty(
            input.shape[:-1], dtype=torch.float32, device=input.device
        )

    if residual is None:
        ops.infer.layer_norm_dynamic_int8(
            input,
            weight,
            bias,
            output,
            output_scales,
            smooth_scales,
            residual_bias,
            eps,
        )
    else:
        ops.infer.residual_layer_norm_dynamic_int8(
            input,
            residual,
            weight,
            bias,
            output,
            output_scales,
            smooth_scales,
            residual_output,
            residual_bias,
            eps,
        )
        residual_output = residual_output if residual_output is not None else residual

    return output, residual_output, output_scales


def ref_layer_norm_2sb_smoothquant(
    input,
    weight1,
    bias1,
    smooth_scales1,
    weight2,
    bias2,
    smooth_scales2,
    i8_output1=None,
    output_scales1=None,
    i8_output2=None,
    output_scales2=None,
    eps=1e-5,
):
    input1 = torch.nn.functional.layer_norm(
        input, [weight1.shape[-1]], weight1, bias1, eps=eps
    )
    input2 = torch.nn.functional.layer_norm(
        input, [weight2.shape[-1]], weight2, bias2, eps=eps
    )

    i8_output1, output_scales1 = ref_dynamic_scaled_quant_dynamic_int8(
        input1, smooth_scales1, i8_output1, output_scales1
    )
    i8_output2, output_scales2 = ref_dynamic_scaled_quant_dynamic_int8(
        input2, smooth_scales2, i8_output2, output_scales2
    )

    return i8_output1, output_scales1, i8_output2, output_scales2


def layer_norm_2sb_smoothquant(
    input,
    weight1,
    bias1,
    smooth_scales1,
    weight2,
    bias2,
    smooth_scales2,
    output1=None,
    output_scales1=None,
    output2=None,
    output_scales2=None,
    eps=1e-5,
):
    """
    Args:
        input:              (..., hidden_size)    torch.float16, torch.bfloat16
        weight1:            (hidden_size)         torch.float16, torch.bfloat16
        bias1:              (hidden_size)         torch.float16, torch.bfloat16
        smooth_scales1:     (hidden_size)         torch.float16, torch.bfloat16
        weight2:            (hidden_size)         torch.float16, torch.bfloat16
        bias2:              (hidden_size)         torch.float16, torch.bfloat16
        smooth_scales2:     (hidden_size)         torch.float16, torch.bfloat16
        eps:                                      float32
    Returns:
        output1:            (..., hidden_size)    torch.float16, torch.bfloat16
        output_scales1:     (...)                 torch.float16, torch.bfloat16
        output2:            (..., hidden_size)    torch.float16, torch.bfloat16
        output_scales2:     (...)                 torch.float16, torch.bfloat16
    """
    if output1 is None:
        output1 = torch.empty(input.shape, dtype=torch.int8, device=input.device)
    if output_scales1 is None:
        output_scales1 = torch.empty(
            input.shape[:-1], dtype=torch.float32, device=input.device
        )
    if output2 is None:
        output2 = torch.empty(input.shape, dtype=torch.int8, device=input.device)
    if output_scales2 is None:
        output_scales2 = torch.empty(
            input.shape[:-1], dtype=torch.float32, device=input.device
        )

    ops.infer.layer_norm_2sb_smoothquant(
        input,
        weight1,
        bias1,
        smooth_scales1,
        weight2,
        bias2,
        smooth_scales2,
        output1,
        output_scales1,
        output2,
        output_scales2,
        eps,
    )

    return output1, output_scales1, output2, output_scales2


def ref_residual_layer_norm_2sb_smoothquant(
    input,
    residual,
    weight1,
    bias1,
    smooth_scales1,
    weight2,
    bias2,
    smooth_scales2,
    i8_output1=None,
    output_scales1=None,
    i8_output2=None,
    output_scales2=None,
    eps=1e-5,
):
    residual_out = input + residual
    input1 = torch.nn.functional.layer_norm(
        residual_out, [weight1.shape[-1]], weight1, bias1, eps=eps
    )
    input2 = torch.nn.functional.layer_norm(
        residual_out, [weight2.shape[-1]], weight2, bias2, eps=eps
    )

    i8_output1, output_scales1 = ref_dynamic_scaled_quant_dynamic_int8(
        input1, smooth_scales1, i8_output1, output_scales1
    )
    i8_output2, output_scales2 = ref_dynamic_scaled_quant_dynamic_int8(
        input2, smooth_scales2, i8_output2, output_scales2
    )

    return residual_out, i8_output1, output_scales1, i8_output2, output_scales2


def residual_layer_norm_2sb_smoothquant(
    input,
    residual,
    weight1,
    bias1,
    smooth_scales1,
    weight2,
    bias2,
    smooth_scales2,
    output1=None,
    output_scales1=None,
    output2=None,
    output_scales2=None,
    eps=1e-5,
):
    """
    Args:
        input:              (..., hidden_size)    torch.float16, torch.bfloat16
        residual:           (..., hidden_size)    torch.float16, torch.bfloat16
        weight1:            (hidden_size)         torch.float16, torch.bfloat16
        bias1:              (hidden_size)         torch.float16, torch.bfloat16
        smooth_scales1:     (hidden_size)         torch.float16, torch.bfloat16
        weight2:            (hidden_size)         torch.float16, torch.bfloat16
        bias2:              (hidden_size)         torch.float16, torch.bfloat16
        smooth_scales2:     (hidden_size)         torch.float16, torch.bfloat16
        eps:                                      float32
    Returns:
        output1:            (..., hidden_size)    torch.float16, torch.bfloat16
        output_scales1:     (...)                 torch.float16, torch.bfloat16
        output2:            (..., hidden_size)    torch.float16, torch.bfloat16
        output_scales2:     (...)                 torch.float16, torch.bfloat16
    """
    if output1 is None:
        output1 = torch.empty(input.shape, dtype=torch.int8, device=input.device)
    if output_scales1 is None:
        output_scales1 = torch.empty(
            input.shape[:-1], dtype=torch.float32, device=input.device
        )
    if output2 is None:
        output2 = torch.empty(input.shape, dtype=torch.int8, device=input.device)
    if output_scales2 is None:
        output_scales2 = torch.empty(
            input.shape[:-1], dtype=torch.float32, device=input.device
        )

    ops.infer.residual_layer_norm_2sb_smoothquant(
        input,
        residual,
        weight1,
        bias1,
        smooth_scales1,
        weight2,
        bias2,
        smooth_scales2,
        output1,
        output_scales1,
        output2,
        output_scales2,
        eps,
    )

    return residual, output1, output_scales1, output2, output_scales2
