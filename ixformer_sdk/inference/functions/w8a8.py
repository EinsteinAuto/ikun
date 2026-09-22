from typing import Optional, Tuple

import ixformer._C as ops
import torch

__all__ = [
    "w8a8",
    "ref_w8a8",
    "dynamic_scaled_int8_quant",
    "ref_dynamic_scaled_int8_quant",
    "static_scaled_int8_quant",
    "ref_static_scaled_int8_quant",
    "scaled_int8_quant",
]


def ref_w8a8(
    input: "torch.Tensor",
    weight: "torch.Tensor",
    i_scales: "torch.Tensor",
    w_scales: "torch.Tensor",
    output: "torch.Tensor",
    format: str = "TN",
    persistent=0,
    bias: torch.Tensor = None,
):
    dtype = output.dtype
    input_f32 = input.to(torch.float32)
    weight_f32 = weight.to(torch.float32)
    assert format in ["TN", "NN", "NT"]
    if format == "TN":
        weight_f32 = weight_f32.transpose(0, 1)
    if format == "NT":
        input_f32 = input_f32.transpose(0, 1)
    output_f32 = (
        torch.matmul(input_f32, weight_f32)
        * i_scales.view(-1, 1)
        * w_scales.view(1, -1)
    )
    if bias is not None:
        bias_f32 = bias.to(torch.float32)
        output_f32 += bias_f32.view(1, -1)
    output.copy_(output_f32.to(dtype))
    return output


def w8a8_gemm(
    input: "torch.Tensor",
    weight: "torch.Tensor",
    i_scales: "torch.Tensor",
    w_scales: "torch.Tensor",
    bias: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    format: str = "TN",
    persistent: bool = False,
    out_dtype: torch.dtype = None,
):
    """
    Args:
        input:         (n, k)                                   torch.int8
        weight:        (m, k)  if format == "TN" else (k, m)    torch.int8
        i_scales:      (n)                                      torch.float32
        w_scales:      (m)                                      torch.float32
        bias:          (m)                                      torch.float32, same as output_type
        format:                                                 str
                       Options include TN, NN and NT
        persistent:    Whether to use overleap                  bool
        out_dtype:                                              torch.float16, torch.bfloat16
    Returns:
        output:        (n, m)                                   torch.float16, torch.bfloat16
    """

    input_shape = input.shape

    if output is None:
        if out_dtype is None:
            raise RuntimeError("w8a8 gemm need out_dtype argument when output is none.")
        output = torch.empty(
            (input_shape[:-1] + (weight.shape[0],)),
            dtype=out_dtype,
            device=input.device,
        )

    output_shape = output.shape

    input = input.view(-1, input_shape[-1])
    output = output.view(-1, output_shape[-1])

    ops.infer.w8a8_gemm(
        output, input, weight, i_scales, w_scales, bias, format, int(persistent)
    )

    return output.view(*output_shape)


def ref_static_scaled_int8_quant(output, input, scale):
    """
    Args:
        output:          [torch.int8]                 [m, k]
        input:           [torch.half,torch.bfloat16]  [m, k]
        scale:           [torch.float32]              [1]
    Returns:
        output:          [torch.int8]                 [m, k]
        scale:           [torch.float32]              [1]
    """
    # [m, 1]
    f_input = input / scale.to(input.dtype)
    i_output = torch.clamp(torch.round(f_input), -127, 127).to(torch.int8)
    output.copy_(i_output)
    return output, scale


# for vllm: https://github.com/vllm-project/vllm/blob/v0.5.4/vllm/_custom_ops.py#L387
def static_scaled_int8_quant(output, input, scale):
    """
    Args:
        output:          [torch.int8]                 [m, k]
        input:           [torch.half,torch.bfloat16]  [m, k]
        scale:           [torch.float32]              [1]
    Returns:
        output:          [torch.int8]                 [m, k]
        scale:           [torch.float32]              [1]
    """
    ops.infer.scaled_int8_quant(output, input, scale, 0)
    return output, scale


def ref_dynamic_scaled_int8_quant(output, input, scale):
    """
    Args:
        output:          [torch.int8]                 [m, k]
        input:           [torch.half,torch.bfloat16]  [m, k]
        scale:           [torch.float32]              [m]
    Returns:
        output:          [torch.int8]                 [m, k]
        scale:           [torch.float32]              [m]
    """
    # [m, 1]
    amax_, _ = torch.max(torch.abs(input), dim=-1, keepdim=True)
    f_scale = amax_.float() / 127.0
    scale.view(-1).copy_(f_scale.view(-1))

    f_input = input / f_scale.to(input.dtype)
    i_output = torch.clamp(torch.round(f_input), -127, 127).to(torch.int8)
    output.copy_(i_output)
    return output, scale.view(input.shape[:-1])


# for vllm: https://github.com/vllm-project/vllm/blob/v0.5.4/vllm/_custom_ops.py#L394
def dynamic_scaled_int8_quant(output, input, scale):
    """
    Args:
        output:          [torch.int8]                 [m, k]
        input:           [torch.half,torch.bfloat16]  [m, k]
        scale:           [torch.float32]              [m]
    Returns:
        output:          [torch.int8]                 [m, k]
        scale:           [torch.float32]              [m]
    """
    ops.infer.scaled_int8_quant(output, input, scale, 1)
    return output, scale


def scaled_int8_quant(
    input: torch.Tensor, scale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize the input tensor to int8 and return the quantized tensor and scale.

    Args:
        input: The input tensor to be quantized to int8.
        scale: Optional scaling factor for the int8 quantization.
            When not provided, we invoke dynamic-per-token quantization.

    Returns:
      Tuple[Torch.Tensor, Torch.Tensor] : Output int8 tensor and scales.
    """
    output = torch.empty_like(input, dtype=torch.int8)
    if scale is not None:
        # static-per-tensor quantization.
        static_scaled_int8_quant(output, input, scale)
        return output, scale

    # dynamic-per-token quantization.
    input_scales = torch.empty(
        (input.numel() // input.shape[-1], 1), device=input.device, dtype=torch.float32
    )
    dynamic_scaled_int8_quant(output, input, input_scales)
    return output, input_scales


def w8a8_gemv(
    input: "torch.Tensor",
    weight: "torch.Tensor",
    i_scales: "torch.Tensor",
    w_scales: "torch.Tensor",
    bias: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    format: str = "TN",
    persistent: bool = False,
    out_dtype: torch.dtype = None,
):
    """
    Args:
        input:         (n, k)                                   torch.int8
        weight:        (m, k)  if format == "TN" else (k, m)    torch.int8
        i_scales:      (n)                                      torch.float32
        w_scales:      (m)                                      torch.float32
        bias:          (m)                                      torch.float32 same as output_type
        format:                                                 str
                       Options include TN and NN
        persistent:    Whether to use overleap                  bool
        out_dtype:                                              torch.float16, torch.bfloat16
    Returns:
        output:        (n, m)                                   torch.float16, torch.bfloat16
    """

    input_shape = input.shape

    if output is None:
        if out_dtype is None:
            raise RuntimeError("w8a8 gemv need out_dtype argument when output is none.")
        output = torch.empty(
            (input_shape[:-1] + (weight.shape[0],)),
            dtype=out_dtype,
            device=input.device,
        )

    input = input.view(-1, input_shape[-1])

    ops.infer.w8a8_gemv(
        output, input, weight, i_scales, w_scales, bias, format, int(persistent)
    )

    return output


def handle_pading(weight: torch.Tensor, format: str, is_gemm: bool):
    """Handle padding alignment for weight matrices
    Args:
        weight: Original weight matrix [m, k]
        format: Matrix format, TN indicates transposed layout
        is_gemm: Whether for GEMM operation (requires extra alignment checks)
    Returns:
        torch.Tensor: Padded weight matrix
    Raises:
        AssertionError: When is_gemm=True requires 4-byte alignment for m/k
    """
    # weight should have been pad before w8a8 is called, handle _padding here just ensure the code run success,
    # but performance is low, please refer to vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py
    m, k = weight.shape
    s = weight.stride(0)
    if s % 64 != 0 and format == "TN":
        pad_k = (s // 64 + 1) * 64
        weight_pad = torch.empty((m, pad_k), dtype=weight.dtype, device=weight.device)
        _weight = weight_pad[:, :k]
        if is_gemm:
            assert m % 4 == 0 and k % 4 == 0
        _weight.copy_(weight)
        return _weight
    else:
        return weight


def w8a8(
    input: torch.Tensor,
    weight: torch.Tensor,
    i_scales: torch.Tensor,
    w_scales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    format: str = "TN",
    persistent: bool = False,
    out_dtype: torch.dtype = None,
):
    """
    Args:
        input:         (n, k)                                   torch.int8
        weight:        (m, k)  if format == "TN" else (k, m)    torch.int8
        i_scales:      (n)                                      torch.float32
        w_scales:      (m)                                      torch.float32
        bias:          (m)                                      torch.float32, same as output_type
        format:                                                 str
                       Options include TN and NN
        persistent:    Whether to use overleap                  bool
        out_dtype:                                              torch.float16, torch.bfloat16
    Returns:
        output:        (n, m)                                   torch.float16, torch.bfloat16
    """
    bs = input.numel() // input.shape[-1]
    gemv_condition = (format == "TN" and bs <= 1) or (format == "NN" and bs <= 16)
    if gemv_condition:
        weight = handle_pading(weight, format, is_gemm=False)
        return w8a8_gemv(
            input,
            weight,
            i_scales,
            w_scales,
            bias=bias,
            output=output,
            format=format,
            persistent=persistent,
            out_dtype=out_dtype,
        )
    else:
        weight = handle_pading(weight, format, is_gemm=True)
        return w8a8_gemm(
            input,
            weight,
            i_scales,
            w_scales,
            bias=bias,
            output=output,
            format=format,
            persistent=persistent,
            out_dtype=out_dtype,
        )
