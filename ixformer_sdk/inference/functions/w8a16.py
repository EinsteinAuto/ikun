import math
from typing import List, Union, Optional

import ixformer._C as ops
import torch
from torch.autograd.function import Function, FunctionCtx

import ixformer
from ixformer.core import config

__all__ = [
    "w8a16_gemm",
    "w8a16_gemv",
    "w8a16",
    "ref_w8a16",
    "wu8a16",
    "ref_wu8a16",
]


def w8a16_gemv(
    inputs: "torch.Tensor",
    qweights: "torch.Tensor",
    scales: "torch.Tensor",
    group_size: int = -1,
    format: str = "unknown",
    output: Optional[torch.Tensor] = None
):
    """
    w8a16 gemv 接口
    input    : bf16|fp16      (bs, ic)
    qweights : int8           TN:(oc, ic) NN:(ic, oc)
    scales   : bf16|fp16      TN: 当groupsize为-1时, shape: (1, oc), 否则,shape: (ic/group_size, oc) NN:(1, oc)
    TN 支持条件： ic % groupSize = 0, oc % 2 = 0, bs<=4
    NN 支持条件： groupsize = -1 or groupsize = ic, oc % 4 = 0, bs<=4
    """
    assert format in ["TN", "NN"]
    assert len(qweights.shape) == 2
    assert len(scales.shape) == 2

    input_shape = list(inputs.shape)
    inputs = inputs.view(-1, input_shape[-1])

    if format == "TN":
        output_shape = input_shape[:-1] + [qweights.shape[0]]
    else:
        output_shape = input_shape[:-1] + [qweights.shape[1]]

    if output is None:
        output = inputs.new_empty(output_shape).view(-1, output_shape[-1])

    ops.infer.w8a16_gemv(output, inputs, qweights, scales, group_size, format)
    return output.view(output_shape)


def w8a16_gemm(
    inputs: "torch.Tensor",
    qweights: "torch.Tensor",
    scales: "torch.Tensor",
    group_size: int = -1,
    format: str = "TN",
    persistent: int = 0,
    output: Optional[torch.Tensor] = None
):
    """
    w8a16 gemm 接口
    1. group_size=-1 or group_size=ic
    input    : bf16|fp16      (bs, ic)
    qweights : int8           TN:(oc, ic) NN:(ic, oc)
    scales   : bf16|fp16      (1, oc)
    NN 支持条件： ic%64==0, oc%64==0

    2. group_size=64
    input    : bf16|fp16      (bs, ic)
    qweights : int8           TN:(oc, ic)
    scales   : bf16|fp16      (ic/64, oc)
    TN 支持条件： oc%2==0, ic%64==0
    NN 不支持
    """

    assert format in ["TN", "NN"]
    assert len(qweights.shape) == 2
    assert len(scales.shape) == 2

    input_shape = list(inputs.shape)
    inputs = inputs.view(-1, input_shape[-1])

    if format == "TN":
        output_shape = input_shape[:-1] + [qweights.shape[0]]
    else:
        output_shape = input_shape[:-1] + [qweights.shape[1]]

    if output is None:
        output = inputs.new_empty(output_shape).view(-1, output_shape[-1])

    ops.infer.w8a16_gemm(
        output, inputs, qweights, scales, group_size, format, persistent
    )
    return output.view(output_shape)


def dequant(qweight, scales, group_size):
    IC, OC = qweight.shape
    weight = qweight.t().reshape(OC, -1, group_size).to(
        torch.float32
    ) * scales.t().unsqueeze(-1)
    return weight.reshape(OC, IC)


def ref_w8a16(
    inputs: "torch.Tensor",
    qweights: "torch.Tensor",
    scales: "torch.Tensor",
    group_size: int = -1,
    format: str = "TN",
):
    if group_size == -1:
        group_size = inputs.shape[1]
    if format == "TN":
        weights = dequant(qweights.transpose(0, 1), scales, group_size)
    elif format == "NN":
        weights = dequant(qweights, scales, group_size)
    return torch.nn.functional.linear(inputs, weights.to(inputs.dtype))


def w8a16(
    inputs: "torch.Tensor",
    qweights: "torch.Tensor",
    scales: "torch.Tensor",
    group_size: int = -1,
    format: str = "TN",
    output: Optional[torch.Tensor] = None,
    persistent: int = 0,
):
    input_shape = inputs.shape
    inputs = inputs.view(-1, input_shape[-1])
    bs = inputs.size(0)
    inputs = inputs.view(input_shape)
    if bs <= config.IXFORMER_GEMV_THRESHOLD:
        return w8a16_gemv(
            inputs=inputs,
            qweights=qweights,
            scales=scales,
            group_size=group_size,
            format=format,
            output=output
        )
    else:
        return w8a16_gemm(
            inputs=inputs,
            qweights=qweights,
            scales=scales,
            group_size=group_size,
            format=format,
            output=output,
            persistent=persistent
        )


def ref_wu8a16(
    inputs: "torch.Tensor",
    qweights: "torch.Tensor",
    scales: "torch.Tensor",
    zeros: "torch.Tensor",
    group_size: int = -1,
    format: str = "TN",
):
    assert format in ["TN"]
    assert len(qweights.shape) == 2
    assert len(scales.shape) == 2
    org_w_shape = qweights.shape
    scales = scales.transpose(0, 1).flatten().view(-1, 1)
    zeros = zeros.transpose(0, 1).flatten().view(-1, 1)

    if group_size != -1:
        qweights = qweights.reshape(-1, group_size)
    w = (qweights - zeros) * scales
    w = w.reshape(org_w_shape)
    output = torch.matmul(inputs, w.t())
    return output


def wu8a16(
    inputs: "torch.Tensor",
    qweights: "torch.Tensor",
    scales: "torch.Tensor",
    zeros: "torch.Tensor",
    group_size: int = -1,
    format: str = "TN",
    persistent: int = 0,
):
    """
    http://confluence.iluvatar.ai:8090/display/SW/cuinferCustomGemm+Interface+Doc
    wu8a16 非对称量化 gemm 接口
    1. group_size=-1
    input    : bf16|fp16      (bs, ic)
    qweights : uint8          TN:(oc, ic)
    scales   : bf16|fp16      (1,oc)
    zeros    : bf16|fp16      (1,oc)
    TN 支持条件： ic % 64 == 0
    NN 不支持

    2. group_size=64
    input    : bf16|fp16      (bs, ic)
    qweights : int8           TN:(oc, ic)
    scales   : bf16|fp16      (ic/64, oc)
    zeros    : bf16|fp16      (ic/64, oc)
    TN 支持条件： oc % 2 == 0 && ic % 64 == 0
    NN 不支持
    """
    assert format in ["TN"]
    assert len(qweights.shape) == 2
    assert len(scales.shape) == 2

    input_shape = list(inputs.shape)
    inputs = inputs.view(-1, input_shape[-1])

    output_shape = input_shape[:-1] + [qweights.shape[0]]

    output = inputs.new_empty(output_shape).view(-1, output_shape[-1])

    ops.infer.wu8a16_gemm(
        output, inputs, qweights, scales, zeros, group_size, format, persistent
    )
    return output.view(output_shape)
