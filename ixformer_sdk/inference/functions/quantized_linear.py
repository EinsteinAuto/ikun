from typing import Union

import ixformer._C as ops
import torch
from torch import Tensor

__all__ = [
    "quantized_linear",
    "quantized_weight_dequant",
    "ref_quantized_weight_dequant",
    "weight_quantize",
]


def quantized_linear(
    inputs: torch.Tensor,
    qweights: torch.Tensor,
    scales: torch.Tensor,
    quant_type: str,
    bits: int,
    qzeros: torch.Tensor = None,
    bias: torch.Tensor = None,
    group_size: int = -1,
    g_idx: torch.Tensor = None,
    format: str = "unknown",
):
    """
    QuantType       inputs                    qweights                                              Scales                           bits           qzeros                                       bias            GroupSize     Format                        ApiCall                  备注
      awq        (bs, ic) bf16/fp16     int32 NN:(ic, oc // 8) TN:(oc, ic // 8)                  (ic // group_size, oc)fp16/bf16      4/8       int32(ic // group_size, oc // 8)    (oc) or None fp16/bf16        32/128       TN/NN                        vllm & auto-awq
      gptq       (bs, ic) bf16/fp16     int32  (ic//8, oc)                                       (ic // group_size, oc)fp16/bf16      4         int32(ic // group_size, oc // 8)    (oc) or None fp16/bf16        ic/128        \                             auto-gptq            bs 只支持到8
      fp4        (bs, ic) bf16/fp16     uint8 (oc * ic // 2, 1)                                  (oc * ic // group_size)fp32          4          \                                  (oc) or None fp16/bf16        64            \                           bitsandbytes           bs 只支持到8
      nf4        (bs, ic) bf16/fp16     uint8 (oc * ic // 2, 1)                                  (oc * ic // group_size)fp32          4          \                                  (oc) or None fp16/bf16        64            \                           bitsandbytes           bs 只支持到8
      int8       (bs, ic) bf16/fp16     int8  TN:(oc, ic) NN:(ic, oc)                            (1, oc)fp16/bf16                     8          \                                  (oc) or None fp16/bf16        -1           TN/NN                        vllm & bitsandbytes

    """
    if isinstance(inputs, torch.Tensor) and not inputs.requires_grad:
        return ops.infer.quantized_linear(
            inputs,
            qweights,
            scales,
            quant_type,
            bits,
            qzeros,
            bias,
            group_size,
            g_idx,
            format,
        )
    raise NotImplementedError()


def quantized_weight_dequant(
    qweights: torch.Tensor,
    scales: torch.Tensor,
    quant_type: str,
    output_type: str,
    bits: int,
    qzeros: torch.Tensor = None,
    group_size: int = -1,
    g_idx: torch.Tensor = None,
):
    """
    Args:
        qweights:           (oc, ic//2) or (ic// (32/bits, oc)   torch.unint8 or torch.int32
        scales:             (oc * ic//g) or (ic // g, oc)        torch.float16, torch.bfloat16, torch.float32
        quant_type:                                              str
                    可选项:fp4/nf4/gptq/gptq-ex
        output_type:                                             str
                    可选项:fp16/bf16
        bits:                                                    int
                    可选项:4/8
        qzeros:             (ic//g, oc//(32/bits))               torch.int32
        group_size:                                              int
                    可选项:-1/64/128
        g_idx:              (ic)                                 torch.int

    Returns:
        Tensor:             (oc, ic) or (ic, oc)                 torch.float16, torch.bfloat16

    quant_type            qweights                             scales                    qzeros                               output_type         bits           group_size    g_idx
     fp4/nf4          (oc, ic//2) uint8                   (oc * ic//g)  fp32                                                  fp16/bf16            /               64/128       /
     gptq/gptq-ex     (ic// (32/bits, oc) int32          (ic // g, oc) fp16/bf16     (ic // g, oc // (32/bits)) int32         fp16/bf16           4/8                -1         (ic)

    """
    if isinstance(qweights, torch.Tensor) and not qweights.requires_grad:
        return ops.infer.quantized_weight_dequant(
            qweights, scales, quant_type, output_type, bits, qzeros, group_size, g_idx
        )
    raise NotImplementedError()


def ref_quantized_weight_dequant(
    qweights: torch.Tensor,
    scales: torch.Tensor,
    quant_type: str,
    output_type: torch.dtype,
    bits: int,
    qzeros: torch.Tensor = None,
    group_size: int = -1,
    g_idx: torch.Tensor = None,
    order_map: list = None,
):
    assert quant_type in ["awq"]
    if quant_type == "awq":
        # qweights:(k, n/8)           int32
        # scale:(k/group_size, n)   f16
        # qzeros:(k/group_size, n/8) int32
        ic, oc = qweights.shape[0], scales.shape[1]
        assert bits == 4
        if order_map is None:
            order_map = [0, 2, 4, 6, 1, 3, 5, 7]
        order_map = torch.Tensor(order_map).to(torch.int32).to(qweights.device)
        order_map = order_map.argsort()

        # (1, 8)
        wf = (
            torch.tensor(list(range(0, 32, bits)), dtype=torch.int32)
            .unsqueeze(0)
            .to(qweights.device)
        )

        # unpack qzeros
        unpack_zeros = torch.bitwise_right_shift(
            torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits), wf.unsqueeze(0)
        ).to(torch.int16 if bits == 8 else torch.int8)
        unpack_zeros = unpack_zeros[:, :, order_map]
        unpack_zeros = torch.bitwise_and(unpack_zeros, (2**bits) - 1)
        # groups, 1, n
        unpack_zeros = unpack_zeros.reshape(unpack_zeros.shape[0], 1, -1)

        # unpack weights
        unpack_weights = torch.bitwise_right_shift(
            torch.unsqueeze(qweights, 2).expand(-1, -1, 32 // bits),
            wf.unsqueeze(0),
        ).to(torch.int16 if bits == 8 else torch.int8)
        unpack_weights = unpack_weights[:, :, order_map]
        unpack_weights = torch.bitwise_and(unpack_weights, (2**bits) - 1)
        # w : groups, group_size, n
        unpack_weights = unpack_weights.reshape(
            -1, group_size, unpack_weights.shape[1] * unpack_weights.shape[2]
        )

        deq_weights = (unpack_weights - unpack_zeros) * scales.reshape(
            -1, 1, scales.shape[-1]
        )
        deq_weights = deq_weights.reshape(ic, oc)
        return deq_weights.to(output_type)


def create_dynamic_map(signed=True, max_exponent_bits=7, total_bits=8):
    """
    Creates the dynamic quantiztion map.

    The dynamic data type is made up of a dynamic exponent and
    fraction. As the exponent increase from 0 to -7 the number
    of bits available for the fraction shrinks.

    This is a generalization of the dynamic type where a certain
    number of the bits and be reserved for the linear quantization
    region (the fraction). n determines the maximum number of
    exponent bits.

    For more details see
    (8-Bit Approximations for Parallelism in Deep Learning)[https://arxiv.org/abs/1511.04561]
    """

    data = []
    # these are additional items that come from the case
    # where all the exponent bits are zero and no
    # indicator bit is present
    non_sign_bits = total_bits - (1 if signed else 1)
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
    for i in range(max_exponent_bits):
        fraction_items = int(
            2 ** (i + non_sign_bits - max_exponent_bits) + 1
            if signed
            else 2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1,
        )
        boundaries = torch.linspace(0.1, 1, fraction_items)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    if additional_items > 0:
        boundaries = torch.linspace(0.1, 1, additional_items + 1)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    data.append(0)
    data.append(1.0)

    assert len(data) == 2**total_bits

    gap = 256 - len(data)
    for i in range(gap):
        data.append(0)

    data.sort()
    return Tensor(data)


def weight_quantize(
    A: torch.Tensor,
    absmax: torch.Tensor,
    out: torch.Tensor,
    blocksize: int,
    n: int,
    quant_dtype: str,
    code: torch.Tensor = None,
):
    """
    Args:
        A:                  (row ,col)                                                              torch.float16, torch.bfloat16, torch.float32
        absmax:             (blocks)                                                                torch.float32
                    blocks = n // blocksize, blocks += 1 if n % blocksize > 0 else 0
        quant_dtype:                                                                                str
                    目前可支持"int8"/"fp4"/"nf4"
        blocksize:                                                                                  int
                    目前只支持4096, 2048, 1024, 512, 256, 128, 64
        n:                                                                                          int
                    n = A.numel()
        out:                (row ,col)                                                              torch.int8
        code:                                                                                       torch.float32
                    the quantization map
    Returns:
        out:                (row ,col)                                                              torch.int8

    """
    assert quant_dtype == "int8" or "fp4" or "nf4"
    if code is None and quant_dtype == "int8":
        code = create_dynamic_map().to(A.device)
    if isinstance(A, torch.Tensor) and not A.requires_grad:
        ops.infer.weight_quantize(A, absmax, out, blocksize, n, quant_dtype, code)
    else:
        raise NotImplementedError()
