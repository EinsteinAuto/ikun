import ixformer._C as ops
import torch

__all__ = [
    "marlin_w4a16",
    "marlin_w4_weight_repack",
    "marlin_w8a16",
    "marlin_w8_weight_repack",
]


def marlin_w4a16(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    bias: torch.Tensor = None,  # TODO
    group_size: int = -1,
    format: str = "k16n32",
    batch_first: bool = True,
    outputs: torch.Tensor = None,
):
    """
    Args:
        inputs:     (batch, m, k) if batch_first else (m, batch, k)         torch.float16, torch.bfloat16
        weights:    (batch, k/16, n/32, 64)                                 torch.int32
        scales:
                    (batch, k_groups, n)    format:k16n32                   torch.float16, torch.bfloat16
                    (batch, n_groups, k)    format:k16n32_grouped_n         torch.float16, torch.bfloat16
        zeros:
                    (batch, k_groups, n/8)  format:k16n32                   torch.int32
                    (batch, n_groups, k/8)  format:k16n32_grouped_n         torch.int32
        group_size:                                                         int
                    group size of quant
        format:                                                             str
                    describe format of weight
        batch_first:                                                        bool
                    describe format of input and output
    Returns:
        outputs:    (batch, m, n) if batch_first else (m, batch, n)         torch.float16, torch.bfloat16
    """
    if outputs is None:
        batch, m = (
            (inputs.shape[0], inputs.shape[1])
            if batch_first
            else (inputs.shape[1], inputs.shape[0])
        )
        if format.startswith("k16n32"):
            n = weights.shape[2] * 32
        outputs = torch.empty(
            (batch, m, n) if batch_first else (m, batch, n),
            dtype=inputs.dtype,
            device=inputs.device,
        )

    ops.infer.marlin_w4a16(
        outputs, inputs, weights, scales, zeros, bias, group_size, format, batch_first
    )
    return outputs


def marlin_w4_weight_repack(
    weights: torch.Tensor,
    scales: torch.Tensor = None,
    zeros: torch.Tensor = None,
    weight_format: str = "gptq",
    reformat: str = "k16n32",
    pack_order: str = "default",
    repack_weight: torch.Tensor = None,
):
    """
    Args:
        weights:
                        (batch, k, n/8)         weight_format:awq               torch.int32
                        (batch, k/8, n)         weight_format:gptq              torch.int32
        scales:
                        (batch, k_groups, n)    format:k16n32                   torch.float16, torch.bfloat16
                        (batch, n_groups, k)    format:k16n32_grouped_n         torch.float16, torch.bfloat16
        zeros:
                        (batch, k_groups, n/8)  format:k16n32                   torch.int32
                        (batch, n_groups, k/8)  format:k16n32_grouped_n         torch.int32
        weight_format:                                                          str
                        describe format of weight
        reformat:                                                               str
                        describe format of repacked weight
        pack_order:                                                             str
                        describe pack order on a pack unit
    Returns:
        repack_weight:  (batch, k/16, n/32, 64)                                 torch.int32
    """
    assert weight_format in ["gptq", "gptq_grouped_n", "awq"]
    assert reformat in ["k16n32", "k16n32_grouped_n"]
    assert pack_order in ["default", "02461357", "01234567"]

    if pack_order == "default":
        default_order = {
            "gptq": "01234567",
            "awq": "02461357",
            "gptq_grouped_n": "02461357",
        }
        pack_order = default_order[weight_format]

    if weight_format.startswith("gptq"):
        batch, pack_k, n = weights.shape
        k = pack_k * 8
    elif weight_format == "awq":
        batch, k, pack_n = weights.shape
        n = pack_n * 8

    repack_scales, repack_zeros = None, None
    if reformat.startswith("k16n32"):
        if repack_weight is None:
            repack_weight = torch.empty(
                (batch, k // 16, n // 32, 64),
                dtype=torch.int32,
                device=weights.device,
            )
        if scales is not None:
            repack_scales = torch.empty_like(scales)
        if zeros is not None:
            repack_zeros = torch.empty_like(zeros)

    ops.infer.marlin_w4_weight_repack(
        weights,
        repack_weight,
        scales,
        repack_scales,
        zeros,
        repack_zeros,
        weight_format,
        reformat,
        pack_order,
    )

    if repack_scales is not None and repack_zeros is not None:
        return repack_weight, repack_scales, repack_zeros
    else:
        return repack_weight


def marlin_w8a16(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor = None,  # TODO
    group_size: int = -1,
    format: str = "k16n16",
    batch_first: bool = True,
    outputs: torch.Tensor = None,
):
    """
    Args:
        inputs:     (batch, m, k) if batch_first else (m, batch, k)         torch.float16, torch.bfloat16
        weights:    (batch, k/16, n/16, 64)                                 torch.int32
        scales:
                    (batch, k_groups, n)    format:k16n16                   torch.float32
                    (batch, n_groups, k)    format:k16n16_grouped_n         torch.float32
        group_size:                                                         int
                    group size of quant
        format:                                                             str
                    describe format of weight
        batch_first:                                                        bool
                    describe format of input and output
    Returns:
        outputs:    (batch, m, n) if batch_first else (m, batch, n)         torch.float16, torch.bfloat16
    """
    if outputs is None:
        batch, m = (
            (inputs.shape[0], inputs.shape[1])
            if batch_first
            else (inputs.shape[1], inputs.shape[0])
        )
        if format.startswith("k16n16"):
            n = weights.shape[2] * 16
        outputs = torch.empty(
            (batch, m, n) if batch_first else (m, batch, n),
            dtype=inputs.dtype,
            device=inputs.device,
        )

    ops.infer.marlin_w8a16(
        outputs, inputs, weights, scales, bias, group_size, format, batch_first
    )
    return outputs


def marlin_w8_weight_repack(
    weights: torch.Tensor,
    scales: torch.Tensor = None,
    weight_format: str = "int8",
    reformat: str = "k16n16",
):
    """
    Args:
        weights:
                                (batch, k, n)           weight_format:int8              torch.int8
        scales:
                                (batch, k_groups, n)    format:k16n16                   torch.float32
                                (batch, n_groups, k)    format:k16n16_grouped_n         torch.float32
        weight_format:                                                                  str
                                describe format of weight
        reformat:                                                                       str
                                describe format of repacked weight
    Returns:
        repack_weight:
                                (batch, k/16, n/16, 64)                                 torch.int32
    """
    assert weight_format in ["int8"]
    assert reformat in ["k16n16", "k16n16_grouped_n"]

    repack_scales = None
    if weight_format == "int8":
        batch, k, n = weights.shape
        repack_weight = torch.empty(
            (batch, k // 16, n // 16, 64),
            dtype=torch.int32,
            device=weights.device,
        )
        if scales is not None:
            repack_scales = torch.empty_like(scales)

    ops.infer.marlin_w8_weight_repack(
        weights,
        repack_weight,
        scales,
        repack_scales,
        weight_format,
        reformat,
    )

    if repack_scales is not None:
        return repack_weight, repack_scales
    else:
        return repack_weight
