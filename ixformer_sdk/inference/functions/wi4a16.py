import ixformer._C as ops
import torch

__all__ = ["wi4a16_gemm", "wi4a16_gemv", "wi4a16", "ref_wi4a16"]


def dequant_weight(tensor, scales, zeros, block_size):
    # from CPM
    """
    tensor: (oc/2, ic)
    scales: (oc, ic/group_size)
    zeros:  (oc, ic/group_size)
    """
    dtype = scales.dtype
    left = tensor >> 4
    right = tensor << 4 >> 4
    left, right = right, left
    ret = torch.cat((left, right), dim=-1).reshape(-1, left.size(-1))
    ret_shape = ret.size()
    ret = ret.view(-1, block_size)
    ret = scales.view(-1, 1) * (ret - zeros.view(-1, 1))
    ret = ret.reshape(ret_shape).to(dtype=dtype)
    return ret


def ref_wi4a16(
    inputs: "torch.Tensor",
    qweights: "torch.Tensor",
    scales: "torch.Tensor",
    zeros: "torch.Tensor",
    group_size: int = -1,
    format: str = "TN",
):
    assert format in ["TN"]
    weights = dequant_weight(
        qweights,
        scales.transpose(0, 1).contiguous(),
        zeros.transpose(0, 1).contiguous(),
        group_size,
    )
    output = torch.nn.functional.linear(inputs, weights.to(inputs.dtype))
    return output


def wi4a16_gemm(
    inputs: "torch.Tensor",
    qweights: "torch.Tensor",
    scales: "torch.Tensor",
    zeros: "torch.Tensor",
    group_size: int = -1,
    format: str = "TN",
    output=None,
):
    """
    wi4a16 gemm 接口
    支持条件:
    format = TN
    group_size = 128
    input    : fp16      (bs, ic)
    qweights : int8           (oc/2, ic)
    scales   : fp16      (ic/group_size, oc)
    zeros    : fp16      (ic/group_size, oc)
    TN 支持条件： oc % 2 == 0 && ic % 128 == 0
    NN 支持条件： 不支持
    """

    assert format in ["TN"]
    assert len(qweights.shape) == 2
    assert len(scales.shape) == 2
    assert len(zeros.shape) == 2

    input_shape = list(inputs.shape)
    inputs = inputs.view(-1, input_shape[-1])

    if output is None:
        output_shape = input_shape[:-1] + [scales.shape[1]]
        output = inputs.new_empty(output_shape).view(-1, output_shape[-1])
    else:
        output_shape = output.shape

    ops.infer.wi4a16_gemm(output, inputs, qweights, scales, zeros, group_size, format)
    return output.view(output_shape)


def wi4a16_gemv(
    inputs: "torch.Tensor",
    qweights: "torch.Tensor",
    scales: "torch.Tensor",
    zeros: "torch.Tensor",
    group_size: int = -1,
    format: str = "TN",
    output=None,
):
    """
    wi4a16 gemv 接口
    支持条件:
    format = TN
    group_size = 128
    input    : bf16|fp16      (bs, ic)
    qweights : int8           (oc/2, ic)
    scales   : bf16|fp16      (ic/group_size, oc)
    zeros    : bf16|fp16      (ic/group_size, oc)
    TN 支持条件： oc % 2 == 0 && ic % 128 == 0
    NN 支持条件： 不支持
    """

    assert format in ["TN"]
    assert len(qweights.shape) == 2
    assert len(scales.shape) == 2
    assert len(zeros.shape) == 2

    input_shape = list(inputs.shape)
    inputs = inputs.view(-1, input_shape[-1])
    
    if output is None:
        output_shape = input_shape[:-1] + [scales.shape[1]]
        output = inputs.new_empty(output_shape).view(-1, output_shape[-1])
    else:
        output_shape = output.shape

    ops.infer.wi4a16_gemv(output, inputs, qweights, scales, zeros, group_size, format)
    return output.view(output_shape)


def wi4a16(
    inputs: "torch.Tensor",
    qweights: "torch.Tensor",
    scales: "torch.Tensor",
    zeros: "torch.Tensor",
    group_size: int = -1,
    format: str = "TN",
    output=None,
):
    input_shape = inputs.shape
    inputs = inputs.view(-1, input_shape[-1])
    bs = inputs.size(0)
    inputs = inputs.view(input_shape)
    if bs <= 1:
        return wi4a16_gemv(
            inputs=inputs,
            qweights=qweights,
            scales=scales,
            zeros=zeros,
            group_size=group_size,
            format=format,
            output=output,
        )
    else:
        return wi4a16_gemm(
            inputs=inputs,
            qweights=qweights,
            scales=scales,
            zeros=zeros,
            group_size=group_size,
            format=format,
            output=output,
        )
