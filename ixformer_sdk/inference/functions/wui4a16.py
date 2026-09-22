import ixformer._C as ops
import torch

__all__ = ["wui4a16_gemm", "wui4a16_gemv", "wui4a16", "ref_wui4a16"]


def dequant_weight(tensor, scales, zeros, block_size):
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


def ref_wui4a16(
    inputs: "torch.Tensor",
    qweights: "torch.Tensor",
    scales: "torch.Tensor",
    zeros: "torch.Tensor",
    bias: "torch.Tensor" = None,
    group_size: int = -1,
    format: str = "NN",
    only_return_weight: bool = False,
):
    """
    format = TN,NN
    group_size = TN(128),NN(128, 32)
    input    : bfloat16|fp16        (bs, ic)
    qweights : int32                NN: (ic, oc // 8)  TN:(oc, ic // 8)
    scales   : bfloat16|fp16        (ic // group_size, oc)
    zeros    : int32                (ic // group_size, oc // 8)
    bias     : bfloat16|fp16        (oc, )
    output   : bfloat16|fp16        (bs, oc)
    """

    def unpack_tensor(x, pack_num=8, order_map=None):
        if order_map is None:
            order_map = [0, 1, 2, 3, 4, 5, 6, 7]
        unit = 32 // pack_num
        rows, cols = x.shape
        res = torch.zeros((rows, cols * pack_num), dtype=torch.int32, device=x.device)
        for col in range(cols):
            for k in range(pack_num):
                res[:, col * pack_num + order_map[k]] = (x[:, col] >> (unit * k)) & 0xF
        return res

    scales = scales.t().contiguous()
    if format == "NN":
        zeros = unpack_tensor(zeros, order_map=[0, 2, 4, 6, 1, 3, 5, 7])
        zeros = zeros.t().contiguous()
        qweights = unpack_tensor(qweights, order_map=[0, 2, 4, 6, 1, 3, 5, 7])
        qweights = qweights.t().contiguous()
    else:
        zeros = unpack_tensor(zeros)
        zeros = zeros.t().contiguous()
        qweights = unpack_tensor(qweights)
    output_dim, input_dim = qweights.shape
    qweights = qweights.view(output_dim, input_dim // group_size, group_size)
    zeros = zeros.view(output_dim, input_dim // group_size, 1)
    scales = scales.view(output_dim, input_dim // group_size, 1)

    qweights = (qweights - zeros) * scales
    qweights = qweights.view(output_dim, input_dim)
    if only_return_weight:
        return qweights
    output = torch.nn.functional.linear(inputs, qweights.to(inputs.dtype))
    return output, qweights


def wui4a16_gemm(
    inputs: "torch.Tensor",
    qweights: "torch.Tensor",
    scales: "torch.Tensor",
    zeros: "torch.Tensor",
    bias: "torch.Tensor" = None,
    group_size: int = 128,
    format: str = "NN",
):
    output_shape = inputs.shape[:-1] + (scales.shape[1],)

    output = ops.infer.wui4a16_gemm(
        inputs, qweights, scales, zeros, bias, group_size, format
    )
    return output.view(output_shape)


def wui4a16_gemv(
    inputs: "torch.Tensor",
    qweights: "torch.Tensor",
    scales: "torch.Tensor",
    zeros: "torch.Tensor",
    bias: "torch.Tensor" = None,
    group_size: int = 128,
    format: str = "NN",
):
    output_shape = inputs.shape[:-1] + (scales.shape[1],)

    output = ops.infer.wui4a16_gemv(
        inputs, qweights, scales, zeros, bias, group_size, format
    )
    return output.view(output_shape)


def wui4a16(
    inputs: "torch.Tensor",
    qweights: "torch.Tensor",
    scales: "torch.Tensor",
    zeros: "torch.Tensor",
    bias: "torch.Tensor" = None,
    group_size: int = 128,
    format: str = "NN",
):
    """
    format = TN,NN
    group_size = TN(128),NN(128, 32)
    input    : bfloat16|fp16        (bs, ic)
    qweights : int32                NN: (ic, oc // 8)  TN:(oc, ic // 8)
    scales   : bfloat16|fp16        (ic // group_size, oc)
    zeros    : int32                (ic // group_size, oc // 8)
    bias     : bfloat16|fp16        (oc, )
    output   : bfloat16|fp16        (bs, oc)
    支持条件  : NN: oc % 8 == 0 && ic % group_size == 0 && ic % 2 == 0
               TN: oc % 2 == 0 && ic % group_size == 0
    """
    batch = inputs.numel() // inputs.shape[-1]
    if batch <= 1:
        return wui4a16_gemv(
            inputs=inputs,
            qweights=qweights,
            scales=scales,
            zeros=zeros,
            bias=bias,
            group_size=group_size,
            format=format,
        )
    else:
        return wui4a16_gemm(
            inputs=inputs,
            qweights=qweights,
            scales=scales,
            zeros=zeros,
            bias=bias,
            group_size=group_size,
            format=format,
        )
