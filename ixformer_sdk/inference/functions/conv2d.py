from typing import Union

import ixformer._C as ops
import torch
import torch.nn.functional as NNF

__all__ = ["conv2d", "ref_conv2d", "ref_conv2d_nhwc", "conv2d_nhwc"]


def is_channels_last(ten):
    return torch._prims_common.suggest_memory_format(ten) == torch.channels_last


def _pair(x):
    if isinstance(x, (list, tuple)):
        return x
    return (x, x)


def ref_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: Union[int, tuple] = 1,
    padding: Union[int, tuple] = 0,
    dilation: Union[int, tuple] = 1,
    groups: int = 1,
):

    output = NNF.conv2d(input, weight, bias, stride, padding, dilation, groups)
    return output


# conv2d官方接口，如果weight是torch.channels_last,输出也是torch.channels_last；如果weight是nchw，那么输出也是nchw;特殊情况，如果输入是nchw，weight是torch.channels_last，输出也是torch.channels_last
def conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: Union[int, tuple] = 1,
    padding: Union[int, tuple] = 0,
    dilation: Union[int, tuple] = 1,
    groups: int = 1,
):
    """
    Args:
        input:              (n,in_c,h,w)                        torch.float16
        weight:             (out_c,in_c/groups,kH,kW)           torch.float16
        bias:               (out_c)                             torch.float16
        stride:                                                 int or tuple
                Stride of the convolution. Default: 1
        padding:                                                int or tuple
                Padding added to all four sides of the input. Default: 0
        dilation:                                               int or tuple
                Spacing between kernel elements. Default: 1
        groups:                                                 int
                Number of blocked connections from input channels to output channels. Default: 1
    Returns:
        Tensor:             (n,out_c,h_out,w_out)               torch.float16
                h_out = (h_in + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
                w_out = (w_in + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;      
    """
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    channel_last = is_channels_last(weight)
    if not is_channels_last(input) and channel_last:
        input = input.to(memory_format=torch.channels_last)

    # compute outshape
    n, in_c, h_in, w_in = input.shape
    out_c, _, kernel_h, kernel_w = weight.shape
    pad_h = padding[0]
    pad_w = padding[1]
    stride_h = stride[0]
    stride_w = stride[1]
    dilation_h = dilation[0]
    dilation_w = dilation[1]
    h_out = (h_in + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    w_out = (w_in + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

    if channel_last:
        output_shape = [n, out_c, h_out, w_out]
        output = torch.empty(
            output_shape,
            memory_format=torch.channels_last,
            dtype=input.dtype,
            device=input.device,
        )
    else:
        output_shape = [n, out_c, h_out, w_out]
        output = input.new_empty(output_shape)

    if channel_last:
        input = input.permute(0, 2, 3, 1)
        weight = weight.permute(0, 2, 3, 1)
        output = output.permute(0, 2, 3, 1)

    if bias is not None:
        bias = bias.float()
    ops.infer.conv2d(
        input, weight, bias, output, stride, padding, dilation, groups, channel_last
    )
    if channel_last:
        output = output.permute(0, 3, 1, 2)

    return output


def ref_conv2d_nhwc(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: Union[int, tuple] = 1,
    padding: Union[int, tuple] = 0,
    dilation: Union[int, tuple] = 1,
    groups: int = 1,
):

    output = NNF.conv2d(
        input.permute(0, 3, 1, 2).contiguous(),
        weight.permute(0, 3, 1, 2).contiguous(),
        bias,
        stride,
        padding,
        dilation,
        groups,
    )
    return output.permute(0, 2, 3, 1).contiguous()


# conv2d_nhwc，
# conv2d官方接口解决两种情况：
# 1、务必输入tensor内存上是nhwc,且tensor属于memory_format=torch.channels_last，
# 2、或者输入tensor内存上是nchw，并且是contiguous；
# conv2d官方接口不能解决，conv2d_nhwc则可处理这种情况的
# 输入tensor内存上是nhwc的，但tensor没有用memory_format=torch.channels_last进行过处理，不会有memory_format=torch.channels_last的标签
def conv2d_nhwc(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: Union[int, tuple] = 1,
    padding: Union[int, tuple] = 0,
    dilation: Union[int, tuple] = 1,
    groups: int = 1,
):
    
    """
    Args:
        input:              (n,h,w,in_c)                        torch.float16
        weight:             (out_c,kH,kW,in_c/groups)           torch.float16
        bias:               (out_c)                             torch.float16
        stride:                                                 int or tuple
                Stride of the convolution. Default: 1
        padding:                                                int or tuple
                Padding added to all four sides of the input. Default: 0
        dilation:                                               int or tuple
                Spacing between kernel elements. Default: 1
        groups:                                                 int
                Number of blocked connections from input channels to output channels. Default: 1
    Returns:
        Tensor:             (n,h_out,w_out,out_c)               torch.float16
                h_out = (h_in + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
                w_out = (w_in + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;      
    """

    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    assert input.is_contiguous()
    assert weight.is_contiguous()

    # compute outshape
    n, h_in, w_in, in_c = input.shape
    (
        out_c,
        kernel_h,
        kernel_w,
        _,
    ) = weight.shape
    pad_h = padding[0]
    pad_w = padding[1]
    stride_h = stride[0]
    stride_w = stride[1]
    dilation_h = dilation[0]
    dilation_w = dilation[1]
    h_out = (h_in + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    w_out = (w_in + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

    output_shape = [n, h_out, w_out, out_c]
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    if bias is not None:
        bias = bias.float()
    ops.infer.conv2d(
        input, weight, bias, output, stride, padding, dilation, groups, True
    )
    return output
