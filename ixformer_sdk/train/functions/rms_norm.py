import numbers
from typing import Union

import ixformer._C as ops
import torch
from torch.nn import init
from torch.nn.parameter import Parameter


# apex interface for trainning add by xuelu.peng 2024/04/07
class FusedRMSNormAffineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, normalized_shape, eps, memory_efficient=False, gradient_accumulation_fusion=False):
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        ctx.memory_efficient = memory_efficient
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion

        input_ = input.contiguous()
        weight_ = weight.contiguous()
        output = torch.empty_like(input_)
        normalized_shape_size = len(normalized_shape)
        assert normalized_shape_size == 1  # 目前只支持normalized_shape_size=1
        invvar = torch.empty(
            input_.shape[:-normalized_shape_size],
            dtype=torch.float,
            device=input_.device,
        )
        ops.train.rms_norm_forward_training(input_, weight_, output, invvar, ctx.eps)

        ctx.save_for_backward(input_, weight_, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight_, invvar = ctx.saved_tensors

        if ctx.gradient_accumulation_fusion:
            if weight_.grad == None:
                weight_.grad = torch.zeros_like(weight_)
            grad_weight = weight_.grad
        else:
            grad_weight = torch.zeros_like(weight_)  # 支持权重梯度累积融合，使用zeros_like，而不是emtpy_like 。

        grad_input = torch.empty_like(input_)

        if input_.numel() < 4096 * 8192:
            ops.train.rms_norm_backward_training(
                input_, invvar, weight_, grad_output, grad_weight, grad_input
            )
        else:  ##llama 34b
            ops.train.rms_norm_backward_training_opt(
                input_, invvar, weight_, grad_output, grad_weight, grad_input
            )
            
        if ctx.gradient_accumulation_fusion:
            grad_weight = None
        return grad_input, grad_weight, None, None, None, None
def fused_rms_norm_affine(
    input, weight, normalized_shape, eps=1e-6, memory_efficient=False, gradient_accumulation_fusion = False
):
    return FusedRMSNormAffineFunction.apply(
        input, weight, normalized_shape, eps, memory_efficient, gradient_accumulation_fusion
    )


class FusedRMSNorm(torch.nn.Module):
    r"""Applies RMS Normalization over a mini-batch of inputs

    Currently only runs on cuda() tensors.

    .. math::
        y = \frac{x}{\mathrm{RMS}[x]} * \gamma

    The root-mean-square is calculated separately over the last
    certain number dimensions which have to be of the shape specified by
    :attr:`normalized_shape`.
    :math:`\gamma` is a learnable affine transform parameter of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    `epsilon` is added to the mean-square, then the root of the sum is taken.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, RMS Normalization applies per-element scale
        with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized}\_\text{shape}[0] \times \text{normalized}\_\text{shape}[1]
                    \times \ldots \times \text{normalized}\_\text{shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 5, 10, 10)
        >>> # With Learnable Parameters
        >>> m = ixformer.FusedRMSNorm(10)
        >>> # Without Learnable Parameters
        >>> m = ixformer.FusedRMSNorm(input.size()[1:], elementwise_affine=False)
        >>> # Normalize over last dimension of size 10 #目前只支持在最后一维norm
        >>> m = ixformer.FusedRMSNorm(10)
        >>> # Activating the module
        >>> output = m(input)

    .. _`Root Mean Square Layer Normalization`: https://arxiv.org/pdf/1910.07467.pdf
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-5,
        elementwise_affine=True,
        memory_efficient=False,
        gradient_accumulation_fusion=False
    ):
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.memory_efficient = memory_efficient
        self.gradient_accumulation_fusion = gradient_accumulation_fusion

        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(*normalized_shape))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)

    def forward(self, input):
        if torch.jit.is_tracing() or torch.jit.is_scripting() or not input.is_cuda:
            raise NotImplementedError()

        if self.elementwise_affine:
            return fused_rms_norm_affine(
                input,
                self.weight,
                self.normalized_shape,
                self.eps,
                self.memory_efficient,
                self.gradient_accumulation_fusion
            )
        else:
            raise NotImplementedError()

    def extra_repr(self):
        return "{normalized_shape}, eps={eps}, " "elementwise_affine={elementwise_affine}".format(**self.__dict__)

class FusedRMSNormResFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, residual, normalized_shape, eps, gradient_accumulation_fusion=False, memory_efficient=False):        
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        ctx.memory_efficient = memory_efficient
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion

        input_ = input.contiguous()
        weight_ = weight.contiguous()
        output = torch.empty_like(input_)
        normalized_shape_size=len(normalized_shape)
        assert normalized_shape_size == 1 #目前只支持normalized_shape_size=1
        invvar = torch.empty(input_.shape[:-normalized_shape_size], dtype=torch.float, device=input_.device)     

        if residual is not None:
            ctx.input_res = True
            out_res = torch.empty_like(input_)
            ops.train.rms_norm_res_forward_training(input_, weight_, output, invvar, ctx.eps, residual, out_res)
        else:
            ctx.input_res = False
            ops.train.rms_norm_forward_training(input_, weight_, output, invvar, ctx.eps)
            out_res = input_

        # input_res 为 True 时 LN 的 input 为 input+redidual
        ctx.save_for_backward(out_res, weight_, invvar)
        return output, out_res

    @staticmethod
    def backward(ctx, grad_output, grad_out_res):
        input_, weight_, invvar = ctx.saved_tensors        

        if ctx.gradient_accumulation_fusion:
            if weight_.grad == None:
                weight_.grad = torch.zeros_like(weight_)
            grad_weight = weight_.grad
        else:
            grad_weight = torch.zeros_like(weight_) # 算子kernel 支持权重梯度累积融合，使用zeros_like，而不是emtpy_like 。

        grad_input = torch.empty_like(input_)

        # rms_norm_res_backward_training 本身支持权重梯度累积融合，当不进行融合时，其输入 grad_weight 必须为 zero_like 。
        if input_.numel()< 4096*8192:
            ops.train.rms_norm_res_backward_training(input_, invvar, weight_,
                        grad_output, grad_weight, grad_input, grad_out_res)
        else:##llama 34b
            ops.train.rms_norm_res_backward_training_opt(input_,invvar, weight_,
                      grad_output,grad_weight,grad_input,grad_out_res)

        if ctx.input_res:
            grad_res = grad_input
        else:
            grad_res = None

        if ctx.gradient_accumulation_fusion:
            grad_weight = None

        return grad_input, grad_weight, grad_res, None, None, None, None
    
class FusedRMSNormRes(torch.nn.Module):
    r"""Applies RMS Normalization and resdiual over a mini-batch of inputs, RMS Normalization part comes from FusedRMSNorm.

    Currently only runs on cuda() tensors.

    .. math::
        y = \frac{x}{\mathrm{RMS}[x]} * \gamma

        if residual None, x is input and output is equal to x, otherwise, x is input+residual and out_res is equal to x.

    The root-mean-square is calculated separately over the last
    certain number dimensions which have to be of the shape specified by
    :attr:`normalized_shape`.
    :math:`\gamma` is a learnable affine transform parameter of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    `epsilon` is added to the mean-square, then the root of the sum is taken.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, RMS Normalization applies per-element scale
        with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized}\_\text{shape}[0] \times \text{normalized}\_\text{shape}[1]
                    \times \ldots \times \text{normalized}\_\text{shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, *)`
        - residual: :math:`(N, *)`   (if not None)
        - Output: :math:`(N, *)` (same shape as input)
        - out_res: :math:`(N, *)` 

    Examples::

        >>> input = torch.randn(20, 5, 10, 10)
        >>> res = torch.randn(20, 5, 10, 10)
        >>> # With Learnable Parameters
        >>> m = ixformer.FusedRMSNorm(10)
        >>> # Without Learnable Parameters
        >>> m = ixformer.FusedRMSNorm(input.size()[1:], elementwise_affine=False)     
        >>> # Normalize over last dimension of size 10 #目前只支持在最后一维norm
        >>> m = ixformer.FusedRMSNorm(10)
        >>> # Activating the module
        >>> output, output_res = m(input, res)

    .. _`Root Mean Square Layer Normalization`: https://arxiv.org/pdf/1910.07467.pdf
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, memory_efficient=False, gradient_accumulation_fusion=False):
        super().__init__()      

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.gradient_accumulation_fusion = gradient_accumulation_fusion
        self.memory_efficient = memory_efficient
        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(*normalized_shape))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)

    def forward(self, input, residual=None):
        if torch.jit.is_tracing() or torch.jit.is_scripting() or not input.is_cuda:
            raise NotImplementedError()

        if self.elementwise_affine:
            return FusedRMSNormResFunction.apply(input, self.weight, residual, self.normalized_shape, self.eps, self.gradient_accumulation_fusion, self.memory_efficient)
        else:
            raise NotImplementedError()

    def extra_repr(self):
        return "{normalized_shape}, eps={eps}, " "elementwise_affine={elementwise_affine}".format(**self.__dict__)
