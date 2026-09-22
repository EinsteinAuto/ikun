import ixformer._C as ops
import torch
from torch.nn import init
from torch.nn.parameter import Parameter


class GN_NHWC_Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, G: int, eps: float, activation: str):
        X_out, means, rstds = ops.train.gn_nhwc_fwd(X, weight, bias, G, eps, activation)
        ctx.save_for_backward(X, weight, bias, means, rstds)
        ctx.G = G
        ctx.activation = activation
        return X_out

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        dy = dy.contiguous(memory_format=torch.channels_last)
        X, weight, bias, means, rstds = ctx.saved_tensors 
        dx, dgamma, dbeta = ops.train.gn_nhwc_bwd(dy, X, weight, bias, means, rstds, ctx.G, ctx.activation)
        return dx, dgamma, dbeta, None, None, None
    

class GroupNorm_nhwc(torch.nn.GroupNorm):
    def __init__(self, num_groups: int, nc: int, activation='identity', **kwargs):
        super().__init__(num_groups, nc, **kwargs)
        assert activation in {'identity', 'silu', 'relu', 'gelu', 'gelu_tanh'}
        if activation == 'identity':
            self.activation = 0
        if activation == 'relu':
            self.activation = 1
        if activation == 'silu':
            self.activation = 2
        if activation == 'gelu':
            self.activation = 3
        if activation == 'gelu_tanh':
            self.activation = 4

    @torch._dynamo.disable
    def forward(self, x):
        #print(x.shape, self.num_channels)
        if len(x.size()) == 3:
            N, C, L = x.shape
        elif len(x.size()) == 4:
            N, C, H, W = x.shape
        else:
            raise ValueError
        G = self.num_groups

        #if C // G > 512:
        #    raise ValueError(f'Error in fwd for X.shape={x.shape}, G={G}: C // G = {C // G} which is greater than 512. This input is not supported.')

        #if H * W % 8 != 0:
        #    raise ValueError(f'Error in fwd for X.shape={x.shape}, G={G}: H * W is not a multiple of 8. This input is not supported.')

        if self.affine:
            return GN_NHWC_Func.apply(x, self.weight, self.bias, self.num_groups, self.eps, self.activation)
        else:
            w = torch.ones((self.num_channels,), device=x.device, dtype=x.dtype)
            b = torch.zeros((self.num_channels,), device=x.device, dtype=x.dtype)
            return GN_NHWC_Func.apply(x, w, b, self.num_groups, self.eps, self.activation)
