from typing import List, Tuple, Union

import ixformer._C as ops
import torch
from torch.autograd.function import Function, FunctionCtx

import ixformer

__all__ = [
    "group_norm",
    "ref_group_norm",
    "ref_fused_group_norm_silu",
    "fused_group_norm_silu",
    "ref_fused_group_norm_silu_nhwc",
    "fused_group_norm_silu_nhwc"
]
def is_channels_last(ten):
    return torch._prims_common.suggest_memory_format(ten) == torch.channels_last

def ref_group_norm(input, num_groups, weight, bias, eps):
    output = torch.nn.functional.group_norm(input, num_groups, weight, bias, eps)
    return output

#group_norm官方接口，如果input是nhwc(channel_last),输出则不是channel_last,而是nchw；如果input是nchw，那么输出也是nchw
def group_norm(
    input: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-05, 
    
):
    """
    Args:
        input:              (n,c,h,w) or (n,c,h) or (n,h,w,c)                       torch.float16
                 "contiguous_format":(n,c,h,w) or (n,c,h) "channels_last": (n,h,w,c)
        num_groups:                                                                 int
        weight:             (c)                                                     torch.float16
        bias:               (c)                                                     torch.float16
        eps:                                                                        float
    Returns:
        Tensor:             (n,c,h,w)                                               torch.float16
    """   

    is_nhwc=is_channels_last(input)    
    out = ops.infer.groupnorm(input, num_groups, weight, bias, eps, is_nhwc, 0)
    if is_nhwc:
        out=out.permute(0,3,1,2).contiguous()
    return out
def ref_fused_group_norm_silu_nhwc(
    input: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-05,
    act_type:int = 0
):
    output = torch.nn.functional.group_norm(input.permute(0,3,1,2).contiguous(), num_groups, weight, bias, eps)    
    if act_type:
        output = output * torch.sigmoid(output)
    output = output.permute(0,2,3,1).contiguous()
    return output
#为了减少permute/contiguous,新接口支持输入输出都是nhwc的，融合silu
def fused_group_norm_silu_nhwc(
    input: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-05, 
    act_type:int = 0    
): 
    """
    Args:
        input:              (n,h,w,c)                                               torch.float16
        num_groups:                                                                 int
        weight:             (c)                                                     torch.float16
        bias:               (c)                                                     torch.float16
        eps:                                                                        float
        act_type:                                                                   int
                0 or 1,if act_type=1, silu  
    Returns:
        Tensor:             (n,h,w,c)                                               torch.float16
        
    """    
    out = ops.infer.groupnorm(input, num_groups, weight, bias, eps, True, act_type)    
    return out

def ref_fused_group_norm_silu(
    input: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-05,
):
    output = torch.nn.functional.group_norm(input, num_groups, weight, bias, eps)
    output = output * torch.sigmoid(output)
    return output


def fused_group_norm_silu(
    input: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-05,
):
    
    """
    Args:
        input:              (n,c,h,w) or (n,c,h)                                    torch.float16
        num_groups:                                                                 int
        weight:             (c)                                                     torch.float16
        bias:               (c)                                                     torch.float16
        eps:                                                                        float
    Returns:
        output:             (n,c,h,w) or (n,c,h)                                    torch.float16
    """ 

    out = ops.infer.groupnorm(input, num_groups, weight, bias, eps, False, 1)
    return out
