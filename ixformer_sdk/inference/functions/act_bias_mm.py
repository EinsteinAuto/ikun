import ixformer._C as ops
import torch
import torch.nn.functional as NNF

__all__ = ["act_bias_mm", "ref_act_bias_mm"]


def ref_act_bias_mm(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    bias: torch.Tensor = None,
    scale: float = 1,
    act_type: str = "none",
    trans_format: str = "NN",
):
    assert len(mat1.shape) >= 2
    assert len(mat2.shape) >= 2
    if trans_format == "NN":
        if bias is not None:
            output = torch.matmul(mat1, mat2) * scale + bias
        else:
            output = torch.matmul(mat1, mat2) * scale
    else:
        if bias is not None:
            output = torch.matmul(mat1, mat2.transpose(-1, -2)) * scale + bias
        else:
            output = torch.matmul(mat1, mat2.transpose(-1, -2)) * scale
    if act_type == "gelu":
        output = NNF.gelu(output)
    elif act_type == "relu":
        output = NNF.relu(output)
    elif act_type == "silu":
        output = NNF.silu(output)
    elif act_type == "none":
        output = output
    else:
        raise NotImplementedError()
    return output


def act_bias_mm(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    bias: torch.Tensor = None,
    output: torch.Tensor = None,
    scale: float = 1,
    act_type: str = "none",
    trans_format: str = "NN",
):
    """
    Args:
        mat1:               [m,k] or [batch_count,m,k]          torch.float16
        mat2:               [k,n] or [n,k]                      torch.float16
                当trans_format为"NN"时[k,n], 当trans_format为"TN"时[n,k]
        bias:               [n]                                 torch.float16
        output:             [m,n]                               torch.float16
        scale:                                                  float
        act_type:           silu/gelu/relu/None                 str
                如果act_type不为None,则bias也不可以为None
        trans_format:       NN or TN                            str
    Returns:
        output:             [m,n]                               torch.float16
    """
    assert len(mat1.shape) >= 2
    assert len(mat2.shape) >= 2
    if output is None:
        output_shape = list(mat1.shape)
        m = mat1.shape[-2]
        if trans_format == "NN":
            n = mat2.shape[-1]
        else:
            n = mat2.shape[-2]
        output_shape[-2] = m
        output_shape[-1] = n
        output = mat1.new_empty(output_shape)

    add_bias = False
    if bias is not None:
        add_bias = True

    if add_bias:
        ops.infer.act_bias_mm(
            mat1, mat2, bias, output, add_bias, scale, act_type, trans_format
        )
    else:
        ops.infer.act_bias_mm(
            mat1, mat2, mat1, output, add_bias, scale, act_type, trans_format
        )
    return output
