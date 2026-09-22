import ixformer._C as ops
import torch
import torch.nn.functional

__all__ = [
    "mm",
    "addmm",
    "fused_addmm_bias_col_act",
    "ref_fused_addmm_bias_col_act",
    "ref_addmm",
    "ref_mm",
    "ref_bmm",
    "bmm",
]


def ref_mm(input, mat, *, out=None):
    out = torch.mm(input, mat, out = out)
    return out


def mm(input, mat, *, out=None):
    
    """
    Args:
        input:              (m,k)                   torch.float16, torch.bfloat16, torch.float32
        mat:                (k,n)                   torch.float16, torch.bfloat16, torch.float32
        out:                (m,n)                   torch.float16, torch.bfloat16, torch.float32
    Returns:
        out:                (m,n)                   torch.float16, torch.bfloat16, torch.float32
    """
    assert input.dim() == mat.dim(), "mm tensors must be 2-D"
    assert input.size(1) == mat.size(
        0
    ), f"mm cannot be multiplied, {input.size(0)}X{input.size(1)} and {mat.size(0)}X{mat.size(1)}"

    m = input.shape[0]
    n = mat.shape[-1]
    if out is None:
        out = input.new_empty([m, n])
    ops.infer.mm(input, mat, out)
    return out


"""ixinfer support activations
/// @ingroup GEMM
typedef enum {
  CUINFER_BLAS_GEMM_CUSTOM_NONE = 0,
  CUINFER_BLAS_GEMM_CUSTOM_BIAS_ADD_ROW_OUT = 1,
  CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS = 2,
  CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_GELU = 3,
  CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_RELU = 4,
  CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_TRANSPOSE = 5,
  CUINFER_BLAS_GEMM_CUSTOM_FLOATBIAS = 6,
  CUINFER_BLAS_GEMM_CUSTOM_FLOATBIAS_GELU = 7,
  CUINFER_BLAS_GEMM_CUSTOM_FLOATBIAS_RELU = 8,
  CUINFER_BLAS_GEMM_CUSTOM_FLOATBIAS_TRANSPOSE = 9,
  CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_SIGMOID = 10,
  CUINFER_BLAS_GEMM_CUSTOM_FLOATBIAS_SIGMOID = 11,
  CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_SILU = 12,
  CUINFER_BLAS_GEMM_CUSTOM_FLOATBIAS_SILU = 13,
  CUINFER_BLAS_GEMM_CUSTOM_SIGMOID = 14,
  CUINFER_BLAS_GEMM_CUSTOM_SILU = 15,
  CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_TANH = 16,
  CUINFER_BLAS_GEMM_CUSTOM_FLOATBIAS_TANH = 17,
  CUINFER_BLAS_GEMM_SPECIAL_INT8_FLOATBIAS = 18,
  CUINFER_BLAS_GEMM_SPECIAL_INT8_FLOATBIAS_GELU = 19,
  CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_SWISH = 20,
  CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_ERF_GELU = 21
} cuinferGEMMCustomOption_t;
"""

activation_to_id = {
    "fused_bias_col": 2,  # support bf16, fp16
    "fused_bias_gelu": 3,  # support fp16
    "fused_bias_relu": 4,  # support fp16
}

id_to_activation = {value: key for key, value in activation_to_id.items()}


def ref_addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    output_pt = torch.addmm(input, mat1, mat2, alpha=alpha, beta=beta)
    return output_pt


def ref_fused_addmm_bias_col_act(
    input, mat1, mat2, *, beta=1, alpha=1, out=None, bias=None, activation=2
):
    if isinstance(activation, int):
        assert activation in id_to_activation
    if isinstance(activation, str):
        assert activation in activation_to_id
        activation = activation_to_id[activation]
    if activation == 2:
        output_pt = torch.addmm(input, mat1, mat2, alpha=alpha, beta=beta) + bias
    elif activation == 3:
        output_pt = torch.nn.functional.gelu(
            torch.addmm(input, mat1, mat2, alpha=alpha, beta=beta) + bias
        )
    else:
        output_pt = torch.nn.functional.relu(
            torch.addmm(input, mat1, mat2, alpha=alpha, beta=beta) + bias
        )
    return output_pt


def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    
    """
    Args:
        input:              (m,n)                   torch.float16, torch.bfloat16, torch.float32
        mat1:               (m,k)                   torch.float16, torch.bfloat16, torch.float32
        mat2:               (k,n)                   torch.float16, torch.bfloat16, torch.float32
        beta:                                       float
        alpha:                                      float
        out:                (m,n)                   torch.float16, torch.bfloat16, torch.float32
    Returns:
        out:                (m,n)                   torch.float16, torch.bfloat16, torch.float32
    """
    assert mat1.dim() == mat2.dim(), "addmm mat1 mat2 tensors must be 2-D"
    assert mat1.size(1) == mat2.size(
        0
    ), f"addmm cannot be multiplied, {mat1.size(0)}X{mat1.size(1)} and {mat2.size(0)}X{mat2.size(1)}"

    m = mat1.shape[0]
    n = mat2.shape[-1]

    if input is not None and len(input.shape) == 1:
        input = input.view(1, -1)

    if out is None:
        out = input.new_empty([m, n])
    if input is None:
        input = out
        beta = 0

    ops.infer.addmm(input, mat1, mat2, beta, alpha, out, None, 0)
    return out


def fused_addmm_bias_col_act(
    input, mat1, mat2, *, beta=1, alpha=1, out=None, bias=None, activation=2
):
    """
    Args:
        input:              (m,n)                                   torch.float16, torch.bfloat16, torch.float32
        mat1:               (m,k)                                   torch.float16, torch.bfloat16, torch.float32
        mat2:               (k,n)                                   torch.float16, torch.bfloat16, torch.float32
        beta:                                                       float
        alpha:                                                      float
        out:                (m,n)                                   torch.float16, torch.bfloat16, torch.float32
            当out的shape为(m,n)时,如果out是is_continouns,则bias 必须为(1,n), 否则,bias为(m,1)
        bias:               (1,n) or (m,1)
        activation:                                                 str or int
             "fused_bias_col": 2, "fused_bias_gelu": 3, "fused_bias_relu": 4                               
    Returns:
        out:                (m,n)                                   torch.float16, torch.bfloat16, torch.float32
    """
    assert mat1.dim() == mat2.dim(), "addmm mat1 mat2 tensors must be 2-D"
    assert mat1.size(1) == mat2.size(
        0
    ), f"addmm cannot be multiplied, {mat1.size(0)}X{mat1.size(1)} and {mat2.size(0)}X{mat2.size(1)}"

    if isinstance(activation, int):
        assert activation in id_to_activation
    if isinstance(activation, str):
        assert activation in activation_to_id
        activation = activation_to_id[activation]

    m = mat1.shape[0]
    n = mat2.shape[-1]

    if out is None:
        out = input.new_empty([m, n])
    if input is None:
        input = out
        beta = 0

    # activations
    if activation == 2:
        assert bias is not None
        if mat1.dtype == torch.float:
            ops.infer.addmm(input, mat1, mat2, beta, alpha, out, None, 0)
            out.add_(bias)
            return out
    elif activation == 3:
        assert bias is not None
        if mat1.dtype == torch.float:
            ops.infer.addmm(input, mat1, mat2, beta, alpha, out, None, 0)
            out.add_(bias)
            out.copy_(torch.nn.functional.gelu(out))
            return out
        elif mat1.dtype == torch.bfloat16:
            ops.infer.addmm(input, mat1, mat2, beta, alpha, out, bias, 2)
            out.copy_(torch.nn.functional.gelu(out))
            return out
    elif activation == 4:
        assert bias is not None
        if mat1.dtype == torch.float:
            ops.infer.addmm(input, mat1, mat2, beta, alpha, out, None, 0)
            out.add_(bias)
            out.copy_(torch.nn.functional.relu(out))
            return out
        elif mat1.dtype == torch.bfloat16:
            ops.infer.addmm(input, mat1, mat2, beta, alpha, out, bias, 2)
            out.copy_(torch.nn.functional.relu(out))
            return out

    ops.infer.addmm(input, mat1, mat2, beta, alpha, out, bias, activation)
    return out


def ref_bmm(
    input: torch.Tensor,
    mat2: torch.Tensor,
    alpha: float = 1,
    format: str = "NN",
    input_scales: torch.Tensor = None,
    mat2_scales: torch.Tensor = None,
    out_dtype: torch.dtype = None,
    out: torch.Tensor = None,
):
    if format[1] == "T":
        input = input.transpose(-1, -2)
    if format[0] == "T":
        mat2 = mat2.transpose(-1, -2)

    m = input.size(-2)
    n = mat2.size(-1)

    bs = input.size(0)
    if input.dtype != torch.int8:
        out_dtype = input.dtype
    if out is None:
        out = torch.empty([bs, m, n], dtype=out_dtype, device=input.device)

    if input.dtype != torch.int8:
        torch.bmm(input, mat2, out=out)
        if alpha != 1:
            out = out * alpha
    else:
        input = input.float() * input_scales.view(1, -1, 1)
        mat2 = mat2.float() * mat2_scales.view(1, 1, -1)
        out = torch.bmm(input.float(), mat2.float()) * alpha
        out = out.to(out_dtype)
    return out


def bmm(
    input: torch.Tensor,
    mat2: torch.Tensor,
    alpha: float = 1,
    format: str = "NN",
    input_scales: torch.Tensor = None,
    mat2_scales: torch.Tensor = None,
    out_dtype: torch.dtype = None,
    out: torch.Tensor = None,
):
    """
    out = (input@mat2)*alpha
    Support three formats:
        format: "NN" input shape: (b, m, k)  mat2 shape: (b, k, n) out shape: (b, m, n).
                If the dtype of input is int8, the following conditions need to be met: n%64==0 k%64==0
        format: "TN" input shape: (b, m, k)  mat2 shape: (b, n, k) out shape: (b, m, n)
                If the dtype of input is int8, the following conditions need to be met: n%2==0 k%64==0
        format: "NT" input shape: (b, k, m)  mat2 shape: (b, k, n) out shape: (b, m, n)
                If the dtype of input is int8, the following conditions need to be met: m%64==0 n%64==0 k%64==0
    If the dtype of input is int8, it is necessary to specify out_dtype.
    Args:
        input:              (b, m, k) or (b, k, m)        torch.float16, torch.bfloat16, int8
        mat2:               (b, k, n) or (b, n, k)        torch.float16, torch.bfloat16, int8
        alpha:                                            float32
        format:             TN,NN,NT                      string
        input_scales:       (m)                           torch.float32
        mat2_scales:        (n)                           torch.float32
        out_dtype:          torch.float16, torch.bfloat16
        out:                (b, m, n)                     torch.float16, torch.bfloat16
    Returns:
        out:                (b, m, n)                     torch.float16, torch.bfloat16
    """

    if format[1] == "N":
        m = input.size(-2)
        k = input.size(-1)
    else:
        m = input.size(-1)
        k = input.size(-2)
    if format[0] == "N":
        n = mat2.size(-1)
    else:
        n = mat2.size(-2)

    if input.dtype == torch.int8:
        if format == "TN":
            assert (
                n % 2 == 0 and k % 64 == 0
            ), f"bmm shape error, m={m} n={n} k={k}."
        elif format == "NT":
            assert (
                m % 64 == 0 and n % 64 == 0 and k % 64 == 0
            ), f"bmm shape error, m={m} n={n} k={k}."
        elif format == "NN":
            assert (
                n % 64 == 0 and k % 64 == 0
            ), f"bmm shape error, m={m} n={n} k={k}." 
    bs = input.size(0)
    if out is None:
        if input.dtype != torch.int8:
            out_dtype = input.dtype
        else:
            assert out_dtype is not None
        out = torch.empty([bs, m, n], dtype=out_dtype, device=input.device)
    ops.infer.bmm(input, mat2, input_scales, mat2_scales, alpha, format, out)
    return out
