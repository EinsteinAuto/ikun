import functools
from typing import TYPE_CHECKING, List, Optional, Tuple, Dict, Any

import torch
import torch.library
import torch.nn.functional as F
import math
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType

try:
    import ixformer.inference.functions as ops
except ModuleNotFoundError:
    import ixformer.functions as ops
from ixformer.distributed import _distributed as cdist

import importlib.util
import os

logger = init_logger(__name__)

_log_count = {}
def _log_once(tag, max_n=3):
    _log_count[tag] = _log_count.get(tag, 0) + 1
    if _log_count[tag] <= max_n:
        logger.info("============== change{%s} [%d/%d] ==============", tag, _log_count[tag], max_n)

# 加载 ix_moe_bridge.so
_bridge = None
_here = os.path.dirname(os.path.abspath(__file__))
for _p in [os.path.join(_here, "ix_moe_bridge.so"),
           "/usr/local/corex/lib64/python3/dist-packages/vllm/ix_moe_bridge.so"]:
    if os.path.isfile(_p):
        try:
            _spec = importlib.util.spec_from_file_location("ix_moe_bridge", _p)
            _bridge = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_bridge)
            logger.info("ix_moe_bridge loaded from %s", _p)
            break
        except Exception as _e:
            logger.warning("ix_moe_bridge load failed from %s: %s", _p, _e)
            _bridge = None

# 加载 xllm_cache.so
_xllm_cache = None
for _p in [os.path.join(_here, "xllm_cache.so"),
           "/usr/local/corex/lib64/python3/dist-packages/vllm/xllm_cache.so"]:
    if os.path.isfile(_p):
        try:
            _spec = importlib.util.spec_from_file_location("xllm_cache", _p)
            _xllm_cache = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_xllm_cache)
            logger.info("xllm_cache loaded from %s", _p)
            break
        except Exception as _e:
            logger.warning("xllm_cache load failed from %s: %s", _p, _e)
            _xllm_cache = None

supports_moe_ops = True

def register_fake(fn):
    return lambda name: fn


# activation ops
def silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    if _bridge is not None:
        _log_once("silu_and_mul: bridge path, out.copy_")
        out.copy_(_bridge.silu_and_mul(x))
    else:
        _log_once("silu_and_mul: ixf_F path")
        ops.silu_and_mul(x, out)


def gelu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    ops.gelu_and_mul(x, out)


def gelu_tanh_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    ops.gelu_tanh_and_mul(x, out)


def fatrelu_and_mul(out: torch.Tensor,
                    x: torch.Tensor,
                    threshold: float = 0.0) -> None:
    raise NotImplementedError("FIX soon")


def gelu_fast(out: torch.Tensor, x: torch.Tensor) -> None:
    out.copy_(F.gelu(x,approximate="tanh"))
    return out


def gelu_new(out: torch.Tensor, x: torch.Tensor) -> None:
    out.copy_(F.gelu(x,approximate="tanh"))
    return out


def gelu_quick(out: torch.Tensor, x: torch.Tensor) -> None:
    out.copy_(F.gelu(x,approximate="tanh"))
    return out


# page attention ops
def paged_attention_v1(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    head_mapping: torch.Tensor,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor] = None,
) -> None:
    _log_once("paged_attention_v1: ixf_F vllm_single_query_cached_kv_attention")
    import ixformer.functions as _ixf
    _ixf.vllm_single_query_cached_kv_attention(
        out, query, key_cache, value_cache,
        head_mapping, scale, block_tables, seq_lens,
        block_size, max_seq_len, alibi_slopes)


def paged_attention_v2(
    out: torch.Tensor,
    exp_sum: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> None:
    if hasattr(ops, 'vllm_single_query_cached_kv_attention_v2'):
        _log_once("paged_attention_v2: ixf_F v2 path")
        ops.vllm_single_query_cached_kv_attention_v2(
            out, exp_sum, max_logits, tmp_out, query,
            key_cache, value_cache, num_kv_heads, scale,
            block_tables, seq_lens, block_size, max_seq_len,
            alibi_slopes)
        return
    _log_once("paged_attention_v2: NOT AVAILABLE, raise")
    raise NotImplementedError("paged_attention_v2 not available in this ixformer build")


def paged_attention_rocm(
    out: torch.Tensor,
    exp_sum: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: Optional[torch.Tensor],
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    raise NotImplementedError("Do not use this in our implement")


def mla_decode_kvcache_cpu(
    out: torch.Tensor,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    raise NotImplementedError("Do not use this in our implement")


# pos encoding ops
def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    ops.vllm_rotary_embedding(positions, query, key, head_size,
                                  cos_sin_cache, is_neox)


def batched_rotary_embedding(positions: torch.Tensor, query: torch.Tensor,
                             key: torch.Tensor, head_size: int,
                             cos_sin_cache: torch.Tensor, is_neox: bool,
                             rot_dim: int,
                             cos_sin_cache_offsets: torch.Tensor) -> None:
    ops.vllm_batched_rotary_embedding(positions, query, key, head_size,
                                          cos_sin_cache, is_neox, rot_dim,
                                          cos_sin_cache_offsets)


# layer norm ops
def rms_norm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
             epsilon: float) -> None:
    ops.rms_norm(input, weight, epsilon, out)


def fused_add_rms_norm(input: torch.Tensor, residual: torch.Tensor,
                       weight: torch.Tensor, epsilon: float,
                       residual_alpha: Optional[float] = 1) -> None:
    if _bridge is not None:
        _log_once("fused_add_rms_norm: bridge path")
        if residual_alpha is not None and residual_alpha != 1:
            residual.mul_(residual_alpha)
        _bridge.fused_add_rms_norm(input, residual, weight, epsilon)
    else:
        _log_once("fused_add_rms_norm: ixf_F residual_rms_norm")
        ops.residual_rms_norm(input=input, weight=weight, residual=residual, eps=epsilon, residual_alpha=residual_alpha)


def advance_step_flashattn(num_seqs: int, num_queries: int, block_size: int,
                           input_tokens: torch.Tensor,
                           sampled_token_ids: torch.Tensor,
                           input_positions: torch.Tensor,
                           seq_lens: torch.Tensor, slot_mapping: torch.Tensor,
                           block_tables: torch.Tensor) -> None:
    """Advance a step on GPU for existing inputs for a multi-step runner"""
    return ops.advance_step_flashattn(num_seqs, num_queries,
                                               block_size, input_tokens,
                                               sampled_token_ids,
                                               input_positions, seq_lens,
                                               slot_mapping, block_tables)


def advance_step_flashinfer(num_seqs: int, num_queries: int, block_size: int,
                            input_tokens: torch.Tensor,
                            sampled_token_ids: torch.Tensor,
                            input_positions: torch.Tensor,
                            seq_lens: torch.Tensor, slot_mapping: torch.Tensor,
                            block_tables: torch.Tensor,
                            paged_kv_indices: torch.Tensor,
                            paged_kv_indptr: torch.Tensor,
                            paged_kv_last_page_len: torch.Tensor,
                            block_table_bound: torch.Tensor) -> None:
    raise NotImplementedError("FIX soon")


# fused quant layer norm ops
def rms_norm_dynamic_per_token_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
    scale_ub: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("FIX soon")
    output = torch.empty_like(input, dtype=quant_dtype)
    scales = torch.empty((input.numel() // input.shape[-1], 1),
                         device=input.device,
                         dtype=torch.float32)

    ops.rms_norm_dynamic_per_token_quant(output, input, weight,
                                                  scales, epsilon, scale_ub,
                                                  residual)
    return output, scales


# quantization ops
# awq
def awq_dequantize(qweight: torch.Tensor, scales: torch.Tensor,
                   zeros: torch.Tensor, split_k_iters: int, thx: int,
                   thy: int) -> torch.Tensor:
    raise NotImplementedError("FIX soon")


def awq_gemm(input: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor,
             pack_factor, group_size: int = 128) -> torch.Tensor:
    return ops.wui4a16(input, qweight, scales, qzeros, None, group_size, "NN")


# gptq
def gptq_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
              b_gptq_qzeros: torch.Tensor, b_gptq_scales: torch.Tensor,
              b_g_idx: torch.Tensor, use_exllama: bool,
              bit: int) -> torch.Tensor:
    return ops.gptq_gemm(a, b_q_weight, b_gptq_qzeros ,b_gptq_scales,
                                  b_g_idx, use_exllama, bit)


if hasattr(ops, "gptq_gemm"):

    @register_fake("_C::gptq_gemm")
    def _gptq_gemm_fake(a: torch.Tensor, b_q_weight: torch.Tensor,
                        b_gptq_qzeros: torch.Tensor,
                        b_gptq_scales: torch.Tensor, b_g_idx: torch.Tensor,
                        use_exllama: bool, bit: int) -> torch.Tensor:
        return torch.empty((a.size(0), b_q_weight.size(1)),
                           dtype=a.dtype,
                           device=a.device)


def gptq_shuffle(q_weight: torch.Tensor, q_perm: torch.Tensor,
                 bit: int) -> None:
    ops.vllm_gptq_shuffle(q_weight, q_perm, bit)


# marlin
def marlin_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
                b_scales: torch.Tensor, workspace: torch.Tensor, size_m: int,
                size_n: int, size_k: int) -> torch.Tensor:
    raise NotImplementedError("FIX soon")


# marlin_24
def gptq_marlin_24_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
                        b_meta: torch.Tensor, b_scales: torch.Tensor,
                        workspace: torch.Tensor, b_q_type: ScalarType,
                        size_m: int, size_n: int, size_k: int) -> torch.Tensor:
    raise NotImplementedError("FIX soon")


if hasattr(ops, "gptq_marlin_24_gemm"):

    @register_fake("_C::gptq_marlin_24_gemm")
    def _gptq_marlin_24_gemm_fake(a: torch.Tensor, b_q_weight: torch.Tensor,
                                  b_meta: torch.Tensor, b_scales: torch.Tensor,
                                  workspace: torch.Tensor,
                                  b_q_type: ScalarType, size_m: torch.SymInt,
                                  size_n: torch.SymInt,
                                  size_k: torch.SymInt) -> torch.Tensor:
        return torch.empty((size_m, size_n), device=a.device, dtype=a.dtype)

    @register_fake("_C::gptq_marlin_gemm")
    def _gptq_marlin_gemm_fake(a: torch.Tensor,
                               b_q_weight: torch.Tensor,
                               b_scales: torch.Tensor,
                               b_zeros: torch.Tensor,
                               g_idx: torch.Tensor,
                               perm: torch.Tensor,
                               workspace: torch.Tensor,
                               b_q_type: ScalarType,
                               size_m: torch.SymInt,
                               size_n: torch.SymInt,
                               size_k: torch.SymInt,
                               is_k_full: bool,
                               has_zp: bool = False,
                               use_atomic_add: bool = False,
                               use_fp32_reduce: bool = False,
                               is_zp_float: bool = False) -> torch.Tensor:
        return torch.empty((size_m, size_n), device=a.device, dtype=a.dtype)

    @register_fake("_C::marlin_qqq_gemm")
    def _marlin_qqq_gemm_fake(a: torch.Tensor, b_q_weight: torch.Tensor,
                              s_tok: torch.Tensor, s_ch: torch.Tensor,
                              s_group: torch.Tensor, workspace: torch.Tensor,
                              size_m: torch.SymInt, size_n: torch.SymInt,
                              size_k: torch.SymInt) -> torch.Tensor:
        return torch.empty((size_m, size_n),
                           dtype=torch.float16,
                           device=a.device)

    @register_fake("_C::marlin_gemm")
    def _marlin_gemm_fake(a: torch.Tensor, b_q_weight: torch.Tensor,
                          b_scales: torch.Tensor, workspace: torch.Tensor,
                          size_m: torch.SymInt, size_n: torch.SymInt,
                          size_k: torch.SymInt) -> torch.Tensor:
        return torch.empty((size_m, size_n),
                           dtype=torch.float16,
                           device=a.device)

    @register_fake("_C::awq_dequantize")
    def _awq_dequantize_fake(qweight: torch.Tensor, scales: torch.Tensor,
                             zeros: torch.Tensor, split_k_iters: torch.SymInt,
                             thx: int, thy: int) -> torch.Tensor:
        in_c = qweight.size(0)
        qout_c = qweight.size(1)
        out_c = qout_c * 8
        return torch.empty((in_c, out_c),
                           dtype=scales.dtype,
                           device=scales.device)

    @register_fake("_C::awq_gemm")
    def _awq_gemm_fake(input: torch.Tensor, qweight: torch.Tensor,
                       qzeros: torch.Tensor, scales: torch.Tensor,
                       split_k_iters: torch.SymInt) -> torch.Tensor:
        num_in_feats = input.size(0)
        return torch.empty((split_k_iters, num_in_feats, qweight.size(1) * 8),
                           dtype=input.dtype,
                           device=input.device).sum(0)

    @register_fake("_C::aqlm_gemm")
    def _aqlm_gemm_fake(input: torch.Tensor, codes: torch.Tensor,
                        codebooks: torch.Tensor, scales: torch.Tensor,
                        codebook_partition_sizes: list[int],
                        bias: Optional[torch.Tensor]) -> torch.Tensor:
        out_features = codes.size(0) * codebooks.size(2)
        flat_input = input.reshape((-1, input.size(-1)))
        flat_output = torch.empty((flat_input.size(0), out_features),
                                  dtype=input.dtype,
                                  device=input.device)

        output_sizes = list(input.shape)
        output_sizes.pop()
        output_sizes.append(-1)
        return flat_output.reshape(tuple(output_sizes))

    @register_fake("_C::aqlm_dequant")
    def _aqlm_dequant_fake(
            codes: torch.Tensor, codebooks: torch.Tensor,
            codebook_partition_sizes: list[int]) -> torch.Tensor:
        in_features = codes.size(1) * 8
        out_features = codes.size(0)
        return torch.empty((out_features, in_features),
                           dtype=codebooks.dtype,
                           device=codebooks.device)

    @register_fake("_C::fp8_marlin_gemm")
    def _fp8_marlin_gemm_fake(a: torch.Tensor, b_q_weight: torch.Tensor,
                              b_scales: torch.Tensor, workspace: torch.Tensor,
                              num_bits: int, size_m: torch.SymInt,
                              size_n: torch.SymInt,
                              size_k: torch.SymInt) -> torch.Tensor:
        return torch.empty((size_m, size_n), dtype=a.dtype, device=a.device)

    @register_fake("_C::machete_mm")
    def machete_mm_fake(
        a: torch.Tensor,
        # b_q Should be the tensor returned by machete_prepack_B
        b_q: torch.Tensor,
        b_type: ScalarType,
        out_type: Optional[torch.dtype] = None,
        b_group_scales: Optional[torch.Tensor] = None,
        b_group_zeros: Optional[torch.Tensor] = None,
        b_group_size: Optional[int] = None,
        b_channel_scales: Optional[torch.Tensor] = None,
        a_token_scales: Optional[torch.Tensor] = None,
        schedule: Optional[str] = None,
    ) -> torch.Tensor:
        m = a.size(0)
        n = b_q.size(1)
        return torch.empty((m, n), device=a.device, dtype=a.dtype)

    @register_fake("_C::machete_prepack_B")
    def machete_prepack_B_fake(
            b_q_weight: torch.Tensor, a_type: torch.dtype, b_type: ScalarType,
            group_scales_type: Optional[torch.dtype]) -> torch.Tensor:
        return torch.empty_like(b_q_weight,
                                memory_format=torch.contiguous_format)


if hasattr(torch.ops._C, "allspark_w8a16_gemm"):

    @register_fake("_C::allspark_w8a16_gemm")
    def _allspark_w8a16_gemm_fake(a: torch.Tensor, b_qweight: torch.Tensor,
                                  b_scales: torch.Tensor,
                                  b_qzeros: Optional[torch.Tensor],
                                  n: torch.SymInt, group_size: torch.SymInt,
                                  sm_count: torch.SymInt,
                                  sm_version: torch.SymInt,
                                  CUBLAS_M_THRESHOLD: torch.SymInt,
                                  has_zp: bool,
                                  n32k16_reorder: bool) -> torch.Tensor:
        m = a.size(0)
        return torch.empty((m, n), device=a.device, dtype=a.dtype)


if hasattr(torch.ops._C, "ggml_dequantize"):

    @register_fake("_C::ggml_dequantize")
    def _ggml_dequantize_fake(
            W: torch.Tensor,
            quant_type: int,
            m: torch.SymInt,
            n: torch.SymInt,
            dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return torch.empty((m, n), dtype=torch.float16, device=W.device)

    @register_fake("_C::ggml_mul_mat_vec_a8")
    def _ggml_mul_mat_vec_a8_fake(
        W: torch.Tensor,
        X: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
    ) -> torch.Tensor:
        return torch.empty((1, row), dtype=X.dtype, device=W.device)

    @register_fake("_C::ggml_mul_mat_a8")
    def _ggml_mul_mat_a8_fake(
        W: torch.Tensor,
        X: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
    ) -> torch.Tensor:
        batch = X.size(0)
        return torch.empty((batch, row), dtype=X.dtype, device=W.device)

    @register_fake("_C::ggml_moe_a8")
    def _ggml_moe_a8_fake(
        X: torch.Tensor,
        W: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
        top_k: torch.SymInt,
        tokens: torch.SymInt,
    ) -> torch.Tensor:
        tokens = X.size(0)
        return torch.empty((tokens * top_k, row),
                           dtype=torch.float16,
                           device=W.device)


# cutlass
def cutlass_scaled_mm_supports_fp4(cuda_device_capability: int) -> bool:
    return torch.ops._C.cutlass_scaled_mm_supports_fp4(cuda_device_capability)


def cutlass_scaled_fp4_mm(a: torch.Tensor, b: torch.Tensor,
                          block_scale_a: torch.Tensor,
                          block_scale_b: torch.Tensor, alpha: torch.Tensor,
                          out_dtype: torch.dtype) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    m, n = a.shape[0], b.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    torch.ops._C.cutlass_scaled_fp4_mm(out, a, b, block_scale_a, block_scale_b,
                                       alpha)
    return out


def cutlass_scaled_mm_supports_fp8(cuda_device_capability: int) -> bool:
    return False


def cutlass_scaled_mm_supports_block_fp8(cuda_device_capability: int) -> bool:
    return False


def cutlass_scaled_mm(a: torch.Tensor,
                      b: torch.Tensor,
                      scale_a: torch.Tensor,
                      scale_b: torch.Tensor,
                      out_dtype: torch.dtype,
                      bias: Optional[torch.Tensor] = None,
                      format: Optional[str] = "TN") -> torch.Tensor:
    """
    `cutlass_scaled_mm` implements a fused version of
        `output = torch.mm((scale_a * a), (scale_b * b)).to(out_dtype)`
    where scale_a * a and scale_b * b are implemented using numpy-style
    broadcasting.

    In order to support blockwise scaling like found in DeepSeek V3 we also
    support extended "group" broadcast rules. We extend the numpy-style
    broadcasting rules with the following rule:
        "if the extent of a dimension in the source shape is between 1 and
        corresponding extent in the target shape we repeat each element along
        that dimension  src_shape[dim] // target_shape[dim] times consecutively"
    example if we have:
          a = [[1, 2], and target_shape = (2, 4)
               [3, 4]]
    then we would expand a to:
          a = [[1, 1, 2, 2],
               [3, 3, 4, 4]]
    currently we only support the case:
        scale_a.shape * [1, 128] == a.shape
        scale_b.shape * [128, 128] == b.shape
    """
    assert (b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0)
    assert (out_dtype is torch.bfloat16 or out_dtype is torch.float16)
    assert bias is None or bias.shape[0] == b.shape[
        1] and bias.dtype == out_dtype

    
    m = a.shape[0]
    n = b.shape[1]
    if format == "TN":
        b = b.t()
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    
    ops.w8a8(a, b, scale_a, scale_b, bias, format=format, output=out, out_dtype=out_dtype)
   
    return out


def cutlass_scaled_mm_azp(a: torch.Tensor,
                          b: torch.Tensor,
                          scale_a: torch.Tensor,
                          scale_b: torch.Tensor,
                          out_dtype: torch.dtype,
                          azp_adj: torch.Tensor,
                          azp: Optional[torch.Tensor] = None,
                          bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    :param azp_adj: In the per-tensor case, this should include the azp.
    Always per-channel.
    :param azp: Only set in the per-token case. Per-token if set.
    """
    raise NotImplementedError("FIX soon")


def cutlass_sparse_scaled_mm_supported(cuda_device_capability: int) -> bool:
    return False


def cutlass_group_gemm_supported(cuda_device_capability: int) -> bool:
    return torch.ops._C.cutlass_group_gemm_supported(cuda_device_capability)

def cutlass_sparse_compress(a: torch.Tensor) \
    -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compresses a sparse matrix for use with Cutlass sparse operations.

    This function takes a dense tensor and compresses it into two components:
    non-zero elements and metadata. The compressed representation is compatible
    with Cutlass sparse kernels.

    Args:
        a (torch.Tensor):
            The input tensor to be compressed. Must have one of the following data types:
            - `torch.int8`
            - `torch.float8_e4m3fn`
            - `torch.bfloat16`
            - `torch.float16`

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            A tuple containing:
            - `a_nzs` (torch.Tensor): A tensor containing non-zero elements of `a`.
            - `a_meta` (torch.Tensor): A tensor containing metadata for the sparse representation.

    Raises:
        ValueError: If the compression operation fails.

    Notes:
        - The `a_meta` tensor has a data type of `torch.uint8`.
        - Each metadata element encodes the sparsity of 4 non-zero elements (i.e., `elemsPerMetaElem = 4`).
        - The shape of `a_nzs` is `(m, k // 2)`, where `m` and `k` are the dimensions of the input tensor.
        - The shape of `a_meta` is `(m, k // 2 // elemsPerMetaElem)`.
    """
    assert (a.dtype in [
        torch.int8, torch.float8_e4m3fn, torch.bfloat16, torch.float16
    ])
    assert (a.is_contiguous())

    # a_meta.dtype: torch.uint8 so elemsPerMetaElem = 8b / 2b_per_nz = 4
    elemsPerMetaElem = 4
    assert (a.shape[1] % (2 * elemsPerMetaElem) == 0)

    return torch.ops._C.cutlass_sparse_compress(a)


def cutlass_scaled_sparse_mm(
        a: torch.Tensor,
        bt_nzs: torch.Tensor,
        bt_meta: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        out_dtype: torch.dtype,
        bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Performs a scaled sparse matrix multiplication using Cutlass.

    Steps:
    1. Create a dense matrix `a` of shape (m, k) on the CUDA device:
    `a = torch.randn((m, k), device='cuda')`.

    2. Create a dense matrix `b` of shape (k, n) on the CUDA device:
    `b = torch.randn((k, n), device='cuda')`.

    3. Prune matrix `b` to 2:4 sparsity along the specified dimension:
    `b = prune_to_2_4(b, dim=0)`.

    4. Compress the transposed sparse matrix `b.t()`:
    `bt_nzs, bt_meta = cutlass_sparse_compress(b.t())`.

    5. Perform sparse matrix multiplication using the compressed matrix,
    applying scaling factors for `a` and `b`, and the output data type:
    `out = cutlass_scaled_sparse_mm(a, bt_nzs, bt_meta, scale_a, scale_b, out_dtype)`.

    Returns:
    - The result of the scaled sparse matrix multiplication.
    """
    assert (bt_nzs.shape[0] % 16 == 0 and bt_nzs.shape[1] % 16 == 0)
    assert (out_dtype is torch.bfloat16 or out_dtype is torch.float16)
    assert bias is None or bias.shape[0] == bt_nzs.shape[0] \
        and bias.dtype == out_dtype

    m = a.shape[0]
    n = bt_nzs.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)

    torch.ops._C.cutlass_scaled_sparse_mm(out, a, bt_nzs, bt_meta, scale_a,
                                          scale_b, bias)

    return out


def get_cutlass_moe_mm_data(
        topk_ids: torch.Tensor, expert_offsets: torch.Tensor,
        problem_sizes1: torch.Tensor, problem_sizes2: torch.Tensor,
        input_permutation: torch.Tensor, output_permutation: torch.Tensor,
        num_experts: int, n: int, k: int):
    """
    Prepare data necessary to perform CUTLASS grouped matrix multiplications
    used in CUTLASS-based fused MoE.

    The function takes in topk_ids (token-expert mapping) and uses it to
    compute:
    - expert_offsets: Indices that mark at which token index each expert begins
                      its computation after the input is sorted with
                      input_permutation. The number of tokens computed with
                      expert E is expert_offsets[E + 1] - expert_offsets[E]
    - problem_sizes1, problem_sizes2: MxNxK sizes of each expert's
                                      multiplication in two grouped MMs used in
                                      the fused MoE operation.
    - input_permutation: Permutation that must be used to shuffle the input
                         before executing the MMs.
    - output_permutation: Permutation that must be used to shuffle the output
                          after executing the MMs.
    """
    torch.ops._C.get_cutlass_moe_mm_data(topk_ids, expert_offsets,
                                         problem_sizes1, problem_sizes2,
                                         input_permutation, output_permutation,
                                         num_experts, n, k)


def cutlass_moe_mm(out_tensors: torch.Tensor, a_tensors: torch.Tensor,
                   b_tensors: torch.Tensor, a_scales: torch.Tensor,
                   b_scales: torch.Tensor, expert_offsets: torch.Tensor,
                   problem_sizes: torch.Tensor, a_strides: torch.Tensor,
                   b_strides: torch.Tensor, c_strides: torch.Tensor):
    """
    A single grouped matrix multiplication used in CUTLASS-based fused MoE.
    The function executes fp8-quantized OUT = AB matrix multiplication.

    - expert_offsets: Indices that mark at which token index each expert begins
                      its computation. The number of tokens computed with
                      expert E is expert_offsets[E + 1] - expert_offsets[E]
    - problem_sizes: MxNxK sizes of each expert's multiplication in two grouped
                     MMs used in the fused MoE operation.
    - a/b/c_strides: The data strides passed to grouped matrix multiplication.
    """
    torch.ops._C.cutlass_moe_mm(out_tensors, a_tensors, b_tensors, a_scales,
                                b_scales, expert_offsets, problem_sizes,
                                a_strides, b_strides, c_strides)


# aqlm
def aqlm_gemm(input: torch.Tensor, codes: torch.Tensor,
              codebooks: torch.Tensor, scales: torch.Tensor,
              codebook_partition_sizes: list[int],
              bias: Optional[torch.Tensor]) -> torch.Tensor:
    raise NotImplementedError("FIX soon")


def aqlm_dequant(codes: torch.Tensor, codebooks: torch.Tensor,
                 codebook_partition_sizes: List[int]) -> torch.Tensor:
    raise NotImplementedError("FIX soon")


# gptq_marlin
def gptq_marlin_repack(b_q_weight: torch.Tensor, perm: torch.Tensor,
                       size_k: int, size_n: int,
                       num_bits: int) -> torch.Tensor:
    raise NotImplementedError("FIX soon")


# gptq_marlin
def awq_marlin_repack(b_q_weight: torch.Tensor, size_k: int, size_n: int,
                      num_bits: int) -> torch.Tensor:
    raise NotImplementedError("FIX soon")


def gptq_marlin_moe_repack(b_q_weight: torch.Tensor, perm: torch.Tensor,
                           size_k: int, size_n: int,
                           num_bits: int) -> torch.Tensor:
    raise NotImplementedError("FIX soon")


def awq_marlin_moe_repack(b_q_weight: torch.Tensor, perm: torch.Tensor,
                          size_k: int, size_n: int,
                          num_bits: int) -> torch.Tensor:
    raise NotImplementedError("FIX soon")


def gptq_marlin_gemm(a: torch.Tensor,
                     b_q_weight: torch.Tensor,
                     b_scales: torch.Tensor,
                     b_zeros: torch.Tensor,
                     g_idx: torch.Tensor,
                     perm: torch.Tensor,
                     workspace: torch.Tensor,
                     b_q_type: ScalarType,
                     size_m: int,
                     size_n: int,
                     size_k: int,
                     is_k_full: bool,
                     has_zp: bool = False,
                     use_atomic_add: bool = False,
                     use_fp32_reduce: bool = False,
                     is_zp_float: bool = False) -> torch.Tensor:
    raise NotImplementedError("FIX soon")


# fp8 marlin
def fp8_marlin_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
                    b_scales: torch.Tensor, workspace: torch.Tensor,
                    num_bits: int, size_m: int, size_n: int,
                    size_k: int) -> torch.Tensor:
    raise NotImplementedError("FIX soon")


# machete
def machete_supported_schedules(
        a_type: torch.dtype,
        b_type: ScalarType,
        group_scales_type: Optional[torch.dtype],
        group_zeros_type: Optional[torch.dtype] = None,
        channel_scales_type: Optional[torch.dtype] = None,
        token_scales_type: Optional[torch.dtype] = None,
        out_type: Optional[torch.dtype] = None) -> List[str]:
    raise NotImplementedError("FIX soon")


def machete_mm(
        a: torch.Tensor,
        # b_q Should be the tensor returned by machete_prepack_B
        b_q: torch.Tensor,
        b_type: ScalarType,
        out_type: Optional[torch.dtype] = None,
        b_group_scales: Optional[torch.Tensor] = None,
        b_group_zeros: Optional[torch.Tensor] = None,
        b_group_size: Optional[int] = None,
        b_channel_scales: Optional[torch.Tensor] = None,
        a_token_scales: Optional[torch.Tensor] = None,
        schedule: Optional[str] = None) -> torch.Tensor:
    raise NotImplementedError("FIX soon")


def machete_prepack_B(
        b_q_weight: torch.Tensor, a_type: torch.dtype, b_type: ScalarType,
        group_scales_type: Optional[torch.dtype]) -> torch.Tensor:
    raise NotImplementedError("FIX soon")


if hasattr(ops, "permute_cols"):

    @register_fake("_C::permute_cols")
    def _permute_cols_fake(a: torch.Tensor,
                           perm: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(a)


def permute_cols(a: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("FIX soon")


# fp4
def scaled_fp4_quant(
        input: torch.Tensor,
        input_global_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP4 and return quantized tensor and scale.

    This function quantizes the last dimension of the given tensor `input`. For
    every 16 consecutive elements, a single dynamically computed scaling factor
    is shared. This scaling factor is quantized using the `input_global_scale`
    and is stored in a swizzled layout (see
    https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x).

    Args:
        input: The input tensor to be quantized to FP4
        input_global_scale: A scalar scaling factor for the entire tensor.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The output tensor in FP4 but every
            two values are packed into a uint8 and float8_e4m3 scaling factors
            in the sizzled layout.
    """
    assert not current_platform.is_rocm()
    assert input.ndim >= 1, (
        f'input.ndim needs to be >= 1, but got {input.ndim}.')
    other_dims = 1 if input.ndim == 1 else -1
    input = input.reshape(other_dims, input.shape[-1])
    m, n = input.shape
    block_size = 16
    device = input.device

    assert n % block_size == 0, (
        f'last dim has to be multiple of 16, but got {n}.')
    assert input.dtype in (torch.float16, torch.bfloat16), (
        f'input.dtype needs to be fp16 or bf16 but got {input.dtype}.')

    # Two fp4 values will be packed into an uint8.
    output = torch.empty((m, n // 2), device=device, dtype=torch.uint8)

    # We use the rounded values to store the swizzled values. Due to the
    # requirement of the Tensor Core, the minimum tile is 128x4 for the scales.
    # So, we first pad the scales to multiples of 128 and 4. Then, the scales
    # (in float8_e4m3fn) are packed into an int32 for every 4 values. More:
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x
    round_up = lambda x, y: (x + y - 1) // y * y
    rounded_m = round_up(m, 128)
    scale_n = n // block_size
    rounded_n = round_up(scale_n, 4)
    output_scale = torch.empty((rounded_m, rounded_n // 4),
                               device=device,
                               dtype=torch.int32)

    torch.ops._C.scaled_fp4_quant(output, input, output_scale,
                                  input_global_scale)
    output_scale = output_scale.view(torch.float8_e4m3fn)
    return output, output_scale


# fp8
def scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    num_token_padding: Optional[int] = None,
    scale_ub: Optional[torch.Tensor] = None,
    use_per_token_if_dynamic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP8 and return quantized tensor and scale.

    This function supports both static and dynamic quantization: If you
    provide the scale, it will use static scaling and if you omit it,
    the scale will be determined dynamically. The function also allows
    optional padding of the output tensors for downstream kernels that
    will benefit from padding.

    Args:
        input: The input tensor to be quantized to FP8
        scale: Optional scaling factor for the FP8 quantization
        scale_ub: Optional upper bound for scaling factor in dynamic
            per token case
        num_token_padding: If specified, pad the first dimension
            of the output to at least this value.
        use_per_token_if_dynamic: Whether to do per_tensor or per_token
            in the dynamic quantization case.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
            scaling factor.
    """
    raise NotImplementedError("FIX soon")


# gptq allspark
def allspark_repack_weight(
        qweight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor] = None,
        has_zp: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rearrange qweight, scale, and zero_point(if asymmetric) to n32k16 format
    for Ampere W8A16 Fused Gemm kernel

    Args:
        qweight: uint8 weight tensor, original k x n format.
        scale: fp16/bf16 weight scale tensor, 1 x n format.
        zero_point: fp16/bf16 weight zero_point tensor, 1 x n format.
            Must be provided for asymmetric quantization.
        has_zp: if use symmetric quantization, has_zp = False.
            if use asymmetric quantization, has_zp = True.

    Returns:
        tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]] :
            rearranged weight, scale, and optionally zero_point.
    """
    K = qweight.shape[0]
    N = qweight.shape[1]
    N_32align = (N + 32 - 1) // 32 * 32

    qweight_reorder = torch.empty((N_32align, K),
                                  device=qweight.device,
                                  dtype=qweight.dtype)
    scale_reorder = torch.empty((1, N_32align),
                                device=scale.device,
                                dtype=scale.dtype)
    zero_point_reorder = None
    if has_zp:
        assert zero_point is not None, (
            "zero_point must be provided for asymmetric quantization.")
        zero_point_reorder = torch.empty((1, N_32align),
                                         device=zero_point.device,
                                         dtype=zero_point.dtype)

    torch.ops._C.rearrange_kn_weight_as_n32k16_order(
        qweight, scale, zero_point, has_zp, qweight_reorder, scale_reorder,
        zero_point_reorder, K, N, N_32align)

    return qweight_reorder, scale_reorder, zero_point_reorder


def allspark_w8a16_gemm(a: torch.Tensor, b_qweight: torch.Tensor,
                        b_scales: torch.Tensor,
                        b_qzeros: Optional[torch.Tensor], n: int,
                        group_size: int, sm_count: int, sm_version: int,
                        CUBLAS_M_THRESHOLD: int, has_zp: bool,
                        n32k16_reorder: bool) -> torch.Tensor:

    return torch.ops._C.allspark_w8a16_gemm(a, b_qweight, b_scales, b_qzeros,
                                            n, group_size, sm_count,
                                            sm_version, CUBLAS_M_THRESHOLD,
                                            has_zp, n32k16_reorder)


# int8
def scaled_int8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    azp: Optional[torch.Tensor] = None,
    symmetric: bool = True
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Quantize the input tensor to int8 and return the quantized tensor and scale, and maybe azp.

    Args:
        input: The input tensor to be quantized to int8.
        scale: Optional scaling factor for the int8 quantization.
            When not provided, we invoke dynamic-per-token quantization.
        azp: Optional zero-point for the int8 quantization.
            Must be provided for asymmetric quantization if `scale` is provided.
        symmetric: Whether to use symmetric quantization (scale only, azp ignored).

    Returns:
      tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]] : Output int8 tensor, scales, and optionally azp.
    """
    output = torch.empty_like(input, dtype=torch.int8)
    if scale is not None:
        # static-per-tensor quantization.
        assert symmetric == (
            azp
            is None), "azp must only be provided for asymmetric quantization."
        ops.static_scaled_int8_quant(output, input, scale)
        return output, scale, azp

    # dynamic-per-token quantization.
    input_scales = torch.empty((input.numel() // input.shape[-1], 1),
                               device=input.device,
                               dtype=torch.float32)
    input_azp = None if symmetric else torch.empty_like(input_scales,
                                                        dtype=torch.int32)
    ops.dynamic_scaled_int8_quant(output, input, input_scales)
    return output, input_scales, input_azp


# qqq ops
def marlin_qqq_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
                    s_tok: torch.Tensor, s_ch: torch.Tensor,
                    s_group: torch.Tensor, workspace: torch.Tensor,
                    size_m: int, size_n: int, size_k: int) -> torch.Tensor:
    raise NotImplementedError("FIX soon")


# gguf
def ggml_dequantize(W: torch.Tensor, quant_type: int, m: int, n: int,
                    dtype: Optional[torch.dtype]) -> torch.Tensor:
    raise NotImplementedError("FIX soon")


def ggml_mul_mat_vec_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    raise NotImplementedError("FIX soon")


def ggml_mul_mat_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    raise NotImplementedError("FIX soon")


def ggml_moe_a8(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    quant_type: int,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    return torch.ops._C.ggml_moe_a8(X, W, sorted_token_ids, expert_ids,
                                    num_tokens_post_padded, quant_type, row,
                                    top_k, tokens)


def ggml_moe_get_block_size(quant_type: int) -> int:
    return torch.ops._C.ggml_moe_get_block_size(quant_type)


# mamba
def causal_conv1d_fwd(x: torch.Tensor, weight: torch.Tensor,
                      bias_: Optional[torch.Tensor],
                      conv_states: Optional[torch.Tensor],
                      query_start_loc: Optional[torch.Tensor],
                      cache_indices: Optional[torch.Tensor],
                      has_initial_state: Optional[torch.Tensor],
                      silu_activation: bool, pad_slot_id: int):
    raise NotImplementedError("FIX soon")


def causal_conv1d_update(x: torch.Tensor, conv_state: torch.Tensor,
                         weight: torch.Tensor, bias_: Optional[torch.Tensor],
                         silu_activation: bool,
                         cache_seqlens: Optional[torch.Tensor],
                         conv_state_indices: Optional[torch.Tensor],
                         pad_slot_id: int):
    raise NotImplementedError("FIX soon")


def selective_scan_fwd(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor,
                       B: torch.Tensor, C: torch.Tensor,
                       D_: Optional[torch.Tensor], z_: Optional[torch.Tensor],
                       delta_bias_: Optional[torch.Tensor],
                       delta_softplus: bool,
                       query_start_loc: Optional[torch.Tensor],
                       cache_indices: Optional[torch.Tensor],
                       has_initial_state: Optional[torch.Tensor],
                       ssm_states: torch.Tensor, pad_slot_id: int):
    raise NotImplementedError("FIX soon")


# moe
def moe_sum(input: torch.Tensor, output: torch.Tensor):
    raise NotImplementedError("FIX soon")


def moe_align_block_size(topk_ids: torch.Tensor, num_experts: int,
                         block_size: int, sorted_token_ids: torch.Tensor,
                         experts_ids: torch.Tensor,
                         num_tokens_post_pad: torch.Tensor) -> None:
    ops.vllm_moe_align_block_size(topk_ids, num_experts, block_size,
                                      sorted_token_ids, experts_ids,
                                      num_tokens_post_pad)


def sgl_moe_align_block_size(topk_ids: torch.Tensor, num_experts: int,
                             block_size: int, sorted_token_ids: torch.Tensor,
                             experts_ids: torch.Tensor,
                             num_tokens_post_pad: torch.Tensor) -> None:
    torch.ops._moe_C.sgl_moe_align_block_size(topk_ids, num_experts,
                                              block_size, sorted_token_ids,
                                              experts_ids, num_tokens_post_pad)


def moe_wna16_gemm(input: torch.Tensor, output: torch.Tensor,
                   b_qweight: torch.Tensor, b_scales: torch.Tensor,
                   b_qzeros: Optional[torch.Tensor],
                   topk_weights: Optional[torch.Tensor],
                   sorted_token_ids: torch.Tensor, experts_ids: torch.Tensor,
                   num_tokens_post_pad: torch.Tensor, top_k: int,
                   BLOCK_SIZE_M: int, BLOCK_SIZE_N: int, BLOCK_SIZE_K: int,
                   bit: int) -> torch.Tensor:
    if not current_platform.is_cuda():
        raise NotImplementedError(
            "The optimized moe_wna16_gemm kernel is only "
            "available on CUDA platforms")
    torch.ops._moe_C.moe_wna16_gemm(input, output, b_qweight, b_scales,
                                    b_qzeros, topk_weights, sorted_token_ids,
                                    experts_ids, num_tokens_post_pad, top_k,
                                    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                                    bit)


def topk_softmax(topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                 token_expert_indicies: torch.Tensor,
                 gating_output: torch.Tensor) -> None:
    # 三级 fallback: bridge(.float()) → ixf_F → corex → PyTorch
    if _bridge is not None:
        _log_once("topk_softmax: bridge path, gating.float()")
        topk = topk_weights.shape[-1]
        w, ids, _tei = _bridge.topk_softmax(gating_output.float(), topk, True)
        topk_weights.copy_(w)
        topk_ids.copy_(ids.to(topk_ids.dtype))
        return
    if hasattr(ops, 'vllm_moe_topk_softmax'):
        _log_once("topk_softmax: ixf_F path")
        ops.vllm_moe_topk_softmax(topk_weights, topk_ids,
                                  token_expert_indicies, gating_output)
        return
    try:
        from vllm import corex_moe_topk_softmax as _cmts
        _log_once("topk_softmax: corex path")
        topk = topk_weights.shape[-1]
        w, ids = _cmts.moe_topk_softmax(gating_output, topk, True)
        topk_weights.copy_(w)
        topk_ids.copy_(ids)
        return
    except (ImportError, Exception):
        pass
    # Pure PyTorch fallback
    _log_once("topk_softmax: PyTorch fallback")
    topk = topk_weights.shape[-1]
    scores = torch.softmax(gating_output.float(), dim=-1)
    tw, ti = torch.topk(scores, topk, dim=-1)
    tw = tw / tw.sum(dim=-1, keepdim=True)
    topk_weights.copy_(tw)
    topk_ids.copy_(ti.to(topk_ids.dtype))


if supports_moe_ops and hasattr(torch.ops._moe_C, "marlin_gemm_moe"):

    @register_fake("_moe_C::marlin_gemm_moe")
    def marlin_gemm_moe_fake(a: torch.Tensor, b_q_weights: torch.Tensor,
                             sorted_ids: torch.Tensor,
                             topk_weights: torch.Tensor,
                             topk_ids: torch.Tensor, b_scales: torch.Tensor,
                             b_zero_points: torch.Tensor, g_idx: torch.Tensor,
                             perm: torch.Tensor, workspace: torch.Tensor,
                             b_q_type: ScalarType, size_m: torch.SymInt,
                             size_n: torch.SymInt, size_k: torch.SymInt,
                             is_k_full: bool, num_experts: int, topk: int,
                             moe_block_size: int, replicate_input: bool,
                             apply_weights: bool) -> torch.Tensor:
        return torch.empty((size_m, topk, size_n),
                           dtype=a.dtype,
                           device=a.device)


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    import ixformer.functions as _ixf
    if hasattr(_ixf, 'vllm_cache_ops_reshape_and_cache'):
        _log_once("reshape_and_cache: ixformer.functions path")
        _ixf.vllm_cache_ops_reshape_and_cache(key, value, key_cache,
                                               value_cache, slot_mapping.to(torch.int32))
    elif _xllm_cache is not None:
        _log_once("reshape_and_cache: xllm_cache path, slot_i32 first")
        slot_mapping_i32 = slot_mapping.flatten().to(torch.int32)
        _xllm_cache.reshape_paged_cache(slot_mapping_i32, key, value,
                                        key_cache, value_cache)
    elif _bridge is not None:
        _log_once("reshape_and_cache: bridge path, slot_i32")
        slot_mapping_i32 = slot_mapping.flatten().to(torch.int32)
        _bridge.reshape_and_cache(key, value, key_cache, value_cache,
                                 slot_mapping_i32)
    else:
        _log_once("reshape_and_cache: ops fallback")
        ops.vllm_cache_ops_reshape_and_cache(key, value, key_cache,
                                             value_cache, slot_mapping)


def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    ops.reshape_and_cache_flash(key, value, key_cache,
                                value_cache, slot_mapping,
                                kv_cache_dtype, 1.0, 1.0)


def concat_and_cache_mla(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    scale: torch.Tensor,
) -> None:
    ops.vllm_concat_and_cache_mla(kv_c, k_pe, kv_cache,
                                  slot_mapping, kv_cache_dtype,
                                  scale)
    
    
def concat_and_cache_mla_int8(
    kv_c_int8: torch.Tensor,
    kv_c_scale: torch.Tensor,
    k_pe_int8: torch.Tensor,
    k_pe_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_cache_scale: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    scale: torch.Tensor,
) -> None:
    ops.vllm_concat_and_cache_mla_int8(kv_c_int8,kv_c_scale, k_pe_int8, k_pe_scale, kv_cache, kv_cache_scale,
                                  slot_mapping, kv_cache_dtype,
                                  scale)


def copy_blocks(key_caches: list[torch.Tensor],
                value_caches: list[torch.Tensor],
                block_mapping: torch.Tensor) -> None:
    ops.vllm_copy_blocks(key_caches, value_caches, block_mapping)


def copy_blocks_mla(kv_caches: list[torch.Tensor],
                    block_mapping: torch.Tensor) -> None:
    torch.ops._C_cache_ops.copy_blocks_mla(kv_caches, block_mapping)


def swap_blocks(src: torch.Tensor, dst: torch.Tensor,
                block_mapping: torch.Tensor) -> None:
    ops.vllm_swap_blocks(src, dst, block_mapping)


def convert_fp8(output: torch.Tensor,
                input: torch.Tensor,
                scale: float = 1.0,
                kv_dtype: str = "fp8") -> None:
    raise NotImplementedError("FIX soon")



def gather_cache(
    src_cache: torch.Tensor,      # [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    dst: torch.Tensor,            # [TOT_TOKENS, ENTRIES...]
    block_table: torch.Tensor,    # [BATCH, BLOCK_INDICES]
    cu_seq_lens: torch.Tensor,    # [BATCH+1]
    batch_size: int,
    seq_starts: torch.Tensor = None
):
    ops.vllm_gather_cache(src_cache, dst, block_table, cu_seq_lens, batch_size, seq_starts)
            
def gather_cache_int8(
    src_cache: torch.Tensor,      # [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    src_cache_scale: torch.Tensor,# [NUM_BLOCKS, BLOCK_SIZE, 2]
    kv_lora_rank: int,
    dst: torch.Tensor,            # [TOT_TOKENS, ENTRIES...]
    block_table: torch.Tensor,    # [BATCH, BLOCK_INDICES]
    cu_seq_lens: torch.Tensor,    # [BATCH+1]
    batch_size: int,
    seq_starts: torch.Tensor = None
):
    ops.vllm_gather_cache_int8(src_cache,src_cache_scale, kv_lora_rank, dst, block_table, cu_seq_lens, batch_size, seq_starts)


def get_device_attribute(attribute: int, device: int) -> int:
    raise NotImplementedError("FIX soon")


def get_max_shared_memory_per_block_device_attribute(device: int) -> int:
    # BI-V100 SMEM = 49152 bytes (48KB), confirmed via ixsmi
    # Was incorrectly hardcoded to 32KB (32768), limiting Triton tile sizes
    # and potentially constraining ixformer internal SMEM allocation.
    return 49152


# custom ar
def init_custom_ar(ipc_tensors: list[torch.Tensor], rank_data: torch.Tensor,
                   rank: int, fully_connected: bool) -> int:
    raise NotImplementedError("Do not use this in our implement")


def all_reduce(fa: int, inp: torch.Tensor, out: torch.Tensor, reg_buffer: int,
               reg_buffer_sz_bytes: int) -> None:
    raise NotImplementedError("Do not use this in our implement")


def dispose(fa: int) -> None:
    raise NotImplementedError("Do not use this in our implement")


def meta_size() -> int:
    raise NotImplementedError("Do not use this in our implement")


def register_buffer(fa: int, ipc_tensors: List[int]) -> None:
    raise NotImplementedError("Do not use this in our implement")


def get_graph_buffer_ipc_meta(fa: int) -> Tuple[List[int], List[int]]:
    raise NotImplementedError("Do not use this in our implement")


def register_graph_buffers(fa: int, handles: List[List[int]],
                           offsets: List[List[int]]) -> None:
    raise NotImplementedError("Do not use this in our implement")


def allocate_shared_buffer_and_handle(size: int) -> tuple[int, torch.Tensor]:
    return torch.ops._C_custom_ar.allocate_shared_buffer_and_handle(size)


def open_mem_handle(mem_handle: torch.Tensor):
    return torch.ops._C_custom_ar.open_mem_handle(mem_handle)


def free_shared_buffer(ptr: int) -> None:
    torch.ops._C_custom_ar.free_shared_buffer(ptr)


def get_flash_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        cache_seqlens: (batch_size), dtype torch.int32.
        num_heads_per_head_k: Equals to seq_len_q * num_heads_q // num_heads_k.
        num_heads_k: num_heads_k.

    Return:
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), dtype torch.int32.
        num_splits: (batch_size + 1), dtype torch.int32.
    """
    return torch.ops._C.get_flash_mla_metadata(cache_seqlens,
                                               num_heads_per_head_k,
                                               num_heads_k)


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim).
        k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
        block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
        cache_seqlens: (batch_size), torch.int32.
        head_dim_v: Head_dim of v.
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), torch.int32, return by get_mla_metadata.
        num_splits: (batch_size + 1), torch.int32, return by get_mla_metadata.
        softmax_scale: float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim).
        causal: bool. Whether to apply causal attention mask.

    Return:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1]**(-0.5)
    out, softmax_lse = torch.ops._C.flash_mla_fwd_kvcache(
        q,
        k_cache,
        None,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
    )
    return out, softmax_lse


# Add our new features here..

# mee
def invoke_fused_moe_kernel(
    A: torch.Tensor, 
    B: torch.Tensor, 
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    topk_weights: torch.Tensor, 
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool, 
    top_k: int,
    config: Dict[str, Any],
    compute_type,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    block_shape: Optional[List[int]] = None,
) -> None:
    ops.vllm_invoke_fused_moe_kernel(
        A,
        B,
        C,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        top_k,
        config['BLOCK_SIZE_M']
    )


# broadcast
class Async_helper():
    # For now, the comm and the other kernels are in the same stream, so we can remove the stream wait..
    def wait(self,):
        return True


def broadcast(tensor, src=0, group=None, async_op=False):
    cdist.broadcast(tensor,src,group,async_op=True)
    if async_op:
        return Async_helper()
    else:
        pass


# w8a16
def linear_w8a16(x: torch.Tensor, qweight: torch.Tensor, scales:torch.Tensor,
                 group_size: int = -1, format: str = "TN")-> torch.Tensor:
    return ops.w8a16(x, qweight, scales, format="TN", group_size=group_size)


## lora sgmv / bgmv
def sbgmv_expand(x: torch.Tensor,
                w_t_all: torch.Tensor,
                y: torch.Tensor,
                b_seq_start_loc: torch.Tensor = None,
                seq_len_tensor: torch.Tensor = None,
                lora_indices_tensor: torch.Tensor = None,
                batches: int = -1,
                max_seq_length: int = -1,
                token_nums: int = -1,
                add_input=True,
                ):
    '''
    x: inputs
    w_t_all: lora weight
    y: output

    y += x@wt_t_all
    '''
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert w_t_all.dtype in [
        torch.float16,
        torch.bfloat16,
    ]

    assert x.is_contiguous()
    # assert y.is_contiguous()
    if x.dtype == torch.float:
        x = x.to(w_t_all.dtype)

    if w_t_all.ndim == 4:  # shape:(lora_num,1,size,rank)
        assert w_t_all.size(1) == 1
        w_t_all = w_t_all.squeeze(dim=1)
    else:
        assert w_t_all.ndim == 3  # shape:(lora_num,size,rank)
    assert w_t_all.is_contiguous()

    assert add_input == True

    lora_indices = lora_indices_tensor.cpu().tolist()
    lora_num = w_t_all.shape[0]

    ## 单一lora model, 且所有request均使用lora
    if lora_num == 1 and all(x == lora_indices[0] for x in lora_indices):
        if lora_indices[0] != -1:
            w_t = w_t_all[0]
            y += torch.matmul(x, w_t.t())
    ## 多个lora model
    else:
        ## prefill
        if batches != -1:
            for i, lora_id, start, seq_len in zip(range(batches), lora_indices, b_seq_start_loc, seq_len_tensor):
                if lora_id != -1:
                    xi = x[start: start+seq_len]
                    w_t = w_t_all[lora_id]
                    y[start:start+seq_len] += (xi @ w_t.t())
        ## decode
        else:
            batches = x.shape[0]
            for i, lora_id in zip(range(batches), lora_indices):
                if lora_id != -1:
                    xi = x[i].unsqueeze(0)
                    w_t = w_t_all[lora_id]
                    y[i] += (xi @ w_t.t()).squeeze(0)

    return y


def sbgmv_shrink(x: torch.Tensor,
                w_t_all: torch.Tensor,
                y: torch.Tensor,
                b_seq_start_loc: torch.Tensor = None,
                seq_len_tensor: torch.Tensor = None,
                lora_indices_tensor: torch.Tensor = None,
                batches: int = -1,
                max_seq_length: int = -1,
                token_nums: int = -1,
                scale: float = 1.0,):
    """
    xx: inputs
    w_t_all: lora weight
    y: output
    scale: float

    y = x@w_t_all * scale
    """
    assert x.dtype == w_t_all.dtype
    assert x.dtype in [torch.float16, torch.bfloat16]
    assert x.is_contiguous()
    assert y.is_contiguous()

    if w_t_all.ndim == 4:  # shape:(lora_num,1,size,rank)
        assert w_t_all.size(1) == 1
        w_t_all = w_t_all.squeeze(dim=1)
    else:
        assert w_t_all.ndim == 3  # shape:(lora_num,size,rank)
    assert w_t_all.is_contiguous()
    
    lora_num = w_t_all.shape[0]
    lora_indices = lora_indices_tensor.cpu().tolist()

    ## 单一lora model, 且所有request均使用lora
    if lora_num == 1 and all(x == lora_indices[0] for x in lora_indices):
        if lora_indices[0] != -1:
            w_t = w_t_all[0]
            y = torch.matmul(x, w_t.t()) * scale
    ## 多个lora model
    else:
        ## prefill
        if batches != -1:
            for i, lora_id, start, seq_len in zip(range(batches), lora_indices, b_seq_start_loc, seq_len_tensor):
                if lora_id != -1:
                    xi = x[start: start+seq_len]
                    w_t = w_t_all[lora_id]
                    y[start:start+seq_len] = (xi @ w_t.t())* scale
        ## decode
        else:
            batches = x.shape[0]
            for i, lora_id in zip(range(batches), lora_indices):
                if lora_id != -1:
                    xi = x[i].unsqueeze(0)
                    w_t = w_t_all[lora_id]
                    y[i] = (xi @ w_t.t()).squeeze(0) * scale

    return y

def dynamic_scaled_quant_dynamic_int8(x, input_scales=None, int8_out=None, scales=None):
    return ops.dynamic_scaled_quant_smoothquant(x, input_scales, int8_out, scales)

weak_ref_tensor = getattr(ops, 'weak_ref_tensor', None)
