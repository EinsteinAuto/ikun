"""
ix_ops_dispatch.py — Runtime C++ kernel dispatcher for BI-V100

Replaces Python fallbacks in vllm's hot path with ixformer::infer C++ calls.
All functions go through ix_full_bridge_v2.so → ixformer::infer namespace.

Upstream reference: xllm/core/kernels/ilu/*.cpp
Bridge reference:   ex_engine/csrc/ix_full_bridge_v2.cpp

Call chain (no fallback allowed):
  vllm._custom_ops.silu_and_mul      → ixformer::infer::silu_and_mul
  vllm._custom_ops.rms_norm          → ixformer::infer::rms_norm
  vllm._custom_ops.fused_add_rms_norm→ ixformer::infer::residual_rms_norm
  vllm._custom_ops.rotary_embedding  → ixformer::infer::xllm_rotary_embedding
  vllm._custom_ops.reshape_and_cache → ixformer::infer::xllm_reshape_and_cache
  MoE topk_softmax                   → ixformer::infer::topk_softmax
  MoE group_gemm                     → ixformer::infer::moe_w16a16_group_gemm
  MoE expand_input                   → ixformer::infer::moe_expand_input
  MoE combine_result                 → ixformer::infer::moe_output_reduce_sum

Not a "connector" — this is the algorithm factor replacement layer.
"""

import importlib
import importlib.util
import logging
import os
import sys
from typing import Optional

import torch

logger = logging.getLogger("ix_ops_dispatch")

# =====================================================================
# Bridge loader: find and load ix_full_bridge_v2.so
# =====================================================================
_bridge = None
_bridge_loaded = False


def _load_bridge():
    """Load the compiled C++ bridge module."""
    global _bridge, _bridge_loaded
    if _bridge_loaded:
        return _bridge

    _bridge_loaded = True

    # Search order for the .so
    search_paths = []

    # 1. Inside vllm package
    try:
        import vllm
        vllm_dir = os.path.dirname(vllm.__file__)
        search_paths.append(os.path.join(vllm_dir, "ex_engine", "ix_full_bridge_v2.so"))
        search_paths.append(os.path.join(vllm_dir, "ix_full_bridge_v2.so"))
    except ImportError:
        pass

    # 2. Prebuilt directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths.append(os.path.join(script_dir, "..", "prebuilt", "ix_full_bridge_v2.so"))
    search_paths.append(os.path.join(script_dir, "..", "prebuilt", "corex-3.2.3-ivcore10", "ix_full_bridge_v2.so"))

    # 3. Workspace
    search_paths.append("/workspace/ex_engine/prebuilt/ix_full_bridge_v2.so")
    search_paths.append("/workspace/qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10/ix_full_bridge_v2.so")

    for path in search_paths:
        if os.path.isfile(path):
            try:
                spec = importlib.util.spec_from_file_location("ix_full_bridge_v2", path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                _bridge = mod
                logger.info("ix_full_bridge_v2 loaded from %s", path)
                return _bridge
            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)

    # 4. Try as already-imported module (from prebuilt .so in VLLM_ROOT)
    try:
        import ix_full_bridge_v2
        _bridge = ix_full_bridge_v2
        logger.info("ix_full_bridge_v2 loaded from sys.path")
        return _bridge
    except ImportError:
        pass

    logger.warning("ix_full_bridge_v2.so not found — C++ dispatch unavailable")
    return None


def get_bridge():
    """Get the loaded bridge module, loading it if necessary."""
    if not _bridge_loaded:
        return _load_bridge()
    return _bridge


# =====================================================================
# Individual op dispatchers — match ixformer::infer signatures
# =====================================================================

def silu_and_mul(input_tensor: torch.Tensor) -> torch.Tensor:
    """SiLU activation: x[:half] * sigmoid(x[:half]) * x[half:]."""
    bridge = get_bridge()
    if bridge is not None and hasattr(bridge, 'silu_and_mul'):
        d = input_tensor.shape[-1]
        out = torch.empty(*input_tensor.shape[:-1], d // 2,
                          dtype=input_tensor.dtype, device=input_tensor.device)
        bridge.silu_and_mul(input_tensor, out)
        return out
    # Direct ixformer Python path (base image has this)
    try:
        import ixformer.functions as ixf_F
        d = input_tensor.shape[-1]
        out = torch.empty(*input_tensor.shape[:-1], d // 2,
                          dtype=input_tensor.dtype, device=input_tensor.device)
        ixf_F.silu_and_mul(input_tensor, out)
        return out
    except (ImportError, AttributeError):
        pass
    raise RuntimeError("silu_and_mul: no C++ implementation available")


def rms_norm(input_tensor: torch.Tensor, weight: torch.Tensor,
             epsilon: float = 1e-6) -> torch.Tensor:
    """RMSNorm: x * rsqrt(mean(x^2) + eps) * weight."""
    bridge = get_bridge()
    if bridge is not None and hasattr(bridge, 'rms_norm'):
        out = torch.empty_like(input_tensor)
        bridge.rms_norm(input_tensor, weight, out, None, epsilon)
        return out
    try:
        import ixformer.functions as ixf_F
        out = torch.empty_like(input_tensor)
        ixf_F.rms_norm(input_tensor, weight, out, epsilon)
        return out
    except (ImportError, AttributeError):
        pass
    raise RuntimeError("rms_norm: no C++ implementation available")


def fused_add_rms_norm(input_tensor: torch.Tensor, residual: torch.Tensor,
                       weight: torch.Tensor, epsilon: float = 1e-6):
    """Fused residual + RMSNorm: output = rms_norm(input + residual)."""
    bridge = get_bridge()
    if bridge is not None and hasattr(bridge, 'residual_rms_norm'):
        out = torch.empty_like(input_tensor)
        residual_out = torch.empty_like(residual)
        bridge.residual_rms_norm(
            input_tensor, residual, weight, out, residual_out,
            None, 1.0, epsilon, False)
        return out, residual_out
    try:
        import ixformer.functions as ixf_F
        ixf_F.fused_add_rms_norm(input_tensor, residual, weight, epsilon)
        return input_tensor, residual
    except (ImportError, AttributeError):
        pass
    raise RuntimeError("fused_add_rms_norm: no C++ implementation available")


def rotary_embedding(positions: torch.Tensor, query: torch.Tensor,
                     key: torch.Tensor, head_size: int,
                     cos_sin_cache: torch.Tensor, is_neox: bool = True):
    """Apply rotary positional embeddings."""
    bridge = get_bridge()
    if bridge is not None and hasattr(bridge, 'rotary_embedding'):
        bridge.rotary_embedding(positions, query, key,
                                head_size, cos_sin_cache, is_neox)
        return
    try:
        import ixformer.functions as ixf_F
        ixf_F.vllm_rotary_embedding_neox(
            positions, query, key, head_size, cos_sin_cache, is_neox)
        return
    except (ImportError, AttributeError):
        pass
    raise RuntimeError("rotary_embedding: no C++ implementation available")


def reshape_and_cache(key: torch.Tensor, value: torch.Tensor,
                      key_cache: torch.Tensor, value_cache: torch.Tensor,
                      slot_mapping: torch.Tensor):
    """Write KV pairs into paged cache."""
    bridge = get_bridge()
    if bridge is not None and hasattr(bridge, 'reshape_and_cache'):
        key_stride = key.stride(0)
        value_stride = value.stride(0)
        bridge.reshape_and_cache(key, value, key_cache, value_cache,
                                 slot_mapping, key_stride, value_stride)
        return
    try:
        import ixformer.functions as ixf_F
        ixf_F.vllm_cache_ops_reshape_and_cache(key, value, key_cache,
                                                value_cache, slot_mapping)
        return
    except (ImportError, AttributeError):
        pass
    raise RuntimeError("reshape_and_cache: no C++ implementation available")


# =====================================================================
# MoE dispatchers — 7-step pipeline from xllm upstream
# =====================================================================

def topk_softmax(gating_output: torch.Tensor, topk: int,
                 renormalize: bool = True):
    """MoE routing: softmax → topk selection."""
    bridge = get_bridge()
    if bridge is not None and hasattr(bridge, 'topk_softmax'):
        num_tokens = gating_output.shape[0]
        topk_weights = torch.empty(num_tokens, topk,
                                   dtype=torch.float32,
                                   device=gating_output.device)
        topk_ids = torch.empty(num_tokens, topk,
                               dtype=torch.int32,
                               device=gating_output.device)
        token_expert_indices = torch.empty(num_tokens, topk,
                                           dtype=torch.int32,
                                           device=gating_output.device)
        bridge.topk_softmax(topk_weights, topk_ids,
                            token_expert_indices, gating_output, renormalize)
        return topk_weights, topk_ids
    # Direct ixformer path
    try:
        import ixformer.functions as ixf_F
        num_tokens = gating_output.shape[0]
        topk_weights = torch.empty(num_tokens, topk,
                                   dtype=torch.float32,
                                   device=gating_output.device)
        topk_ids = torch.empty(num_tokens, topk,
                               dtype=torch.int32,
                               device=gating_output.device)
        token_expert_indices = torch.empty(num_tokens, topk,
                                           dtype=torch.int32,
                                           device=gating_output.device)
        ixf_F.topk_softmax(topk_weights, topk_ids,
                           token_expert_indices, gating_output, renormalize)
        return topk_weights, topk_ids
    except (ImportError, AttributeError):
        pass
    # Prebuilt corex_moe_topk_softmax.so
    try:
        import corex_moe_topk_softmax
        return corex_moe_topk_softmax.forward(gating_output, topk, renormalize)
    except (ImportError, AttributeError):
        pass
    raise RuntimeError("topk_softmax: no C++ implementation available")


def moe_compute_token_index(topk_ids: torch.Tensor, num_experts: int,
                            start_expert: int = 0):
    """Compute permutation indices for MoE expert dispatch."""
    bridge = get_bridge()
    if bridge is not None and hasattr(bridge, 'moe_compute_token_index'):
        end_expert = start_expert + num_experts
        flat_ids = topk_ids.view(-1)
        total_tokens = flat_ids.shape[0]
        src_dst = torch.empty(total_tokens, dtype=torch.int32,
                              device=topk_ids.device)
        dst_src = torch.empty(total_tokens, dtype=torch.int32,
                              device=topk_ids.device)
        expert_sizes = torch.empty(num_experts, dtype=torch.int32,
                                   device=topk_ids.device)
        bridge.moe_compute_token_index(
            flat_ids, src_dst, dst_src, expert_sizes,
            None, None, None,
            start_expert, end_expert, num_experts)
        return src_dst, dst_src, expert_sizes
    raise RuntimeError("moe_compute_token_index: no C++ implementation available")


def moe_expand_input(hidden_states: torch.Tensor, dst_to_src: torch.Tensor,
                     topk: int) -> torch.Tensor:
    """Expand input tokens for MoE expert dispatch."""
    bridge = get_bridge()
    if bridge is not None and hasattr(bridge, 'moe_expand_input'):
        num_dst = dst_to_src.shape[0]
        expanded = torch.empty(num_dst, hidden_states.shape[-1],
                               dtype=hidden_states.dtype,
                               device=hidden_states.device)
        bridge.moe_expand_input(expanded, hidden_states, dst_to_src,
                                None, num_dst, topk)
        return expanded
    raise RuntimeError("moe_expand_input: no C++ implementation available")


def moe_group_gemm(inputs: torch.Tensor, weights: torch.Tensor,
                   expert_sizes: torch.Tensor, output_n: int) -> torch.Tensor:
    """Group GEMM for MoE experts — one cublas call for all experts."""
    bridge = get_bridge()
    if bridge is not None and hasattr(bridge, 'moe_w16a16_group_gemm'):
        output = torch.empty(inputs.shape[0], output_n,
                             dtype=inputs.dtype, device=inputs.device)
        bridge.moe_w16a16_group_gemm(
            output, inputs, weights, expert_sizes,
            None, None, "NT", 0, output_n)
        return output
    raise RuntimeError("moe_group_gemm: no C++ implementation available")


def moe_output_reduce_sum(outputs: torch.Tensor, weights: torch.Tensor,
                          scaling_factor: float = 1.0) -> torch.Tensor:
    """Weighted combine of expert outputs."""
    bridge = get_bridge()
    if bridge is not None and hasattr(bridge, 'moe_output_reduce_sum'):
        result = torch.empty_like(outputs)
        bridge.moe_output_reduce_sum(result, outputs, weights,
                                     None, None, scaling_factor)
        return result
    raise RuntimeError("moe_output_reduce_sum: no C++ implementation available")


# =====================================================================
# Attention dispatchers
# =====================================================================

def paged_attention_v1(out: torch.Tensor, query: torch.Tensor,
                       key_cache: torch.Tensor, value_cache: torch.Tensor,
                       num_kv_heads: int, scale: float,
                       block_tables: torch.Tensor,
                       context_lens: torch.Tensor,
                       block_size: int, max_context_len: int,
                       **kwargs):
    """Paged attention v1 via ixformer::infer."""
    bridge = get_bridge()
    if bridge is not None and hasattr(bridge, 'paged_attention'):
        return bridge.paged_attention(
            out, query, key_cache, value_cache,
            num_kv_heads, scale, block_tables, context_lens,
            block_size, max_context_len,
            kwargs.get('alibi_slopes'), True,
            kwargs.get('window_left', -1), kwargs.get('window_right', -1),
            kwargs.get('softcap', 0.0), False, False, None)
    try:
        import ixformer.functions as ixf_F
        return ixf_F.vllm_single_query_cached_kv_attention(
            out, query, key_cache, value_cache,
            num_kv_heads, scale, block_tables, context_lens,
            block_size, max_context_len,
            kwargs.get('alibi_slopes'))
    except (ImportError, AttributeError):
        pass
    raise RuntimeError("paged_attention_v1: no C++ implementation available")


def flash_attn_with_block_tables(query: torch.Tensor,
                                 key_cache: torch.Tensor,
                                 value_cache: torch.Tensor,
                                 block_tables: torch.Tensor,
                                 cu_seq_q: torch.Tensor,
                                 cu_seq_k: torch.Tensor,
                                 max_seq_q: int, max_seq_k: int,
                                 scale: float, **kwargs):
    """Flash attention with block tables via ixformer::infer."""
    bridge = get_bridge()
    if bridge is not None and hasattr(bridge, 'flash_attn_with_block_tables'):
        out = torch.empty_like(query)
        return bridge.flash_attn_with_block_tables(
            query, key_cache, value_cache, out, block_tables,
            cu_seq_q, cu_seq_k, max_seq_q, max_seq_k,
            True, -1, -1, scale, 0.0, False, None, None, None)
    try:
        import ixformer.functions as ixf_F
        out = torch.empty_like(query)
        return ixf_F.ixinfer_flash_attn_unpad_with_block_tables(
            query, key_cache, value_cache, out, block_tables,
            cu_seq_q, cu_seq_k, max_seq_q, max_seq_k,
            True, -1, -1, scale, 0.0, False, None, None, None)
    except (ImportError, AttributeError):
        pass
    raise RuntimeError("flash_attn_with_block_tables: no C++ implementation available")


# =====================================================================
# Availability check
# =====================================================================

def check_availability():
    """Report which ops are available through the C++ bridge."""
    bridge = get_bridge()
    ops = [
        'silu_and_mul', 'rms_norm', 'residual_rms_norm',
        'rotary_embedding', 'reshape_and_cache',
        'topk_softmax', 'moe_compute_token_index', 'moe_expand_input',
        'moe_w16a16_group_gemm', 'moe_output_reduce_sum',
        'paged_attention', 'flash_attn_with_block_tables',
    ]
    available = {}
    for op in ops:
        available[op] = bridge is not None and hasattr(bridge, op)
    return available


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    avail = check_availability()
    print("ix_ops_dispatch availability:")
    for op, ok in avail.items():
        print(f"  {op}: {'✓' if ok else '✗'}")
    total = sum(avail.values())
    print(f"\n{total}/{len(avail)} ops available via C++ bridge")
