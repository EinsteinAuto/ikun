"""
ix_ops.py — Drop-in operator replacements via ix_moe_bridge.so

Architecture (CCCL dispatch pattern):
    CCCL:  compute_capability → policy_selector → tuned_kernel
    EX:    base_image_so     → ix_moe_bridge   → ixformer::infer

This module provides torch.nn.Module-compatible replacements for:
    1. RMSNorm         → residual_rms_norm / rms_norm   (fused kernel)
    2. SiluAndMul      → silu_and_mul                   (fused activation)
    3. RotaryEmbedding → xllm_rotary_embedding          (fused RoPE)
    4. reshape_and_cache → xllm_reshape_and_cache       (fused KV write)
    5. paged_attention → xllm_paged_attention            (fused decode attn)
    6. flash_attn_prefill → ixinfer_flash_attn_unpad     (fused prefill attn)
    7. linear          → ixformer_linear / linear_ex     (GEMM)

Loading: tries prebuilt ix_moe_bridge.so first, then JIT-compiles
ix_moe_bridge_v2.cpp as fallback.

Source mapping:
    upstream_ref/xllm_latest/core/kernels/ilu/*.cpp  → this file (Python side)
    ex_engine/csrc/ix_moe_bridge_v2.cpp              → .so (C++ side)
    ixformer::infer namespace (base image)            → actual CUDA kernels
"""

import os
import sys
import logging
import importlib
import importlib.util
import glob
import torch
from typing import Optional, Tuple, List

logger = logging.getLogger("ex_engine.ix_ops")

# =========================================================================
# Bridge loader
# =========================================================================
_bridge = None
_loaded = False
_available = False


def _try_prebuilt():
    """Load prebuilt ix_moe_bridge.so."""
    search = [
        # Deployed by patch_ops.sh into vllm package
        "/usr/local/corex/lib/python3/dist-packages/vllm/ix_moe_bridge.so",
    ]
    # Also check vllm package dir
    try:
        import vllm
        vd = os.path.dirname(vllm.__file__)
        search.insert(0, os.path.join(vd, "ix_moe_bridge.so"))
    except ImportError:
        pass
    # Check prebuilt dir
    here = os.path.dirname(os.path.abspath(__file__))
    search.append(os.path.join(here, "..", "..", "qwen3_6_scripts", "prebuilt",
                               "corex-3.2.3-ivcore10", "ix_moe_bridge.so"))

    for path in search:
        path = os.path.normpath(path)
        if not os.path.isfile(path):
            continue
        try:
            spec = importlib.util.spec_from_file_location("ix_moe_bridge", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            fns = [x for x in dir(mod) if not x.startswith("_")]
            logger.info("ix_ops: loaded prebuilt %s: %s", path, fns)
            return mod
        except Exception as e:
            logger.debug("ix_ops: prebuilt %s failed: %s", path, e)
    return None


def _try_jit():
    """JIT compile ix_moe_bridge_v2.cpp."""
    here = os.path.dirname(os.path.abspath(__file__))
    cpp_candidates = [
        os.path.join(here, "..", "csrc", "ix_moe_bridge_v2.cpp"),
        os.path.join(here, "..", "csrc", "ix_moe_bridge.cpp"),
        "/workspace/ex_engine/csrc/ix_moe_bridge_v2.cpp",
        "/workspace/qwen3_6_scripts/ix_moe_bridge_v2.cpp",
    ]
    cpp_file = None
    for c in cpp_candidates:
        c = os.path.normpath(c)
        if os.path.isfile(c):
            cpp_file = c
            break
    if cpp_file is None:
        return None

    extra_ldflags = []
    # Link ixformer .so libraries
    try:
        import ixformer
        ixf_dir = os.path.dirname(ixformer.__file__)
        for so in glob.glob(os.path.join(ixf_dir, "*.so")):
            extra_ldflags.append(so)
        extra_ldflags.append(f"-Wl,-rpath,{ixf_dir}")
    except ImportError:
        pass
    # Also link corex libraries
    corex_lib = "/usr/local/corex/lib64"
    if os.path.isdir(corex_lib):
        for lib in ["libixattn.so", "libixformer.so", "libcublas.so"]:
            p = os.path.join(corex_lib, lib)
            if os.path.isfile(p):
                extra_ldflags.append(p)
        extra_ldflags.append(f"-Wl,-rpath,{corex_lib}")

    try:
        from torch.utils.cpp_extension import load
        logger.info("ix_ops: JIT compiling %s", cpp_file)
        mod = load(
            name="ix_moe_bridge_v2",
            sources=[cpp_file],
            extra_cflags=["-O2", "-std=c++17"],
            extra_ldflags=extra_ldflags,
            verbose=False,
        )
        fns = [x for x in dir(mod) if not x.startswith("_")]
        logger.info("ix_ops: JIT compiled: %s", fns)
        return mod
    except Exception as e:
        logger.warning("ix_ops: JIT compile failed: %s", e)
        return None


def _ensure_loaded():
    global _bridge, _loaded, _available
    if _loaded:
        return _available
    _loaded = True
    _bridge = _try_prebuilt()
    if _bridge is None:
        _bridge = _try_jit()
    _available = _bridge is not None
    if _available:
        logger.info("ix_ops: bridge available with %d functions",
                     len([x for x in dir(_bridge) if not x.startswith("_")]))
    else:
        logger.warning("ix_ops: bridge NOT available, all ops will be no-op")
    return _available


def is_available() -> bool:
    return _ensure_loaded()


def get_bridge():
    if not _ensure_loaded():
        raise RuntimeError("ix_ops bridge not available")
    return _bridge


# =========================================================================
# Feature probes — check what the loaded bridge supports
# =========================================================================
def has_silu_and_mul() -> bool:
    return is_available() and hasattr(_bridge, "silu_and_mul")

def has_rms_norm() -> bool:
    return is_available() and hasattr(_bridge, "rms_norm")

def has_fused_add_rms_norm() -> bool:
    return is_available() and hasattr(_bridge, "fused_add_rms_norm")

def has_rotary_embedding() -> bool:
    return is_available() and hasattr(_bridge, "rotary_embedding")

def has_reshape_and_cache() -> bool:
    return is_available() and hasattr(_bridge, "reshape_and_cache")

def has_paged_attention() -> bool:
    return is_available() and hasattr(_bridge, "paged_attention")

def has_flash_attn_prefill() -> bool:
    return is_available() and hasattr(_bridge, "flash_attn_prefill")

def has_linear() -> bool:
    return is_available() and hasattr(_bridge, "linear")

def has_topk_softmax() -> bool:
    return is_available() and hasattr(_bridge, "topk_softmax")

def has_fused_moe_forward() -> bool:
    return is_available() and hasattr(_bridge, "fused_moe_forward")


# =========================================================================
# Op wrappers — match xllm upstream signatures
# Source: upstream_ref/xllm_latest/core/kernels/ilu/*.cpp
# =========================================================================

def silu_and_mul(input: torch.Tensor) -> torch.Tensor:
    """Fused SiLU activation + element-wise multiply.

    Source: xllm/core/kernels/ilu/activation.cpp → infer::silu_and_mul
    input: (T, 2*I) → output: (T, I)
    """
    return _bridge.silu_and_mul(input)


def rms_norm(output: torch.Tensor, input: torch.Tensor,
             weight: torch.Tensor, eps: float = 1e-6) -> None:
    """RMSNorm: output = rms_norm(input, weight, eps).

    Source: xllm/core/kernels/ilu/norm.cpp → infer::rms_norm
    """
    _bridge.rms_norm(output, input, weight, eps)


def fused_add_rms_norm(input: torch.Tensor, residual: torch.Tensor,
                       weight: torch.Tensor, output: torch.Tensor,
                       residual_output: torch.Tensor,
                       eps: float = 1e-6) -> None:
    """Fused residual addition + RMSNorm.

    Source: xllm/core/kernels/ilu/norm.cpp → infer::residual_rms_norm
    The C++ function is in-place: modifies input → rms_norm(input+residual)*weight,
    and residual → input+residual. We copy results to output/residual_output.
    """
    # C++ signature: fused_add_rms_norm_forward(input, residual, weight, eps, alpha)
    # It modifies input and residual in-place.
    inp_clone = input.clone()
    res_clone = residual.clone()
    _bridge.fused_add_rms_norm(inp_clone, res_clone, weight, eps)
    output.copy_(inp_clone)
    residual_output.copy_(res_clone)


def rotary_embedding(positions: torch.Tensor, query: torch.Tensor,
                     key: torch.Tensor, head_size: int,
                     cos_sin_cache: torch.Tensor,
                     is_neox: bool = True) -> None:
    """Fused rotary position embedding (in-place on query and key).

    Source: xllm/core/kernels/ilu/rope.cpp → infer::xllm_rotary_embedding
    """
    _bridge.rotary_embedding(positions, query, key, head_size,
                              cos_sin_cache, is_neox)


def reshape_and_cache(key: torch.Tensor, value: torch.Tensor,
                      key_cache: torch.Tensor, value_cache: torch.Tensor,
                      slot_mapping: torch.Tensor) -> None:
    """Write KV to paged cache.

    Source: xllm/core/kernels/ilu/attention.cpp → infer::xllm_reshape_and_cache
    """
    _bridge.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)


def paged_attention(output: torch.Tensor, query: torch.Tensor,
                    key_cache: torch.Tensor, value_cache: torch.Tensor,
                    num_kv_heads: int, scale: float,
                    block_tables: torch.Tensor, seq_lens: torch.Tensor,
                    block_size: int, max_context_len: int,
                    alibi_slopes: Optional[torch.Tensor] = None
                    ) -> torch.Tensor:
    """Paged attention decode.

    Source: xllm/core/kernels/ilu/attention.cpp → infer::xllm_paged_attention
    """
    return _bridge.paged_attention(
        output, query, key_cache, value_cache,
        num_kv_heads, scale, block_tables, seq_lens,
        block_size, max_context_len, alibi_slopes)


def flash_attn_prefill(query: torch.Tensor, key_cache: torch.Tensor,
                       value_cache: torch.Tensor, output: torch.Tensor,
                       block_tables: torch.Tensor,
                       cu_seq_q: torch.Tensor, cu_seq_k: torch.Tensor,
                       max_query_len: int, max_seq_len: int,
                       scale: float, is_causal: bool = True,
                       window_left: int = -1,
                       window_right: int = -1) -> torch.Tensor:
    """Flash attention prefill with paged KV cache.

    Source: xllm/core/kernels/ilu/attention.cpp →
            infer::ixinfer_flash_attn_unpad_with_block_tables
    """
    return _bridge.flash_attn_prefill(
        query, key_cache, value_cache, output, block_tables,
        cu_seq_q, cu_seq_k, max_query_len, max_seq_len,
        scale, is_causal, window_left, window_right)


def linear(input: torch.Tensor, weight: torch.Tensor,
           bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """GEMM via ixformer (auto-selects linear vs linear_ex).

    Source: xllm/core/kernels/ilu/matmul.cpp → infer::ixformer_linear[_ex]
    """
    return _bridge.linear(input, weight, bias)


# =========================================================================
# MoE ops — full 7-step pipeline
# Source: xllm/core/layers/ilu/fused_moe.cpp
# =========================================================================
def topk_softmax(gating_output: torch.Tensor, topk: int,
                 renormalize: bool = True):
    """Fused topk + softmax routing."""
    return _bridge.topk_softmax(gating_output, topk, renormalize)


def moe_gen_idx(expert_id: torch.Tensor, expert_num: int):
    """Build expert permutation maps."""
    return _bridge.moe_gen_idx(expert_id, expert_num)


def moe_expand_input(input: torch.Tensor, gather_index: torch.Tensor,
                     combine_idx: torch.Tensor, topk: int):
    """Expand input tokens by expert assignment."""
    return _bridge.moe_expand_input(input, gather_index, combine_idx, topk)


def group_gemm(inputs: torch.Tensor, weights: torch.Tensor,
               token_count: torch.Tensor, output_n: int):
    """Batched expert GEMM."""
    return _bridge.group_gemm(inputs, weights, token_count, output_n)


def moe_combine_result(input: torch.Tensor, weight: torch.Tensor):
    """Weighted scatter-back of expert outputs."""
    return _bridge.moe_combine_result(input, weight)


def fused_moe_forward(hidden_states: torch.Tensor,
                      router_logits: torch.Tensor,
                      w13: torch.Tensor, w2: torch.Tensor,
                      topk: int, num_experts: int,
                      renormalize: bool = True) -> torch.Tensor:
    """Full fused MoE forward (7-step pipeline).

    Source: xllm/core/layers/ilu/fused_moe.cpp → FusedMoEImpl::forward_experts
    Pipeline: topk → gen_idx → expand → gemm1(w13) → silu → gemm2(w2) → combine
    """
    return _bridge.fused_moe_forward(
        hidden_states, router_logits, w13, w2,
        topk, num_experts, renormalize)
