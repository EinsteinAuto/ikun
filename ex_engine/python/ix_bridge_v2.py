"""
ix_bridge_v2.py — Complete ixformer bridge loader (14 functions).

Loads ix_full_bridge_v2.so via JIT compilation, linking against ALL
ixformer .so files in the base image.

Functions exposed:
  MoE:       topk_softmax, moe_gen_idx, moe_expand_input, group_gemm,
             silu_and_mul, moe_combine_result, fused_moe_forward
  Attention: paged_attention, flash_attn_prefill
  Norm:      rms_norm, fused_add_rms_norm
  RoPE:      rotary_embedding
  Cache:     reshape_and_cache
  Linear:    linear
"""

import os
import logging
import glob
import torch
from typing import Tuple, Optional, List

logger = logging.getLogger("ex_engine.ix_bridge_v2")

_bridge = None
_loaded = False
_available = False


def _find_cpp():
    """Find ix_full_bridge_v2.cpp in known locations."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "csrc", "ix_full_bridge_v2.cpp"),
        os.path.join("/workspace/ex_engine/csrc", "ix_full_bridge_v2.cpp"),
        # fallback to v1
        os.path.join(here, "..", "csrc", "ix_full_bridge.cpp"),
        os.path.join("/workspace/ex_engine/csrc", "ix_full_bridge.cpp"),
    ]
    for c in candidates:
        p = os.path.normpath(c)
        if os.path.exists(p):
            return p
    return None


def _collect_ixformer_libs():
    """Collect all ixformer .so files for linking."""
    extra_ldflags = []
    rpath_dirs = set()

    # From ixformer Python package
    try:
        import ixformer
        ixf_dir = os.path.dirname(ixformer.__file__)
        for so in glob.glob(os.path.join(ixf_dir, "*.so")):
            extra_ldflags.append(so)
            rpath_dirs.add(os.path.dirname(so))
        # Also the _ixformer_torch extension
        for so in glob.glob(os.path.join(ixf_dir, "_ixformer_torch*.so")):
            if so not in extra_ldflags:
                extra_ldflags.append(so)
    except ImportError:
        pass

    # From corex lib64
    corex_lib = "/usr/local/corex/lib64"
    if os.path.isdir(corex_lib):
        for lib in ["libixattn.so", "libixformer.so", "libcublas.so",
                     "libcudart.so", "libcudnn.so"]:
            p = os.path.join(corex_lib, lib)
            if os.path.exists(p) and p not in extra_ldflags:
                extra_ldflags.append(p)
                rpath_dirs.add(corex_lib)

    # From ixformer subdirectory
    ixf_subdir = os.path.join(corex_lib, "python3/dist-packages/ixformer")
    if os.path.isdir(ixf_subdir):
        for so in glob.glob(os.path.join(ixf_subdir, "*.so")):
            if so not in extra_ldflags:
                extra_ldflags.append(so)
                rpath_dirs.add(ixf_subdir)

    # Add rpath
    for d in rpath_dirs:
        extra_ldflags.append(f"-Wl,-rpath,{d}")

    return extra_ldflags


def _load_bridge():
    """JIT compile and load the bridge."""
    global _bridge, _loaded, _available
    if _loaded:
        return _available
    _loaded = True

    cpp_path = _find_cpp()
    if cpp_path is None:
        logger.warning("ix_full_bridge_v2.cpp not found")
        return False

    extra_ldflags = _collect_ixformer_libs()
    logger.info("ix_bridge_v2: compiling %s", cpp_path)
    logger.info("ix_bridge_v2: ldflags count=%d", len(extra_ldflags))

    try:
        from torch.utils.cpp_extension import load
        mod_name = "ix_full_bridge_v2" if "v2" in cpp_path else "ix_full_bridge"
        _bridge = load(
            name=mod_name,
            sources=[cpp_path],
            extra_cflags=["-O2", "-std=c++17"],
            extra_ldflags=extra_ldflags,
            verbose=False,
        )
        _available = True
        fns = [x for x in dir(_bridge) if not x.startswith("_")]
        logger.info("ix_bridge_v2 loaded: %s", fns)
        return True
    except Exception as e:
        logger.error("ix_bridge_v2 JIT compile failed: %s", e)
        return False


def is_available() -> bool:
    if not _loaded:
        _load_bridge()
    return _available


def _get():
    if not is_available():
        raise RuntimeError("ix_bridge_v2 not available")
    return _bridge


# =========================================================================
# MoE
# =========================================================================
def topk_softmax(gating_output, topk, renormalize=True):
    """Returns (topk_weights, topk_ids, token_expert_indices)."""
    return _get().topk_softmax(gating_output, topk, renormalize)

def moe_gen_idx(expert_id, expert_num):
    """Returns [src_dst, dst_src, expert_sizes_gpu, expert_sizes_cumsum]."""
    return _get().moe_gen_idx(expert_id, expert_num)

def moe_expand_input(input, gather_index, combine_idx, topk):
    return _get().moe_expand_input(input, gather_index, combine_idx, topk)

def group_gemm(inputs, weights, token_count, output_n):
    return _get().group_gemm(inputs, weights, token_count, output_n)

def silu_and_mul(input):
    return _get().silu_and_mul(input)

def moe_combine_result(input, weight):
    return _get().moe_combine_result(input, weight)

def fused_moe_forward(hidden_states, router_logits, w13, w2,
                      topk, num_experts, renormalize=True):
    return _get().fused_moe_forward(
        hidden_states, router_logits, w13, w2, topk, num_experts, renormalize)

# =========================================================================
# Attention
# =========================================================================
def paged_attention(output, query, key_cache, value_cache,
                    num_kv_heads, scale, block_tables, seq_lens,
                    block_size, max_context_len, alibi_slopes=None):
    return _get().paged_attention(
        output, query, key_cache, value_cache,
        num_kv_heads, scale, block_tables, seq_lens,
        block_size, max_context_len, alibi_slopes)

def flash_attn_prefill(query, key_cache, value_cache, output, block_tables,
                       cu_seq_q, cu_seq_k, max_query_len, max_seq_len,
                       scale, is_causal=True, window_left=-1, window_right=-1):
    return _get().flash_attn_prefill(
        query, key_cache, value_cache, output, block_tables,
        cu_seq_q, cu_seq_k, max_query_len, max_seq_len,
        scale, is_causal, window_left, window_right)

# =========================================================================
# Norm
# =========================================================================
def rms_norm(output, input, weight, eps=1e-6):
    return _get().rms_norm(output, input, weight, eps)

def fused_add_rms_norm(input, residual, weight, output, residual_output, eps=1e-6):
    return _get().fused_add_rms_norm(input, residual, weight, output, residual_output, eps)

# =========================================================================
# RoPE
# =========================================================================
def rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox=True):
    return _get().rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox)

# =========================================================================
# Cache
# =========================================================================
def reshape_and_cache(key, value, key_cache, value_cache, slot_mapping):
    return _get().reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

# =========================================================================
# Linear
# =========================================================================
def linear(input, weight, bias=None):
    return _get().linear(input, weight, bias)
