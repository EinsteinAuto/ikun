"""
ix_bridge.py — Full ixformer bridge loader.

Loads ix_full_bridge.so (all 14 ixformer::infer functions) or falls back
to ix_moe_bridge.so (MoE-only 6 functions).

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
import torch
from typing import Tuple, Optional, List

logger = logging.getLogger("ex_engine.ix_bridge")

_bridge = None
_loaded = False
_available = False

# All .cpp sources to try, in priority order
_CPP_NAMES = ["ix_full_bridge.cpp", "ix_moe_bridge.cpp"]


def _find_cpp(name):
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "csrc", name),
        os.path.join(here, name),
        os.path.join("/workspace/ex_engine/csrc", name),
        os.path.join("/workspace/qwen3_6_scripts", name),
    ]
    for c in candidates:
        p = os.path.normpath(c)
        if os.path.exists(p):
            return p
    return None


def _load_bridge():
    global _bridge, _loaded, _available
    if _loaded:
        return _available
    _loaded = True

    from torch.utils.cpp_extension import load
    import glob

    # Find ixformer .so libraries to link against
    extra_ldflags = []
    ixf_lib_dirs = set()
    try:
        import ixformer
        ixf_dir = os.path.dirname(ixformer.__file__)
        # Link against all .so in the ixformer package
        for so in glob.glob(os.path.join(ixf_dir, "*.so")):
            if "cpython" not in so:  # skip the Python extension .so
                extra_ldflags.append(so)
                ixf_lib_dirs.add(os.path.dirname(so))
        # Also try the _C and _ixformer_torch extensions
        for so in glob.glob(os.path.join(ixf_dir, "_ixformer_torch*.so")):
            extra_ldflags.append(so)
    except ImportError:
        pass

    # Also check /usr/local/corex/lib64 for libixattn etc
    corex_lib = "/usr/local/corex/lib64"
    if os.path.isdir(corex_lib):
        for lib in ["libixattn.so", "libixformer.so", "libcublas.so"]:
            p = os.path.join(corex_lib, lib)
            if os.path.exists(p) and p not in extra_ldflags:
                extra_ldflags.append(p)
                ixf_lib_dirs.add(corex_lib)

    # Add rpath so the .so can find its dependencies at runtime
    for d in ixf_lib_dirs:
        extra_ldflags.append(f"-Wl,-rpath,{d}")

    logger.info("ix_bridge extra_ldflags: %s", extra_ldflags)

    for cpp_name in _CPP_NAMES:
        cpp_path = _find_cpp(cpp_name)
        if cpp_path is None:
            continue
        mod_name = cpp_name.replace(".cpp", "").replace(".", "_")
        try:
            logger.info("JIT-compiling %s from %s ...", cpp_name, cpp_path)
            _bridge = load(
                name=mod_name,
                sources=[cpp_path],
                extra_cflags=["-O2", "-std=c++17"],
                extra_ldflags=extra_ldflags,
                verbose=False,
            )
            _available = True
            fns = [x for x in dir(_bridge) if not x.startswith("_")]
            logger.info("ix_bridge loaded (%s): %s", cpp_name, fns)
            return True
        except Exception as e:
            logger.warning("JIT compile %s failed: %s — trying next", cpp_name, e)

    logger.warning("All ix_bridge sources failed to compile")
    return False


def is_available() -> bool:
    if not _loaded:
        _load_bridge()
    return _available


def _get():
    if not is_available():
        raise RuntimeError("ix_bridge not available")
    return _bridge


# =========================================================================
# MoE
# =========================================================================
def topk_softmax(gating_output, topk, renormalize=True):
    return _get().topk_softmax(gating_output, topk, renormalize)

def moe_gen_idx(expert_id, expert_num):
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

def flash_attn_prefill(query, key, value, output, block_tables,
                       cu_seq_q, cu_seq_k, max_query_len, max_seq_len,
                       scale, is_causal=True, window_left=-1, window_right=-1):
    return _get().flash_attn_prefill(
        query, key, value, output, block_tables,
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
