"""
xllm_ops.py — NO-FALLBACK xllm kernel loader for vllm hot path

Function name mapping (verified via `nm -D` + `strings` on real BI-V100):
    xllm_cache.so:      reshape_paged_cache   (NOT reshape_and_cache)
    xllm_norm.so:       rms_norm, fused_add_rms_norm  (NOT residual_rms_norm)
    xllm_moe.so:        moe_fused_topk        (NOT topk_softmax)
    xllm_moe.so:        moe_compute_index     (NOT moe_compute_token_index)
    ix_moe_bridge.so:   ix_paged_attention, ix_linear  (NOT in ix_full_bridge.so)

C++ argument order verified against *_bind.cpp pybind11 source:
    xllm_norm_bind.cpp:       rms_norm(output, input, weight, eps)
    xllm_activation_bind.cpp: silu_and_mul(out, input)
    xllm_cache_bind.cpp:      reshape_paged_cache(slot_ids, keys, values, kc, vc)
    ix_full_bridge_v2.cpp:    ix_paged_attention(out, q, kc, vc, head_mapping, scale, ...)
    xllm_moe_bind.cpp:        moe_fused_topk(gating, topk) → returns (w, ids)

NO FALLBACK: If a .so fails to load, we raise immediately.
"""

import os
import sys
import importlib.util
import logging

import torch
from typing import Optional, Dict, Any

logger = logging.getLogger("ex_engine.xllm_ops")

# =========================================================================
# .so search paths
# =========================================================================
_SEARCH_DIRS = []

def _init_search_dirs():
    """Build list of directories to search for .so files."""
    global _SEARCH_DIRS
    if _SEARCH_DIRS:
        return

    here = os.path.dirname(os.path.abspath(__file__))

    # 1. vllm package dir (deployed by patch_ops.sh)
    try:
        import vllm
        _SEARCH_DIRS.append(os.path.dirname(vllm.__file__))
    except ImportError:
        pass

    # 2. prebuilt dir
    _SEARCH_DIRS.append(os.path.join(here, "..", "..", "qwen3_6_scripts",
                                      "prebuilt", "corex-3.2.3-ivcore10"))

    # 3. build output dir
    _SEARCH_DIRS.append(os.path.join(here, "..", "build"))

    # 4. /workspace paths (inside docker)
    _SEARCH_DIRS.append("/workspace/qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10")
    _SEARCH_DIRS.append("/workspace/ex_engine/build")

    # Normalize
    _SEARCH_DIRS = [os.path.normpath(d) for d in _SEARCH_DIRS if os.path.isdir(d)]


def _load_so(name: str) -> Any:
    """Load a .so by name. Raises RuntimeError if not found."""
    _init_search_dirs()

    for d in _SEARCH_DIRS:
        path = os.path.join(d, f"{name}.so")
        if not os.path.isfile(path):
            continue
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            fns = [x for x in dir(mod) if not x.startswith("_")]
            logger.info("xllm_ops: loaded %s from %s (%d functions: %s)",
                       name, path, len(fns), ", ".join(fns[:8]))
            return mod
        except Exception as e:
            logger.warning("xllm_ops: %s at %s failed: %s", name, path, e)
            continue

    raise RuntimeError(
        f"xllm_ops: CANNOT load {name}.so — searched {_SEARCH_DIRS}. "
        f"Build with: bash ex_engine/build_xllm_kernels.sh"
    )


# =========================================================================
# Module registry — lazy-loaded, no fallback
# =========================================================================
_modules: Dict[str, Any] = {}

def _get(name: str) -> Any:
    if name not in _modules:
        _modules[name] = _load_so(name)
    return _modules[name]


# =========================================================================
# Public API — C++ signatures verified against *_bind.cpp pybind source
# =========================================================================

# --- Norm (xllm_norm.so) ---
# C++ rms_norm(output, input, weight, eps) — output FIRST
def rms_norm(input, weight, epsilon):
    """RMSNorm. C++ takes (output, input, weight, eps)."""
    output = torch.empty_like(input)
    _get("xllm_norm").rms_norm(output, input, weight, epsilon)
    return output

# C++ fused_add_rms_norm(input&, residual&, weight&, epsilon) — in-place
def residual_rms_norm(input, residual, weight, epsilon):
    """Fused residual + RMSNorm. Modifies input and residual in-place."""
    _get("xllm_norm").fused_add_rms_norm(input, residual, weight, epsilon)
    return input, residual

# --- RoPE (xllm_rope.so) ---
# C++ rotary_embedding(positions, query, key, cos_sin_cache, is_neox)
def rotary_embedding(positions, query, key, cos_sin_cache, is_neox=True):
    """Fused rotary embedding. Signature matches C++ directly."""
    return _get("xllm_rope").rotary_embedding(positions, query, key,
                                               cos_sin_cache, is_neox)

# --- Activation (xllm_activation.so) ---
# C++ silu_and_mul(out, input) — out FIRST
def silu_and_mul(input, output=None):
    """Fused SiLU activation. C++ takes (out, input)."""
    if output is None:
        d = input.shape[-1] // 2
        output = torch.empty(*input.shape[:-1], d, dtype=input.dtype,
                             device=input.device)
    _get("xllm_activation").silu_and_mul(output, input)
    return output

# C++ gelu_and_mul(out, input) — out FIRST
def gelu_and_mul(input, output=None):
    """Fused GeLU activation. C++ takes (out, input)."""
    if output is None:
        d = input.shape[-1] // 2
        output = torch.empty(*input.shape[:-1], d, dtype=input.dtype,
                             device=input.device)
    _get("xllm_activation").gelu_and_mul(output, input)
    return output

# --- Cache (xllm_cache.so) ---
# C++ reshape_paged_cache(slot_ids, keys, values, key_cache, value_cache)
#   — slot_ids FIRST (not last!)
#   — slot_ids must be int32 (C++ uses data_ptr<int>), vllm passes int64
def reshape_and_cache(key, value, key_cache, value_cache, slot_mapping):
    """Write KV to paged cache. C++ takes slot_ids as FIRST arg, dtype=int32."""
    slot_mapping_i32 = slot_mapping.to(torch.int32)
    return _get("xllm_cache").reshape_paged_cache(slot_mapping_i32, key, value,
                                                   key_cache, value_cache)

# --- Attention (ix_moe_bridge.so) ---
# C++ ix_paged_attention(output, query, key_cache, value_cache,
#                        head_mapping, scale, block_tables, context_lens,
#                        block_size, max_context_len, num_kv_heads,
#                        alibi_slopes)
def paged_attention(out, query, key_cache, value_cache,
                    num_kv_heads, scale, block_tables, context_lens,
                    block_size, max_context_len, alibi_slopes=None):
    """Paged attention decode. C++ needs head_mapping tensor at position 5."""
    bridge = _get("ix_moe_bridge")
    num_q_heads = query.shape[1]
    head_mapping = torch.arange(num_q_heads, dtype=torch.int32,
                                device=query.device)
    if num_kv_heads != num_q_heads:
        head_mapping = head_mapping // (num_q_heads // num_kv_heads)
    return bridge.paged_attention(
        out, query, key_cache, value_cache,
        head_mapping, scale, block_tables, context_lens,
        block_size, max_context_len, num_kv_heads, alibi_slopes
    )

def flash_attn_prefill(query, key_cache, value_cache, out,
                       block_tables, cu_seq_q, cu_seq_k,
                       max_seq_q, max_seq_k, scale,
                       is_causal=True):
    """Flash attention prefill. .so export: fused_paged_prefill_forward."""
    return _get("corex_fused_paged_prefill").fused_paged_prefill_forward(
        query, key_cache, value_cache, out,
        block_tables, cu_seq_q, max_seq_q, scale
    )

# --- MoE (xllm_moe.so) ---
# C++ moe_fused_topk(gating_output, topk, renormalize=true,
#                     correction_bias=None, scoring_func="softmax")
#   → returns (topk_weights, topk_ids)  (C++ allocates internally)
def topk_softmax(topk_weights, topk_ids, token_expert_ids, gating_output, topk):
    """MoE topk+softmax. C++ returns new tensors; we copy into pre-allocated."""
    weights, ids = _get("xllm_moe").moe_fused_topk(gating_output, topk)
    topk_weights.copy_(weights)
    topk_ids.copy_(ids)
    return topk_weights, topk_ids, token_expert_ids

# C++ moe_compute_index(expert_id, num_experts)
#   → returns (sorted_token_ids, expert_ids, num_tokens_post_padded)
def moe_compute_token_index(sorted_token_ids, expert_ids, num_tokens_post_padded,
                            token_expert_ids, num_experts, block_size):
    """MoE token routing. C++ takes only (expert_id, num_experts)."""
    s_ids, e_ids, n_post = _get("xllm_moe").moe_compute_index(
        token_expert_ids, num_experts
    )
    sorted_token_ids.copy_(s_ids[:sorted_token_ids.numel()].reshape_as(sorted_token_ids))
    expert_ids.copy_(e_ids[:expert_ids.numel()].reshape_as(expert_ids))
    num_tokens_post_padded.copy_(n_post[:num_tokens_post_padded.numel()].reshape_as(num_tokens_post_padded))
    return sorted_token_ids, expert_ids, num_tokens_post_padded

# --- Linear (ix_moe_bridge.so: ix_linear) ---
def ixformer_linear(input, weight, act_type=0, bias=None, out=None):
    """GEMM via ixformer. .so export: ix_linear in ix_moe_bridge.so."""
    bridge = _get("ix_moe_bridge")
    return bridge.linear(input, weight, bias)

# --- Fused QK-Norm + RoPE ---
def fused_qknorm_rope(query, key, cos_sin_cache, positions,
                      qk_norm_weight, epsilon, interleave=False):
    """Fused QK normalization + rotary embedding (saves 128 kernel launches)."""
    return _get("xllm_fused_qknorm_rope").fused_qknorm_rope(
        query, key, cos_sin_cache, positions, qk_norm_weight, epsilon, interleave
    )


# =========================================================================
# Availability check — call at startup to verify ALL .so are loadable
# =========================================================================
def check_all(strict=True):
    """Verify all required .so files are loadable.

    Args:
        strict: If True, raise on any missing .so (NO FALLBACK mode).
                If False, return dict of {name: loaded_bool}.
    """
    required = [
        "ix_moe_bridge",      # attention (ix_paged_attention) + linear (ix_linear)
        "xllm_norm",          # rms_norm, fused_add_rms_norm
        "xllm_cache",         # reshape_paged_cache
        "xllm_moe",           # moe_fused_topk, moe_compute_index
    ]

    optional = [
        "ix_full_bridge",          # legacy bridge (not used in hot path)
        "xllm_rope",               # rotary_embedding
        "xllm_activation",         # silu_and_mul
        "xllm_fused_qknorm_rope",  # fused QK-norm + RoPE
        "corex_fused_paged_prefill",  # flash attention prefill
    ]

    results = {}
    missing = []

    for name in required:
        try:
            _get(name)
            results[name] = True
        except RuntimeError:
            results[name] = False
            missing.append(name)

    for name in optional:
        try:
            _get(name)
            results[name] = True
        except RuntimeError:
            results[name] = False
            logger.info("xllm_ops: optional %s not available", name)

    if strict and missing:
        raise RuntimeError(
            f"xllm_ops: {len(missing)} required .so MISSING: {missing}. "
            f"Score will be ~683 without these. Build with: "
            f"bash ex_engine/build_xllm_kernels.sh"
        )

    loaded = sum(1 for v in results.values() if v)
    total = len(results)
    logger.info("xllm_ops: %d/%d .so loaded", loaded, total)

    return results