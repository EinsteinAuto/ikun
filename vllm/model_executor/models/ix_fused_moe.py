"""
ix_fused_moe.py — Fused MoE pipeline via ixformer C++ API

Replaces the entire Python expert loop in qwen3_5.py with xllm's 7-step
fused pipeline:
    topk_softmax → gen_idx → expand → group_gemm → silu → group_gemm → combine

Source: upstream_ref/xllm_latest/core/layers/ilu/fused_moe.cpp
Bridge: ex_engine/csrc/ix_moe_bridge.cpp → ixformer::infer namespace

Loading strategy:
    1. Try prebuilt ix_moe_bridge.so from known locations
    2. Try JIT compile from .cpp source
    3. Return unavailable (caller falls back to Python loop)
"""

import os
import logging
import importlib
import torch

logger = logging.getLogger("ix_fused_moe")

_bridge = None
_loaded = False


def _try_load_prebuilt():
    """Load prebuilt ix_moe_bridge.so without JIT compilation."""
    search_paths = [
        # Deployed by patch_ops.sh into vllm package
        os.path.join(os.path.dirname(__file__), "ix_moe_bridge.so"),
        # Prebuilt directory
        os.path.join(os.path.dirname(__file__), "prebuilt",
                     "corex-3.2.3-ivcore10", "ix_moe_bridge.so"),
        # Workspace deployment
        "/workspace/qwen3_6_scripts/ix_moe_bridge.so",
    ]

    # Also check the vllm package directory
    try:
        import vllm
        vllm_dir = os.path.dirname(vllm.__file__)
        search_paths.append(os.path.join(vllm_dir, "ix_moe_bridge.so"))
    except ImportError:
        pass

    for path in search_paths:
        if os.path.isfile(path):
            try:
                spec = importlib.util.spec_from_file_location(
                    "ix_moe_bridge", path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                fns = [x for x in dir(mod) if not x.startswith("_")]
                logger.info("ix_moe_bridge loaded from %s: %s", path, fns)
                return mod
            except Exception as e:
                logger.debug("Failed to load %s: %s", path, e)

    return None


def _try_jit_compile():
    """JIT compile ix_moe_bridge.cpp using torch.utils.cpp_extension."""
    import glob

    cpp_paths = [
        os.path.join(os.path.dirname(__file__), "..", "ex_engine",
                     "csrc", "ix_moe_bridge.cpp"),
        os.path.join(os.path.dirname(__file__), "ix_moe_bridge.cpp"),
        "/workspace/ex_engine/csrc/ix_moe_bridge.cpp",
        "/workspace/qwen3_6_scripts/ix_moe_bridge.cpp",
    ]

    cpp_file = None
    for p in cpp_paths:
        p = os.path.normpath(p)
        if os.path.isfile(p):
            cpp_file = p
            break

    if cpp_file is None:
        logger.debug("ix_moe_bridge.cpp not found in search paths")
        return None

    # Build link flags to find ixformer symbols
    extra_ldflags = []
    try:
        import ixformer
        ixf_dir = os.path.dirname(ixformer.__file__)
        for so in glob.glob(os.path.join(ixf_dir, "*.so")):
            extra_ldflags.append(so)
        extra_ldflags.append(f"-Wl,-rpath,{ixf_dir}")
    except ImportError:
        pass

    corex_lib = "/usr/local/corex/lib64"
    if os.path.isdir(corex_lib):
        for lib in ["libixattn.so", "libixformer.so", "libcublas.so"]:
            p = os.path.join(corex_lib, lib)
            if os.path.isfile(p):
                extra_ldflags.append(p)
        extra_ldflags.append(f"-Wl,-rpath,{corex_lib}")

    try:
        from torch.utils.cpp_extension import load
        logger.info("JIT compiling ix_moe_bridge from %s", cpp_file)
        mod = load(
            name="ix_moe_bridge",
            sources=[cpp_file],
            extra_cflags=["-O2", "-std=c++17"],
            extra_ldflags=extra_ldflags,
            verbose=False,
        )
        fns = [x for x in dir(mod) if not x.startswith("_")]
        logger.info("ix_moe_bridge JIT compiled: %s", fns)
        return mod
    except Exception as e:
        logger.warning("JIT compile failed: %s", e)
        return None


def _ensure_loaded():
    global _bridge, _loaded
    if _loaded:
        return _bridge is not None
    _loaded = True

    _bridge = _try_load_prebuilt()
    if _bridge is not None:
        return True

    _bridge = _try_jit_compile()
    if _bridge is not None:
        return True

    logger.info("ix_moe_bridge unavailable — MoE will use Python loop fallback")
    return False


def is_available():
    """Check if the fused MoE bridge is available."""
    return _ensure_loaded()


# =========================================================================
# Public API — matches xllm's 7-step pipeline
# =========================================================================

def fused_moe_forward(
    hidden_states: torch.Tensor,    # (T, H)
    router_logits: torch.Tensor,    # (T, E)
    w13: torch.Tensor,              # (E, 2*I, H)
    w2: torch.Tensor,               # (E, H, I)
    topk: int,
    num_experts: int,
    renormalize: bool = True,
) -> torch.Tensor:
    """Full fused MoE forward — replaces _pure_pytorch_experts().

    Pipeline (matching xllm/core/layers/ilu/fused_moe.cpp):
        1. topk_softmax    — router_logits → (weights, expert_ids)
        2. moe_gen_idx     — expert_ids → permutation maps
        3. moe_expand_input — gather tokens by expert
        4. group_gemm 1    — w13 projection (gate+up)
        5. silu_and_mul    — fused activation
        6. group_gemm 2    — w2 projection (down)
        7. combine_result  — weighted scatter back
    """
    if _bridge is None:
        raise RuntimeError("ix_fused_moe not loaded")
    return _bridge.fused_moe_forward(
        hidden_states, router_logits, w13, w2,
        topk, num_experts, renormalize)


def topk_softmax(gating_output, topk, renormalize=True):
    """Fused topk + softmax routing."""
    if _bridge is None:
        raise RuntimeError("ix_fused_moe not loaded")
    return _bridge.topk_softmax(gating_output, topk, renormalize)


def moe_gen_idx(expert_id, expert_num):
    """Build expert permutation maps."""
    if _bridge is None:
        raise RuntimeError("ix_fused_moe not loaded")
    return _bridge.moe_gen_idx(expert_id, expert_num)


def group_gemm(inputs, weights, token_count, output_n):
    """Batched expert GEMM via ixformer."""
    if _bridge is None:
        raise RuntimeError("ix_fused_moe not loaded")
    return _bridge.group_gemm(inputs, weights, token_count, output_n)
