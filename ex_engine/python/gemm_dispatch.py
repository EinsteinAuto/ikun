"""gemm_dispatch.py — Unified GEMM dispatch for MoE group matmul.

AST Layer 2: selects best available GEMM backend on real device.

Backend priority:
  1. gemm_grouped.so  (cutlass Cu10 TensorOp, per-expert GEMM)
  2. ix_moe_bridge.so (cuinferCustomGemm, per-expert loop)
  3. corex_batched_gemm.so (cutlass batched, decode-only)
  4. hgemm.so (blocktiling kernel from siboehm)
  5. torch.mm loop (PyTorch fallback)

Reference: ex_engine/python/ix_ops_dispatch.py (407L)
"""
import os
import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger("gemm_dispatch")

# --- Backend loading ---
_cutlass_grouped = None
_moe_bridge = None
_batched_gemm = None
_hgemm = None
_backend = "torch"


def _try_load(name):
    """Try to load a .so module by name."""
    # Search paths
    search = [
        os.path.join(os.path.dirname(__file__), f"{name}.so"),
        os.path.join(os.path.dirname(__file__), "..", "prebuilt", f"{name}.so"),
        os.path.join(os.path.dirname(__file__), "..", f"{name}.so"),
    ]
    for p in search:
        if os.path.isfile(p):
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(name, p)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
            except Exception as e:
                logger.debug(f"[gemm] Failed to load {p}: {e}")
    # Try direct import
    try:
        import importlib
        return importlib.import_module(name)
    except ImportError:
        return None


def _init_backends():
    global _cutlass_grouped, _moe_bridge, _batched_gemm, _hgemm, _backend

    _cutlass_grouped = _try_load("gemm_grouped")
    if _cutlass_grouped and hasattr(_cutlass_grouped, "moe_group_gemm"):
        _backend = "cutlass_grouped"
        logger.info("[gemm] Backend: cutlass_grouped (Cu10 TensorOp)")
        return

    _moe_bridge = _try_load("ix_moe_bridge")
    if _moe_bridge and hasattr(_moe_bridge, "group_gemm"):
        _backend = "cuinfer"
        logger.info("[gemm] Backend: cuinfer (via ix_moe_bridge)")
        return

    _batched_gemm = _try_load("corex_batched_gemm")
    if _batched_gemm and hasattr(_batched_gemm, "batched_gemm_fp16"):
        _backend = "cutlass_batched"
        logger.info("[gemm] Backend: cutlass_batched")
        return

    _hgemm = _try_load("hgemm")
    if _hgemm and hasattr(_hgemm, "moe_expert_gemm"):
        _backend = "hgemm"
        logger.info("[gemm] Backend: hgemm (blocktiling)")
        return

    _backend = "torch"
    logger.info("[gemm] Backend: torch (F.linear fallback)")


_init_backends()


# ============================================================================
# Public API
# ============================================================================

def group_gemm(input_tokens, weights, expert_counts, output_dim):
    """Per-expert GEMM: output[offset:offset+count] = input[offset:offset+count] @ W[e]^T

    Args:
        input_tokens: (total_tokens, K) fp16
        weights:      (num_experts, N, K) fp16, TN layout
        expert_counts: (num_experts,) int32
        output_dim:   N (output dimension)

    Returns:
        (total_tokens, N) fp16
    """
    if _backend == "cutlass_grouped":
        return _cutlass_grouped.moe_group_gemm(input_tokens, weights, expert_counts)

    if _backend == "cuinfer":
        return _moe_bridge.group_gemm(input_tokens, weights, expert_counts, output_dim)

    if _backend == "hgemm":
        return _hgemm.moe_expert_gemm(input_tokens, weights, expert_counts)

    # torch fallback
    return _torch_group_gemm(input_tokens, weights, expert_counts)


def moe_decode_gemm(hidden, w13_sel, w2_sel, topk_weights):
    """Single-token MoE decode: batched GEMM over topk experts.

    Args:
        hidden:       (1, H) fp16
        w13_sel:      (topk, 2*I, H) fp16
        w2_sel:       (topk, H, I) fp16
        topk_weights: (topk,) float32

    Returns:
        (1, H) fp16
    """
    if _backend == "cutlass_grouped" and hasattr(_cutlass_grouped, "moe_decode_cutlass"):
        return _cutlass_grouped.moe_decode_cutlass(hidden, w13_sel, w2_sel, topk_weights)

    if _backend == "cutlass_batched" and _batched_gemm is not None:
        return _batched_gemm.moe_decode_fused(hidden, w13_sel, w2_sel, topk_weights)

    # torch fallback
    return _torch_moe_decode(hidden, w13_sel, w2_sel, topk_weights)


def get_backend():
    return _backend


# ============================================================================
# Fallbacks
# ============================================================================

def _torch_group_gemm(input_tokens, weights, expert_counts):
    """PyTorch fallback: per-expert F.linear loop."""
    num_experts = weights.size(0)
    N = weights.size(1)
    output = torch.zeros(input_tokens.size(0), N,
                          device=input_tokens.device, dtype=input_tokens.dtype)

    counts_cpu = expert_counts.cpu().to(torch.int32)
    offset = 0
    for e in range(num_experts):
        cnt = counts_cpu[e].item()
        if cnt <= 0:
            offset += cnt
            continue
        x = input_tokens[offset:offset+cnt]
        w = weights[e]  # (N, K)
        output[offset:offset+cnt] = F.linear(x, w)
        offset += cnt

    return output


def _torch_moe_decode(hidden, w13_sel, w2_sel, topk_weights):
    """PyTorch fallback for single-token MoE decode."""
    topk = w13_sel.size(0)
    results = []
    for k in range(topk):
        gate_up = F.linear(hidden, w13_sel[k])
        inter = gate_up.shape[-1] // 2
        act = torch.silu(gate_up[:, :inter]) * gate_up[:, inter:]
        down = F.linear(act, w2_sel[k])
        results.append(down * topk_weights[k].to(down.dtype))
    return sum(results)
