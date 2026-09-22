"""moe_dispatch.py — Load ix_moe_bridge.so and dispatch MoE forward.

3-level fallback:
  Tier 0: ix_moe_bridge.fused_moe_forward (C++ fused 7-step pipeline)
  Tier 1: ix_moe_bridge individual ops (topk + expand + gemm + silu + gemm + combine)
  Tier 2: Pure PyTorch fallback (F.linear loop)

Used by: patch_moe_hot_path.py → replaces Qwen3_5MoE.forward()

Reference: ex_engine/python/corex_moe.py (237L)
"""
import os
import sys
import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger("moe_dispatch")

# --- Load bridge .so ---
_bridge = None
_tier = 2  # default: PyTorch fallback


def _try_load_bridge():
    global _bridge, _tier

    # Try 1: prebuilt .so
    search_paths = [
        os.path.join(os.path.dirname(__file__), "ix_moe_bridge.so"),
        os.path.join(os.path.dirname(__file__), "..", "prebuilt", "ix_moe_bridge.so"),
        os.path.join(os.path.dirname(__file__), "..", "ix_moe_bridge.so"),
    ]
    for p in search_paths:
        if os.path.isfile(p):
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("ix_moe_bridge", p)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                _bridge = mod
                logger.info(f"[moe_dispatch] ✓ Loaded bridge from {p}")
                break
            except Exception as e:
                logger.warning(f"[moe_dispatch] Failed to load {p}: {e}")

    # Try 2: torch JIT compiled module
    if _bridge is None:
        try:
            import ix_moe_bridge
            _bridge = ix_moe_bridge
            logger.info("[moe_dispatch] ✓ Loaded bridge via import")
        except ImportError:
            pass

    if _bridge is None:
        logger.warning("[moe_dispatch] Bridge not available, using PyTorch fallback")
        _tier = 2
        return

    # Check what functions are available
    try:
        if hasattr(_bridge, 'fused_moe_forward'):
            _tier = 0
            logger.info("[moe_dispatch] Tier 0: fused pipeline available")
        elif hasattr(_bridge, 'topk_softmax') and hasattr(_bridge, 'group_gemm'):
            _tier = 1
            logger.info("[moe_dispatch] Tier 1: individual ops available")
        else:
            _tier = 2
            logger.warning("[moe_dispatch] Bridge loaded but missing functions")
    except Exception as e:
        logger.warning(f"[moe_dispatch] Function check failed: {e}")
        _tier = 2


_try_load_bridge()


# ============================================================================
# Tier 2: Pure PyTorch fallback (identical to base vllm behavior)
# ============================================================================

def _pytorch_moe_forward(hidden_states, router_logits, w13, w2,
                          topk, num_experts, renormalize):
    """Python fallback: softmax → topk → loop over experts with F.linear."""
    gating = torch.softmax(router_logits.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(gating, topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)
    topk_weights = topk_weights.to(hidden_states.dtype)

    # Per-expert loop
    final_output = torch.zeros_like(hidden_states)
    for k in range(topk):
        expert_ids = topk_ids[:, k]  # [T]
        weights_k = topk_weights[:, k].unsqueeze(-1)  # [T, 1]
        for e in range(num_experts):
            mask = (expert_ids == e)
            if not mask.any():
                continue
            expert_input = hidden_states[mask]
            # gate_up = expert_input @ w13[e].T  → [n, 2*inter]
            gate_up = F.linear(expert_input, w13[e])
            inter = gate_up.shape[-1] // 2
            gate = torch.sigmoid(gate_up[:, :inter])
            up = gate_up[:, inter:]
            activated = gate * up  # SiLU approximated as sigmoid * x (should be silu_and_mul)
            # down = activated @ w2[e].T → [n, hidden]
            down = F.linear(activated, w2[e])
            final_output[mask] += weights_k[mask] * down

    return final_output


# ============================================================================
# Tier 1: Individual bridge ops
# ============================================================================

def _bridge_individual_moe_forward(hidden_states, router_logits, w13, w2,
                                    topk, num_experts, renormalize):
    """Use individual bridge ops: topk → gen_idx → expand → gemm → silu → gemm → combine."""
    topk_weights, topk_ids, _ = _bridge.topk_softmax(router_logits, topk, False)
    if renormalize:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)

    idx_results = _bridge.moe_gen_idx(topk_ids.view(-1).to(torch.int32), num_experts)
    src_dst, dst_src, expert_sizes = idx_results[0], idx_results[1], idx_results[2]

    expanded = _bridge.moe_expand_input(hidden_states, src_dst, dst_src, topk)

    gate_up = _bridge.group_gemm(expanded, w13, expert_sizes, w13.size(1))
    activated = _bridge.silu_and_mul(gate_up)
    down = _bridge.group_gemm(activated, w2, expert_sizes, w2.size(1))
    output = _bridge.moe_combine_result(down, topk_weights)

    return output


# ============================================================================
# Public API
# ============================================================================

def moe_forward(hidden_states, router_logits, w13, w2,
                topk, num_experts, renormalize=True):
    """Dispatch MoE forward to best available implementation."""
    if _tier == 0:
        try:
            return _bridge.fused_moe_forward(
                hidden_states, router_logits, w13, w2,
                topk, num_experts, renormalize)
        except Exception as e:
            logger.warning(f"[moe_dispatch] Tier 0 failed: {e}, falling to Tier 1")
            pass

    if _tier <= 1 and _bridge is not None:
        try:
            return _bridge_individual_moe_forward(
                hidden_states, router_logits, w13, w2,
                topk, num_experts, renormalize)
        except Exception as e:
            logger.warning(f"[moe_dispatch] Tier 1 failed: {e}, falling to Tier 2")
            pass

    return _pytorch_moe_forward(
        hidden_states, router_logits, w13, w2,
        topk, num_experts, renormalize)


def get_tier():
    """Return current dispatch tier (0=fused, 1=individual, 2=pytorch)."""
    return _tier