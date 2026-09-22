"""patch_moe_hot_path.py — Replace Qwen3_5MoE.forward() with bridge dispatch.

This is the key performance patch: replaces the Python expert-loop MoE
with a single C++ call that does all 7 steps fused.

Called by: patch_ops.sh during Docker build
Target:    vllm.model_executor.models.qwen3_5.Qwen3_5MoE

Reference: ex_engine/python/patch_vllm_hot_path.py (200L)
"""
import sys
import logging
import torch

logger = logging.getLogger("patch_moe_hot_path")


def apply_moe_patch():
    """Monkey-patch Qwen3_5MoE.forward to use moe_dispatch."""
    try:
        from ex_engine.python.moe_dispatch import moe_forward, get_tier
    except ImportError:
        try:
            from moe_dispatch import moe_forward, get_tier
        except ImportError:
            logger.warning("[moe_patch] moe_dispatch not available, skipping patch")
            return False

    tier = get_tier()
    logger.info(f"[moe_patch] moe_dispatch tier={tier}")

    # Find the MoE class
    moe_cls = None
    try:
        from vllm.model_executor.models.qwen3_5 import Qwen3_5MoE
        moe_cls = Qwen3_5MoE
    except ImportError:
        pass

    if moe_cls is None:
        # Try to find it in sys.modules (may be registered under different name)
        for mod_name, mod in sys.modules.items():
            if hasattr(mod, 'Qwen3_5MoE'):
                moe_cls = getattr(mod, 'Qwen3_5MoE')
                break

    if moe_cls is None:
        logger.warning("[moe_patch] Qwen3_5MoE class not found")
        return False

    # Save original forward
    _original_forward = moe_cls.forward

    def patched_forward(self, hidden_states, *args, **kwargs):
        """Patched MoE forward using bridge dispatch."""
        # Get router logits
        # In Qwen3_5, the gate + shared_expert_gate are concatenated:
        #   router_and_shared_gate = self.gate(hidden_states)
        #   router_logits = router_and_shared_gate[..., :self.num_experts]
        #   shared_gate = router_and_shared_gate[..., -1]
        router_and_shared_gate = self.gate(hidden_states)
        router_logits = router_and_shared_gate[..., :self.num_experts]

        # Shared expert (if any) — run in parallel
        shared_output = None
        if hasattr(self, 'shared_expert') and self.shared_expert is not None:
            if hasattr(self, 'shared_expert_gate'):
                shared_gate = torch.sigmoid(
                    router_and_shared_gate[..., -1].unsqueeze(-1))
            else:
                shared_gate = None

        # Routed experts via bridge
        try:
            routed_output = moe_forward(
                hidden_states.view(-1, hidden_states.shape[-1]),
                router_logits.view(-1, router_logits.shape[-1]),
                self.w13_weight if hasattr(self, 'w13_weight') else self.experts.w13_weight,
                self.w2_weight if hasattr(self, 'w2_weight') else self.experts.w2_weight,
                topk=self.top_k,
                num_experts=self.num_experts,
                renormalize=True,
            )
            routed_output = routed_output.view_as(hidden_states)
        except Exception as e:
            logger.warning(f"[moe_patch] Bridge failed ({e}), using original forward")
            return _original_forward(self, hidden_states, *args, **kwargs)

        # Add shared expert output
        if hasattr(self, 'shared_expert') and self.shared_expert is not None:
            shared_out = self.shared_expert(hidden_states)
            if shared_gate is not None:
                shared_out = shared_out * shared_gate
            routed_output = routed_output + shared_out

        return routed_output

    # Only patch if we have a real bridge (not pure Python fallback)
    if tier < 2:
        moe_cls.forward = patched_forward
        logger.info(f"[moe_patch] ✓ Patched Qwen3_5MoE.forward (tier={tier})")
        return True
    else:
        logger.info("[moe_patch] Tier 2 (Python only), not patching")
        return False


if __name__ == "__main__":
    apply_moe_patch()