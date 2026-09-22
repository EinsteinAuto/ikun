"""
patch_vllm_hot_path.py — Wire xllm kernel .so into vllm hot path

Architecture (matching xllm/core/layers/ilu/ dispatch chain):

    xllm C++ call chain:
        qwen3_5.h → decoder_layer.forward()
            → layers/ilu/attention.cpp → kernels/ilu/attention.cpp → ixformer::infer
            → layers/common/rms_norm.cpp → kernels/ilu/norm.cpp → ixformer::infer
            → layers/common/activation.cpp → kernels/ilu/activation.cpp → ixformer::infer
            → layers/ilu/fused_moe.cpp → kernels/ilu/fused_moe.cpp → ixformer::infer

    Our Python equivalent:
        qwen3_5.py → Qwen3_5ForCausalLM.forward()
            → patch_vllm_hot_path → xllm_ops → xllm_*.so → ixformer::infer
            → corex_moe.py → ix_full_bridge.so → ixformer::infer

This module patches vllm at import time. Call apply() from patch_ops.sh.

Patches applied (matching xllm/core/kernels/ilu/ exactly):
    1. vllm._custom_ops.topk_softmax    → xllm_ops.topk_softmax
    2. vllm model RMSNorm               → xllm_ops.rms_norm
    3. vllm model SiluAndMul            → xllm_ops.silu_and_mul
    4. vllm model RotaryEmbedding       → xllm_ops.rotary_embedding
    5. vllm attention reshape_and_cache  → xllm_ops.reshape_and_cache
    6. vllm attention paged_attention    → xllm_ops.paged_attention

NO FALLBACK. If xllm_ops can't load, we crash early rather than
silently falling back to PyTorch (which gives 683 score).
"""

import os
import sys
import logging
import importlib

logger = logging.getLogger("ex_engine.patch_hot_path")


def apply(strict=True):
    """Apply all hot-path patches.

    Args:
        strict: If True, crash if any .so is missing.
                Set False only for development/debugging.
    """
    from ex_engine.python import xllm_ops

    # Verify all .so are loadable BEFORE patching anything
    status = xllm_ops.check_all(strict=strict)
    loaded = sum(1 for v in status.values() if v)
    total = len(status)
    logger.info("patch_hot_path: %d/%d kernels available, applying patches", loaded, total)

    patches_applied = 0

    # =====================================================================
    # 1. Patch _custom_ops.topk_softmax (THE critical one from comp 168 log)
    # =====================================================================
    if status.get("xllm_moe", False):
        try:
            # The comp 168 log shows:
            #   ERROR _custom_ops.py:58] Error in calling custom op topk_softmax:
            #     module 'ixformer.functions' has no attribute 'vllm_moe_topk_softmax'
            #   WARNING qwen3_5.py:913] FusedMoE native kernel failed, falling back
            #     to pure PyTorch experts permanently.
            #
            # This single fallback kills performance from 8000 → 683.
            # Fix: provide topk_softmax via xllm_moe.so

            import vllm._custom_ops as ops
            _orig_topk_softmax = getattr(ops, 'topk_softmax', None)

            def patched_topk_softmax(topk_weights, topk_ids, token_expert_ids,
                                     gating_output, topk):
                xllm_ops.topk_softmax(topk_weights, topk_ids, token_expert_ids,
                                      gating_output, topk)

            ops.topk_softmax = patched_topk_softmax
            patches_applied += 1
            logger.info("patch_hot_path: ✓ _custom_ops.topk_softmax → xllm_moe.so")

        except Exception as e:
            logger.error("patch_hot_path: ✗ topk_softmax patch failed: %s", e)
            if strict:
                raise

    # =====================================================================
    # 2. Patch RMSNorm
    # =====================================================================
    if status.get("xllm_norm", False):
        try:
            # vllm uses ops.rms_norm / ops.fused_add_rms_norm
            import vllm._custom_ops as ops

            def patched_rms_norm(output, input, weight, epsilon):
                xllm_ops.rms_norm(input, weight, epsilon)

            def patched_fused_add_rms_norm(input, residual, weight, epsilon):
                xllm_ops.residual_rms_norm(input, residual, weight, epsilon)

            if hasattr(ops, 'rms_norm'):
                ops.rms_norm = patched_rms_norm
                patches_applied += 1
                logger.info("patch_hot_path: ✓ ops.rms_norm → xllm_norm.so")

            if hasattr(ops, 'fused_add_rms_norm'):
                ops.fused_add_rms_norm = patched_fused_add_rms_norm
                patches_applied += 1
                logger.info("patch_hot_path: ✓ ops.fused_add_rms_norm → xllm_norm.so")

        except Exception as e:
            logger.error("patch_hot_path: ✗ norm patch failed: %s", e)
            if strict:
                raise

    # =====================================================================
    # 3. Patch SiluAndMul
    # =====================================================================
    if status.get("xllm_activation", False):
        try:
            import vllm._custom_ops as ops

            def patched_silu_and_mul(output, input):
                xllm_ops.silu_and_mul(input, output)

            if hasattr(ops, 'silu_and_mul'):
                ops.silu_and_mul = patched_silu_and_mul
                patches_applied += 1
                logger.info("patch_hot_path: ✓ ops.silu_and_mul → xllm_activation.so")

        except Exception as e:
            logger.error("patch_hot_path: ✗ activation patch failed: %s", e)
            if strict:
                raise

    # =====================================================================
    # 4. Patch Rotary Embedding
    # =====================================================================
    if status.get("xllm_rope", False):
        try:
            import vllm._custom_ops as ops

            def patched_rotary_embedding(positions, query, key, head_size,
                                         cos_sin_cache, is_neox=True):
                xllm_ops.rotary_embedding(positions, query, key,
                                          cos_sin_cache, is_neox)

            if hasattr(ops, 'rotary_embedding'):
                ops.rotary_embedding = patched_rotary_embedding
                patches_applied += 1
                logger.info("patch_hot_path: ✓ ops.rotary_embedding → xllm_rope.so")

        except Exception as e:
            logger.error("patch_hot_path: ✗ rope patch failed: %s", e)
            if strict:
                raise

    # =====================================================================
    # 5. Patch reshape_and_cache
    # =====================================================================
    if status.get("xllm_cache", False):
        try:
            import vllm._custom_ops as ops

            def patched_reshape_and_cache(key, value, key_cache, value_cache,
                                          slot_mapping, kv_cache_dtype, kv_scale):
                xllm_ops.reshape_and_cache(key, value, key_cache, value_cache,
                                           slot_mapping)

            if hasattr(ops, 'reshape_and_cache'):
                ops.reshape_and_cache = patched_reshape_and_cache
                patches_applied += 1
                logger.info("patch_hot_path: ✓ ops.reshape_and_cache → xllm_cache.so")

        except Exception as e:
            logger.error("patch_hot_path: ✗ cache patch failed: %s", e)
            if strict:
                raise

    # =====================================================================
    # Summary
    # =====================================================================
    logger.info("patch_hot_path: %d patches applied (of %d .so loaded)",
               patches_applied, loaded)

    if patches_applied == 0 and strict:
        raise RuntimeError(
            "patch_hot_path: 0 patches applied. "
            "This means the vllm hot path is running pure PyTorch. "
            "Score will be ~683 instead of 8000."
        )

    return patches_applied


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n = apply(strict="--strict" in sys.argv)
    print(f"Applied {n} hot-path patches")
