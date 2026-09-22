"""
patch_fused_linear_allreduce.py — Fuse linear + allreduce into single kernel launch

Current RowParallelLinear.forward() does:
    output = self.quant_method.apply(self, input, bias=bias_)   # GEMM
    if self.reduce_results and self.tp_size > 1:
        output = tensor_model_parallel_all_reduce(output)       # NCCL allreduce

This patch replaces it with:
    output = ix_full_bridge_fused_ar.linear_allreduce(input, weight, bias)  # fused

Per decode step savings:
    32 attention o_proj + 4 GDN out_proj + 36 shared_expert_down = 72 RowParallel calls
    Each saves 1 kernel launch (~10-25us Python dispatch overhead)

Usage:
    from patch_fused_linear_allreduce import apply_patch
    apply_patch()  # call once at startup
"""

import logging
import os
import importlib.util

import torch

logger = logging.getLogger("patch_fused_linear_allreduce")

_bridge_fused_ar = None
_bridge_loaded = False


def _load_bridge():
    """Load ix_full_bridge_fused_ar.so (prebuilt or JIT)."""
    global _bridge_fused_ar, _bridge_loaded
    if _bridge_loaded:
        return _bridge_fused_ar is not None
    _bridge_loaded = True

    # Search paths for the prebuilt .so
    # patch_ops.sh deploys to vllm's ex_engine/ and model_executor/models/
    search = []
    # Dynamic: find vllm install path
    try:
        import vllm
        vllm_root = os.path.dirname(vllm.__file__)
        search.append(os.path.join(vllm_root, "ex_engine", "ix_full_bridge_fused_ar.so"))
        search.append(os.path.join(vllm_root, "model_executor", "models", "ix_full_bridge_fused_ar.so"))
    except ImportError:
        pass
    search.extend([
        "ex_engine/prebuilt/ix_full_bridge_fused_ar.so",
        "qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10/ix_full_bridge_fused_ar.so",
        "/workspace/ex_engine/prebuilt/ix_full_bridge_fused_ar.so",
        "/workspace/qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10/ix_full_bridge_fused_ar.so",
        "/workspace/qwen3_6_scripts/ex_engine/prebuilt/ix_full_bridge_fused_ar.so",
    ])

    for path in search:
        if os.path.isfile(path):
            try:
                # Use importlib with RTLD_GLOBAL so libc10 symbols are visible
                import sys, ctypes
                old_flags = sys.getdlopenflags()
                sys.setdlopenflags(old_flags | ctypes.RTLD_GLOBAL)
                spec = importlib.util.spec_from_file_location(
                    "ix_full_bridge_fused_ar", path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                sys.setdlopenflags(old_flags)
                if hasattr(mod, "linear_allreduce"):
                    _bridge_fused_ar = mod
                    logger.info("Loaded ix_full_bridge_fused_ar from %s", path)
                    return True
            except Exception as e:
                logger.debug("Failed to load %s: %s", path, e)

    logger.warning("ix_full_bridge_fused_ar.so not found — fused linear_allreduce unavailable")
    return False


def _fused_row_parallel_forward(self, input_):
    """
    Replacement forward for RowParallelLinear.
    Uses fused linear_allreduce when:
      1. Bridge is available
      2. reduce_results=True and tp_size>1 (i.e. needs allreduce)
      3. No bias on non-rank-0 (standard vllm behavior)
      4. fp16 (the SDK function expects fp16)
    Falls back to original forward otherwise.
    """
    if self.input_is_parallel:
        input_parallel = input_
    else:
        from vllm.model_executor.parallel_utils.communication_op import (
            split_tensor_along_last_dim)
        tp_rank = self.tp_rank
        splitted_input = split_tensor_along_last_dim(
            input_, num_partitions=self.tp_size)
        input_parallel = splitted_input[tp_rank].contiguous()

    # Decide whether to use fused path
    # CRITICAL: linear_allreduce will segfault if NCCL process group is not initialized
    use_fused = (
        _bridge_fused_ar is not None
        and self.reduce_results
        and self.tp_size > 1
        and torch.distributed.is_initialized()
        and input_parallel.dtype == torch.float16
        and hasattr(self, 'weight')
        and self.weight.dtype == torch.float16
    )

    if use_fused:
        # Bias handling: only rank 0 adds bias (same as original)
        bias = None
        if self.tp_rank == 0 and not self.skip_bias_add and self.bias is not None:
            bias = self.bias

        try:
            inp = input_parallel.contiguous()
            wt = self.weight
            output = _bridge_fused_ar.linear_allreduce(
                inp, wt,
                bias if bias is not None else None)

            output_bias = self.bias if self.skip_bias_add else None
            return output, output_bias

        except Exception as e:
            # Fall through to original on any error
            logger.debug("linear_allreduce failed: %s, falling back", e)

    # Original path
    return self._original_forward(input_)


_patched = False


def apply_patch():
    """
    Monkey-patch RowParallelLinear.forward to use fused linear_allreduce.
    Safe to call multiple times (idempotent).
    """
    global _patched
    if _patched:
        return

    if not _load_bridge():
        logger.info("Skipping fused linear_allreduce patch (bridge not available)")
        return

    try:
        from vllm.model_executor.layers.linear import RowParallelLinear
    except ImportError:
        logger.warning("Cannot import RowParallelLinear — patch skipped")
        return

    if hasattr(RowParallelLinear, '_original_forward'):
        logger.info("RowParallelLinear already patched")
        _patched = True
        return

    # Save original and install replacement
    RowParallelLinear._original_forward = RowParallelLinear.forward
    RowParallelLinear.forward = _fused_row_parallel_forward
    _patched = True
    logger.info("RowParallelLinear.forward patched with fused linear_allreduce "
                "(saves 72 kernel launches per decode step)")


def revert_patch():
    """Revert the monkey-patch."""
    global _patched
    if not _patched:
        return
    try:
        from vllm.model_executor.layers.linear import RowParallelLinear
        if hasattr(RowParallelLinear, '_original_forward'):
            RowParallelLinear.forward = RowParallelLinear._original_forward
            del RowParallelLinear._original_forward
    except ImportError:
        pass
    _patched = False
    logger.info("RowParallelLinear.forward reverted to original")