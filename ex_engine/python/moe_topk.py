"""
ex_engine/python/moe_topk.py — MoE topk_softmax CUDA kernel loader

Loads the xllm-derived CUB-based fused softmax+topk kernel.
JIT compiled via torch.utils.cpp_extension.load() on BI-V100.

Usage:
    from ex_engine.python.moe_topk import moe_topk_softmax
    moe_topk_softmax(topk_weights, topk_ids, token_expert_indices, gating_output)
"""

import os
import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger("ex_engine.moe_topk")

_EXT = None


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    if not torch.cuda.is_available():
        raise RuntimeError("MoE topk_softmax kernel requires CUDA.")

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0;7.5")

    csrc_dir = Path(__file__).parent.parent / "csrc" / "moe"

    # Try precompiled .so first
    build_dir = Path(__file__).parent.parent / "build"
    if build_dir.is_dir():
        so_files = list(build_dir.glob("ex_moe_topk*.so"))
        if so_files:
            try:
                from torch.utils.cpp_extension import load
                _EXT = load(
                    name="ex_moe_topk_softmax",
                    sources=[],
                    build_directory=str(build_dir),
                    verbose=False,
                )
                return _EXT
            except Exception:
                pass

    # JIT compile
    from torch.utils.cpp_extension import load
    sources = [str(csrc_dir / "moe_topk_softmax_ext.cu")]
    _EXT = load(
        name="ex_moe_topk_softmax",
        sources=sources,
        extra_cuda_cflags=["-O3", "-I" + str(csrc_dir)],
        extra_cflags=["-O3"],
        verbose=bool(int(os.environ.get("EX_MOE_VERBOSE_BUILD", "0"))),
    )
    logger.info("MoE topk_softmax CUDA kernel compiled successfully")
    return _EXT


def moe_topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> None:
    """
    Drop-in replacement for ixf_F.vllm_moe_topk_softmax.

    Interface matches _custom_ops.topk_softmax() exactly:
      topk_weights:           [num_tokens, topk] float32, output
      topk_ids:               [num_tokens, topk] int32, output
      token_expert_indices:   [num_tokens, topk] int32, output
      gating_output:          [num_tokens, num_experts] input
    """
    ext = _load_ext()
    ext.topk_softmax(topk_weights, topk_ids, token_expert_indices,
                     gating_output, renormalize)
