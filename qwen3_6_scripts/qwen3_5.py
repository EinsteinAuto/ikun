# Inference-only Qwen3.6-35B-A3B (Qwen3_5 MoE architecture) for Iluvatar BI-V100.
# Pure-PyTorch DeltaNet (no fla / causal_conv1d dependency).
# Includes the native Qwen3.6 vision tower; MTP remains unsupported.

from functools import lru_cache, partial
import hashlib
import os
import sys
import time
print("[qwen3_5] module load START", file=sys.stderr, flush=True)
from typing import (Any, Dict, Iterable, List, Literal, Mapping, Optional,
                    Tuple, TypedDict, Union)

def _bi100_model_trace(message: str) -> None:
    if os.getenv("BI100_EXECUTOR_STARTUP_DEBUG") == "1":
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rank = os.getenv("RANK", os.getenv("LOCAL_RANK", "?"))
        print(f"[BI100 STARTUP] {stamp} pid={os.getpid()} rank={rank} {message}",
              file=sys.stderr, flush=True)


_bi100_model_trace("qwen3_5 stdlib imports complete; importing torch and vLLM")

import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from transformers.image_utils import (ChannelDimension, get_image_size,
                                      infer_channel_dimension_format,
                                      to_numpy_array)
from transformers.models.qwen2_vl import (
    image_processing_qwen2_vl as _qwen2_vl_image_processing)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    Qwen2VLImageProcessor, smart_resize)


def _compat_make_batched_images(images):
    return images if isinstance(images, list) else [images]


def _compat_make_batched_videos(videos):
    if isinstance(videos, list) and videos and isinstance(videos[0], list):
        return videos
    return [videos]


# The CoreX image pins transformers 4.55.3, while its vLLM Qwen2-VL module
# imports helpers introduced by another transformers build.
if not hasattr(_qwen2_vl_image_processing, "make_batched_images"):
    _qwen2_vl_image_processing.make_batched_images = \
        _compat_make_batched_images
if not hasattr(_qwen2_vl_image_processing, "make_batched_videos"):
    _qwen2_vl_image_processing.make_batched_videos = \
        _compat_make_batched_videos

from vllm.attention import Attention, AttentionMetadata
from vllm.config import (CacheConfig, LoRAConfig, MultiModalConfig,
                         SchedulerConfig)
from vllm.distributed import (get_tensor_model_parallel_rank,
                               get_tensor_model_parallel_world_size,
                               tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.fused_moe import FusedMoE
# [PR #2269] Apply EP patch to FusedMoE before any layers are constructed.
# When VLLM_ENABLE_EXPERT_PARALLEL=1, this replaces TP-sharded MoE weights
# with EP-sharded MoE weights (each card holds num_experts/ep_size experts
# with full intermediate_size), solving the OOM under TP=2.
try:
    from vllm.ep_fused_moe_patch import patch_fused_moe_for_ep as _patch_ep
    _patch_ep()
except ImportError:
    pass  # EP patch not installed — TP mode unchanged
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding, _apply_rotary_emb)
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, sharded_weight_loader)
from vllm.model_executor.models.mamba_cache import MambaCacheManager
from vllm.model_executor.models.qwen2_vl import (Qwen2VisionAttention,
                                                 Qwen2VisionRotaryEmbedding)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_weight_attrs
from vllm.inputs import INPUT_REGISTRY, InputContext, TokenInputs as LLMInputs
from vllm.multimodal import (MULTIMODAL_REGISTRY, MultiModalDataDict,
                             MultiModalKwargs as MultiModalInputs)
from vllm.sequence import IntermediateTensors, SequenceData
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.logger import init_logger
from vllm.bi100_env import env_bool, env_int
from vllm.bi100_profile import (bi100_profile_event_enabled,
                                bi100_profile_flush,
                                bi100_profile_transaction, bi100_timer)

try:
    from vllm import corex_gdn_causal_conv as _corex_gdn_causal_conv
except ImportError:
    _corex_gdn_causal_conv = None

try:
    from vllm import corex_gdn_gated_norm as _corex_gdn_gated_norm
except ImportError:
    _corex_gdn_gated_norm = None

try:
    from vllm import corex_gdn_beta_decay as _corex_gdn_beta_decay
except ImportError:
    _corex_gdn_beta_decay = None

try:
    from vllm import corex_gdn_qk_map as _corex_gdn_qk_map
except ImportError:
    _corex_gdn_qk_map = None

try:
    from vllm import corex_gdn_packed_decode as _corex_gdn_packed_decode
except ImportError:
    _corex_gdn_packed_decode = None

try:
    from vllm import corex_attn_head_rms_norm as _corex_attn_head_rms_norm
except ImportError:
    _corex_attn_head_rms_norm = None

try:
    from vllm import corex_moe_exact_reduce as _corex_moe_exact_reduce
except ImportError:
    _corex_moe_exact_reduce = None

try:
    from vllm import corex_moe_weight_gather as _corex_moe_weight_gather
except ImportError:
    _corex_moe_weight_gather = None

try:
    from vllm import corex_moe_direct_routed as _corex_moe_direct_routed
except ImportError:
    _corex_moe_direct_routed = None

try:
    from vllm import corex_batched_gemm as _corex_batched_gemm
except ImportError:
    _corex_batched_gemm = None

try:
    from vllm import gemm_grouped as _gemm_grouped
except ImportError:
    _gemm_grouped = None

try:
    from vllm import corex_moe_topk_softmax as _corex_moe_topk_softmax
except ImportError:
    _corex_moe_topk_softmax = None

try:
    from vllm import corex_moe_index_combine as _corex_moe_index_combine
except ImportError:
    _corex_moe_index_combine = None

try:
    from vllm import xllm_moe as _xllm_moe
except ImportError:
    try:
        import xllm_moe as _xllm_moe
    except ImportError:
        _xllm_moe = None

# --- xllm prebuilt kernel loading (PRD build) ---
def _load_xllm_prebuilt(name):
    """Load a prebuilt xllm .so from corex-3.2.3-ivcore10 directory."""
    import importlib.util as _ilu
    _search = [
        f"/workspace/qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10/{name}.so",
        os.path.join(os.path.dirname(__file__), "prebuilt",
                     "corex-3.2.3-ivcore10", f"{name}.so"),
        # When patch_ops.sh copies this file into vllm package, __file__
        # points to vllm/model_executor/models/ — look back up to workspace
        f"/home/dylan/0814/project_6/qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10/{name}.so",
        # .so installed to VLLM_ROOT by install_prebuilt_corex.sh
        os.path.join(os.path.dirname(__file__), "..", "..", f"{name}.so"),
    ]
    for _p in _search:
        if os.path.isfile(_p):
            print(f"[xllm] loading {name} from {_p} ...", file=sys.stderr, flush=True)
            try:
                _spec = _ilu.spec_from_file_location(name, _p)
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                print(f"[xllm] {name} OK: {[x for x in dir(_mod) if not x.startswith('_')]}", file=sys.stderr, flush=True)
                return _mod
            except Exception as _e:
                print(f"[xllm] {name} FAILED: {_e}", file=sys.stderr, flush=True)
                return None
    print(f"[xllm] {name} not found", file=sys.stderr, flush=True)
    return None

print("[xllm] loading prebuilt kernels ...", file=sys.stderr, flush=True)
_xllm_norm = _load_xllm_prebuilt("xllm_norm")
_xllm_rope = _load_xllm_prebuilt("xllm_rope")
_xllm_activation = _load_xllm_prebuilt("xllm_activation")
_xllm_cache = _load_xllm_prebuilt("xllm_cache")
_xllm_fused_qknorm_rope = _load_xllm_prebuilt("xllm_fused_qknorm_rope")
print("[xllm] prebuilt kernel loading done", file=sys.stderr, flush=True)

# ix_moe_bridge: direct GEMV (4.1x faster than F.linear for decode M=1)
# Benchmark: F.linear 133us vs br.linear 32us on BI-V100
try:
    from vllm import ix_moe_bridge as _ix_moe_bridge
except ImportError:
    try:
        import importlib.util as _ilu
        for _p in ["/workspace/qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10/ix_moe_bridge.so",
                   os.path.join(os.path.dirname(__file__), "prebuilt",
                                "corex-3.2.3-ivcore10", "ix_moe_bridge.so")]:
            if os.path.isfile(_p):
                _spec = _ilu.spec_from_file_location("ix_moe_bridge", _p)
                _ix_moe_bridge = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_ix_moe_bridge)
                break
        else:
            _ix_moe_bridge = None
    except Exception:
        _ix_moe_bridge = None
_HAS_BRIDGE_LINEAR = (
    _ix_moe_bridge is not None
    and hasattr(_ix_moe_bridge, 'linear'))
if _HAS_BRIDGE_LINEAR:
    print("[xllm] ix_moe_bridge.linear ENABLED", file=sys.stderr, flush=True)


def _fast_linear(x: torch.Tensor, weight: torch.Tensor,
                 bias=None) -> torch.Tensor:
    """Drop-in F.linear replacement using ix_moe_bridge GEMV.
    4.1x faster than F.linear for M=1 decode on BI-V100."""
    if _HAS_BRIDGE_LINEAR and x.dtype == torch.float16 and weight.dtype == torch.float16:
        return _ix_moe_bridge.linear(x, weight, bias)
    return F.linear(x, weight, bias)

try:
    from vllm import corex_gdn_chunk_recurrent as _corex_gdn_chunk_recurrent
except ImportError:
    _corex_gdn_chunk_recurrent = None

_HAS_COREX_GDN_CHUNK = _corex_gdn_chunk_recurrent is not None

from vllm.model_executor.models.interfaces import (HasInnerState, SupportsLoRA,
                                                   SupportsMultiModal)

logger = init_logger(__name__)

_bi100_model_trace("qwen3_5 runtime imports complete")

# ---------------------------------------------------------------------------
# BI100 flash_attn compatibility patch
# The corex flash_attn_cuda.varlen_fwd kernel has a different signature than
# upstream flash_attn >= 2.5.  We monkey-patch flash_attn.flash_attn_interface
# so that flash_attn_varlen_func works transparently on BI100.
# ---------------------------------------------------------------------------
_flash_attn_patched = False


def _patch_flash_attn_varlen_for_bi100():
    """Patch flash_attn for BI100 corex compatibility.

    The BI100 corex flash_attn package adds three extra required parameters
    to both ``_flash_attn_varlen_forward`` and ``flash_attn_varlen_func``:
        use_alibi (bool), alibi_mode (int), imp_mode (int)
    Upstream callers (e.g. qwen2_vl.py) don't pass these, so we wrap the
    low-level ``_flash_attn_varlen_forward`` to supply defaults.
    """
    global _flash_attn_patched
    if _flash_attn_patched:
        return
    _flash_attn_patched = True

    import flash_attn.flash_attn_interface as _fai

    _orig = _fai._flash_attn_varlen_forward

    def _compat_flash_attn_varlen_forward(
        q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        dropout_p, softmax_scale, causal,
        window_size=(-1, -1), alibi_slopes=None,
        return_softmax=False,
        use_alibi=False, alibi_mode=1, imp_mode=0,
    ):
        return _orig(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            dropout_p, softmax_scale, causal,
            window_size, alibi_slopes, return_softmax,
            use_alibi, alibi_mode, imp_mode,
        )

    _fai._flash_attn_varlen_forward = _compat_flash_attn_varlen_forward
    logger.info("BI100: patched _flash_attn_varlen_forward — "
                "added use_alibi/alibi_mode/imp_mode defaults")

_ALLOW_GDN_NAN_ZERO = env_bool("BI100_GDN_ALLOW_NAN_ZERO", False)
_GDN_FINITE_CHECK = (env_bool("BI100_GDN_FINITE_CHECK", False)
                     or _ALLOW_GDN_NAN_ZERO)
_DNN_CHUNK_SIZE = env_int("BI100_DNN_CHUNK", 4096, 64, 65536)
_USE_COREX_GDN_CAUSAL_CONV = (
    _corex_gdn_causal_conv is not None
    and env_bool("BI100_GDN_COREX_CAUSAL_CONV", True))
_USE_COREX_GDN_GATED_NORM = (
    _corex_gdn_gated_norm is not None
    and env_bool("BI100_GDN_COREX_GATED_NORM", True))
_USE_COREX_GDN_BETA_DECAY = (
    _corex_gdn_beta_decay is not None
    and env_bool("BI100_GDN_COREX_BETA_DECAY", True))
_USE_COREX_GDN_QK_MAP = (
    _corex_gdn_qk_map is not None
    and env_bool("BI100_GDN_COREX_QK_MAP", True))
_USE_COREX_GDN_COMBINED_QK_NORM = (
    _USE_COREX_GDN_QK_MAP
    and env_bool("BI100_GDN_COMBINED_QK_NORM", False))
_USE_COREX_GDN_PACKED_DECODE = (
    _corex_gdn_packed_decode is not None
    and env_bool("BI100_GDN_COREX_PACKED_DECODE", False))
_USE_COREX_ATTN_HEAD_RMS_NORM = (
    _corex_attn_head_rms_norm is not None
    and env_bool("BI100_ATTN_COREX_HEAD_RMS_NORM", True))
_USE_COREX_MOE_EXACT_REDUCE = (
    _corex_moe_exact_reduce is not None
    and env_bool("BI100_MOE_COREX_EXACT_REDUCE", True))
_USE_COREX_MOE_WEIGHT_GATHER = (
    _corex_moe_weight_gather is not None
    and env_bool("BI100_MOE_COREX_WEIGHT_GATHER", True))
_USE_COREX_MOE_DIRECT_ROUTED = (
    _corex_moe_direct_routed is not None
    and env_bool("BI100_MOE_COREX_DIRECT_ROUTED", False))
_USE_COREX_BATCHED_GEMM = (
    _corex_batched_gemm is not None
    and env_bool("BI100_MOE_BATCHED_GEMM", True))
_USE_GEMM_GROUPED = (
    _gemm_grouped is not None
    and env_bool("BI100_MOE_GEMM_GROUPED", True))
if _USE_GEMM_GROUPED:
    logger.info("gemm_grouped ENABLED — CUTLASS Cu10 grouped GEMM for MoE prefill")
_USE_COREX_MOE_TOPK_SOFTMAX = (
    _corex_moe_topk_softmax is not None
    and env_bool("BI100_MOE_COREX_TOPK_SOFTMAX", True))
_USE_COREX_MOE_INDEX_COMBINE = (
    _corex_moe_index_combine is not None
    and env_bool("BI100_MOE_COREX_INDEX_COMBINE", True))
_USE_XLLM_MOE = (
    _xllm_moe is not None
    and env_bool("BI100_MOE_XLLM", True))
if _USE_XLLM_MOE:
    logger.info("xllm_moe ENABLED — fused_topk + compute_index + combine_result")
_USE_FUSED_MOE_ACTIVATION = env_bool("BI100_MOE_FUSED_ACTIVATION", True)

# --- xllm CUDA kernel flags ---
_USE_XLLM_NORM = (
    _xllm_norm is not None
    and env_bool("BI100_XLLM_NORM", True))
_USE_XLLM_ROPE = (
    _xllm_rope is not None
    and env_bool("BI100_XLLM_ROPE", True))
_USE_XLLM_ACTIVATION = (
    _xllm_activation is not None
    and env_bool("BI100_XLLM_ACTIVATION", True))
_USE_XLLM_CACHE = (
    _xllm_cache is not None
    and env_bool("BI100_XLLM_CACHE", True))
_USE_XLLM_FUSED_QKNORM_ROPE = (
    _xllm_fused_qknorm_rope is not None
    and env_bool("BI100_XLLM_FUSED_QKNORM_ROPE", True))
if _USE_XLLM_NORM:
    logger.info("xllm_norm ENABLED — fused RMSNorm CUDA kernel")
if _USE_XLLM_ROPE:
    logger.info("xllm_rope ENABLED — fused RoPE CUDA kernel")
if _USE_XLLM_ACTIVATION:
    logger.info("xllm_activation ENABLED — fused SiLU-and-Mul CUDA kernel")
if _USE_XLLM_CACHE:
    logger.info("xllm_cache ENABLED — fused reshape_paged_cache CUDA kernel")
if _USE_XLLM_FUSED_QKNORM_ROPE:
    logger.info("xllm_fused_qknorm_rope ENABLED — fused QK-Norm+RoPE (saves 128 launches/fwd)")

# ---------------------------------------------------------------------------
# xllm kernel monkey-patches: replace PyTorch fallback → fused CUDA kernels
# ---------------------------------------------------------------------------
print("[xllm] applying monkey-patches ...", file=sys.stderr, flush=True)

# --- 1. RMSNorm: xllm_norm replaces GemmaRMSNorm.forward_cuda ---
# GemmaRMSNorm uses  x * (1 + weight)  while xllm kernel uses  x * weight.
# We pre-compute (1 + weight) and cache it so the kernel sees the correct
# effective weight.  The cached tensor is lazily materialised on first call
# and invalidated if shape/device/dtype change (should never happen after
# model init).
if _USE_XLLM_NORM:
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm as _GemmaRMSNorm

    def _xllm_rms_norm_forward_cuda(self, x, residual=None):
        """Drop-in for GemmaRMSNorm.forward_cuda using xllm_norm CUDA kernel."""
        # Lazily compute and cache  effective_weight = 1 + weight
        _ew = getattr(self, '_xllm_eff_weight', None)
        if (_ew is None
                or _ew.shape != self.weight.shape
                or _ew.device != self.weight.device
                or _ew.dtype != self.weight.dtype):
            _ew = (1.0 + self.weight.data.float()).to(self.weight.dtype)
            self._xllm_eff_weight = _ew

        if residual is not None:
            # fused_add_rms_norm:  input = RMSNorm(input + residual),
            #                     residual = input + residual  (before norm)
            # The kernel modifies input and residual IN-PLACE.
            _xllm_norm.fused_add_rms_norm(
                x, residual, _ew, self.variance_epsilon)
            return x, residual
        else:
            out = torch.empty_like(x)
            _xllm_norm.rms_norm(out, x, _ew, self.variance_epsilon)
            return out

    _GemmaRMSNorm.forward_cuda = _xllm_rms_norm_forward_cuda
    logger.info("xllm_norm PATCHED — GemmaRMSNorm.forward_cuda → xllm CUDA kernel")


# --- 2. SiluAndMul: xllm_activation replaces SiluAndMul.forward_cuda ---
if _USE_XLLM_ACTIVATION:

    def _xllm_silu_and_mul_forward_cuda(self, x):
        """Drop-in for SiluAndMul.forward_cuda using xllm_activation kernel."""
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        # Reuse cached output tensor for stable shapes (decode)
        _cache_key = (output_shape, x.dtype, x.device)
        _cached = getattr(self, '_out_cache', {}).get(_cache_key)
        if _cached is not None and _cached.shape == output_shape:
            out = _cached
        else:
            out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
            if not hasattr(self, '_out_cache'):
                self._out_cache = {}
            self._out_cache[_cache_key] = out
        _xllm_activation.silu_and_mul(out, x)
        return out

    SiluAndMul.forward_cuda = _xllm_silu_and_mul_forward_cuda
    logger.info("xllm_activation PATCHED — SiluAndMul.forward_cuda → xllm CUDA kernel")


# --- 3. Cache ops: xllm_cache (reshape_paged_cache / block_copy) ---
# Replace ixformer vllm_cache_ops_reshape_and_cache with xllm_cache.
# xllm_cache signature: reshape_paged_cache(slot_ids, keys, values, kc, vc)
# vllm calls:           ops.reshape_and_cache(key, value, kc, vc, slot_mapping, ...)
# Difference: arg order, slot_ids must be int32.
if _USE_XLLM_CACHE:
    import vllm._custom_ops as _vllm_ops

    _orig_reshape_and_cache = _vllm_ops.reshape_and_cache

    def _xllm_reshape_and_cache(key, value, key_cache, value_cache,
                                slot_mapping, kv_cache_dtype="auto",
                                k_scale=1.0, v_scale=1.0):
        slot_ids = slot_mapping.flatten().to(torch.int32)
        _xllm_cache.reshape_paged_cache(slot_ids, key, value,
                                        key_cache, value_cache)

    _vllm_ops.reshape_and_cache = _xllm_reshape_and_cache
    logger.info("xllm_cache PATCHED — reshape_and_cache → xllm CUDA kernel")


# --- 4. RoPE: xllm_rope ---
# Qwen3.5 uses interleaved multi-axis RoPE (MRoPE) with partial rotary
# factor 0.25.  The xllm_rope kernel accepts (positions, query, key,
# cos_sin_cache, is_neox) and modifies q/k in-place, but it applies RoPE
# to the FULL head dim.  Qwen3.5's forward only rotates the first 25% of
# each head (rotary_dim = 64 out of 256), then concatenates the unrotated
# tail.  The multi-axis interleaving of cos/sin across T/H/W axes also
# requires Python-level assembly before calling any kernel.
#
# A safe replacement would need to:
#  1. Assemble the interleaved cos_sin for the partial dim.
#  2. Call xllm_rope on just the rotary_dim slice of q and k.
#  3. Skip the concat step since the kernel modifies in-place.
#
# This is doable but requires careful per-position indexing that differs
# from the standard "positions → cos_sin_cache[positions]" pattern.
# We mark this as TODO and leave the current _apply_rotary_emb path.
if _USE_XLLM_ROPE:
    logger.info("xllm_rope LOADED but NOT PATCHED — Qwen3.5 interleaved MRoPE "
                "requires adapter; using PyTorch _apply_rotary_emb fallback")


# ix_fused_moe: full 7-step fused MoE pipeline via ixformer C++ API
# Source: xllm/core/layers/ilu/fused_moe.cpp → ix_moe_bridge.so
try:
    from vllm.model_executor.models import ix_fused_moe as _ix_fused_moe
    _HAS_IX_FUSED_MOE = _ix_fused_moe.is_available()
except ImportError:
    try:
        import ix_fused_moe as _ix_fused_moe
        _HAS_IX_FUSED_MOE = _ix_fused_moe.is_available()
    except ImportError:
        _ix_fused_moe = None
        _HAS_IX_FUSED_MOE = False
_USE_IX_FUSED_MOE = (
    _HAS_IX_FUSED_MOE
    and env_bool("BI100_MOE_IX_FUSED", True))
if _USE_IX_FUSED_MOE:
    logger.info("ix_fused_moe ENABLED — full 7-step fused MoE pipeline")
else:
    logger.info("ix_fused_moe unavailable — using point-optimized Python MoE")

# naive_batched_moe_forward: ported from ds_vllm NaiveBatchedExperts
# Uses view transpose + @ operator (cublas transB), no physical transpose
try:
    from ex_engine.moe.naive_batched_experts import naive_batched_moe_forward
    _HAS_NAIVE_BATCHED_MOE = True
except ImportError:
    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from ex_engine.moe.naive_batched_experts import naive_batched_moe_forward
        _HAS_NAIVE_BATCHED_MOE = True
    except ImportError:
        _HAS_NAIVE_BATCHED_MOE = False
        naive_batched_moe_forward = None
_USE_NAIVE_BATCHED_MOE = (
    _HAS_NAIVE_BATCHED_MOE
    and env_bool("BI100_MOE_NAIVE_BATCHED", True))


# ---------------------------------------------------------------------------
# Qwen3.6 vision tower and vLLM 0.6 multimodal input integration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Hot path patch: replace vllm ops with xllm .so (THE performance fix)
# Source: ex_engine/python/patch_vllm_hot_path.py
# Savings: ~12.2ms/token from linear alone (F.linear 115µs → bridge 31µs)
# ---------------------------------------------------------------------------
if env_bool("BI100_HOT_PATH_PATCH", True):
    try:
        from ex_engine.python.patch_vllm_hot_path import apply as _apply_hot_path
        _hot_path_count = _apply_hot_path(strict=False)
        print(f"[xllm] hot path: {_hot_path_count} patches applied",
              file=sys.stderr, flush=True)
    except Exception as _e:
        print(f"[xllm] hot path patch FAILED: {_e}", file=sys.stderr, flush=True)

_MAX_IMAGE_TOKENS = 1280


@lru_cache(maxsize=None)
def _cached_get_qwen36_image_processor(model_path: str):
    # The fast processor in transformers 4.55 calls torch.compiler APIs that
    # are absent from the evaluator's torch 2.1 CoreX build.
    return Qwen2VLImageProcessor.from_pretrained(model_path)


@lru_cache(maxsize=None)
def _cached_get_qwen36_tokenizer(model_path: str, trust_remote_code: bool):
    return get_tokenizer(model_path, trust_remote_code=trust_remote_code)


def _image_cache_marker_tokens(image, tokenizer) -> List[int]:
    array = to_numpy_array(image)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes())
    marker = f"[image-cache-key:{digest.hexdigest()[:16]}]"
    return tokenizer.encode(marker, add_special_tokens=False)


def _make_batched_images(images):
    if isinstance(images, list):
        if images and isinstance(images[0], list):
            return [image for batch in images for image in batch]
        return images
    return [images]


class Qwen3_5ImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    image_grid_thw: torch.Tensor


class Qwen3_5ImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor


Qwen3_5ImageInputs = Union[Qwen3_5ImagePixelInputs,
                           Qwen3_5ImageEmbeddingInputs]


def _vision_pos_embed_interpolate(
    embed_weight: torch.Tensor,
    t: int,
    h: int,
    w: int,
    num_grid_per_side: int,
    merge_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if h % merge_size or w % merge_size:
        raise ValueError(
            f"vision grid {(t, h, w)} is not divisible by merge_size="
            f"{merge_size}")
    hidden_dim = embed_weight.shape[1]
    device = embed_weight.device
    h_idxs = torch.linspace(0, num_grid_per_side - 1, h,
                            dtype=torch.float32, device=device)
    w_idxs = torch.linspace(0, num_grid_per_side - 1, w,
                            dtype=torch.float32, device=device)
    h_floor = h_idxs.long()
    w_floor = w_idxs.long()
    h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
    w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)
    dh = h_idxs - h_floor
    dw = w_idxs - w_floor
    dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
    hf_grid, wf_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
    hc_grid, wc_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")
    w11 = dh_grid * dw_grid
    w10 = dh_grid - w11
    w01 = dw_grid - w11
    w00 = 1 - dh_grid - w01
    h_grid = torch.stack([hf_grid, hf_grid, hc_grid, hc_grid])
    w_grid = torch.stack([wf_grid, wc_grid, wf_grid, wc_grid])
    indices = (h_grid * num_grid_per_side + w_grid).reshape(4, -1)
    weights = torch.stack([w00, w01, w10, w11], dim=0)
    weights = weights.reshape(4, -1, 1).to(dtype=dtype)
    combined = (embed_weight[indices] * weights).sum(dim=0)
    combined = combined.reshape(
        h // merge_size, merge_size,
        w // merge_size, merge_size, hidden_dim)
    combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
    return combined.expand(t, -1, -1).reshape(-1, hidden_dim).to(dtype)


class Qwen3_5VisionPatchEmbed(nn.Module):
    def __init__(self, vision_config) -> None:
        super().__init__()
        self.patch_size = vision_config.patch_size
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.hidden_size = vision_config.hidden_size
        kernel = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(
            vision_config.in_channels,
            self.hidden_size,
            kernel_size=kernel,
            stride=kernel,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[0]
        x = x.view(length, -1, self.temporal_patch_size,
                   self.patch_size, self.patch_size)
        return self.proj(x).view(length, self.hidden_size)


class Qwen3_5VisionMLP(nn.Module):
    def __init__(self, vision_config,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()
        self.linear_fc1 = ColumnParallelLinear(
            vision_config.hidden_size,
            vision_config.intermediate_size,
            bias=True,
            quant_config=quant_config,
        )
        self.linear_fc2 = RowParallelLinear(
            vision_config.intermediate_size,
            vision_config.hidden_size,
            bias=True,
            quant_config=quant_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.linear_fc1(x)
        x = F.gelu(x, approximate="tanh")
        x, _ = self.linear_fc2(x)
        return x


class Qwen3_5VisionBlock(nn.Module):
    def __init__(self, vision_config,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()
        dim = vision_config.hidden_size
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Qwen2VisionAttention(
            embed_dim=dim,
            num_heads=vision_config.num_heads,
            projection_size=dim,
            quant_config=quant_config,
        )
        # BI100: keep FLASH_ATTN backend but patch flash_attn_cuda.varlen_fwd
        # to be compatible with the corex kernel signature.
        from vllm.platforms import _Backend as _Bk
        if hasattr(self.attn, 'attn_backend'):
            self.attn.attn_backend = _Bk.FLASH_ATTN
            _patch_flash_attn_varlen_for_bi100()
        self.mlp = Qwen3_5VisionMLP(vision_config, quant_config)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor,
                rotary_pos_emb: torch.Tensor,
                max_seqlen: Optional[int] = None,
                seqlens: Optional[list] = None) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            max_seqlen=max_seqlen,
            seqlens=seqlens,
        )
        return x + self.mlp(self.norm2(x))


class Qwen3_5VisionPatchMerger(nn.Module):
    def __init__(self, vision_config,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()
        self.hidden_size = (vision_config.hidden_size
                            * vision_config.spatial_merge_size ** 2)
        self.norm = nn.LayerNorm(vision_config.hidden_size, eps=1e-6)
        self.linear_fc1 = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
        )
        self.linear_fc2 = RowParallelLinear(
            self.hidden_size,
            vision_config.out_hidden_size,
            bias=True,
            quant_config=quant_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).view(-1, self.hidden_size)
        x, _ = self.linear_fc1(x)
        x = F.gelu(x)
        x, _ = self.linear_fc2(x)
        return x


class Qwen3_5VisionTransformer(nn.Module):
    def __init__(self, vision_config,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.num_grid_per_side = int(vision_config.num_position_embeddings ** .5)
        self.patch_embed = Qwen3_5VisionPatchEmbed(vision_config)
        self.pos_embed = nn.Embedding(
            vision_config.num_position_embeddings, self.hidden_size)
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = Qwen2VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([
            Qwen3_5VisionBlock(vision_config, quant_config)
            for _ in range(vision_config.depth)
        ])
        self.merger = Qwen3_5VisionPatchMerger(vision_config, quant_config)

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def _rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for t, h, w in grid_thw.tolist():
            h_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            w_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            h_ids = h_ids.reshape(
                h // self.spatial_merge_size, self.spatial_merge_size,
                w // self.spatial_merge_size, self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            w_ids = w_ids.reshape(
                h // self.spatial_merge_size, self.spatial_merge_size,
                w // self.spatial_merge_size, self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([h_ids, w_ids], dim=-1).repeat(t, 1))
        pos_ids_t = torch.cat(pos_ids, dim=0).to(self.device)
        max_grid_size = int(grid_thw[:, 1:].max().item())
        return self.rotary_pos_emb(max_grid_size)[pos_ids_t].flatten(1)

    def _absolute_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            _vision_pos_embed_interpolate(
                self.pos_embed.weight, int(t), int(h), int(w),
                self.num_grid_per_side, self.spatial_merge_size, self.dtype)
            for t, h, w in grid_thw.tolist()
        ], dim=0)

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.device, dtype=self.dtype)
        grid_thw = grid_thw.to(device=self.device)
        x = self.patch_embed(x)
        x = x + self._absolute_pos_emb(grid_thw)
        rotary_pos_emb = self._rot_pos_emb(grid_thw)
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0],
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)
        x = x.unsqueeze(1)
        # Pre-compute seqlens for xformers attn mask
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        for block in self.blocks:
            x = block(x, cu_seqlens, rotary_pos_emb,
                      max_seqlen=max_seqlen, seqlens=seqlens)
        return self.merger(x)


class Qwen3_5InterleavedMRotaryEmbedding(MRotaryEmbedding):
    """Qwen3.5 frequency-interleaved T/H/W rotary embedding."""

    def forward(self, positions: torch.Tensor, query: torch.Tensor,
                key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if positions.ndim not in (1, 2):
            raise ValueError(f"invalid MRoPE positions shape {positions.shape}")
        num_tokens = positions.shape[-1]
        cos_sin = self.cos_sin_cache[positions]
        cos_all, sin_all = cos_sin.chunk(2, dim=-1)
        if positions.ndim == 2:
            if not self.mrope_section:
                raise ValueError("mrope_section is required")
            cos = cos_all[0].clone()
            sin = sin_all[0].clone()
            for dim, offset in enumerate((1, 2), start=1):
                stop = self.mrope_section[dim] * 3
                cos[..., offset:stop:3] = cos_all[dim, ..., offset:stop:3]
                sin[..., offset:stop:3] = sin_all[dim, ..., offset:stop:3]
        else:
            cos, sin = cos_all, sin_all

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = _apply_rotary_emb(
            query[..., :self.rotary_dim], cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query[..., self.rotary_dim:]), dim=-1)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = _apply_rotary_emb(
            key[..., :self.rotary_dim], cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key[..., self.rotary_dim:]), dim=-1)
        return query.reshape(query_shape), key.reshape(key_shape)


def _qwen36_pixel_limits(image_processor) -> Tuple[int, int]:
    min_pixels = 256 * 256
    configured_max = 4096 * 4096
    runtime_max = _MAX_IMAGE_TOKENS * (
        image_processor.patch_size * image_processor.merge_size) ** 2
    return min_pixels, min(configured_max, runtime_max)


def _qwen36_image_token_count(image, image_processor) -> int:
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
    image_array = to_numpy_array(image)
    height, width = get_image_size(
        image_array, channel_dim=ChannelDimension.LAST)
    min_pixels, max_pixels = _qwen36_pixel_limits(image_processor)
    if getattr(image_processor, "do_resize", True):
        height, width = smart_resize(
            height=height,
            width=width,
            factor=image_processor.patch_size * image_processor.merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    return (height // image_processor.patch_size
            * width // image_processor.patch_size
            // image_processor.merge_size ** 2)


def qwen36_image_input_mapper(
    ctx: InputContext,
    data: object,
) -> MultiModalInputs:
    if isinstance(data, dict):
        return MultiModalInputs({
            "image_embeds": data.get("image_embeds"),
            "image_grid_thw": data.get("image_grid_thw"),
        })
    image_processor = _cached_get_qwen36_image_processor(
        ctx.model_config.model)
    min_pixels, max_pixels = _qwen36_pixel_limits(image_processor)
    batch_data = image_processor.preprocess(
        images=data,
        return_tensors="pt",
        size={"shortest_edge": min_pixels, "longest_edge": max_pixels},
        do_convert_rgb=True,
        input_data_format=ChannelDimension.LAST,
    ).data
    return MultiModalInputs(batch_data)


def get_max_qwen36_image_tokens(_ctx: InputContext) -> int:
    return _MAX_IMAGE_TOKENS


def dummy_data_for_qwen36(
    ctx: InputContext,
    seq_len: int,
    mm_counts: Mapping[str, int],
    **kwargs,
) -> "DummyData":
    from vllm.inputs.registry import DummyData
    num_images = mm_counts.get("image", 0)
    image_tokens = _MAX_IMAGE_TOKENS * num_images
    if seq_len < image_tokens + 2:
        raise RuntimeError(
            f"Qwen3.6 needs {image_tokens + 2} tokens for {num_images} "
            f"max-size image(s), but max_model_len is {seq_len}")
    config = ctx.model_config.hf_config
    seq_data = SequenceData.from_prompt_token_counts(
        (config.vision_start_token_id, 1),
        (config.image_token_id, image_tokens),
        (config.vision_end_token_id, 1),
        (0, seq_len - image_tokens - 2),
    )
    dummy_image = Image.new("RGB", (1280, 1024), color=0)
    mm_data = {
        "image": (dummy_image if num_images == 1
                  else [dummy_image] * num_images)
    }
    return DummyData(seq_data=seq_data, multi_modal_data=mm_data)


def input_processor_for_qwen36(ctx: InputContext,
                               llm_inputs: LLMInputs) -> LLMInputs:
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if not multi_modal_data or "image" not in multi_modal_data:
        return llm_inputs
    images = multi_modal_data["image"]
    prompt_token_ids = llm_inputs.get("prompt_token_ids")
    if prompt_token_ids is None:
        raise ValueError("Qwen3.6 image requests require tokenized prompt input")
    config = ctx.model_config.hf_config
    image_processor = _cached_get_qwen36_image_processor(
        ctx.model_config.model)
    tokenizer = _cached_get_qwen36_tokenizer(
        ctx.model_config.tokenizer,
        ctx.model_config.trust_remote_code,
    )
    batched_images = _make_batched_images(images)
    image_indices = [
        idx for idx, token in enumerate(prompt_token_ids)
        if token == config.image_token_id
    ]
    if len(image_indices) != len(batched_images):
        raise ValueError(
            f"found {len(image_indices)} image placeholders for "
            f"{len(batched_images)} image(s)")
    expanded = []
    previous = 0
    for index, image in zip(image_indices, batched_images):
        vision_start = index - 1
        if (vision_start < previous
                or prompt_token_ids[vision_start]
                != config.vision_start_token_id):
            raise ValueError("image token is not preceded by vision_start")
        expanded.extend(prompt_token_ids[previous:vision_start])
        expanded.extend(_image_cache_marker_tokens(image, tokenizer))
        expanded.extend(prompt_token_ids[vision_start:index])
        expanded.extend([config.image_token_id]
                        * _qwen36_image_token_count(image, image_processor))
        previous = index + 1
    expanded.extend(prompt_token_ids[previous:])
    return LLMInputs(
        prompt_token_ids=expanded,
        prompt=llm_inputs["prompt"],
        multi_modal_data=multi_modal_data,
    )


# ---------------------------------------------------------------------------
# Pure-PyTorch DeltaNet kernels (fallbacks from transformers 5.2.0)
# ---------------------------------------------------------------------------

def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _check_gdn_finite(tensor: torch.Tensor, *, layer_idx: int,
                      stage: str) -> torch.Tensor:
    if not _GDN_FINITE_CHECK:
        return tensor
    if torch.isfinite(tensor).all():
        return tensor
    bad = (~torch.isfinite(tensor)).float().mean().item()
    msg = (
        f"non-finite values in {stage} GatedDeltaNet layer {layer_idx} "
        f"(frac={bad:.4f})"
    )
    if not _ALLOW_GDN_NAN_ZERO:
        raise RuntimeError(msg)
    logger.warning("%s; replacing with zeros because BI100_GDN_ALLOW_NAN_ZERO=1",
                   msg)
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def _gdn_segment_ends(seq_len: int, chunk_size: int,
                      capture_offsets: Iterable[int]) -> List[int]:
    ends = list(range(chunk_size, seq_len, chunk_size))
    ends.append(seq_len)
    ends.extend(offset for offset in capture_offsets
                if 0 < offset < seq_len)
    return sorted(set(ends))


def _validate_gdn_prefix_key(key: Any) -> Tuple[int, bytes]:
    if (not isinstance(key, tuple) or len(key) != 2
            or not isinstance(key[0], int) or key[0] <= 0
            or not isinstance(key[1], bytes) or len(key[1]) != 32):
        raise RuntimeError(f"invalid GDN prefix key: {key!r}")
    return key


def _torch_causal_conv1d_update(
    hidden_states: torch.Tensor,   # (batch, channels, seq=1)
    conv_state: torch.Tensor,       # (batch, channels, state_len)  modified in-place
    weight: torch.Tensor,           # (channels, kernel_size)
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> torch.Tensor:
    _, channels, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    cat = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(cat[:, :, -state_len:])
    out = F.conv1d(cat, weight.unsqueeze(1), bias, padding=0, groups=channels)
    out = out[:, :, -seq_len:]
    if activation is not None:
        out = F.silu(out)
    return out.to(hidden_states.dtype)


def _torch_chunk_gated_delta_rule(
    query: torch.Tensor,   # (batch, seq, num_heads, head_k_dim)
    key: torch.Tensor,
    value: torch.Tensor,   # (batch, seq, num_heads, head_v_dim)
    g: torch.Tensor,       # (batch, seq, num_heads)
    beta: torch.Tensor,    # (batch, seq, num_heads)
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query)
        key = _l2norm(key)
    # Transpose to (batch, num_heads, seq, dim)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]
    batch, num_heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]
    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad))
    key = F.pad(key, (0, 0, 0, pad))
    value = F.pad(value, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    g = F.pad(g, (0, pad))
    total_len = seq_len + pad
    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask_upper = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask_upper, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    last_state = (
        torch.zeros(batch, num_heads, k_dim, v_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_out = torch.zeros_like(value)
    mask_upper2 = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1)

    for i in range(total_len // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_i = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask_upper2, 0)
        v_prime = k_cumdecay[:, :, i] @ last_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_state
        core_out[:, :, i] = attn_inter + attn_i @ v_new
        last_state = (
            last_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None])
            .transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_state = None
    core_out = core_out.reshape(batch, num_heads, -1, v_dim)[:, :, :seq_len]
    core_out = core_out.transpose(1, 2).contiguous()
    return core_out, last_state

def _torch_recurrent_gated_delta_rule(
    query: torch.Tensor,   # (batch, 1, num_heads, head_k_dim)
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,       # (batch, 1, num_heads)
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query)
        key = _l2norm(key)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]
    batch, num_heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]
    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_out = torch.zeros(batch, num_heads, seq_len, v_dim,
                           dtype=value.dtype, device=value.device)
    last_state = (
        torch.zeros(batch, num_heads, k_dim, v_dim,
                    dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    for t in range(seq_len):
        q_t = query[:, :, t]
        k_t = key[:, :, t]
        v_t = value[:, :, t]
        g_t = g[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, t].unsqueeze(-1)
        last_state = last_state * g_t
        kv_mem = (last_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_state = last_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_out[:, :, t] = (last_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_state = None
    core_out = core_out.transpose(1, 2).contiguous()
    return core_out, last_state


# ---------------------------------------------------------------------------
# Gated RMSNorm (for DeltaNet output normalisation)
# ---------------------------------------------------------------------------

class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor,
                gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hs = hidden_states.to(torch.float32)
        variance = hs.pow(2).mean(-1, keepdim=True)
        hs = hs * torch.rsqrt(variance + self.variance_epsilon)
        hs = self.weight * hs.to(input_dtype)
        return (hs * F.silu(gate.to(torch.float32))).to(input_dtype)

    def forward_decode(self, hidden_states: torch.Tensor,
                       gate: torch.Tensor) -> torch.Tensor:
        if (_USE_COREX_GDN_GATED_NORM
                and hidden_states.dtype == torch.float32
                and gate.dtype == torch.float16
                and self.weight.dtype == torch.float16
                and hidden_states.shape[-1] == 128):
            hs = hidden_states.float()
            inverse = torch.rsqrt(
                hs.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
            return _corex_gdn_gated_norm.apply_inverse(
                hs, gate, self.weight, inverse)
        return self.forward(hidden_states, gate).to(gate.dtype)


def _load_gdn_projection_weight(params_dict, name: str,
                                loaded_weight: torch.Tensor,
                                text_cfg) -> bool:
    projections = {
        "in_proj_qkv": None,
        "in_proj_z": 3,
        "in_proj_b": 4,
        "in_proj_a": 5,
    }
    source = next((projection for projection in projections
                   if f".linear_attn.{projection}." in name), None)
    if source is None:
        return False

    target_name = name.replace(
        f".linear_attn.{source}.",
        ".linear_attn.in_proj_qkvzba.",
    )
    if target_name not in params_dict:
        raise ValueError(f"missing fused GDN projection parameter: {target_name}")
    param = params_dict[target_name]
    weight_loader = getattr(param, "weight_loader", default_weight_loader)

    if source == "in_proj_qkv":
        key_dim = (text_cfg.linear_num_key_heads
                   * text_cfg.linear_key_head_dim)
        value_dim = (text_cfg.linear_num_value_heads
                     * text_cfg.linear_value_head_dim)
        shard_sizes = (key_dim, key_dim, value_dim)
        if loaded_weight.shape[0] != sum(shard_sizes):
            raise ValueError(
                "unexpected fused QKV output size: "
                f"{loaded_weight.shape[0]} != {sum(shard_sizes)}")
        for shard_id, shard in enumerate(
                torch.split(loaded_weight, shard_sizes, dim=0)):
            weight_loader(param, shard, shard_id)
    else:
        weight_loader(param, loaded_weight, projections[source])
    return True


def _load_full_attention_qgkv_weight(params_dict, name: str,
                                     loaded_weight: torch.Tensor,
                                     text_cfg) -> bool:
    projections = {"q_proj": 0, "k_proj": 1, "v_proj": 2}
    source = next((projection for projection in projections
                   if f".self_attn.{projection}." in name), None)
    if source is None:
        return False
    target_name = name.replace(
        f".self_attn.{source}.", ".self_attn.qgkv_proj.")
    if target_name not in params_dict:
        return False

    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    qg_dim = text_cfg.num_attention_heads * text_cfg.head_dim * 2
    if qg_dim % tp_size != 0:
        raise ValueError(f"QG output size {qg_dim} is not divisible by TP {tp_size}")
    local_qg_dim = qg_dim // tp_size
    kv_dim = text_cfg.num_key_value_heads * text_cfg.head_dim
    expected_rows = qg_dim if source == "q_proj" else kv_dim
    if loaded_weight.shape[0] != expected_rows:
        raise ValueError(
            f"unexpected full-attention {source} output size: "
            f"{loaded_weight.shape[0]} != {expected_rows}")

    if source == "q_proj":
        loaded_weight = loaded_weight.narrow(
            0, tp_rank * local_qg_dim, local_qg_dim)
        offset = 0
    elif source == "k_proj":
        offset = local_qg_dim
    else:
        offset = local_qg_dim + kv_dim
    param = params_dict[target_name]
    default_weight_loader(
        param[offset:offset + loaded_weight.shape[0]], loaded_weight)
    return True


# ---------------------------------------------------------------------------
# Gated DeltaNet  (linear_attention layers)
# ---------------------------------------------------------------------------

class GatedDeltaNet(nn.Module):
    def __init__(
        self,
        text_cfg,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = text_cfg.hidden_size
        self.num_v_heads = text_cfg.linear_num_value_heads   # checkpoint: 32
        self.num_k_heads = text_cfg.linear_num_key_heads     # checkpoint: 16
        self.head_k_dim = text_cfg.linear_key_head_dim       # 128
        self.head_v_dim = text_cfg.linear_value_head_dim     # 128
        self.key_dim = self.num_k_heads * self.head_k_dim    # 2048
        self.value_dim = self.num_v_heads * self.head_v_dim  # checkpoint: 4096
        self.conv_dim = self.key_dim * 2 + self.value_dim    # checkpoint: 8192
        self.conv_kernel_size = text_cfg.linear_conv_kernel_dim  # 4
        self.head_expand_ratio = self.num_v_heads // self.num_k_heads  # checkpoint: 2

        tp_size = get_tensor_model_parallel_world_size()

        # Keep each logical projection independently TP-sharded while executing
        # one GEMM. Per-rank output order is [q, k, v, z, beta, decay].
        self.in_proj_qkvzba = MergedColumnParallelLinear(
            self.hidden_size,
            [self.key_dim, self.key_dim, self.value_dim, self.value_dim,
             self.num_v_heads, self.num_v_heads],
            bias=False, quant_config=quant_config)
        self.out_proj = RowParallelLinear(
            self.value_dim, self.hidden_size,
            bias=False, quant_config=quant_config)

        # Depthwise conv weight — sharded along channel dim (dim 0)
        local_conv_dim = self.conv_dim // tp_size
        self.conv1d_weight = nn.Parameter(
            torch.empty(local_conv_dim, 1, self.conv_kernel_size))
        set_weight_attrs(self.conv1d_weight, {
            "weight_loader": self._conv1d_weight_loader})

        # Per-head scalar parameters — sharded along dim 0
        local_num_v = self.num_v_heads // tp_size
        self.A_log = nn.Parameter(torch.zeros(local_num_v))
        self.dt_bias = nn.Parameter(torch.zeros(local_num_v))
        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        # Gated RMSNorm on head_v_dim — replicated (head_v_dim=128 is small)
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim,
                                        eps=text_cfg.rms_norm_eps)
        self.captured_conv_states: Dict[int, torch.Tensor] = {}
        self.captured_temporal_states: Dict[int, torch.Tensor] = {}

    def _conv1d_weight_loader(self, param: torch.Tensor,
                              loaded_weight: torch.Tensor) -> None:
        # loaded_weight is ordered as [q, k, v] along its channel dimension.
        # Must gather channels in the same non-contiguous pattern that
        # MergedColumnParallelLinear uses for in_proj_qkv, so that each rank's
        # conv1d_weight[i] applies to the correct in_proj_qkv output channel.
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        key_local = self.key_dim // tp_size    # 512 with TP=4
        val_local = self.value_dim // tp_size  # 1024 with TP=4
        q_s = loaded_weight[tp_rank * key_local : (tp_rank + 1) * key_local]
        k_s = loaded_weight[self.key_dim + tp_rank * key_local :
                            self.key_dim + (tp_rank + 1) * key_local]
        v_s = loaded_weight[2 * self.key_dim + tp_rank * val_local :
                            2 * self.key_dim + (tp_rank + 1) * val_local]
        param.data.copy_(torch.cat([q_s, k_s, v_s], dim=0))

    def forward(
        self,
        hidden_states: torch.Tensor,      # (total_tokens, hidden_size)
        attn_metadata: AttentionMetadata,
        conv_state: torch.Tensor,          # (batch, local_conv_dim, kernel-1)  in-place
        temporal_state: torch.Tensor,      # (batch, local_v_heads, k_dim, v_dim)  in-place
        capture_offsets: Optional[Iterable[int]] = None,
        segment_offsets: Optional[Iterable[int]] = None,
    ) -> torch.Tensor:
        tp_size = get_tensor_model_parallel_world_size()
        local_key_dim = self.key_dim // tp_size
        local_val_dim = self.value_dim // tp_size
        local_num_v = self.num_v_heads // tp_size
        local_num_k = self.num_k_heads // tp_size
        local_conv_dim = self.conv_dim // tp_size
        self.captured_conv_states = {}
        self.captured_temporal_states = {}

        is_prefill = attn_metadata.num_prefill_tokens > 0

        projected, _ = self.in_proj_qkvzba(hidden_states)
        mixed_qkv_all, z_all, b_all, a_all = torch.split(
            projected,
            [local_conv_dim, local_val_dim, local_num_v, local_num_v],
            dim=-1,
        )

        if is_prefill:
            seq_starts = attn_metadata.query_start_loc.tolist()
            outputs = []
            state_len = self.conv_kernel_size - 1
            weight_2d = self.conv1d_weight.squeeze(1)  # (local_conv_dim, kernel)

            for si in range(len(seq_starts) - 1):
                s, e = int(seq_starts[si]), int(seq_starts[si + 1])
                seq_len = e - s

                # Shape: (1, local_conv_dim, seq_len)
                mixed_qkv = (mixed_qkv_all[s:e]
                             .transpose(0, 1).unsqueeze(0)
                             .to(weight_2d.dtype))

                # Load prev conv state BEFORE overwriting (needed for causal conv padding).
                # For first prefill of a request: mamba_cache is zeros → correct.
                # For chunked prefill chunk 2+: carries last state_len tokens from prev chunk.
                prev_conv = conv_state[si:si + 1].clone().to(weight_2d.dtype)  # [1, local_conv_dim, state_len]

                # Save conv state (last state_len positions)
                if seq_len >= state_len:
                    conv_state[si].copy_(mixed_qkv[0, :, -state_len:])
                else:
                    conv_state[si, :, state_len - seq_len:].copy_(
                        mixed_qkv[0])
                    conv_state[si, :, :state_len - seq_len] = 0

                # Causal conv: left-pad with previous conv state (not zeros).
                padded = torch.cat([prev_conv, mixed_qkv], dim=2)
                seq_capture_offsets = (set(capture_offsets or ())
                                       if si == 0 else set())
                seq_segment_offsets = (set(segment_offsets or ())
                                       if si == 0 else set())
                for capture_offset in seq_capture_offsets:
                    if 0 < capture_offset < seq_len:
                        self.captured_conv_states[capture_offset] = padded[
                            0, :, capture_offset:
                            capture_offset + state_len].clone()
                mixed_qkv_conv = F.conv1d(
                    padded, self.conv1d_weight,
                    bias=None, padding=0, groups=local_conv_dim)
                mixed_qkv_conv = F.silu(mixed_qkv_conv)
                # (1, seq_len, local_conv_dim)
                mixed_qkv_conv = mixed_qkv_conv.squeeze(0).transpose(0, 1).unsqueeze(0)

                q, k, v = torch.split(
                    mixed_qkv_conv,
                    [local_key_dim, local_key_dim, local_val_dim], dim=-1)
                q = q.reshape(1, seq_len, local_num_k, self.head_k_dim)
                k = k.reshape(1, seq_len, local_num_k, self.head_k_dim)
                v = v.reshape(1, seq_len, local_num_v, self.head_v_dim)

                beta = b_all[s:e].sigmoid().unsqueeze(0)  # (1, seq_len, local_num_v)
                g = (-self.A_log.float().exp()
                     * F.softplus(a_all[s:e].float() + self.dt_bias)
                     ).unsqueeze(0)  # (1, seq_len, local_num_v)

                # Expand k/q to match num_v_heads
                q = q.repeat_interleave(self.head_expand_ratio, dim=2)
                k = k.repeat_interleave(self.head_expand_ratio, dim=2)

                # Sub-sequence chunking: call _torch_chunk_gated_delta_rule
                # on _DNN_CHUNK tokens at a time to cap peak memory.
                # Full 18K: tensors [1,6,282,64,64]=220 MB each → ~990 MB/call.
                # With _DNN_CHUNK=4096: [1,6,64,64,64]=6 MB each → ~137 MB/call.
                # State is chained via initial_state / output_final_state.
                cur_state = temporal_state[si:si + 1].clone()
                core_out_parts = []
                segment_ends = _gdn_segment_ends(
                    seq_len, _DNN_CHUNK_SIZE,
                    seq_capture_offsets | seq_segment_offsets)
                sc_start = 0
                _chunk_fn = (
                    _corex_gdn_chunk_recurrent.torch_chunk_gated_delta_rule
                    if _HAS_COREX_GDN_CHUNK
                    else _torch_chunk_gated_delta_rule
                )
                with bi100_timer(f"L{self.layer_idx}.gdn.prefill"):
                    for sc_end in segment_ends:
                        c_out, cur_state = _chunk_fn(
                            q[:, sc_start:sc_end],
                            k[:, sc_start:sc_end],
                            v[:, sc_start:sc_end],
                            g[:, sc_start:sc_end],
                            beta[:, sc_start:sc_end],
                            initial_state=cur_state,
                            output_final_state=True,
                            use_qk_l2norm_in_kernel=True,
                        )
                        core_out_parts.append(c_out)
                        if sc_end in seq_capture_offsets:
                            self.captured_temporal_states[sc_end] = (
                                cur_state[0].clone())
                        sc_start = sc_end
                if cur_state is not None:
                    temporal_state[si].copy_(cur_state[0])
                # [1, seq_len, num_v_heads, head_v_dim]
                core_out = torch.cat(core_out_parts, dim=1)

                # Gate + norm + output proj
                z = z_all[s:e].reshape(seq_len, local_num_v, self.head_v_dim)
                core_out = core_out.reshape(seq_len, local_num_v, self.head_v_dim)
                normed = self.norm(
                    core_out.reshape(-1, self.head_v_dim),
                    z.reshape(-1, self.head_v_dim))
                normed = _check_gdn_finite(
                    normed, layer_idx=self.layer_idx,
                    stage="prefill-norm").reshape(seq_len, -1)
                normed = normed.to(z_all.dtype)
                out, _ = self.out_proj(normed)
                outputs.append(out)

            result = torch.cat(outputs, dim=0)
            return _check_gdn_finite(
                result, layer_idx=self.layer_idx, stage="prefill-output")

        else:
            # Decode: one token per sequence
            num_seqs = hidden_states.shape[0]
            weight_2d = self.conv1d_weight.squeeze(1)

            # (num_seqs, local_conv_dim, 1)
            mixed_qkv = (mixed_qkv_all
                         .to(weight_2d.dtype)
                         .unsqueeze(-1)
                         .contiguous())

            if _USE_COREX_GDN_CAUSAL_CONV:
                mixed_qkv_conv = _corex_gdn_causal_conv.causal_conv_update(
                    conv_state.contiguous(), mixed_qkv, weight_2d)
            else:
                mixed_qkv_conv = _torch_causal_conv1d_update(
                    mixed_qkv, conv_state, weight_2d,
                    bias=None, activation='silu')
            # (num_seqs, local_conv_dim, 1) → (num_seqs, 1, local_conv_dim)
            mixed_qkv_conv = mixed_qkv_conv.squeeze(-1).unsqueeze(1)

            packed_mixed_qkv = mixed_qkv_conv.squeeze(1)
            use_corex_packed_decode = (
                _USE_COREX_GDN_PACKED_DECODE
                and num_seqs == 1
                and local_num_k == 4
                and local_num_v == 8
                and self.head_k_dim == 128
                and self.head_v_dim == 128
                and packed_mixed_qkv.dtype == torch.float16
                and packed_mixed_qkv.shape == (1, 2048)
                and packed_mixed_qkv.is_contiguous()
                and b_all.dtype == torch.float16
                and b_all.shape == (1, 8)
                and b_all.is_contiguous()
                and a_all.dtype == torch.float16
                and a_all.shape == (1, 8)
                and a_all.is_contiguous()
                and self.A_log.dtype == torch.float16
                and self.A_log.shape == (8,)
                and self.A_log.is_contiguous()
                and self.dt_bias.dtype == torch.float16
                and self.dt_bias.shape == (8,)
                and self.dt_bias.is_contiguous()
                and temporal_state.dtype == torch.float32
                and temporal_state.shape == (1, 8, 128, 128)
                and temporal_state.is_contiguous())
            if use_corex_packed_decode:
                with bi100_timer(f"L{self.layer_idx}.gdn.decode"):
                    core_out = _corex_gdn_packed_decode.packed_decode(
                        temporal_state, packed_mixed_qkv, b_all, a_all,
                        self.A_log, self.dt_bias)
            else:
                q, k, v = torch.split(
                    mixed_qkv_conv,
                    [local_key_dim, local_key_dim, local_val_dim], dim=-1)
                q = q.reshape(num_seqs, 1, local_num_k, self.head_k_dim)
                k = k.reshape(num_seqs, 1, local_num_k, self.head_k_dim)
                v = v.reshape(num_seqs, 1, local_num_v, self.head_v_dim)

                use_corex_beta_decay = (
                    _USE_COREX_GDN_BETA_DECAY
                    and b_all.dtype == torch.float16
                    and a_all.dtype == torch.float16
                    and self.A_log.dtype == torch.float16
                    and self.dt_bias.dtype == torch.float16
                    and b_all.is_contiguous()
                    and a_all.is_contiguous())
                if use_corex_beta_decay:
                    beta_decay = _corex_gdn_beta_decay.beta_decay(
                        b_all, a_all, self.A_log, self.dt_bias)
                    bt = beta_decay[0]
                    g_t = beta_decay[1]
                else:
                    beta = b_all.sigmoid()
                    g = (-self.A_log.float().exp()
                         * F.softplus(a_all.float() + self.dt_bias))
                    bt = beta.float()
                    g_t = g.float().exp_()

                # Inlined decode recurrent step (seq_len=1).
                # Uses bmm/baddbmm_ to avoid large intermediate tensors.
                _scale = self.head_k_dim ** -0.5
                q_raw = q.squeeze(1)
                k_raw = k.squeeze(1)
                use_corex_qk_map = (
                    _USE_COREX_GDN_QK_MAP
                    and q_raw.dtype == torch.float16
                    and k_raw.dtype == torch.float16
                    and self.head_k_dim == 128
                    and q_raw.is_contiguous()
                    and k_raw.is_contiguous())
                if use_corex_qk_map:
                    use_combined_qk_norm = (
                        _USE_COREX_GDN_COMBINED_QK_NORM
                        and num_seqs == 1
                        and local_num_k == 4
                        and local_num_v == 8
                        and packed_mixed_qkv.dtype == torch.float16
                        and packed_mixed_qkv.shape == (1, 2048)
                        and packed_mixed_qkv.is_contiguous())
                    if use_combined_qk_norm:
                        raw_qk = packed_mixed_qkv.narrow(
                            1, 0, 2 * local_key_dim).view(
                                num_seqs, 2 * local_num_k,
                                self.head_k_dim)
                        normalized_qk = _l2norm(raw_qk)
                        normalized_q, normalized_k = torch.split(
                            normalized_qk, local_num_k, dim=1)
                    else:
                        normalized_q = _l2norm(q_raw)
                        normalized_k = _l2norm(k_raw)
                    qk_mapped = _corex_gdn_qk_map.qk_map(
                        normalized_q, normalized_k, local_num_v)
                    q_t = qk_mapped[0]
                    k_t = qk_mapped[1]
                else:
                    q_expanded = q_raw.repeat_interleave(
                        self.head_expand_ratio, dim=1)
                    k_expanded = k_raw.repeat_interleave(
                        self.head_expand_ratio, dim=1)
                    q_t = _l2norm(q_expanded).float() * _scale
                    k_t = _l2norm(k_expanded).float()
                v_t = v.squeeze(1).float()

                with bi100_timer(f"L{self.layer_idx}.gdn.decode"):
                    # State shape is (B, H_v, k_dim, v_dim).
                    temporal_state.mul_(g_t[:, :, None, None])
                    ts_flat = temporal_state.view(
                        -1, self.head_k_dim, self.head_v_dim)
                    BH = ts_flat.shape[0]
                    kv_mem = torch.bmm(
                        k_t.view(BH, 1, self.head_k_dim), ts_flat
                    ).view(num_seqs, local_num_v, self.head_v_dim)
                    delta = (v_t - kv_mem) * bt[:, :, None]
                    ts_flat.baddbmm_(
                        k_t.view(BH, self.head_k_dim, 1),
                        delta.view(BH, 1, self.head_v_dim),
                    )
                    core_out = torch.bmm(
                        q_t.view(BH, 1, self.head_k_dim), ts_flat
                    ).view(num_seqs, local_num_v, self.head_v_dim)
            # core_out: (B, H_v, v_dim) = (num_seqs, local_num_v, head_v_dim) already

            z = z_all.reshape(num_seqs, local_num_v, self.head_v_dim)
            normed = self.norm.forward_decode(
                core_out.reshape(-1, self.head_v_dim),
                z.reshape(-1, self.head_v_dim))
            normed = _check_gdn_finite(
                normed, layer_idx=self.layer_idx,
                stage="decode-norm").reshape(num_seqs, -1)
            out, _ = self.out_proj(normed)
            return _check_gdn_finite(
                out, layer_idx=self.layer_idx, stage="decode-output")


# ---------------------------------------------------------------------------
# Full Attention  (with gated q — unique to Qwen3.5)
# ---------------------------------------------------------------------------

class Qwen3_5AttentionHeadRMSNorm(GemmaRMSNorm):
    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ):
        if (_USE_COREX_ATTN_HEAD_RMS_NORM
                and residual is None
                and x.dtype == torch.float16
                and self.weight.dtype == torch.float16
                and x.dim() == 3
                and x.shape[0] == 1
                and x.shape[-1] == 256
                and x.is_contiguous()
                and self.weight.is_contiguous()):
            original_shape = x.shape
            converted, squares = _corex_attn_head_rms_norm.prepare(
                x.view(-1, 256))
            inverse = torch.rsqrt(
                squares.mean(dim=-1, keepdim=True)
                + self.variance_epsilon)
            return _corex_attn_head_rms_norm.apply_inverse(
                converted, self.weight, inverse).view(original_shape)
        return super().forward_cuda(x, residual)


class Qwen3_5FullAttention(nn.Module):
    def __init__(
        self,
        text_cfg,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = text_cfg.hidden_size               # 5120
        self.num_heads = text_cfg.num_attention_heads         # 24
        self.num_kv_heads = text_cfg.num_key_value_heads      # 4
        self.head_dim = text_cfg.head_dim                     # 256
        self.rms_norm_eps = text_cfg.rms_norm_eps

        tp_size = get_tensor_model_parallel_world_size()
        self.local_num_heads = self.num_heads // tp_size
        self.scaling = self.head_dim ** -0.5
        self.use_packed_local_qgkv = tp_size > self.num_kv_heads

        # When num_kv_heads < tp_size we cannot shard KV further (would give
        # fractional heads per rank).  Use ReplicatedLinear so every rank holds
        # all KV heads; local_num_kv_heads equals the full count.
        # When num_kv_heads >= tp_size standard ColumnParallel sharding applies.
        if tp_size > self.num_kv_heads:
            # GQA-aware TP sharding: ixformer kernel only supports num_kv_heads=1
            # per rank.  With num_kv_heads=2 < tp_size=4 we cannot shard KV
            # evenly, but we CAN assign each rank the ONE KV head that serves
            # its Q heads:
            #   q_per_kv = num_heads // num_kv_heads  (e.g. 16//2 = 8)
            #   Rank r uses KV head  r * local_num_heads // q_per_kv
            # e.g. ranks 0,1 → KV head 0;  ranks 2,3 → KV head 1.
            # We replicate all KV heads to every rank and select in forward().
            self.proj_kv_heads = self.num_kv_heads  # heads available from projection
            self.local_num_kv_heads = 1             # heads after rank-local selection
            self.q_per_kv_global = self.num_heads // self.num_kv_heads
            local_qg_dim = self.local_num_heads * self.head_dim * 2
            replicated_kv_dim = self.num_kv_heads * self.head_dim
            self.qgkv_proj = ReplicatedLinear(
                self.hidden_size, local_qg_dim + 2 * replicated_kv_dim,
                bias=False, quant_config=quant_config,
                prefix=f"{prefix}.qgkv_proj")
        else:
            # Standard sharding: each rank gets num_kv_heads // tp_size heads.
            self.local_num_kv_heads = self.num_kv_heads // tp_size
            self.proj_kv_heads = self.local_num_kv_heads  # already sharded
            self.q_per_kv_global = None
            self.k_proj = ColumnParallelLinear(
                self.hidden_size, self.num_kv_heads * self.head_dim,
                bias=False, quant_config=quant_config,
                prefix=f"{prefix}.k_proj")
            self.v_proj = ColumnParallelLinear(
                self.hidden_size, self.num_kv_heads * self.head_dim,
                bias=False, quant_config=quant_config,
                prefix=f"{prefix}.v_proj")

        self.local_q_dim = self.local_num_heads * self.head_dim
        self.local_kv_dim = self.local_num_kv_heads * self.head_dim

        if not self.use_packed_local_qgkv:
            # q_proj includes gate: output = num_heads * head_dim * 2
            self.q_proj = ColumnParallelLinear(
                self.hidden_size, self.num_heads * self.head_dim * 2,
                bias=False, quant_config=quant_config,
                prefix=f"{prefix}.q_proj")
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim, self.hidden_size,
            bias=False, quant_config=quant_config,
            prefix=f"{prefix}.o_proj")

        self.q_norm = Qwen3_5AttentionHeadRMSNorm(
            self.head_dim, eps=self.rms_norm_eps)
        self.k_norm = Qwen3_5AttentionHeadRMSNorm(
            self.head_dim, eps=self.rms_norm_eps)

        # Partial RoPE: rotary_dim = head_dim * partial_rotary_factor = 256 * 0.25 = 64
        rope_params = getattr(text_cfg, "rope_parameters", {}) or {}
        rope_theta = rope_params.get("rope_theta", 10_000_000)
        partial_factor = rope_params.get("partial_rotary_factor", 0.25)
        rotary_dim = int(self.head_dim * partial_factor)

        self.rotary_emb = Qwen3_5InterleavedMRotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=rotary_dim,
            max_position_embeddings=text_cfg.max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            dtype=torch.get_default_dtype(),
            mrope_section=rope_params.get("mrope_section", [11, 11, 10]),
        )

        self.attn = Attention(
            self.local_num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.local_num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        total_tokens = hidden_states.shape[0]

        with bi100_timer("full_attn.project_qgkv"):
            if self.use_packed_local_qgkv:
                projected, _ = self.qgkv_proj(hidden_states)
                qg, k, v = torch.split(
                    projected,
                    [self.local_num_heads * self.head_dim * 2,
                     self.proj_kv_heads * self.head_dim,
                     self.proj_kv_heads * self.head_dim],
                    dim=-1)
            else:
                qg, _ = self.q_proj(hidden_states)
                k, _ = self.k_proj(hidden_states)
                v, _ = self.v_proj(hidden_states)

        with bi100_timer("full_attn.norm_rope"):
            # q projection output includes gate (dim doubled).
            qg = qg.view(total_tokens, self.local_num_heads,
                         self.head_dim * 2)
            q = qg[:, :, :self.head_dim].reshape(total_tokens, -1)
            gate = qg[:, :, self.head_dim:].reshape(total_tokens, -1)

            # Select the one rank-local KV head before k_norm and RoPE.
            if self.q_per_kv_global is not None:
                tp_rank = get_tensor_model_parallel_rank()
                kv_idx = ((tp_rank * self.local_num_heads)
                          // self.q_per_kv_global)
                k = (k.view(total_tokens, self.proj_kv_heads, self.head_dim)
                      [:, kv_idx, :].contiguous())
                v = (v.view(total_tokens, self.proj_kv_heads, self.head_dim)
                      [:, kv_idx, :].contiguous())

            # --- Fused QK-Norm + RoPE path (saves 4 kernel launches per layer) ---
            # Only for 1D positions (decode / text-only prefill).
            # 2D MRoPE positions (vision prefill) fall back to separate ops.
            if (_USE_XLLM_FUSED_QKNORM_ROPE
                    and positions.ndim == 1
                    and q.is_contiguous() and k.is_contiguous()):
                # Pack Q, K, V into contiguous [T, (Hq+Hk+Hv)*D] for fused kernel
                v_flat = v.view(total_tokens, -1)
                qkv = torch.cat([q, k.view(total_tokens, -1), v_flat], dim=-1)
                # GemmaRMSNorm weight convention: kernel uses x*w, Gemma uses x*(1+w)
                _q_ew = getattr(self, '_fused_q_ew', None)
                if _q_ew is None:
                    _q_ew = (1.0 + self.q_norm.weight.data.float()).to(
                        self.q_norm.weight.dtype)
                    self._fused_q_ew = _q_ew
                _k_ew = getattr(self, '_fused_k_ew', None)
                if _k_ew is None:
                    _k_ew = (1.0 + self.k_norm.weight.data.float()).to(
                        self.k_norm.weight.dtype)
                    self._fused_k_ew = _k_ew
                _xllm_fused_qknorm_rope.fused_qk_norm_rope(
                    qkv,
                    self.local_num_heads,
                    self.local_num_kv_heads,
                    self.local_num_kv_heads,
                    self.head_dim,
                    self.rms_norm_eps,
                    _q_ew,
                    _k_ew,
                    self.rotary_emb.cos_sin_cache,
                    True,  # interleaved (Qwen3.5 uses interleaved RoPE)
                    positions.to(torch.int64))
                # Unpack
                q_dim = self.local_num_heads * self.head_dim
                k_dim = self.local_num_kv_heads * self.head_dim
                q = qkv[:, :q_dim]
                k = qkv[:, q_dim:q_dim + k_dim]
                # v is untouched by fused kernel, keep original
            else:
                # Fallback: separate q_norm, k_norm, rotary_emb
                q = self.q_norm.forward_cuda(
                    q.view(total_tokens, self.local_num_heads, self.head_dim)
                    .contiguous()).view(total_tokens, -1)
                k = self.k_norm.forward_cuda(
                    k.view(total_tokens, self.local_num_kv_heads, self.head_dim)
                    .contiguous()).view(total_tokens, -1)
                q, k = self.rotary_emb(positions, q, k)

        with bi100_timer("full_attn.attention"):
            with bi100_timer(f"L{self.layer_idx}.full_attn"):
                attn_out = self.attn(q, k, v,
                                     kv_cache=kv_cache,
                                     attn_metadata=attn_metadata)

        with bi100_timer("full_attn.gate"):
            attn_out = (attn_out
                        * torch.sigmoid(gate.float()).to(attn_out.dtype))
        with bi100_timer("full_attn.output_proj"):
            output, _ = self.o_proj(attn_out)
        return output


# ---------------------------------------------------------------------------
# MLP  (SwiGLU, same as Qwen2/Qwen3)
# ---------------------------------------------------------------------------

class Qwen3_5MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False, quant_config=quant_config)
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size,
            bias=False, quant_config=quant_config)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}")
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


# ---------------------------------------------------------------------------
# MoE sparse block  (Qwen3.5-MoE / Qwen3.6-35B-A3B)
# ---------------------------------------------------------------------------

class Qwen3_5MoeSparseBlock(nn.Module):
    """Replaces Qwen3_5MLP for qwen3_5_moe_text layers.

    FusedMoE is used ONLY for weight storage and loading (create_weights /
    weight_loader are pure PyTorch).  Its forward kernel is bypassed because
    ixformer on BI-V100 lacks vllm_moe_topk_softmax / vllm_invoke_fused_moe_kernel.
    Routing and expert computation use a pure-PyTorch loop instead.

    Shared expert uses RowParallelLinear(reduce_results=False) so both paths
    produce partial (pre-all-reduce) outputs that are combined before a single
    all-reduce.
    """

    def __init__(
        self,
        text_cfg,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        hidden_size = text_cfg.hidden_size
        self.num_experts = text_cfg.num_experts
        self.top_k = text_cfg.num_experts_per_tok

        # Router and scalar shared-expert gate read the same hidden state. Keep
        # their checkpoint shards in one replicated weight so forward needs a
        # single GEMM for 256 + 1 outputs.
        self.router_shared_gate = ReplicatedLinear(
            hidden_size, text_cfg.num_experts + 1,
            bias=False, quant_config=quant_config)
        self.router_shared_gate.weight.weight_loader = \
            self._router_shared_gate_weight_loader

        # FusedMoE: weight storage + weight_loader ONLY.
        # Forward is NEVER called — _pure_pytorch_experts() handles everything.
        # In EP mode, _ep_enabled/start_expert_id/num_experts_per_rank attrs
        # are set by ep_fused_moe_patch.py, used by _pure_pytorch_experts()
        # to mask non-local experts and remap ids.
        self.experts = FusedMoE(
            num_experts=text_cfg.num_experts,
            top_k=text_cfg.num_experts_per_tok,
            hidden_size=hidden_size,
            intermediate_size=text_cfg.moe_intermediate_size,
            reduce_results=False,   # we do the all-reduce ourselves below
            renormalize=True,
            quant_config=quant_config,
        )

        # Shared expert: defer all-reduce to combine with routed output first
        shared_size = text_cfg.shared_expert_intermediate_size
        self.shared_expert_gate_up = MergedColumnParallelLinear(
            hidden_size, [shared_size] * 2, bias=False,
            quant_config=quant_config)
        self.shared_expert_down = RowParallelLinear(
            shared_size, hidden_size, bias=False, reduce_results=False,
            quant_config=quant_config)
        self.act_fn = SiluAndMul()

        # Pre-transposed weight cache for bmm decode path
        self._w13_t = None
        self._w2_t = None

    def _router_shared_gate_weight_loader(
        self,
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        shard_id: int,
    ) -> None:
        if shard_id == 0:
            offset = 0
            rows = self.num_experts
        elif shard_id == 1:
            offset = self.num_experts
            rows = 1
        else:
            raise ValueError(f"unexpected router/shared gate shard: {shard_id}")

        expected = (rows, param.shape[1])
        if tuple(loaded_weight.shape) != expected:
            raise ValueError(
                "unexpected router/shared gate weight shape: "
                f"expected {expected}, got {tuple(loaded_weight.shape)}")
        param.data.narrow(0, offset, rows).copy_(loaded_weight)

    def _pure_pytorch_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """MoE expert dispatch — fused C++ pipeline when available.

        w13_weight: (num_experts, 2*inter_per_partition, hidden)  [TP-sharded]
        w2_weight:  (num_experts, hidden,  inter_per_partition)   [TP-sharded]
        Output is partial (pre-all-reduce), same contract as FusedMoE
        with reduce_results=False.
        """
        # ---------------------------------------------------------------
        # Tier 0: Full fused MoE via ix_moe_bridge (xllm 7-step pipeline)
        # topk → gen_idx → expand → group_gemm → silu → group_gemm → combine
        # Source: xllm/core/layers/ilu/fused_moe.cpp
        # NOTE: Only use for prefill (T>1). For decode (T=1), group_gemm
        # does 8× M=1 GEMMs that are completely memory-bound (<5% GPU util).
        # The Tier 1 T=1 path below uses corex_moe_direct_routed or
        # corex_batched_gemm.moe_decode_fused, which are purpose-built
        # fused kernels for single-token MoE dispatch.
        # ---------------------------------------------------------------
        if _USE_IX_FUSED_MOE:
            w13 = self.experts.w13_weight  # (E, 2*I, H)
            w2 = self.experts.w2_weight    # (E, H, I)
            _ix_cnt = getattr(self, '_ix_diag_cnt', 0)
            _in_nan = hidden_states.isnan().any().item()
            if _ix_cnt < 5 or (_in_nan and _ix_cnt < 200):
                self._ix_diag_cnt = _ix_cnt + 1
                logger.info(
                    "[IX_MOE] call=%d T=%d in_nan=%s in_norm=%.4f "
                    "logits_range=[%.4f,%.4f] w13=%s",
                    _ix_cnt, hidden_states.shape[0], _in_nan,
                    hidden_states.float().norm().item() if not _in_nan else -1,
                    router_logits.min().item() if not _in_nan else -1,
                    router_logits.max().item() if not _in_nan else -1,
                    list(w13.shape))
            out = _ix_fused_moe.fused_moe_forward(
                hidden_states, router_logits.float(),
                w13, w2,
                self.top_k, w13.shape[0],
                True)  # renormalize
            if _ix_cnt < 5:
                logger.info(
                    "[IX_MOE] call=%d out_nan=%s out_norm=%.4f",
                    _ix_cnt, bool(out.isnan().any()),
                    out.float().norm().item() if not out.isnan().any() else -1)
            return out

        # ---------------------------------------------------------------
        # Tier 0.5: NaiveBatchedExperts from ds_vllm
        # Per-expert loop with view transpose + @ (cublas transB)
        # No physical transpose, no weight gather copy
        # Source: ds_vllm/vllm/.../experts/fused_batched_moe.py
        # ---------------------------------------------------------------
        if _USE_NAIVE_BATCHED_MOE and hidden_states.shape[0] > 1:
            w13 = self.experts.w13_weight  # (E, 2*I, H)
            w2 = self.experts.w2_weight    # (E, H, I)

            # topk routing (reuse existing corex/xllm/pytorch topk)
            if _USE_XLLM_MOE:
                topk_weights, topk_ids = _xllm_moe.moe_fused_topk(
                    router_logits.float(), self.top_k, True, None, "softmax")
                topk_ids = topk_ids.to(torch.int64)
                topk_weights = topk_weights.to(hidden_states.dtype)
            elif _USE_COREX_MOE_TOPK_SOFTMAX:
                topk_weights, topk_ids = _corex_moe_topk_softmax.moe_topk_softmax(
                    router_logits.float(), self.top_k, True)
                topk_ids = topk_ids.to(torch.int64)
                topk_weights = topk_weights.to(hidden_states.dtype)
            else:
                topk_logits, topk_ids = torch.topk(
                    router_logits.float(), self.top_k, dim=-1)
                topk_weights = torch.softmax(topk_logits, dim=-1)
                topk_weights = topk_weights.to(hidden_states.dtype)

            return naive_batched_moe_forward(
                hidden_states, w13, w2,
                topk_ids, topk_weights,
                act_fn=self.act_fn)

        # ---------------------------------------------------------------
        # Tier 1: Point-optimized Python loop (individual corex .so)
        # ---------------------------------------------------------------
        # Fused topk+softmax: single CUB kernel vs 2 PyTorch ops.
        # Source: xllm/core/kernels/cuda/moe/moe_topk_softmax_kernels.cuh
        if _USE_XLLM_MOE:
            topk_weights, topk_ids = _xllm_moe.moe_fused_topk(
                router_logits.float(), self.top_k, True, None, "softmax")
            topk_ids = topk_ids.to(torch.int64)
            topk_weights = topk_weights.to(hidden_states.dtype)
        elif _USE_COREX_MOE_TOPK_SOFTMAX:
            topk_weights, topk_ids = _corex_moe_topk_softmax.moe_topk_softmax(
                router_logits.float(), self.top_k, True)
            topk_ids = topk_ids.to(torch.int64)
            topk_weights = topk_weights.to(hidden_states.dtype)
        else:
            topk_logits, topk_ids = torch.topk(
                router_logits.float(), self.top_k, dim=-1)     # (T, top_k)
            topk_weights = torch.softmax(topk_logits, dim=-1)
            topk_weights = topk_weights.to(hidden_states.dtype)

        w13 = self.experts.w13_weight  # (E_local, 2*I_tp, H) or (E_global, 2*I_tp, H)
        w2  = self.experts.w2_weight   # (E_local, H, I_tp) or (E_global, H, I_tp)

        # --- EP: mask non-local experts, remap global ids to local ---
        # Ported from tpu-inference PR #2137 (7163afb1) ragged_gather:
        #   In EP mode, instead of assigning ghost tokens to local experts
        #   (weight=0 but still computed), we FILTER THEM OUT entirely.
        #   This reduces per-expert GEMM from T*topk to ~T*topk/ep_size.
        _ep = getattr(self.experts, '_ep_enabled', False)
        if _ep:
            from vllm.ep_fused_moe_patch import ep_mask_and_remap
            topk_ids, topk_weights, _local_mask = ep_mask_and_remap(
                topk_ids, topk_weights,
                self.experts._start_expert_id,
                self.experts._num_experts_per_rank,
                self.experts._ep_rank)
            if not hasattr(self, '_ep_diag_done'):
                self._ep_diag_done = True
                _lc = _local_mask.sum().item()
                _tc = _local_mask.numel()
                logger.info(
                    "[EP] rank=%d experts=[%d,%d) local_hits=%d/%d (%.1f%%) "
                    "w13=%s w2=%s",
                    self.experts._ep_rank,
                    self.experts._start_expert_id,
                    self.experts._start_expert_id + self.experts._num_experts_per_rank,
                    _lc, _tc, 100.0 * _lc / max(_tc, 1),
                    list(w13.shape), list(w2.shape))

        T = hidden_states.shape[0]
        if T == 1:
            # Fast path: single token (decode).
            eids    = topk_ids[0]                              # (K,)
            ws      = topk_weights[0].to(hidden_states.dtype)  # (K,)

            # --- EP T=1: skip ghost experts ---
            # In EP mode, ~6/8 experts have weight=0 (non-local).
            # Must skip them: EP intermediate_size=1024 (full) vs TP=256 (1/4).
            # 8 experts × 1024 = 4× more compute than 8 × 256.
            # Skipping to ~2 local experts: 2 × 1024 = 2048, same as TP's 8 × 256.
            #
            # .item() here does ONE GPU→CPU sync per layer. CUDA pipelines
            # the 8 scalar transfers into one sync. 40 layers × ~30μs = ~1.2ms,
            # far less than the ~50ms saved by skipping 6 experts' GEMM.
            if _ep:
                valid = ws != 0
                K_local = valid.sum().item()
                if K_local == 0:
                    return torch.zeros(1, hidden_states.shape[-1],
                                       dtype=hidden_states.dtype,
                                       device=hidden_states.device)
                eids = eids[valid]                              # (K_local,)
                ws   = ws[valid]                                # (K_local,)

                w13_sel = w13[eids]                            # (K_local, 2*I, H)
                w2_sel = w2[eids]                              # (K_local, H, I)
                H = hidden_states.shape[-1]

                gate_up = _fast_linear(
                    hidden_states,
                    w13_sel.reshape(-1, H),                    # (K_local*2*I, H)
                )                                              # (1, K_local*2*I)
                gate_up = gate_up.view(K_local, -1)            # (K_local, 2*I)

                if _USE_FUSED_MOE_ACTIVATION:
                    act = self.act_fn(gate_up)
                else:
                    gate, up = gate_up.chunk(2, dim=-1)
                    act = F.silu(gate) * up

                expert_out = torch.bmm(w2_sel, act.unsqueeze(-1)).squeeze(-1)
                out = (expert_out * ws.unsqueeze(-1)).sum(
                    0, keepdim=True).to(hidden_states.dtype)   # (1, H)
            else:
                # --- TP mode: all 8 experts are local ---
                # xllm warp64-safe path: gather weights → fused GEMM → combine
                K = eids.shape[0]
                H = hidden_states.shape[-1]
                w13_sel = w13[eids]                                # (K, 2*I, H)
                w2_sel = w2[eids]                                  # (K, H, I)

                # FC1: single large GEMM via _fast_linear (ix_moe_bridge GEMV)
                gate_up = _fast_linear(
                    hidden_states,
                    w13_sel.reshape(-1, H),                        # (K*2*I, H)
                )                                                  # (1, K*2*I)
                gate_up = gate_up.view(K, -1)                      # (K, 2*I)

                if _USE_FUSED_MOE_ACTIVATION:
                    act = self.act_fn(gate_up)                      # (K, I)
                else:
                    gate, up = gate_up.chunk(2, dim=-1)
                    act = F.silu(gate) * up

                # FC2: bmm (K, H, I) @ (K, I, 1) → (K, H)
                expert_out = torch.bmm(w2_sel, act.unsqueeze(-1)).squeeze(-1)

                # Combine: xllm fused kernel or PyTorch weighted sum
                if _USE_XLLM_MOE:
                    # expert_out is (K, H), need (1*K, H) for combine
                    out = _xllm_moe.moe_combine_result(
                        expert_out, ws.float().unsqueeze(0), 1, K) # (1, H)
                else:
                    out = (expert_out * ws.unsqueeze(-1)).sum(
                        0, keepdim=True).to(hidden_states.dtype)   # (1, H)
        else:
            # General path (prefill / multi-seq): group assignments once.
            out = torch.zeros_like(hidden_states)
            flat_eids = topk_ids.reshape(-1)
            flat_weights = topk_weights.reshape(-1)

            # --- EP: upstream _process_tokens_locally approach ---
            # (PR #2137, 7163afb1 — fused_moe_gmm.py lines 674-718)
            #
            # Upstream sorts by GLOBAL expert id (no mask, no remap),
            # then uses prefix-sum (group_offsets) to locate the
            # contiguous [start, end) range of locally-routed tokens.
            # The per-expert loop only iterates over local experts.
            #
            # Key: NO boolean mask, NO .item(), NO dynamic shape.
            # The sort + prefix-sum are all static-shape tensor ops.
            #
            # In EP mode, ep_mask_and_remap already remapped to local ids.
            # But the upstream approach is different: it keeps global ids
            # for sorting, and only narrows the loop range.
            #
            # For our code: since ep_mask_and_remap already remapped to
            # local [0, E_local) and zeroed non-local weights, we sort
            # by local id. Non-local entries have weight=0 and are
            # distributed across local experts (via mod). The per-expert
            # loop processes ALL entries including ghosts, but ghosts
            # contribute nothing due to weight=0. This matches the upstream
            # behavior where ragged_gather_reduce applies valid_rows_mask
            # to zero out non-local entries (line 264-270 in moe_gmm_local).

            if _USE_XLLM_MOE:
                # xllm CUDA: histogram + prefix_sum + place
                src_dst, dst_src, expert_sizes = \
                    _xllm_moe.moe_compute_index(flat_eids, w13.shape[0])
                sorted_tok_ids = torch.arange(
                    T, device=topk_ids.device
                ).repeat_interleave(self.top_k)[dst_src.long()]
                sorted_weights = flat_weights[dst_src.long()]
                expert_counts = expert_sizes.tolist()
            elif _USE_COREX_MOE_INDEX_COMBINE:
                # Fused CUDA: histogram + prefix_sum + place (11.5x faster)
                src_dst, dst_src, expert_sizes = \
                    _corex_moe_index_combine.moe_compute_index(
                        flat_eids, w13.shape[0])
                sorted_tok_ids = torch.arange(
                    T, device=topk_ids.device
                ).repeat_interleave(self.top_k)[dst_src.long()]
                sorted_weights = flat_weights[dst_src.long()]
                expert_counts = expert_sizes.tolist()
            else:
                order = torch.argsort(flat_eids, stable=True)
                sorted_tok_ids = torch.arange(
                    T, device=topk_ids.device
                ).repeat_interleave(self.top_k)[order]
                sorted_weights = flat_weights[order]
                expert_counts = torch.bincount(
                    flat_eids, minlength=w13.shape[0]).tolist()

            # --- CUTLASS grouped GEMM path (replaces per-expert F.linear loop) ---
            if _USE_GEMM_GROUPED and hidden_states.dtype == torch.float16:
                # Sort tokens into expert order
                sorted_hidden = hidden_states[sorted_tok_ids]  # (T*topk, H)
                expert_counts_t = torch.tensor(
                    expert_counts, dtype=torch.int32,
                    device=hidden_states.device) if not isinstance(
                    expert_counts, torch.Tensor) else expert_counts

                # Step 4: grouped GEMM w13 (gate_proj + up_proj)
                gemm1_out = _gemm_grouped.moe_group_gemm(
                    sorted_hidden, w13, expert_counts_t)  # (T*topk, 2*I)
                gate, up = gemm1_out.chunk(2, dim=-1)
                act_out = F.silu(gate) * up  # (T*topk, I)

                # Step 6: grouped GEMM w2 (down_proj)
                gemm2_out = _gemm_grouped.moe_group_gemm(
                    act_out, w2, expert_counts_t)  # (T*topk, H)

                # Step 7: weighted combine back to token order
                flat_weights = sorted_weights.unsqueeze(-1)  # (T*topk, 1)
                weighted = (gemm2_out * flat_weights).to(out.dtype)
                out.index_add_(0, sorted_tok_ids, weighted)
            else:
                # Fallback: per-expert F.linear loop
                start = 0
                for eid, count in enumerate(expert_counts):
                    end = start + count
                    if count == 0:
                        start = end
                        continue
                    tok_ids = sorted_tok_ids[start:end]
                    tokens = hidden_states[tok_ids]                # (n, H)
                    gate_up = _fast_linear(tokens, w13[eid])           # (n, 2*I)
                    gate, up = gate_up.chunk(2, dim=-1)
                    act = F.silu(gate) * up                        # (n, I)
                    expert_out = _fast_linear(act, w2[eid])            # (n, H)
                    weights = sorted_weights[start:end].unsqueeze(-1)
                    out.index_add_(0, tok_ids, (expert_out * weights).to(out.dtype))
                    start = end

        # One-shot diagnostic: log routed output norm after first forward
        if not hasattr(self, '_routed_diag_done'):
            self._routed_diag_done = True
            _ep = getattr(self.experts, '_ep_enabled', False)
            logger.info(
                "[MoE_OUT] EP=%s T=%d out_shape=%s out_norm=%.6f "
                "out_abs_max=%.6f out_has_nan=%s out_has_inf=%s",
                _ep, hidden_states.shape[0], list(out.shape),
                out.float().norm().item(),
                out.float().abs().max().item(),
                bool(out.isnan().any()), bool(out.isinf().any()))
        return out  # partial, all-reduce done in forward()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with bi100_timer("moe.router"):
            router_and_shared_gate, _ = self.router_shared_gate(hidden_states)
            router_logits = router_and_shared_gate[..., :self.num_experts]
            gate_score = router_and_shared_gate[..., self.num_experts:]
        with bi100_timer("moe.routed"):
            routed_out = self._pure_pytorch_experts(hidden_states, router_logits)

        with bi100_timer("moe.shared"):
            gate_up, _ = self.shared_expert_gate_up(hidden_states)
            shared_out = self.act_fn(gate_up)
            shared_out, _ = self.shared_expert_down(shared_out)
            shared_out = shared_out * torch.sigmoid(gate_score)

        # --- Reduction ---
        # Ported from tpu-inference:
        #   PR #2679 (df7f5b35): scatter_results / defer_all_reduce
        #   PR #3435 (57987c2):  shared expert reduce axis under attn DP
        #
        # TP mode: routed_out + shared_out are both TP-partial → single all-reduce
        # EP mode: routed_out is EP-partial, shared_out is TP-partial
        #          Since TP group == EP group == WORLD → same single all-reduce
        #          (defer_all_reduce pattern: combine first, reduce once)
        _ep = getattr(self.experts, '_ep_enabled', False)
        if _ep:
            from vllm.ep_fused_moe_patch import ep_reduce_output
            with bi100_timer("moe.ep_reduce"):
                out = ep_reduce_output(routed_out, shared_out)
        else:
            with bi100_timer("moe.combine"):
                out = routed_out + shared_out
            if self.experts.tp_size > 1:
                with bi100_timer("moe.all_reduce"):
                    out = tensor_model_parallel_all_reduce(out)
        _fwd_cnt = getattr(self, '_fwd_diag_cnt', 0)
        if _fwd_cnt < 3:
            self._fwd_diag_cnt = _fwd_cnt + 1
            logger.info(
                "[MoE_FWD] ep=%s call=%d T=%d routed=%.4f shared=%.4f final=%.4f",
                _ep, _fwd_cnt, hidden_states.shape[0],
                routed_out.float().norm().item(),
                shared_out.float().norm().item(),
                out.float().norm().item())
        return out


# ---------------------------------------------------------------------------
# Decoder layer  (dispatches to GatedDeltaNet or Qwen3_5FullAttention)
# ---------------------------------------------------------------------------


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(
        self,
        text_cfg,
        layer_idx: int,
        layer_type: str,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self._diagnostic_trace_pending = (
            os.getenv("BI100_DIAGNOSTIC_LAYER_TRACE") == "1")
        self.input_layernorm = GemmaRMSNorm(text_cfg.hidden_size,
                                           eps=text_cfg.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(text_cfg.hidden_size,
                                                     eps=text_cfg.rms_norm_eps)

        if layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(text_cfg, layer_idx,
                                             quant_config=quant_config)
        else:
            self.self_attn = Qwen3_5FullAttention(
                text_cfg, layer_idx,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"layers.{layer_idx}.self_attn",
            )

        if getattr(text_cfg, 'model_type', '') == 'qwen3_5_moe_text':
            self.mlp = Qwen3_5MoeSparseBlock(text_cfg, quant_config=quant_config)
        else:
            self.mlp = Qwen3_5MLP(
                hidden_size=text_cfg.hidden_size,
                intermediate_size=text_cfg.intermediate_size,
                hidden_act=text_cfg.hidden_act,
                quant_config=quant_config,
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        # Only for linear_attention layers:
        conv_state: Optional[torch.Tensor] = None,
        temporal_state: Optional[torch.Tensor] = None,
        gdn_capture_offsets: Optional[Iterable[int]] = None,
        gdn_segment_offsets: Optional[Iterable[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with bi100_timer("layer.input_norm"):
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(
                    hidden_states, residual)

        if self.layer_type == "linear_attention":
            with bi100_timer("layer.gdn"):
                hidden_states = self.linear_attn(
                    hidden_states, attn_metadata, conv_state, temporal_state,
                    capture_offsets=gdn_capture_offsets,
                    segment_offsets=gdn_segment_offsets)
        else:
            with bi100_timer("layer.full_attn"):
                hidden_states = self.self_attn(
                    positions, hidden_states, kv_cache, attn_metadata)

        with bi100_timer("layer.post_attn_norm"):
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)

        with bi100_timer("layer.moe"):
            hidden_states = self.mlp(hidden_states)

        if self._diagnostic_trace_pending:
            self._diagnostic_trace_pending = False
            rank = os.getenv("RANK", os.getenv("LOCAL_RANK", "?"))
            print(
                "[BI100 DIAGNOSTIC] "
                f"rank={rank} layer={self.layer_idx} "
                f"attention={self.layer_type} "
                f"mlp={type(self.mlp).__name__} stage=completed",
                file=sys.stderr,
                flush=True,
            )

        return hidden_states, residual


# ---------------------------------------------------------------------------
# Full transformer model
# ---------------------------------------------------------------------------

def _validate_qwen_kv_cache_count(configured_count, kv_caches):
    # vllm allocates num_hidden_layers KV caches; we only use the first
    # configured_count (full_attention layers). Accept >= instead of ==.
    if len(kv_caches) < configured_count:
        raise RuntimeError(
            "Qwen3.5 KV cache count insufficient: "
            f"need {configured_count}, received {len(kv_caches)}")


class Qwen3_5Model(nn.Module):
    def __init__(
        self,
        text_cfg,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        kv_cache_count: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.text_cfg = text_cfg
        full_attention_count = sum(
            layer_type == "full_attention"
            for layer_type in text_cfg.layer_types)
        if kv_cache_count is None:
            kv_cache_count = full_attention_count
        if (not isinstance(kv_cache_count, int) or isinstance(kv_cache_count, bool)
                or kv_cache_count < full_attention_count):
            raise RuntimeError(
                "Qwen3.5 configured KV cache count must cover every "
                f"full-attention layer: configured {kv_cache_count}, "
                f"required {full_attention_count}")
        self.kv_cache_count = kv_cache_count
        self.embed_tokens = VocabParallelEmbedding(
            text_cfg.vocab_size, text_cfg.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3_5DecoderLayer(
                text_cfg, i, text_cfg.layer_types[i],
                cache_config=cache_config, quant_config=quant_config)
            for i in range(text_cfg.num_hidden_layers)
        ])
        self.norm = GemmaRMSNorm(text_cfg.hidden_size, eps=text_cfg.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        conv_states: torch.Tensor,     # (num_linear_layers, batch, ...)
        temporal_states: torch.Tensor, # (num_linear_layers, batch, ...)
        inputs_embeds: Optional[torch.Tensor] = None,
        gdn_capture_offsets: Optional[Iterable[int]] = None,
        gdn_segment_offsets: Optional[Iterable[int]] = None,
    ) -> torch.Tensor:
        _validate_qwen_kv_cache_count(self.kv_cache_count, kv_caches)
        with bi100_timer("model.embed"):
            hidden_states = (self.embed_tokens(input_ids)
                             if inputs_embeds is None else inputs_embeds)
        residual = None

        attn_idx = 0
        linear_idx = 0
        capture_offsets = tuple(gdn_capture_offsets or ())
        captured_conv_states: Dict[int, List[torch.Tensor]] = {
            offset: [] for offset in capture_offsets
        }
        captured_temporal_states: Dict[int, List[torch.Tensor]] = {
            offset: [] for offset in capture_offsets
        }
        for layer in self.layers:
            if layer.layer_type == "linear_attention":
                hidden_states, residual = layer(
                    positions, hidden_states,
                    kv_cache=None,
                    attn_metadata=attn_metadata,
                    residual=residual,
                    conv_state=conv_states[linear_idx],
                    temporal_state=temporal_states[linear_idx],
                    gdn_capture_offsets=capture_offsets,
                    gdn_segment_offsets=gdn_segment_offsets,
                )
                for offset in capture_offsets:
                    captured_conv_states[offset].append(
                        layer.linear_attn.captured_conv_states[offset])
                    captured_temporal_states[offset].append(
                        layer.linear_attn.captured_temporal_states[offset])
                linear_idx += 1
            else:
                kv_cache = kv_caches[attn_idx]
                hidden_states, residual = layer(
                    positions, hidden_states,
                    kv_cache=kv_cache,
                    attn_metadata=attn_metadata,
                    residual=residual,
                )
                attn_idx += 1

        with bi100_timer("model.final_norm"):
            hidden_states, _ = self.norm(hidden_states, residual)
        self.captured_conv_states = {
            offset: torch.stack(states)
            for offset, states in captured_conv_states.items()
        }
        self.captured_temporal_states = {
            offset: torch.stack(states)
            for offset, states in captured_temporal_states.items()
        }
        return hidden_states


# ---------------------------------------------------------------------------
# Top-level CausalLM wrapper with MambaCacheManager
# ---------------------------------------------------------------------------

class Qwen3_5ForCausalLM(nn.Module, HasInnerState, SupportsLoRA,
                         SupportsMultiModal):

    has_inner_state = True
    supports_lora = True

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    supported_lora_modules = [
        "gate_up_proj",
        "down_proj",
        "o_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(
        self,
        config=None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        scheduler_config: Optional[SchedulerConfig] = None,
        multimodal_config: Optional[MultiModalConfig] = None,
        prefix: str = "",
        vllm_config=None,
        **kwargs,
    ) -> None:
        # BI100: support both new-style (vllm_config=) and old-style init
        if vllm_config is not None:
            config = config or vllm_config.model_config.hf_config
            cache_config = cache_config or vllm_config.cache_config
            quant_config = quant_config or getattr(vllm_config, 'quant_config', None)
            lora_config = lora_config or getattr(vllm_config, 'lora_config', None)
            scheduler_config = scheduler_config or vllm_config.scheduler_config
            multimodal_config = multimodal_config or getattr(
                vllm_config, 'multimodal_config',
                getattr(vllm_config.model_config, 'multimodal_config', None))
        # Apply ix_bridge operator patches on first model init (safe: GPU is ready)
        try:
            from vllm import ix_startup_patch
            ix_startup_patch.apply()
        except Exception:
            pass
        _bi100_model_trace("Qwen3_5ForCausalLM initialization begin")
        super().__init__()
        self._vllm_config = vllm_config  # kept for new-style MambaCacheManager
        self.config = config
        self.scheduler_config = scheduler_config
        self.multimodal_config = multimodal_config

        # The text config holds all architecture parameters
        text_cfg = config.text_config
        self.text_cfg = text_cfg
        rope_parameters = getattr(text_cfg, "rope_parameters", {}) or {}
        mrope_sections = rope_parameters.get("mrope_section", [11, 11, 10])
        if getattr(config, "rope_scaling", None) is None:
            config.rope_scaling = {
                "type": "mrope",
                "mrope_section": mrope_sections,
            }

        # Pre-compute counts
        self.num_linear_layers = sum(
            1 for lt in text_cfg.layer_types if lt == "linear_attention")
        self.num_attn_layers = sum(
            1 for lt in text_cfg.layer_types if lt == "full_attention")
        # Use text_cfg.layer_types directly: only full_attention layers need KV cache
        self.num_kv_cache_layers = self.num_attn_layers
        if self.num_kv_cache_layers < self.num_attn_layers:
            raise RuntimeError(
                "Qwen3.5 KV accounting provides fewer caches than "
                f"full-attention layers: {self.num_kv_cache_layers} < "
                f"{self.num_attn_layers}")
        accounting_mode = getattr(
            config, "bi100_hybrid_kv_accounting_mode", "legacy40")
        accounting_env = os.getenv("BI100_HYBRID_KV_ACCOUNTING", "<unset>")
        tp_rank = get_tensor_model_parallel_rank()
        full_attention_ordinals = ",".join(
            str(index) for index, layer_type in enumerate(text_cfg.layer_types)
            if layer_type == "full_attention")
        logger.info(
            "[BI100] Qwen hybrid KV accounting; tp_rank=%d "
            "env_mode=%s config_mode=%s "
            "configured_kv_layers=%d full_attention_layers=%d "
            "full_attention_ordinals=%s",
            tp_rank,
            accounting_env,
            accounting_mode,
            self.num_kv_cache_layers,
            self.num_attn_layers,
            full_attention_ordinals,
        )

        # DeltaNet state dimensions (per layer, per sequence, TP-sharded)
        tp_size = get_tensor_model_parallel_world_size()
        self.conv_dim = (text_cfg.linear_num_key_heads * text_cfg.linear_key_head_dim * 2
                         + text_cfg.linear_num_value_heads * text_cfg.linear_value_head_dim)
        self.num_v_heads = text_cfg.linear_num_value_heads
        self.head_k_dim = text_cfg.linear_key_head_dim
        self.head_v_dim = text_cfg.linear_value_head_dim
        self.conv_kernel_size = text_cfg.linear_conv_kernel_dim

        self.model = Qwen3_5Model(
            text_cfg,
            cache_config=cache_config,
            quant_config=quant_config,
            kv_cache_count=self.num_kv_cache_layers,
        )

        self.visual = Qwen3_5VisionTransformer(
            config.vision_config,
            quant_config=None,
        )

        self.lm_head = ParallelLMHead(
            text_cfg.vocab_size, text_cfg.hidden_size,
            quant_config=quant_config,
        )

        self.logits_processor = LogitsProcessor(text_cfg.vocab_size)
        self.sampler = Sampler()

        # Lazy initialised in first forward call
        self.mamba_cache: Optional[MambaCacheManager] = None

        # Scheduler-owned recurrent prefix states. Keys are stable chained
        # content hashes, never recyclable physical KV block ids.
        self._gdn_prefix_cache: Dict[
            Tuple[int, bytes], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._block_size: int = (cache_config.block_size
                                  if cache_config is not None else 16)
        self._startup_forward_traced = False
        _bi100_model_trace("Qwen3_5ForCausalLM initialization complete")

    def _get_mamba_cache_shape(self):
        tp_size = get_tensor_model_parallel_world_size()
        # Each sequence's state is stored in float32
        conv_state_shape = (self.conv_dim // tp_size, self.conv_kernel_size - 1)
        temporal_state_shape = (
            self.num_v_heads // tp_size, self.head_k_dim, self.head_v_dim)
        return conv_state_shape, temporal_state_shape

    @staticmethod
    def _validate_and_reshape_mm_tensor(
        mm_input: Union[torch.Tensor, List[torch.Tensor]],
        name: str,
    ) -> torch.Tensor:
        if isinstance(mm_input, list):
            return torch.cat(mm_input)
        if not isinstance(mm_input, torch.Tensor):
            raise ValueError(f"incorrect type for {name}: {type(mm_input)}")
        if mm_input.ndim == 2:
            return mm_input
        if mm_input.ndim == 3:
            return torch.cat(list(mm_input))
        raise ValueError(
            f"{name} must be a 2D tensor or batched 3D tensor, got "
            f"shape={tuple(mm_input.shape)}")

    def _parse_and_validate_image_input(
        self,
        **kwargs: object,
    ) -> Optional[Qwen3_5ImageInputs]:
        pixel_values = kwargs.get("pixel_values")
        image_embeds = kwargs.get("image_embeds")
        image_grid_thw = kwargs.get("image_grid_thw")
        if pixel_values is None and image_embeds is None:
            return None
        if pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("image_grid_thw is required with pixel_values")
            return Qwen3_5ImagePixelInputs(
                type="pixel_values",
                data=self._validate_and_reshape_mm_tensor(
                    pixel_values, "image pixel values"),
                image_grid_thw=self._validate_and_reshape_mm_tensor(
                    image_grid_thw, "image grid_thw"),
            )
        return Qwen3_5ImageEmbeddingInputs(
            type="image_embeds",
            data=self._validate_and_reshape_mm_tensor(
                image_embeds, "image embeddings"),
        )

    def _process_image_input(
        self,
        image_input: Qwen3_5ImageInputs,
    ) -> torch.Tensor:
        if image_input["type"] == "image_embeds":
            return image_input["data"].to(dtype=self.visual.dtype,
                                           device=self.visual.device)
        return self.visual(
            image_input["data"],
            grid_thw=image_input["image_grid_thw"],
        )

    @bi100_profile_transaction
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs,
    ) -> torch.Tensor:
        if not self._startup_forward_traced:
            self._startup_forward_traced = True
            _bi100_model_trace("first model forward entered")
        if self.mamba_cache is None:
            conv_shape, temporal_shape = self._get_mamba_cache_shape()
            # New-style MambaCacheManager takes (vllm_config, dtype,
            # num_mamba_layers, conv_state_shape, temporal_state_shape)
            import inspect as _inspect
            _mcm_params = list(
                _inspect.signature(MambaCacheManager.__init__).parameters)
            if "vllm_config" in _mcm_params:
                self.mamba_cache = MambaCacheManager(
                    self._vllm_config,
                    torch.float32,
                    self.num_linear_layers,
                    conv_shape,
                    temporal_shape,
                )
            else:
                if self.scheduler_config is not None:
                    max_batch_size = self.scheduler_config.max_num_seqs
                else:
                    max_batch_size = 256
                self.mamba_cache = MambaCacheManager(
                    torch.float32,
                    self.num_linear_layers,
                    max_batch_size,
                    *self._get_mamba_cache_shape(),
                )

        gdn_restore_key = kwargs.pop("gdn_restore_key", None)
        gdn_capture_points = kwargs.pop("gdn_capture_points", None) or []
        gdn_evict_keys = kwargs.pop("gdn_evict_keys", None) or []
        gdn_segment_offsets = kwargs.pop("gdn_segment_offsets", None) or []

        mamba_cache_params = self.mamba_cache.current_run_tensors(**kwargs)
        # New API returns MambaCacheParams with .conv_state, .ssm_state,
        # .state_indices_tensor; old API returned (conv_states, temporal_states)
        _mamba_state_indices = None
        if hasattr(mamba_cache_params, 'conv_state'):
            conv_states = mamba_cache_params.conv_state
            temporal_states = mamba_cache_params.ssm_state
            if hasattr(mamba_cache_params, 'state_indices_tensor'):
                _mamba_state_indices = mamba_cache_params.state_indices_tensor.long()
                conv_states = conv_states[:, _mamba_state_indices].contiguous()
                temporal_states = temporal_states[:, _mamba_state_indices].contiguous()
        else:
            conv_states, temporal_states = mamba_cache_params

        _is_single_seq_prefill = (
            attn_metadata is not None
            and attn_metadata.num_prefill_tokens > 0
            and conv_states.shape[1] == 1               # batch == 1
            and getattr(attn_metadata, 'context_lens_tensor', None) is not None
        )
        has_gdn_actions = (gdn_restore_key is not None
                           or bool(gdn_capture_points)
                           or bool(gdn_evict_keys)
                           or bool(gdn_segment_offsets))
        if has_gdn_actions and not _is_single_seq_prefill:
            raise RuntimeError(
                "GDN prefix-cache actions require a single-sequence prefill")

        for evict_key in gdn_evict_keys:
            self._gdn_prefix_cache.pop(_validate_gdn_prefix_key(evict_key),
                                       None)

        if gdn_restore_key is not None:
            restore_key = _validate_gdn_prefix_key(gdn_restore_key)
            saved_state = self._gdn_prefix_cache.get(restore_key)
            if saved_state is None:
                raise RuntimeError(
                    "scheduler requested a missing GDN prefix state: "
                    f"blocks={restore_key[0]} digest={restore_key[1].hex()}")
            saved_conv, saved_temporal = saved_state
            with bi100_timer("gdn_prefix.restore"):
                conv_states[:, 0].copy_(
                    saved_conv.to(device=conv_states.device,
                                  dtype=conv_states.dtype),
                    non_blocking=True)
                temporal_states[:, 0].copy_(
                    saved_temporal.to(device=temporal_states.device,
                                      dtype=temporal_states.dtype),
                    non_blocking=True)

        query_len = (int(attn_metadata.num_prefill_tokens)
                     if _is_single_seq_prefill else 0)
        capture_keys: Dict[int, Tuple[int, bytes]] = {}
        for capture_point in gdn_capture_points:
            if not isinstance(capture_point, tuple) or len(capture_point) != 2:
                raise RuntimeError(
                    f"invalid GDN capture point: {capture_point!r}")
            offset, capture_key = capture_point
            if (not isinstance(offset, int) or offset <= 0
                    or offset > query_len or offset in capture_keys):
                raise RuntimeError(
                    f"invalid GDN capture offset: {offset!r} "
                    f"for query_len={query_len}")
            capture_keys[offset] = _validate_gdn_prefix_key(capture_key)
        if len(capture_keys) > 2:
            raise RuntimeError("at most two GDN capture points are supported")
        interior_capture_offsets = tuple(
            offset for offset in capture_keys if offset < query_len)
        segment_offsets = set()
        for offset in gdn_segment_offsets:
            if (not isinstance(offset, int) or offset <= 0
                    or offset >= query_len):
                raise RuntimeError(
                    f"invalid GDN segment offset: {offset!r} "
                    f"for query_len={query_len}")
            segment_offsets.add(offset)
        if len(segment_offsets) > 128:
            raise RuntimeError("at most 128 GDN segment offsets are supported")
        interior_segment_offsets = tuple(sorted(segment_offsets))

        inputs_embeds = None
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is not None:
            image_mask = input_ids == self.config.image_token_id
            num_placeholders = int(image_mask.sum().item())
            if num_placeholders:
                inputs_embeds = self.model.embed_tokens(input_ids)
                image_embeds = self._process_image_input(image_input)
                if num_placeholders > image_embeds.shape[0]:
                    raise ValueError(
                        f"image token count ({num_placeholders}) exceeds "
                        f"vision embeddings ({image_embeds.shape[0]})")
                # Prefix caching can consume the leading image tokens while
                # vLLM 0.6 still supplies the full pixel tensor. The query's
                # remaining placeholders always form a suffix of the flattened
                # visual token stream.
                image_embeds = image_embeds[-num_placeholders:]
                inputs_embeds[image_mask, :] = image_embeds.to(
                    inputs_embeds.dtype)

        with bi100_timer("model.forward"):
            hidden_states = self.model(
                input_ids, positions, kv_caches, attn_metadata,
                conv_states, temporal_states,
                inputs_embeds=inputs_embeds,
                gdn_capture_offsets=interior_capture_offsets,
                gdn_segment_offsets=interior_segment_offsets)

        # Scatter modified GDN states back into the full cache
        if _mamba_state_indices is not None:
            mamba_cache_params.conv_state[:, _mamba_state_indices] = conv_states
            mamba_cache_params.ssm_state[:, _mamba_state_indices] = temporal_states

        for offset, capture_key in capture_keys.items():
            if offset == query_len:
                captured_conv = conv_states[:, 0]
                captured_temporal = temporal_states[:, 0]
            else:
                captured_conv = self.model.captured_conv_states[offset]
                captured_temporal = self.model.captured_temporal_states[offset]
            with bi100_timer("gdn_prefix.save"):
                self._gdn_prefix_cache[capture_key] = (
                    captured_conv.detach().cpu().clone(),
                    captured_temporal.detach().cpu().clone(),
                )

        if bi100_profile_event_enabled():
            profile_prefill_tokens = int(
                getattr(attn_metadata, "num_prefill_tokens", 0) or 0)
            profile_decode_tokens = int(
                getattr(attn_metadata, "num_decode_tokens", 0) or 0)
            profile_context_len = 0
            if profile_prefill_tokens > 0:
                profile_seq_lens = getattr(attn_metadata, "seq_lens", None)
                if (not isinstance(profile_seq_lens, list)
                        or len(profile_seq_lens) != 1
                        or not isinstance(profile_seq_lens[0], int)):
                    raise RuntimeError(
                        "BI100 profile requires one host-visible prefill "
                        "sequence length")
                profile_context_len = (
                    profile_seq_lens[0] - profile_prefill_tokens)
                if profile_context_len < 0:
                    raise RuntimeError(
                        "BI100 profile observed a negative prefill context")
            bi100_profile_flush(
                tp_rank=get_tensor_model_parallel_rank(),
                phase=("prefill" if profile_prefill_tokens > 0 else "decode"),
                prefill_tokens=profile_prefill_tokens,
                decode_tokens=profile_decode_tokens,
                context_len=profile_context_len,
                gdn_restore=bool(gdn_restore_key is not None),
                gdn_capture_points=len(gdn_capture_points),
                gdn_evict_keys=len(gdn_evict_keys),
            )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        # All TP ranks must call logits_processor to participate in the NCCL
        # gather inside lm_head. Non-driver ranks return None after the gather.
        # With chunked prefill, intermediate chunks have seq_groups=None on all
        # ranks; _apply_logits_processors is guarded against this in
        # logits_processor.py (patched by patch_xformers_sdpa_seq.py).
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        if logits is not None:
            _cnt = getattr(self, '_logits_diag_cnt', 0)
            if _cnt < 5:
                self._logits_diag_cnt = _cnt + 1
                try:
                    top5_vals, top5_ids = logits[-1].topk(5)
                    logger.info(
                        "[LOGITS] call=%d shape=%s last_row_norm=%.4f "
                        "top5_ids=%s top5_vals=%s "
                        "has_nan=%s min=%.4f max=%.4f",
                        _cnt, list(logits.shape),
                        logits[-1].float().norm().item(),
                        top5_ids.tolist(), top5_vals.tolist(),
                        bool(logits.isnan().any()),
                        logits.min().item(), logits.max().item())
                except Exception as e:
                    logger.info("[LOGITS] call=%d diag failed: %s", _cnt, e)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.sampler(logits, sampling_metadata)

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.mamba_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.mamba_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        _bi100_model_trace("dense load_weights begin")
        loaded_count = 0
        stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            loaded_count += 1
            # Skip vision and MTP branches
            if (name.startswith("model.visual")
                    or name.startswith("mtp.")
                    or name.startswith("model.mtp")):
                continue

            # Prefix remapping: checkpoint may wrap under language_model
            if name.startswith("model.language_model."):
                name = "model." + name[len("model.language_model."):]

            # Skip positional embedding caches
            if "rotary_emb.inv_freq" in name:
                continue

            if _load_full_attention_qgkv_weight(
                    params_dict, name, loaded_weight, self.text_cfg):
                continue

            if _load_gdn_projection_weight(
                    params_dict, name, loaded_weight, self.text_cfg):
                continue

            # Remap conv1d.weight → conv1d_weight
            # The conv has depth (1) dim in the checkpoint that we handle separately
            if ".linear_attn.conv1d.weight" in name:
                name = name.replace(".linear_attn.conv1d.weight",
                                    ".linear_attn.conv1d_weight")

            # Stacked param loading (gate_up_proj)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    break
                if name not in params_dict:
                    break
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
        _bi100_model_trace(f"dense load_weights complete items={loaded_count}")


# ---------------------------------------------------------------------------
# Qwen3.6-35B-A3B  (Qwen3_5-MoE architecture)
# ---------------------------------------------------------------------------

@MULTIMODAL_REGISTRY.register_image_input_mapper(qwen36_image_input_mapper)
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_qwen36_image_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_qwen36)
@INPUT_REGISTRY.register_input_processor(input_processor_for_qwen36)
class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM):
    """Qwen3.6-35B-A3B: same hybrid-attention backbone as 27B, dense MLP
    replaced by Qwen3_5MoeSparseBlock (256 routed experts + shared expert).
    Only load_weights differs from the dense variant.
    """

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        _bi100_model_trace("MoE load_weights begin")
        loaded_count = 0
        vision_loaded_count = 0
        # Checkpoint key format for this model (transformers Qwen3_5MoeExperts):
        #   mlp.experts.gate_up_proj  shape (num_experts, 2*intermediate, hidden)
        #   mlp.experts.down_proj     shape (num_experts, hidden, intermediate)
        #   mlp.gate.weight           shape (num_experts, hidden)   [router]
        #   mlp.shared_expert_gate.weight shape (1, hidden)
        #   mlp.shared_expert.{gate,up,down}_proj.weight            [shared MLP]
        # Our FusedMoE stores:
        #   mlp.experts.w13_weight    shape (num_experts, 2*intermediate//tp, hidden)
        #   mlp.experts.w2_weight     shape (num_experts, hidden, intermediate//tp)
        # Our router/shared gate stores both tensors in one (num_experts+1, H)
        # replicated weight. Our shared expert stores:
        #   mlp.shared_expert_gate_up.weight  (merged gate+up)
        #   mlp.shared_expert_down.weight

        stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            # shared expert
            ("shared_expert_gate_up", "shared_expert.gate_proj", 0),
            ("shared_expert_gate_up", "shared_expert.up_proj",   1),
            # linear_attention dense proj (same as 27B)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj",   1),
        ]

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            loaded_count += 1
            if name.startswith("model.visual."):
                name = "visual." + name[len("model.visual."):]
                if "attn.qkv.weight" in name:
                    num_heads = self.config.vision_config.num_heads
                    hidden_size = self.config.vision_config.hidden_size
                    head_size = hidden_size // num_heads
                    loaded_weight = loaded_weight.view(
                        3, num_heads, head_size, hidden_size)
                    loaded_weight = loaded_weight.transpose(0, 1).reshape(
                        -1, hidden_size)
                elif "attn.qkv.bias" in name:
                    num_heads = self.config.vision_config.num_heads
                    hidden_size = self.config.vision_config.hidden_size
                    head_size = hidden_size // num_heads
                    loaded_weight = loaded_weight.view(
                        3, num_heads, head_size)
                    loaded_weight = loaded_weight.transpose(0, 1).reshape(-1)
                if name not in params_dict:
                    raise ValueError(f"unexpected Qwen3.6 vision weight: {name}")
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                vision_loaded_count += 1
                continue

            # MTP is not used by the fixed evaluator command.
            if (name.startswith("mtp.")
                    or name.startswith("model.mtp")):
                continue

            # Prefix remapping for VL checkpoint (Qwen3_5MoeForConditionalGeneration):
            #   model.language_model.model.{layers,embed_tokens,norm} -> model.{...}
            #   model.language_model.lm_head                          -> lm_head
            # Prefix remapping: checkpoint may wrap under language_model
            if name.startswith("model.language_model."):
                name = "model." + name[len("model.language_model."):]

            if "rotary_emb.inv_freq" in name:
                continue

            if _load_full_attention_qgkv_weight(
                    params_dict, name, loaded_weight, self.text_cfg):
                continue

            if _load_gdn_projection_weight(
                    params_dict, name, loaded_weight, self.text_cfg):
                continue

            if name.endswith(".mlp.gate.weight"):
                fused_name = name[:-len("gate.weight")] \
                    + "router_shared_gate.weight"
                if fused_name not in params_dict:
                    raise ValueError(
                        f"missing fused router/shared gate: {fused_name}")
                params_dict[fused_name].weight_loader(
                    params_dict[fused_name], loaded_weight, 0)
                continue

            if name.endswith(".mlp.shared_expert_gate.weight"):
                fused_name = name[:-len("shared_expert_gate.weight")] \
                    + "router_shared_gate.weight"
                if fused_name not in params_dict:
                    raise ValueError(
                        f"missing fused router/shared gate: {fused_name}")
                params_dict[fused_name].weight_loader(
                    params_dict[fused_name], loaded_weight, 1)
                continue

            if ".linear_attn.conv1d.weight" in name:
                name = name.replace(".linear_attn.conv1d.weight",
                                    ".linear_attn.conv1d_weight")

            # --- Fused routed-expert weights (all experts in one tensor) ---

            if "mlp.experts.gate_up_proj" in name:
                # loaded_weight: (num_experts, 2*intermediate, hidden)
                w13_name = name.replace("mlp.experts.gate_up_proj",
                                        "mlp.experts.w13_weight")
                if w13_name not in params_dict:
                    continue
                param = params_dict[w13_name]
                n_exp = loaded_weight.shape[0]
                inter = loaded_weight.shape[1] // 2
                gate_w = loaded_weight[:, :inter, :].contiguous()
                up_w   = loaded_weight[:, inter:, :].contiguous()
                for eid in range(n_exp):
                    param.weight_loader(param, gate_w[eid], "w1_weight", "w1", eid)
                    param.weight_loader(param, up_w[eid],   "w3_weight", "w3", eid)
                continue

            if "mlp.experts.down_proj" in name:
                # loaded_weight: (num_experts, hidden, intermediate)
                w2_name = name.replace("mlp.experts.down_proj",
                                       "mlp.experts.w2_weight")
                if w2_name not in params_dict:
                    continue
                param = params_dict[w2_name]
                n_exp = loaded_weight.shape[0]
                for eid in range(n_exp):
                    param.weight_loader(param, loaded_weight[eid], "w2_weight", "w2", eid)
                continue

            # --- Shared expert down_proj rename ---
            if "mlp.shared_expert.down_proj" in name:
                name = name.replace("mlp.shared_expert.down_proj",
                                    "mlp.shared_expert_down")
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                continue

            # --- Individual expert weights (FT checkpoint: experts.{i}.{proj}.weight) ---
            # Standard transformers fine-tuning saves each expert separately instead of
            # the pre-merged (num_experts, ...) tensors in the original checkpoint.
            if ".mlp.experts." in name:
                parts = name.split(".mlp.experts.", 1)
                expert_rest = parts[1]          # e.g. "0.gate_proj.weight"
                dot_pos = expert_rest.find(".")
                if dot_pos > 0 and expert_rest[:dot_pos].isdigit():
                    eid = int(expert_rest[:dot_pos])
                    proj_raw = expert_rest[dot_pos + 1:]
                    proj = proj_raw[:-7] if proj_raw.endswith(".weight") else proj_raw
                    prefix = parts[0]           # e.g. "model.layers.0"
                    if proj == "gate_proj":
                        w13_name = f"{prefix}.mlp.experts.w13_weight"
                        if w13_name in params_dict:
                            param = params_dict[w13_name]
                            param.weight_loader(param, loaded_weight, "w1_weight", "w1", eid)
                    elif proj == "up_proj":
                        w13_name = f"{prefix}.mlp.experts.w13_weight"
                        if w13_name in params_dict:
                            param = params_dict[w13_name]
                            param.weight_loader(param, loaded_weight, "w3_weight", "w3", eid)
                    elif proj == "down_proj":
                        w2_name = f"{prefix}.mlp.experts.w2_weight"
                        if w2_name in params_dict:
                            param = params_dict[w2_name]
                            param.weight_loader(param, loaded_weight, "w2_weight", "w2", eid)
                    continue

            # --- Stacked / standard weights ---
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    break
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
        _bi100_model_trace(
            f"MoE load_weights complete items={loaded_count} "
            f"vision_items={vision_loaded_count}")
print("[qwen3_5] module load COMPLETE", file=sys.stderr, flush=True)