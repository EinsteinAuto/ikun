"""
ex_engine/python/ex_loader.py — EX Engine Python loader

Architecture:
  CCCL: compute_capability → policy_selector → kernel template instantiation
  EX:   hardware_id        → ctypes.dlopen   → factor.kernel() via torch stream

This module loads the compiled .so factors and provides torch-compatible
wrappers that the vllm model code can call directly.

Usage:
    from ex_engine.python.ex_loader import EXEngine

    engine = EXEngine("/workspace/ex_engine/build")
    engine.load_all()

    # Replace MoE topk+softmax (was: torch.softmax + torch.topk, 36× per layer)
    topk_w, topk_ids = engine.moe_topk_softmax(router_logits, top_k=8)

    # Replace GDN prefill (was: _torch_chunk_gated_delta_rule producing NaN)
    output, new_state = engine.gdn_chunk_fwd(q, k, v, gate, beta, state)
"""

import ctypes
import os
import logging
import torch
from typing import Optional, Tuple

logger = logging.getLogger("ex_engine")

# ---------------------------------------------------------------------------
# C struct mirrors (must match ex_engine.h exactly)
# ---------------------------------------------------------------------------

class ExHardware(ctypes.Structure):
    _fields_ = [
        ("sm_major", ctypes.c_int),
        ("sm_minor", ctypes.c_int),
        ("sm_count", ctypes.c_int),
        ("max_threads_per_sm", ctypes.c_int),
        ("shared_mem_per_sm", ctypes.c_int),
        ("l2_cache_size", ctypes.c_int),
        ("memory_bus_width", ctypes.c_int),
        ("memory_bandwidth", ctypes.c_float),
    ]

class ExTuning(ctypes.Structure):
    _fields_ = [
        ("threads_per_block", ctypes.c_int),
        ("items_per_thread", ctypes.c_int),
        ("vec_size", ctypes.c_int),
        ("shared_mem_bytes", ctypes.c_int),
        ("num_warps", ctypes.c_int),
        ("num_stages", ctypes.c_int),
    ]

class ExFactor(ctypes.Structure):
    _fields_ = [
        ("factor_id", ctypes.c_int),
        ("name", ctypes.c_char_p),
        ("version", ctypes.c_char_p),
        ("tuning", ExTuning),
        ("kernel", ctypes.c_void_p),
        ("kernel_fallback", ctypes.c_void_p),
    ]


# Factor IDs (must match ex_engine.h)
EX_FACTOR_MOE_TOPK_SOFTMAX  = 0
EX_FACTOR_MOE_ALIGN_BLOCK   = 1
EX_FACTOR_MOE_FUSED_GEMM    = 2
EX_FACTOR_GELU_TANH_MUL     = 3
EX_FACTOR_BATCHED_ROTARY    = 4
EX_FACTOR_GDN_CHUNK_FWD     = 5
EX_FACTOR_GDN_RECURRENT     = 6
EX_FACTOR_CACHE_APPEND       = 7
EX_FACTOR_RESHAPE_CACHE_FLASH = 8
EX_FACTOR_COUNT              = 9


# BI-V100 default hardware
BI_V100_HARDWARE = ExHardware(
    sm_major=7, sm_minor=0, sm_count=16,
    max_threads_per_sm=2048, shared_mem_per_sm=49152,
    l2_cache_size=6 * 1024 * 1024, memory_bus_width=4096,
    memory_bandwidth=900.0
)


class EXEngine:
    """
    EX Engine: Algorithm Factor Replacement System

    Loads .so factors via dlopen at runtime, provides torch-compatible
    wrappers for each replaced algorithm.

    CCCL parallel:
        CCCL DispatchReduce → selects policy → launches kernel
        EXEngine.dispatch() → selects factor .so → calls kernel via ctypes
    """

    def __init__(self, build_dir: str = "/workspace/ex_engine/build",
                 hardware: Optional[ExHardware] = None):
        self.build_dir = build_dir
        self.hardware = hardware or BI_V100_HARDWARE
        self._factors = {}       # factor_id → ctypes handle
        self._so_handles = {}    # factor_id → dlopen handle
        self._available = set()  # set of loaded factor IDs

    def load_factor(self, factor_id: int, so_path: str) -> bool:
        """Load a single factor .so file."""
        if not os.path.exists(so_path):
            logger.warning("Factor %d .so not found: %s", factor_id, so_path)
            return False

        try:
            handle = ctypes.CDLL(so_path, mode=ctypes.RTLD_LOCAL)

            # Call ex_get_factor(hardware) → ExFactor*
            get_factor = handle.ex_get_factor
            get_factor.argtypes = [ctypes.POINTER(ExHardware)]
            get_factor.restype = ctypes.POINTER(ExFactor)

            hw = ExHardware()
            ctypes.memmove(ctypes.byref(hw), ctypes.byref(self.hardware),
                          ctypes.sizeof(ExHardware))
            factor_ptr = get_factor(ctypes.byref(hw))

            if not factor_ptr:
                logger.error("Factor %d: ex_get_factor returned NULL", factor_id)
                return False

            factor = factor_ptr.contents
            if factor.factor_id != factor_id:
                logger.error("Factor ID mismatch: expected %d, got %d",
                           factor_id, factor.factor_id)
                return False

            self._so_handles[factor_id] = handle
            self._factors[factor_id] = factor
            self._available.add(factor_id)

            name = factor.name.decode() if factor.name else "?"
            ver = factor.version.decode() if factor.version else "?"
            t = factor.tuning
            logger.info(
                "EX loaded factor %d (%s v%s) threads=%d items=%d smem=%d",
                factor_id, name, ver,
                t.threads_per_block, t.items_per_thread, t.shared_mem_bytes
            )
            return True

        except OSError as e:
            logger.error("Factor %d dlopen failed: %s", factor_id, e)
            return False

    def load_all(self) -> int:
        """Load all available factor .so files from build_dir or co-located."""
        loaded = 0
        # Search paths: build_dir first, then directory containing this module
        search_dirs = [self.build_dir]
        module_dir = os.path.dirname(os.path.abspath(__file__))
        if module_dir not in search_dirs:
            search_dirs.append(module_dir)
        # Also check parent's build dir
        parent_build = os.path.join(os.path.dirname(module_dir), "build")
        if parent_build not in search_dirs:
            search_dirs.append(parent_build)

        for fid in range(EX_FACTOR_COUNT):
            for d in search_dirs:
                so_path = os.path.join(d, f"ex_factor_{fid}.so")
                if os.path.exists(so_path):
                    if self.load_factor(fid, so_path):
                        loaded += 1
                    break
        logger.info("EX Engine: loaded %d/%d factors from %s", loaded, EX_FACTOR_COUNT,
                    search_dirs)
        return loaded

    def has_factor(self, factor_id: int) -> bool:
        return factor_id in self._available

    # ===================================================================
    # Torch-compatible wrappers for each factor
    # ===================================================================

    def moe_topk_softmax(
        self,
        router_logits: torch.Tensor,  # (T, E) float32
        top_k: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fused softmax + topk for MoE routing.

        Replaces:
            probs = torch.softmax(router_logits, dim=-1)
            topk_w, topk_ids = torch.topk(probs, top_k, dim=-1)
            topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

        Returns:
            topk_weights: (T, top_k) float32, renormalized
            topk_ids: (T, top_k) int32
        """
        if not self.has_factor(EX_FACTOR_MOE_TOPK_SOFTMAX):
            # Fallback to PyTorch
            probs = torch.softmax(router_logits.float(), dim=-1)
            topk_w, topk_ids = torch.topk(probs, top_k, dim=-1)
            topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
            return topk_w.to(router_logits.dtype), topk_ids.to(torch.int32)

        T, E = router_logits.shape
        logits = router_logits.float().contiguous()
        topk_weights = torch.empty(T, top_k, dtype=torch.float32,
                                   device=logits.device)
        topk_ids = torch.empty(T, top_k, dtype=torch.int32,
                               device=logits.device)

        # Get CUDA stream from torch
        stream = torch.cuda.current_stream().cuda_stream

        # Call kernel via ctypes
        handle = self._so_handles[EX_FACTOR_MOE_TOPK_SOFTMAX]
        kernel_fn = handle.ex_dispatch_moe_topk_softmax
        kernel_fn.argtypes = [
            ctypes.c_void_p,  # topk_weights
            ctypes.c_void_p,  # topk_ids
            ctypes.c_void_p,  # logits
            ctypes.c_int,     # T
            ctypes.c_int,     # E
            ctypes.c_int,     # top_k
            ctypes.c_void_p,  # stream
        ]
        kernel_fn.restype = ctypes.c_int

        ret = kernel_fn(
            topk_weights.data_ptr(),
            topk_ids.data_ptr(),
            logits.data_ptr(),
            T, E, top_k,
            stream
        )

        if ret != 0:
            logger.warning("moe_topk_softmax kernel returned %d, fallback", ret)
            probs = torch.softmax(logits, dim=-1)
            topk_w, topk_i = torch.topk(probs, top_k, dim=-1)
            topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
            return topk_w, topk_i.to(torch.int32)

        return topk_weights, topk_ids

    def gdn_chunk_fwd(
        self,
        query: torch.Tensor,     # (B, L, H, D) half
        key: torch.Tensor,       # (B, L, H, D) half
        value: torch.Tensor,     # (B, L, H, D) half
        gate: torch.Tensor,      # (B, L, H) float32
        beta: torch.Tensor,      # (B, L, H) float32
        state_in: torch.Tensor,  # (B, H, D, D) float32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GatedDeltaNet chunked prefill forward.

        Replaces _torch_chunk_gated_delta_rule which produces NaN.
        Full fp32 accumulation prevents overflow.

        Returns:
            output: (B, L, H, D) half
            state_out: (B, H, D, D) float32
        """
        if not self.has_factor(EX_FACTOR_GDN_CHUNK_FWD):
            # Cannot fallback safely — the PyTorch version produces NaN
            # Return zeros as a safe default (matches nan_to_num behavior)
            B, L, H, D = query.shape
            output = torch.zeros_like(query)
            state_out = state_in.clone()
            logger.warning("GDN factor not loaded, returning zeros (NaN prevention)")
            return output, state_out

        B, L, H, D = query.shape
        output = torch.empty_like(query)
        state_out = torch.empty_like(state_in)

        stream = torch.cuda.current_stream().cuda_stream

        # Direct kernel call via factor dispatch
        dims = (ctypes.c_int64 * 4)(B, L, H, D)
        aux = (ctypes.c_void_p * 6)(
            key.data_ptr(),
            value.data_ptr(),
            gate.data_ptr(),
            beta.data_ptr(),
            state_in.data_ptr(),
            state_out.data_ptr(),
        )

        handle = self._so_handles[EX_FACTOR_GDN_CHUNK_FWD]
        # Use the generic ex_get_factor → factor.kernel path
        get_factor = handle.ex_get_factor
        get_factor.argtypes = [ctypes.POINTER(ExHardware)]
        get_factor.restype = ctypes.POINTER(ExFactor)

        hw = self.hardware
        factor_ptr = get_factor(ctypes.byref(hw))
        factor = factor_ptr.contents

        # Cast kernel function pointer
        KERNEL_FN = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,          # output
            ctypes.c_void_p,          # input (query)
            ctypes.POINTER(ctypes.c_void_p),  # aux_inputs
            ctypes.c_int,             # n_aux
            ctypes.POINTER(ctypes.c_int64),   # dims
            ctypes.c_int,             # n_dims
            ctypes.c_void_p,          # stream
        )
        kernel = KERNEL_FN(factor.kernel)

        ret = kernel(
            output.data_ptr(),
            query.data_ptr(),
            aux,
            6,
            dims,
            4,
            stream,
        )

        if ret != 0:
            logger.warning("gdn_chunk_fwd kernel returned %d, returning zeros", ret)
            output.zero_()
            state_out.copy_(state_in)

        return output, state_out


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_engine: Optional[EXEngine] = None

def get_engine(build_dir: str = "/workspace/ex_engine/build") -> EXEngine:
    """Get or create the global EX Engine instance."""
    global _engine
    if _engine is None:
        _engine = EXEngine(build_dir)
        _engine.load_all()
    return _engine
