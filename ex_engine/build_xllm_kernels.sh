#!/usr/bin/env bash
# build_xllm_kernels.sh — Compile xllm CUDA kernels into .so for BI-V100
#
# Architecture (CCCL compile pattern):
#   CCCL: CMakePresets.json → cmake --preset cub-cpp20 → ninja → .so
#   EX:   torch.utils.cpp_extension → clang --cuda-gpu-arch=ivcore10 → .so
#
# Usage:
#   bash ex_engine/build_xllm_kernels.sh [--output-dir /path/to/output]
#
# Prerequisites:
#   - BI-V100 machine with corex SDK
#   - PyTorch with CUDA support
#   - corex clang/16 compiler
#
# Outputs:
#   xllm_fused_qknorm_rope.so — Fused QK-Norm + RoPE (saves 128 kernel launches/fwd)

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNELS_DIR="${SCRIPT_DIR}/xllm_kernels/cuda"
HEADERS_DIR="${KERNELS_DIR}/headers"
BINDINGS_DIR="${KERNELS_DIR}/bindings"
OUTPUT_DIR="${1:-${SCRIPT_DIR}/../qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10}"

mkdir -p "${OUTPUT_DIR}"

echo "[build] KERNELS_DIR=${KERNELS_DIR}"
echo "[build] HEADERS_DIR=${HEADERS_DIR}"
echo "[build] OUTPUT_DIR=${OUTPUT_DIR}"

# Common compile flags for BI-V100 (ivcore10 = SM70-class)
CUDA_FLAGS="-O2 --cuda-gpu-arch=ivcore10"
CXX_FLAGS="-O2 -std=c++17"
INCLUDE_FLAGS="-I${HEADERS_DIR}"

# Use torch's cpp_extension for JIT compile
build_so() {
    local name=$1
    local sources=$2
    local extra_flags="${3:-}"

    echo "[build] Building ${name}.so from: ${sources}"

    python3 -c "
import os, sys
from torch.utils.cpp_extension import load

sources = '${sources}'.split()
abs_sources = [os.path.join('${SCRIPT_DIR}', '..', s) if not os.path.isabs(s) else s for s in sources]
abs_sources = [os.path.abspath(s) for s in abs_sources]

for s in abs_sources:
    if not os.path.exists(s):
        print(f'ERROR: source not found: {s}', file=sys.stderr)
        sys.exit(1)

try:
    mod = load(
        name='${name}',
        sources=abs_sources,
        extra_cuda_cflags=['-O2'],
        extra_cflags=['-O2', '-std=c++17'],
        extra_include_paths=['${HEADERS_DIR}'],
        build_directory='/tmp/build_${name}',
        verbose=True,
    )
    # Find the compiled .so
    import glob
    sos = glob.glob('/tmp/build_${name}/${name}*.so')
    if sos:
        import shutil
        dst = os.path.join('${OUTPUT_DIR}', '${name}.so')
        shutil.copy2(sos[0], dst)
        print(f'[build] SUCCESS: {dst}')
    else:
        print('[build] WARN: .so not found after build', file=sys.stderr)
except Exception as e:
    print(f'[build] FAIL ${name}: {e}', file=sys.stderr)
    sys.exit(1)
" || echo "[build] FAILED: ${name}"
}

# ============================================================================
# Build targets
# ============================================================================

# 1. xllm_fused_qknorm_rope — Fused QK-Norm + RoPE
#    Source: upstream xllm fused_qknorm_rope.cu
#    Note: Requires corex_compat_utils.h instead of glog-dependent utils.h
#    The .cu includes "cuda_ops_api.h" and "utils.h" — we need to make sure
#    the include path resolves to our corex-compat headers first.
echo ""
echo "============================================================"
echo "  1. xllm_fused_qknorm_rope.so"
echo "============================================================"
build_so "xllm_fused_qknorm_rope" \
    "ex_engine/xllm_kernels/cuda/fused_qknorm_rope.cu ex_engine/xllm_kernels/cuda/bindings/xllm_fused_qknorm_rope_bind.cpp"

# 2. xllm_norm — RMSNorm + Fused Add RMSNorm
#    Source: upstream xllm norm.cu
#    Hot path: called 2× per decoder layer = 72× per forward pass
echo ""
echo "============================================================"
echo "  2. xllm_norm.so"
echo "============================================================"
build_so "xllm_norm" \
    "ex_engine/xllm_kernels/cuda/norm.cu ex_engine/xllm_kernels/cuda/bindings/xllm_norm_bind.cpp"

# 3. xllm_rope — Rotary Position Embedding
#    Source: upstream xllm rope.cu
#    Hot path: called 1× per attention layer = 36× per forward pass
echo ""
echo "============================================================"
echo "  3. xllm_rope.so"
echo "============================================================"
build_so "xllm_rope" \
    "ex_engine/xllm_kernels/cuda/rope.cu ex_engine/xllm_kernels/cuda/bindings/xllm_rope_bind.cpp"

# 4. xllm_activation — SiLU-and-Mul fused activation
#    Source: upstream xllm activation.cu
#    Hot path: called 1× per MLP = 36× per forward pass
echo ""
echo "============================================================"
echo "  4. xllm_activation.so"
echo "============================================================"
build_so "xllm_activation" \
    "ex_engine/xllm_kernels/cuda/activation.cu ex_engine/xllm_kernels/cuda/bindings/xllm_activation_bind.cpp"

# 5. xllm_cache — Reshape + block copy for KV cache
#    Source: upstream xllm reshape_paged_cache.cu + block_copy.cu
#    Hot path: called every prefill + decode step
echo ""
echo "============================================================"
echo "  5. xllm_cache.so"
echo "============================================================"
build_so "xllm_cache" \
    "ex_engine/xllm_kernels/cuda/reshape_paged_cache.cu ex_engine/xllm_kernels/cuda/block_copy.cu ex_engine/xllm_kernels/cuda/bindings/xllm_cache_bind.cpp"

# 6. xllm_moe — MoE topk + index + combine + fused pipeline
#    Source: upstream xllm moe_fused_topk.cu + moe_compute_index.cu + moe_combine.cu + fused_moe.cpp
#    THE critical .so: replaces Python for-loop over 64 experts
echo ""
echo "============================================================"
echo "  6. xllm_moe.so"
echo "============================================================"
build_so "xllm_moe" \
    "ex_engine/xllm_kernels/cuda/moe/moe_fused_topk.cu ex_engine/xllm_kernels/cuda/moe/moe_compute_index.cu ex_engine/xllm_kernels/cuda/moe/moe_combine.cu ex_engine/xllm_kernels/cuda/moe/fused_moe.cpp ex_engine/xllm_kernels/cuda/bindings/xllm_moe_bind.cpp"

echo ""
echo "============================================================"
echo "  Build complete. Output:"
echo "============================================================"
ls -la "${OUTPUT_DIR}"/*.so 2>/dev/null | tail -30
echo ""
echo "Total .so count: $(ls "${OUTPUT_DIR}"/*.so 2>/dev/null | wc -l)"
