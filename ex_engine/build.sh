#!/bin/bash
# ex_engine/build.sh — Compile EX Engine factor .so libraries
#
# Toolchain: corex clang/16 (BI-V100) with --cuda-gpu-arch=ivcore10
# Based on: real compile log from user test showing exact flags
#
# Usage:
#   ./ex_engine/build.sh              # auto-detect toolchain
#   ./ex_engine/build.sh --nvcc       # force nvcc (development)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
CSRC_DIR="${SCRIPT_DIR}/csrc"
INCLUDE_DIR="${SCRIPT_DIR}/include"

mkdir -p "$BUILD_DIR"

COREX_ROOT="/usr/local/corex"
COMPILER=""

detect_toolchain() {
    if [[ "${1:-auto}" != "--nvcc" ]] && [[ -x "${COREX_ROOT}/bin/clang++" ]]; then
        COMPILER="corex"
        echo "[EX] Using corex clang/16 at ${COREX_ROOT}/bin/clang++"
    elif command -v nvcc &>/dev/null; then
        COMPILER="nvcc"
        echo "[EX] Using nvcc"
    else
        echo "[EX] ERROR: No CUDA compiler found"
        exit 1
    fi
}

compile_factor() {
    local factor_id=$1
    local cu_file=$2
    local so_name="ex_factor_${factor_id}.so"
    local so_path="${BUILD_DIR}/${so_name}"

    echo "[EX] Compiling factor ${factor_id}: $(basename ${cu_file}) → ${so_name}"

    if [[ "$COMPILER" == "corex" ]]; then
        # Exact flags from real BI-V100 compile log:
        # --cuda-gpu-arch=ivcore10 (NOT sm_70!)
        # -D__ILUVATAR__ -D__ILUVATAR_WORKAROUND__ -D__ILUVATAR_DIAG__
        # -cl-single-precision-constant
        "${COREX_ROOT}/bin/clang++" \
            -x cuda \
            --cuda-gpu-arch=ivcore10 \
            --cuda-path="${COREX_ROOT}" \
            -std=c++17 \
            -O3 \
            -D__ILUVATAR__ \
            -D__ILUVATAR_WORKAROUND__ \
            -D__ILUVATAR_DIAG__ \
            -cl-single-precision-constant \
            -fPIC \
            -mllvm --bonus-inst-threshold=0 \
            -shared \
            -I"${INCLUDE_DIR}" \
            -I"${COREX_ROOT}/include" \
            -L"${COREX_ROOT}/lib64" \
            -lcudart \
            -o "${so_path}" \
            "${cu_file}" 2>&1 || {
                echo "[EX] ✗ FAILED: ${so_name}"
                return 1
            }
    else
        nvcc \
            -arch=sm_70 \
            -std=c++17 \
            -O3 \
            --compiler-options '-fPIC' \
            -shared \
            -I"${INCLUDE_DIR}" \
            -o "${so_path}" \
            "${cu_file}" 2>&1 || {
                echo "[EX] ✗ FAILED: ${so_name}"
                return 1
            }
    fi

    if [[ -f "${so_path}" ]]; then
        local size=$(stat -c%s "${so_path}" 2>/dev/null || stat -f%z "${so_path}" 2>/dev/null)
        echo "[EX] ✓ ${so_name} (${size} bytes)"
    fi
}

compile_registry() {
    local so_path="${BUILD_DIR}/libex_registry.so"
    echo "[EX] Compiling registry → libex_registry.so"
    gcc -O2 -shared -fPIC \
        -I"${INCLUDE_DIR}" \
        -o "${so_path}" \
        "${CSRC_DIR}/ex_registry.c" \
        -ldl
    echo "[EX] ✓ libex_registry.so"
}

# ============================================================================
# Main
# ============================================================================
detect_toolchain "${1:-auto}"

echo ""
echo "========================================"
echo "  EX Engine Build (Algorithm Factor Replacement)"
echo "  Toolchain: ${COMPILER}"
echo "  Output: ${BUILD_DIR}/"
echo "========================================"
echo ""

compile_registry

# Factor mapping
FACTORS=(
    "0:factor_moe_topk_softmax.cu"
    "2:factor_moe_fused_gemm.cu"
)
# Note: Factor 5 (GDN) uses FlashQLA Python extension, NOT a .so

TOTAL=0
SUCCESS=0
for entry in "${FACTORS[@]}"; do
    fid="${entry%%:*}"
    cu_file="${CSRC_DIR}/${entry##*:}"
    TOTAL=$((TOTAL + 1))
    if [[ -f "$cu_file" ]]; then
        if compile_factor "$fid" "$cu_file"; then
            SUCCESS=$((SUCCESS + 1))
        fi
    else
        echo "[EX] SKIP factor ${fid}: ${cu_file} not found"
    fi
done

echo ""
echo "========================================"
echo "  Build complete: ${SUCCESS}/${TOTAL} factors (.so)"
echo "  GDN: via FlashQLA (JIT compiled on hardware)"
echo "  Output: ${BUILD_DIR}/"
echo "========================================"
ls -la "${BUILD_DIR}/" 2>/dev/null || true
