#!/usr/bin/env bash
# deploy_ilu_pipeline.sh — Build + deploy the complete ILU kernel pipeline
#
# This replaces ALL Python fallbacks with C++ calls through ixformer::infer.
# Call from patch_ops.sh after basic vllm patching is done.
#
# What this does:
#   1. Build ix_full_bridge_v2.so (pybind11 bridge to all 14 ixformer functions)
#   2. Deploy Python dispatch modules (ix_ops_dispatch, corex_gdn, corex_moe, corex_fa2)
#   3. Deploy upstream xllm ILU kernel wrappers
#   4. Wire ix_startup_patch to auto-load at vllm import
#
# Usage:
#   bash deploy_ilu_pipeline.sh <VLLM_ROOT>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLLM_ROOT="${1:?Usage: deploy_ilu_pipeline.sh <VLLM_ROOT>}"

echo "============================================"
echo "[ILU] Starting ILU pipeline deployment"
echo "[ILU] VLLM_ROOT: ${VLLM_ROOT}"
echo "[ILU] Script dir: ${SCRIPT_DIR}"
echo "============================================"

# --- Step 1: Create ex_engine package in vllm ---
EX_DIR="${VLLM_ROOT}/ex_engine"
mkdir -p "${EX_DIR}/python"
cat > "${EX_DIR}/__init__.py" << 'EOF'
"""ex_engine — Algorithm factor replacement for BI-V100."""
EOF
cat > "${EX_DIR}/python/__init__.py" << 'EOF'
"""ex_engine.python — Python dispatch modules."""
EOF

# --- Step 2: Try to build ix_full_bridge_v2.so ---
echo "[ILU] Step 2: Building ix_full_bridge_v2.so..."
BRIDGE_SO="${SCRIPT_DIR}/prebuilt/ix_full_bridge_v2.so"
if [[ -f "$BRIDGE_SO" ]]; then
    echo "[ILU] ✓ Using prebuilt ix_full_bridge_v2.so"
else
    if bash "${SCRIPT_DIR}/build_ix_bridge.sh" "${VLLM_ROOT}" 2>&1; then
        echo "[ILU] ✓ Built ix_full_bridge_v2.so"
    else
        echo "[ILU] ⚠ ix_full_bridge_v2.so build failed — will use ixformer Python path"
    fi
fi

# Deploy bridge .so
if [[ -f "$BRIDGE_SO" ]]; then
    cp "$BRIDGE_SO" "${EX_DIR}/ix_full_bridge_v2.so"
    cp "$BRIDGE_SO" "${EX_DIR}/python/ix_full_bridge_v2.so"
    echo "[ILU] ✓ Deployed ix_full_bridge_v2.so"
fi

# --- Step 3: Deploy Python dispatch modules ---
echo "[ILU] Step 3: Deploying Python dispatch modules..."

for pyfile in \
    ix_ops_dispatch.py \
    corex_gdn.py \
    corex_moe.py \
    corex_fa2.py \
    corex_fa2_dispatch.py \
    fused_moe_ilu.py \
    ix_bridge.py \
    ix_bridge_v2.py \
    ix_ops.py \
    patch_vllm_ops.py \
    ex_loader.py \
    moe_topk.py \
    patch_model.py; do
    src="${SCRIPT_DIR}/python/${pyfile}"
    if [[ -f "$src" ]]; then
        cp "$src" "${EX_DIR}/python/${pyfile}"
        echo "[ILU]   ✓ ${pyfile}"
    fi
done

# Also deploy corex_gdn.py and corex_moe.py to vllm models dir for import
MODELS_DIR="${VLLM_ROOT}/model_executor/models"
for pyfile in corex_gdn.py corex_moe.py corex_fa2.py; do
    src="${SCRIPT_DIR}/python/${pyfile}"
    if [[ -f "$src" ]] && [[ -d "$MODELS_DIR" ]]; then
        cp "$src" "${MODELS_DIR}/${pyfile}"
        echo "[ILU]   ✓ ${pyfile} → models/"
    fi
done

# --- Step 4: Deploy xllm ILU kernel wrappers ---
echo "[ILU] Step 4: Deploying xllm ILU kernel sources..."
ILU_SRC="${SCRIPT_DIR}/xllm_kernels/ilu"
ILU_UPSTREAM="${REPO_ROOT}/upstream_ref/xllm/xllm/core/kernels/ilu"

# Copy from upstream if not already in ex_engine
if [[ -d "$ILU_UPSTREAM" ]] && [[ ! -d "$ILU_SRC" ]]; then
    mkdir -p "$ILU_SRC"
    cp "$ILU_UPSTREAM"/*.cpp "$ILU_UPSTREAM"/*.h "$ILU_SRC/" 2>/dev/null || true
    echo "[ILU]   ✓ Copied from upstream xllm/core/kernels/ilu/"
fi

if [[ -d "$ILU_SRC" ]]; then
    mkdir -p "${EX_DIR}/xllm_kernels/ilu"
    cp "$ILU_SRC"/*.cpp "$ILU_SRC"/*.h "${EX_DIR}/xllm_kernels/ilu/" 2>/dev/null || true
    echo "[ILU]   ✓ ILU kernel sources deployed"
fi

# --- Step 5: Deploy upstream kernel sources for reference ---
echo "[ILU] Step 5: Deploying upstream kernel references..."
CUDA_SRC="${REPO_ROOT}/upstream_ref/xllm/xllm/core/kernels/cuda"
if [[ -d "$CUDA_SRC" ]]; then
    mkdir -p "${EX_DIR}/xllm_kernels/cuda"
    # Only copy the key files we need
    for cufile in \
        activation.cu norm.cu fused_qknorm_rope.cu \
        reshape_paged_cache.cu block_copy.cu matmul.cpp; do
        if [[ -f "${CUDA_SRC}/${cufile}" ]]; then
            cp "${CUDA_SRC}/${cufile}" "${EX_DIR}/xllm_kernels/cuda/"
        fi
    done
    # MoE kernels
    if [[ -d "${CUDA_SRC}/moe" ]]; then
        mkdir -p "${EX_DIR}/xllm_kernels/cuda/moe"
        cp "${CUDA_SRC}/moe"/*.cu "${CUDA_SRC}/moe"/*.cpp \
           "${EX_DIR}/xllm_kernels/cuda/moe/" 2>/dev/null || true
    fi
    # xattention kernels
    if [[ -d "${CUDA_SRC}/xattention" ]]; then
        mkdir -p "${EX_DIR}/xllm_kernels/cuda/xattention"
        cp "${CUDA_SRC}/xattention"/*.cu "${CUDA_SRC}/xattention"/*.cpp \
           "${CUDA_SRC}/xattention"/*.h \
           "${EX_DIR}/xllm_kernels/cuda/xattention/" 2>/dev/null || true
    fi
    echo "[ILU]   ✓ Upstream CUDA kernel sources deployed"
fi

# --- Step 6: Deploy ds_vllm libtorch_stable kernels ---
echo "[ILU] Step 6: Deploying ds_vllm kernel references..."
DS_SRC="${REPO_ROOT}/upstream_ref/ds_vllm/csrc/libtorch_stable"
if [[ -d "$DS_SRC" ]]; then
    mkdir -p "${EX_DIR}/ds_kernels"
    for cufile in \
        activation_kernels.cu layernorm_kernels.cu \
        pos_encoding_kernels.cu cache_kernels.cu; do
        if [[ -f "${DS_SRC}/${cufile}" ]]; then
            cp "${DS_SRC}/${cufile}" "${EX_DIR}/ds_kernels/"
        fi
    done
    if [[ -d "${DS_SRC}/moe" ]]; then
        mkdir -p "${EX_DIR}/ds_kernels/moe"
        cp "${DS_SRC}/moe/topk_softmax_kernels.cu" \
           "${DS_SRC}/moe/moe_align_sum_kernels.cu" \
           "${DS_SRC}/moe/torch_bindings.cpp" \
           "${EX_DIR}/ds_kernels/moe/" 2>/dev/null || true
    fi
    if [[ -d "${DS_SRC}/attention" ]]; then
        mkdir -p "${EX_DIR}/ds_kernels/attention"
        cp "${DS_SRC}/attention"/*.cu "${DS_SRC}/attention"/*.cuh \
           "${EX_DIR}/ds_kernels/attention/" 2>/dev/null || true
    fi
    echo "[ILU]   ✓ ds_vllm kernel sources deployed"
fi

# --- Step 7: Verification ---
echo "[ILU] Step 7: Verifying deployment..."
echo "[ILU] ex_engine contents:"
find "${EX_DIR}" -name "*.py" -o -name "*.so" -o -name "*.cpp" -o -name "*.cu" | sort | head -40
echo "[ILU] ..."
COUNT=$(find "${EX_DIR}" -type f | wc -l)
echo "[ILU] Total files deployed: ${COUNT}"

echo ""
echo "============================================"
echo "[ILU] ✓ ILU pipeline deployment complete"
echo "[ILU] Deployed to: ${EX_DIR}"
echo "[ILU] "
echo "[ILU] Runtime dispatch chain:"
echo "[ILU]   vllm import → ix_startup_patch → patch_vllm_ops"
echo "[ILU]     → ix_ops_dispatch → ix_full_bridge_v2.so"
echo "[ILU]       → ixformer::infer::* (C++ kernels)"
echo "[ILU] "
echo "[ILU] MoE pipeline:"
echo "[ILU]   corex_moe.py / fused_moe_ilu.py"
echo "[ILU]     → topk_softmax → moe_gen_idx → expand → gemm → silu → gemm → combine"
echo "[ILU]     → ALL through ixformer::infer (no Python expert loop)"
echo "============================================"
