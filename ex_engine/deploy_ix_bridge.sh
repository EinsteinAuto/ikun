#!/bin/bash
# ex_engine/deploy_ix_bridge.sh — Deploy ix_full_bridge + Python ops into vllm
#
# Architecture (CCCL build pattern):
#   CCCL: cmake → compile → install to site-packages
#   EX:   torch.utils.cpp_extension → compile bridge → deploy to vllm pkg
#
# What this does:
#   1. Find ixformer .so libraries in base image
#   2. Either use prebuilt ix_full_bridge.so or JIT-compile from source
#   3. Deploy .so + Python modules into vllm package
#   4. Verify dlopen chain works
#
# Source mapping:
#   ex_engine/csrc/ix_full_bridge_v2.cpp → pybind11 bridge to ixformer::infer
#   ex_engine/python/ix_ops.py           → Python API layer
#   ex_engine/python/patch_vllm_ops.py   → vllm monkey-patches
#
# Called from: qwen3_6_scripts/patch_ops.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VLLM_ROOT="${1:-$(python3 -c 'import vllm; import os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || echo '/usr/local/corex/lib/python3/dist-packages/vllm')}"

echo "[ix_bridge] VLLM_ROOT=${VLLM_ROOT}"
echo "[ix_bridge] SCRIPT_DIR=${SCRIPT_DIR}"

# =========================================================================
# Step 1: Deploy prebuilt .so if available
# =========================================================================
PREBUILT="${SCRIPT_DIR}/../qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10"
BRIDGE_SO=""

if [[ -f "${PREBUILT}/ix_full_bridge.so" ]]; then
    cp "${PREBUILT}/ix_full_bridge.so" "${VLLM_ROOT}/ix_full_bridge.so"
    BRIDGE_SO="${VLLM_ROOT}/ix_full_bridge.so"
    echo "[ix_bridge] deployed prebuilt ix_full_bridge.so"
fi

# Deploy all corex_*.so and xllm_*.so
if [[ -d "$PREBUILT" ]]; then
    for so_file in "${PREBUILT}"/*.so; do
        base=$(basename "$so_file")
        if [[ "$base" != "ix_full_bridge.so" ]]; then
            cp "$so_file" "${VLLM_ROOT}/${base}" 2>/dev/null || true
            echo "[ix_bridge] deployed ${base}"
        fi
    done
fi

# =========================================================================
# Step 2: Deploy Python integration modules
# =========================================================================
# Create ex_engine package in vllm
EX_PKG="${VLLM_ROOT}/ex_engine"
mkdir -p "${EX_PKG}"

cat > "${EX_PKG}/__init__.py" << 'PYEOF'
"""ex_engine — Algorithm factor replacement engine for BI-V100."""
PYEOF

# Deploy ix_ops.py
cp "${SCRIPT_DIR}/python/ix_ops.py" "${EX_PKG}/ix_ops.py"
echo "[ix_bridge] deployed ix_ops.py"

# Deploy patch_vllm_ops.py
cp "${SCRIPT_DIR}/python/patch_vllm_ops.py" "${EX_PKG}/patch_vllm_ops.py"
echo "[ix_bridge] deployed patch_vllm_ops.py"

# Also make ix_ops importable from vllm.ex_engine
# and from the top-level ex_engine path
SITE_EX="${SCRIPT_DIR}/python"
if [[ -d "$SITE_EX" ]]; then
    # Ensure __init__.py exists
    touch "${SITE_EX}/../__init__.py" 2>/dev/null || true
fi

# =========================================================================
# Step 3: Create auto-patch entry point
# =========================================================================
# This script is sourced by patch_ops.sh to ensure ix_ops patches
# are applied at vllm startup
cat > "${VLLM_ROOT}/ix_startup_patch.py" << 'PYEOF'
"""
ix_startup_patch.py — Apply ix_ops patches at vllm startup.

Import this module early in the vllm startup to replace PyTorch fallbacks
with fused C++ kernels from the base image.

Architecture (CCCL dispatch pattern):
    import vllm → vllm.__init__ → ix_startup_patch → patch_vllm_ops
"""
import logging
logger = logging.getLogger("ix_startup_patch")

def apply():
    """Apply all available ix_ops patches."""
    try:
        from vllm.ex_engine.patch_vllm_ops import apply_all_patches
        n = apply_all_patches()
        if n > 0:
            logger.info("ix_startup_patch: %d patches applied", n)
        return n
    except Exception as e:
        logger.warning("ix_startup_patch failed: %s", e)
        return 0

# Auto-apply on import
_n_patches = apply()
PYEOF
echo "[ix_bridge] deployed ix_startup_patch.py"

# =========================================================================
# Step 4: Deploy bridge C++ source for JIT fallback
# =========================================================================
CSRC_DEST="${VLLM_ROOT}/ex_engine/csrc"
mkdir -p "${CSRC_DEST}"
for cpp in "${SCRIPT_DIR}/csrc/ix_full_bridge_v2.cpp" \
           "${SCRIPT_DIR}/csrc/ix_full_bridge.cpp" \
           "${SCRIPT_DIR}/csrc/ix_moe_bridge.cpp"; do
    if [[ -f "$cpp" ]]; then
        cp "$cpp" "${CSRC_DEST}/"
        echo "[ix_bridge] deployed $(basename $cpp) for JIT fallback"
    fi
done

# =========================================================================
# Step 5: Verify deployment
# =========================================================================
echo ""
echo "[ix_bridge] === Deployment Summary ==="
echo "[ix_bridge] Bridge .so: ${BRIDGE_SO:-'(JIT compile at runtime)'}"
echo "[ix_bridge] Python ops: ${EX_PKG}/ix_ops.py"
echo "[ix_bridge] vllm patches: ${EX_PKG}/patch_vllm_ops.py"
echo "[ix_bridge] Startup hook: ${VLLM_ROOT}/ix_startup_patch.py"

# Quick Python import test
python3 -c "
import sys
sys.path.insert(0, '${VLLM_ROOT}')
try:
    from vllm.ex_engine import ix_ops
    print('[ix_bridge] ✓ ix_ops importable')
except Exception as e:
    print(f'[ix_bridge] ✗ ix_ops import failed: {e}')
try:
    from vllm.ex_engine import patch_vllm_ops
    print('[ix_bridge] ✓ patch_vllm_ops importable')
except Exception as e:
    print(f'[ix_bridge] ✗ patch_vllm_ops import failed: {e}')
" 2>&1 || true

echo "[ix_bridge] === Done ==="
