#!/bin/bash
# deploy_corex_modules.sh — Deploy corex_gdn.py + corex_moe.py into vllm
#
# Competitor 168's Docker had these at:
#   $VLLM/model_executor/models/corex_gdn.py
#   $VLLM/model_executor/models/corex_moe.py
#
# Our qwen3_5.py already has import fallback for these (lines 117-125):
#   from vllm.model_executor.models import corex_gdn as _corex_gdn_module
#   from vllm.model_executor.models import corex_moe as _corex_moe_module
#
# This script copies our implementations there so the imports succeed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/python"

# Find vllm install path
VLLM_MODELS=""
for candidate in \
    /usr/local/corex/lib/python3/dist-packages/vllm/model_executor/models \
    /usr/local/corex/lib64/python3/dist-packages/vllm/model_executor/models \
    /usr/local/lib/python3.10/site-packages/vllm/model_executor/models \
    /workspace/vllm/model_executor/models; do
    if [[ -d "$candidate" ]]; then
        VLLM_MODELS="$candidate"
        break
    fi
done

if [[ -z "$VLLM_MODELS" ]]; then
    # Try Python detection
    VLLM_MODELS=$(python3 -c "
import os, vllm
print(os.path.join(os.path.dirname(vllm.__file__), 'model_executor', 'models'))
" 2>/dev/null || true)
fi

if [[ -z "$VLLM_MODELS" ]] || [[ ! -d "$VLLM_MODELS" ]]; then
    echo "[COREX] ERROR: Cannot find vllm models directory"
    exit 1
fi

echo "[COREX] Deploying to: $VLLM_MODELS"

# Deploy corex_gdn.py
if [[ ! -f "${VLLM_MODELS}/corex_gdn.py" ]]; then
    cp "${SRC_DIR}/corex_gdn.py" "${VLLM_MODELS}/corex_gdn.py"
    echo "[COREX] ✓ Deployed corex_gdn.py"
else
    echo "[COREX] ✓ corex_gdn.py already exists (base image or prior deploy)"
fi

# Deploy corex_moe.py
if [[ ! -f "${VLLM_MODELS}/corex_moe.py" ]]; then
    cp "${SRC_DIR}/corex_moe.py" "${VLLM_MODELS}/corex_moe.py"
    echo "[COREX] ✓ Deployed corex_moe.py"
else
    echo "[COREX] ✓ corex_moe.py already exists (base image or prior deploy)"
fi

# Deploy corex_fa2.py
if [[ ! -f "${VLLM_MODELS}/corex_fa2.py" ]]; then
    cp "${SRC_DIR}/corex_fa2.py" "${VLLM_MODELS}/corex_fa2.py"
    echo "[COREX] ✓ Deployed corex_fa2.py"
else
    echo "[COREX] ✓ corex_fa2.py already exists (base image or prior deploy)"
fi

# Also deploy to ex_engine location (backup import path)
mkdir -p /workspace/ex_engine/python 2>/dev/null || true
cp "${SRC_DIR}/corex_gdn.py" /workspace/ex_engine/python/ 2>/dev/null || true
cp "${SRC_DIR}/corex_moe.py" /workspace/ex_engine/python/ 2>/dev/null || true
cp "${SRC_DIR}/corex_fa2.py" /workspace/ex_engine/python/ 2>/dev/null || true

echo "[COREX] Deploy complete"
echo "[COREX] Expected log on startup:"
echo "  corex_gdn.py:NN → Loaded fused CoreX GDN decode operator ..."
echo "  corex_gdn.py:NN → Using fused CoreX GDN prefill operator"
echo "  corex_moe.py:NN → Using CoreX fused MoE prefill operator: tokens=N, kernel=expert-grouped-wmma"
echo "  corex_fa2.py:NN → Using CoreX FA2 packed prefill: B=N Hq=4 Hkv=1 D=256 ..."
echo "  corex_fa2.py:NN → Using CoreX paged decode: B=N Hq=4 Hkv=1 D=256 ..."
