#!/usr/bin/env bash
# deploy_local.sh — Deploy prebuilt .so + patches to local vllm/ directory
# Usage: bash deploy_local.sh
# For real-machine testing without docker build
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VLLM_ROOT="${PROJECT_ROOT}/vllm"

echo "[deploy] VLLM_ROOT=${VLLM_ROOT}"

# 1. Deploy prebuilt CoreX .so files
BUNDLE="${SCRIPT_DIR}/prebuilt/corex-3.2.3-ivcore10"
if [[ -d "$BUNDLE" ]]; then
    count=0
    for so in "${BUNDLE}"/*.so; do
        name=$(basename "$so")
        cp "$so" "${VLLM_ROOT}/${name}"
        count=$((count+1))
    done
    echo "[deploy] copied ${count} prebuilt .so to ${VLLM_ROOT}/"
else
    echo "[deploy] WARNING: prebuilt bundle not found at ${BUNDLE}"
fi

# 2. Build corex_moe_index_combine.so (missing from prebuilt)
if [[ ! -f "${VLLM_ROOT}/corex_moe_index_combine.so" ]]; then
    COREX_ROOT=""
    for c in /usr/local/corex-3.2.3 /usr/local/corex; do
        if [[ -x "${c}/bin/clang++" ]]; then
            COREX_ROOT="$c"
            break
        fi
    done
    if [[ -n "$COREX_ROOT" ]]; then
        echo "[deploy] building corex_moe_index_combine.so ..."
        COREX_ROOT="$COREX_ROOT" bash "${SCRIPT_DIR}/build_corex_moe_index_combine.sh" "${VLLM_ROOT}" 2>&1 || {
            echo "[deploy] WARNING: corex_moe_index_combine build failed"
        }
    fi
fi

# 3. List deployed .so
echo ""
echo "[deploy] Deployed .so files:"
ls -la "${VLLM_ROOT}"/*.so 2>/dev/null || echo "  (none)"

echo ""
echo "[deploy] Done. Now run: python3 probe_bi100.py"
