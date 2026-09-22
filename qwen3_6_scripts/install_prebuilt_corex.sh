#!/usr/bin/env bash
set -euo pipefail
VLLM_ROOT=${1:?usage: install_prebuilt_corex.sh VLLM_ROOT}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR=${SCRIPT_DIR}/prebuilt/corex-3.2.3-ivcore10
MANIFEST=${BUNDLE_DIR}/SHA256SUMS
[[ -d "$VLLM_ROOT" ]] || { printf 'vLLM root does not exist: %s\n' "$VLLM_ROOT" >&2; exit 2; }
[[ -f "$MANIFEST" ]] || { printf 'prebuilt CoreX manifest is missing: %s\n' "$MANIFEST" >&2; exit 2; }
mapfile -t artifacts < <(awk '{print $2}' "$MANIFEST")
(cd "$BUNDLE_DIR" && sha256sum --strict --check SHA256SUMS)
for artifact in "${artifacts[@]}"; do
    install -m 0755 "$BUNDLE_DIR/$artifact" "$VLLM_ROOT/$artifact"
done
for artifact in "${artifacts[@]}"; do
    [[ -f "$VLLM_ROOT/$artifact" ]] && echo "[ok] installed $artifact" || echo "[FAIL] $artifact"
done
