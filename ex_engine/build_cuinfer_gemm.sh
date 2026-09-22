#!/usr/bin/env bash
# build_cuinfer_gemm.sh — Compile cuinfer GEMM wrapper
#
# Links: libcuinfer.so (from /usr/local/corex/lib64/)
# Output: cuinfer_gemm_wrapper.so
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SCRIPT_DIR}/cuinfer_gemm_wrapper.cu"
HDR="${SCRIPT_DIR}/cuinfer_handle.h"

echo "[cuinfer_gemm] Building cuinfer_gemm_wrapper.so"

COREX_ROOT="${COREX_ROOT:-/usr/local/corex}"
CUINFER_LIB=""
for d in "${COREX_ROOT}/lib64" "${COREX_ROOT}/lib"; do
    if [[ -f "${d}/libcuinfer.so" ]]; then
        CUINFER_LIB="${d}"
        break
    fi
done

python3 << PYEOF
import os, sys, shutil

src = "${SRC}"
hdr_dir = "${SCRIPT_DIR}"
cuinfer_lib = "${CUINFER_LIB}"

ldflags = []
if cuinfer_lib:
    ldflags = [f"-L{cuinfer_lib}", "-lcuinfer", f"-Wl,-rpath,{cuinfer_lib}"]

try:
    from torch.utils.cpp_extension import load
    mod = load(
        name="cuinfer_gemm_wrapper",
        sources=[src],
        extra_include_paths=[hdr_dir],
        extra_cflags=["-O2", "-std=c++17"],
        extra_cuda_cflags=["-O2"],
        extra_ldflags=ldflags,
        verbose=True,
    )
    print("[cuinfer_gemm] ✓ OK")

    import importlib
    spec = importlib.util.find_spec("cuinfer_gemm_wrapper")
    if spec and spec.origin:
        shutil.copy2(spec.origin, os.path.join(hdr_dir, "cuinfer_gemm_wrapper.so"))
        print(f"[cuinfer_gemm] ✓ Saved")

except Exception as e:
    print(f"[cuinfer_gemm] ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
