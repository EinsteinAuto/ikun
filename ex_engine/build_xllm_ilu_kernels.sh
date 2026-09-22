#!/usr/bin/env bash
# build_xllm_ilu_kernels.sh — Compile xllm upstream ILU kernel wrappers
#
# Source:  upstream_ref/xllm/xllm/core/kernels/ilu/*.cpp
# Already: ex_engine/xllm_kernels/ilu/ (copied from upstream)
# Header:  upstream_ref/xllm/xllm/core/kernels/ilu/ixformer.h
#
# These .cpp files are thin wrappers that call ixformer::infer C++ functions.
# They're already proven to work on BI-V100 (xllm uses them in production).
# We compile them into xllm_ilu_ops.so with pybind11 bindings.
#
# Usage:
#   bash build_xllm_ilu_kernels.sh [VLLM_ROOT]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Source locations — prefer ex_engine copy, fall back to upstream_ref
ILU_DIR="${SCRIPT_DIR}/xllm_kernels/ilu"
if [[ ! -d "$ILU_DIR" ]]; then
    ILU_DIR="${REPO_ROOT}/upstream_ref/xllm/xllm/core/kernels/ilu"
fi

if [[ ! -d "$ILU_DIR" ]]; then
    echo "[xllm_ilu] ERROR: ILU kernel source not found" >&2
    exit 1
fi

# Header with ixformer::infer declarations
IXFORMER_H="${ILU_DIR}/ixformer.h"
if [[ ! -f "$IXFORMER_H" ]]; then
    # Copy from upstream
    cp "${REPO_ROOT}/upstream_ref/xllm/xllm/core/kernels/ilu/ixformer.h" \
       "${ILU_DIR}/ixformer.h" 2>/dev/null || true
    cp "${REPO_ROOT}/upstream_ref/xllm/xllm/core/kernels/ilu/utils.h" \
       "${ILU_DIR}/utils.h" 2>/dev/null || true
fi

echo "[xllm_ilu] Source dir: ${ILU_DIR}"
echo "[xllm_ilu] Files:"
ls -la "$ILU_DIR"/*.cpp "$ILU_DIR"/*.h 2>/dev/null || true

# --- Compile via torch.utils.cpp_extension ---
VLLM_ROOT="${1:-}"

python3 << PYEOF
import os
import sys
import glob

# Set up paths
ilu_dir = "${ILU_DIR}"
script_dir = "${SCRIPT_DIR}"
vllm_root = "${VLLM_ROOT}" if "${VLLM_ROOT}" else None

# Find all .cpp files in the ILU directory
cpp_files = sorted(glob.glob(os.path.join(ilu_dir, "*.cpp")))
if not cpp_files:
    print("[xllm_ilu] ERROR: No .cpp files found in", ilu_dir)
    sys.exit(1)

print(f"[xllm_ilu] Found {len(cpp_files)} source files:")
for f in cpp_files:
    print(f"  {os.path.basename(f)}")

# Find ixformer .so files for linking
corex_root = os.environ.get("COREX_ROOT", "/usr/local/corex")
ix_so_files = []
rpath_dirs = set()
for search_dir in [
    os.path.join(corex_root, "lib", "python3", "dist-packages", "ixformer"),
    os.path.join(corex_root, "lib64", "python3", "dist-packages", "ixformer"),
    os.path.join(corex_root, "lib64"),
]:
    if os.path.isdir(search_dir):
        rpath_dirs.add(search_dir)
        for so in glob.glob(os.path.join(search_dir, "*.so")):
            ix_so_files.append(so)
        for so in glob.glob(os.path.join(search_dir, "lib*.so")):
            if so not in ix_so_files:
                ix_so_files.append(so)

extra_ldflags = list(ix_so_files)
for d in rpath_dirs:
    extra_ldflags.append(f"-Wl,-rpath,{d}")

print(f"[xllm_ilu] Linking against {len(ix_so_files)} ixformer .so files")

try:
    from torch.utils.cpp_extension import load
    mod = load(
        name="xllm_ilu_ops",
        sources=cpp_files,
        extra_include_paths=[ilu_dir],
        extra_cflags=["-O2", "-std=c++17"],
        extra_ldflags=extra_ldflags,
        verbose=True,
    )
    print("[xllm_ilu] ✓ Compilation successful")

    # Save the .so
    import torch
    so_path = os.path.join(script_dir, "prebuilt", "xllm_ilu_ops.so")
    os.makedirs(os.path.dirname(so_path), exist_ok=True)

    # Find the compiled .so in the torch cache
    import importlib
    spec = importlib.util.find_spec("xllm_ilu_ops")
    if spec and spec.origin:
        import shutil
        shutil.copy2(spec.origin, so_path)
        print(f"[xllm_ilu] ✓ Saved to {so_path}")

        if vllm_root:
            dst = os.path.join(vllm_root, "ex_engine", "xllm_ilu_ops.so")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(spec.origin, dst)
            print(f"[xllm_ilu] ✓ Deployed to {dst}")

except Exception as e:
    print(f"[xllm_ilu] ERROR: {e}")
    sys.exit(1)
PYEOF

echo "[xllm_ilu] Done"
