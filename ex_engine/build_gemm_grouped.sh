#!/usr/bin/env bash
# build_gemm_grouped.sh — Compile grouped GEMM kernel + bindings
#
# Requires: corex clang/16 + cutlass headers (on BI-V100 device)
# Output: gemm_grouped.so (importable from Python)
#
# Reference: ex_engine/xllm_kernels/build_test_cutlass_batched.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source files
GEMM_CU="${SCRIPT_DIR}/csrc/gemm_grouped.cu"
BIND_CPP="${SCRIPT_DIR}/csrc/gemm_grouped_bind.cpp"
BATCHED_CU="${SCRIPT_DIR}/../xllm_kernels/cuda/corex_batched_gemm_kernel.cu"

echo "[gemm] Building gemm_grouped.so"

# Find cutlass include path
SAMPLES="/usr/local/corex-samples-3.2.3_x86_64/samples/cutlass"
CUTLASS_INCLUDE=""
for d in "${SAMPLES}/include" "/usr/local/corex/include/cutlass" "/usr/include/cutlass"; do
    if [[ -d "$d" ]]; then
        CUTLASS_INCLUDE="$d"
        break
    fi
done

if [[ -z "$CUTLASS_INCLUDE" ]]; then
    echo "[gemm] ERROR: cutlass include not found"
    exit 1
fi
echo "[gemm] cutlass: ${CUTLASS_INCLUDE}"

python3 << PYEOF
import os, sys, shutil

script_dir = "${SCRIPT_DIR}"
cutlass_inc = "${CUTLASS_INCLUDE}"

sources = [
    "${GEMM_CU}",
    "${BIND_CPP}",
    "${BATCHED_CU}",
]
sources = [s for s in sources if os.path.isfile(s)]

print(f"[gemm] Compiling {len(sources)} source files")
for s in sources:
    print(f"  {os.path.basename(s)}")

try:
    from torch.utils.cpp_extension import load
    mod = load(
        name="gemm_grouped",
        sources=sources,
        extra_include_paths=[cutlass_inc, script_dir],
        extra_cflags=["-O2", "-std=c++17"],
        extra_ldflags=["/usr/local/corex/lib64/libcuinfer.so", "-Wl,-rpath,/usr/local/corex/lib64"],
        extra_cuda_cflags=["-O2", "",
                           f"-I{cutlass_inc}"],
        verbose=True,
    )
    print("[gemm] ✓ Compilation successful")

    import importlib
    spec = importlib.util.find_spec("gemm_grouped")
    if spec and spec.origin:
        dst = os.path.join(script_dir, "gemm_grouped.so")
        shutil.copy2(spec.origin, dst)
        print(f"[gemm] ✓ Saved to {dst}")

except Exception as e:
    print(f"[gemm] ERROR: {e}", file=sys.stderr)
    import traceback; traceback.print_exc()
    sys.exit(1)
PYEOF

echo "[gemm] Done"
