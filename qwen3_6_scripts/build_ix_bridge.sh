#!/bin/bash
# Build ix_full_bridge.so — bridges ixformer_torch_ext C++ symbols to Python
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VLLM_ROOT="${1:?usage: build_ix_bridge.sh VLLM_ROOT}"
BRIDGE_SRC="${SCRIPT_DIR}/ix_full_bridge.cpp"

if [ ! -f "$BRIDGE_SRC" ]; then
    echo "[bridge] SKIP: $BRIDGE_SRC not found"
    exit 0
fi

# Find ixformer .so with the real symbols
IX_TORCH_SO=""
for f in /usr/local/corex/lib/python3/dist-packages/ixformer/_ixformer_torch*.so; do
    if [ -f "$f" ]; then
        IX_TORCH_SO="$f"
        break
    fi
done

IX_LIB_SO=""
for f in /usr/local/corex/lib/python3/dist-packages/ixformer/libixformer.so; do
    if [ -f "$f" ]; then
        IX_LIB_SO="$f"
        break
    fi
done

TORCH_LIB=$(python3 -c "import torch; print(torch.__path__[0] + '/lib')" 2>/dev/null)
IX_DIR=$(python3 -c "import ixformer; import os; print(os.path.dirname(ixformer.__file__))" 2>/dev/null)

echo "[bridge] IX_TORCH_SO=${IX_TORCH_SO}"
echo "[bridge] IX_LIB_SO=${IX_LIB_SO}"
echo "[bridge] TORCH_LIB=${TORCH_LIB}"

# Clear cached build (namespace changed)
rm -rf /root/.cache/torch_extensions/py310_cu102/ix_full_bridge

python3 << PYEOF
import torch
from torch.utils.cpp_extension import load
import shutil, os

extra_ldflags = []

# Link against _ixformer_torch .so (has ixformer_torch_ext:: symbols)
ix_torch = "${IX_TORCH_SO}"
if ix_torch and os.path.exists(ix_torch):
    extra_ldflags.append(ix_torch)
    extra_ldflags.append(f"-Wl,-rpath,{os.path.dirname(ix_torch)}")

# Also link libixformer.so (has launcher symbols)
ix_lib = "${IX_LIB_SO}"
if ix_lib and os.path.exists(ix_lib):
    extra_ldflags.append(ix_lib)

# torch lib rpath
torch_lib = "${TORCH_LIB}"
if torch_lib and os.path.isdir(torch_lib):
    extra_ldflags.append(f"-Wl,-rpath,{torch_lib}")

print(f"[bridge] ldflags: {extra_ldflags}")

mod = load(
    name="ix_full_bridge",
    sources=["${BRIDGE_SRC}"],
    extra_cflags=["-O2", "-std=c++17"],
    extra_ldflags=extra_ldflags,
    verbose=True,
)

# Find the compiled .so and copy to VLLM_ROOT
import importlib
spec = importlib.util.find_spec("ix_full_bridge")
if spec and spec.origin:
    dest = os.path.join("${VLLM_ROOT}", "ix_full_bridge.so")
    shutil.copy2(spec.origin, dest)
    print(f"[bridge] SUCCESS: {dest}")
    fns = [x for x in dir(mod) if not x.startswith("_")]
    print(f"[bridge] functions: {fns}")
else:
    # Search in cache
    import glob
    for so in glob.glob(os.path.expanduser("~/.cache/torch_extensions/**/ix_full_bridge*.so"), recursive=True):
        dest = os.path.join("${VLLM_ROOT}", "ix_full_bridge.so")
        shutil.copy2(so, dest)
        print(f"[bridge] SUCCESS: {so} -> {dest}")
        break
    else:
        print("[bridge] WARNING: could not find compiled .so")
PYEOF

echo "[bridge] Build complete"
