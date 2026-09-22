#!/bin/bash
# build_ix_moe_bridge.sh — Build ix_moe_bridge.so on real BI-V100
#
# This compiles ex_engine/csrc/ix_moe_bridge.cpp into a prebuilt .so
# that can be deployed without JIT compilation in Docker.
#
# Run on real machine: bash qwen3_6_scripts/build_ix_moe_bridge.sh
# Output: qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10/ix_moe_bridge.so

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CPP_SOURCE="${PROJECT_DIR}/ex_engine/csrc/ix_moe_bridge.cpp"
PREBUILT_DIR="${SCRIPT_DIR}/prebuilt/corex-3.2.3-ivcore10"

if [ ! -f "$CPP_SOURCE" ]; then
    # Also try the local copy
    CPP_SOURCE="${SCRIPT_DIR}/ix_moe_bridge.cpp"
fi

if [ ! -f "$CPP_SOURCE" ]; then
    echo "ERROR: ix_moe_bridge.cpp not found"
    exit 1
fi

echo "=== Building ix_moe_bridge.so ==="
echo "Source: $CPP_SOURCE"
echo "Output: $PREBUILT_DIR/ix_moe_bridge.so"

python3 -c "
import os, sys, glob
from torch.utils.cpp_extension import load

cpp_source = '$CPP_SOURCE'
extra_ldflags = []

# Find ixformer .so to link against
try:
    import ixformer
    ixf_dir = os.path.dirname(ixformer.__file__)
    for so in glob.glob(os.path.join(ixf_dir, '*.so')):
        extra_ldflags.append(so)
    extra_ldflags.append(f'-Wl,-rpath,{ixf_dir}')
except ImportError:
    pass

corex_lib = '/usr/local/corex/lib64'
if os.path.isdir(corex_lib):
    for lib in ['libixattn.so', 'libixformer.so', 'libcublas.so']:
        p = os.path.join(corex_lib, lib)
        if os.path.isfile(p):
            extra_ldflags.append(p)
    extra_ldflags.append(f'-Wl,-rpath,{corex_lib}')

print(f'Linking: {extra_ldflags}')

mod = load(
    name='ix_moe_bridge',
    sources=[cpp_source],
    extra_cflags=['-O2', '-std=c++17'],
    extra_ldflags=extra_ldflags,
    verbose=True,
)

# Find the compiled .so and copy to prebuilt
import torch.utils.cpp_extension as ext
build_dir = ext._get_build_directory('ix_moe_bridge', verbose=False)
print(f'Build dir: {build_dir}')

import shutil
for f in glob.glob(os.path.join(build_dir, '*.so')):
    dst = '$PREBUILT_DIR/ix_moe_bridge.so'
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(f, dst)
    sz = os.path.getsize(dst)
    print(f'✓ ix_moe_bridge.so ({sz} bytes) → {dst}')
    break

# Verify
fns = [x for x in dir(mod) if not x.startswith('_')]
print(f'Functions: {fns}')
print('=== Build SUCCESS ===')
"
