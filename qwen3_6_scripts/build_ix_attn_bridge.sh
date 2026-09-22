#!/bin/bash
# build_ix_attn_bridge.sh — Build ix_attn_bridge.so on real BI-V100
#
# Compiles ix_attn_bridge.cpp → prebuilt .so for Docker deployment.
# Functions: prefill_attention, decode_attention, linear, residual_rms_norm
#
# Run on real machine: bash qwen3_6_scripts/build_ix_attn_bridge.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CPP_SOURCE="${SCRIPT_DIR}/ix_attn_bridge.cpp"
PREBUILT_DIR="${SCRIPT_DIR}/prebuilt/corex-3.2.3-ivcore10"

if [ ! -f "$CPP_SOURCE" ]; then
    echo "ERROR: ix_attn_bridge.cpp not found at $CPP_SOURCE"
    exit 1
fi

echo "=== Building ix_attn_bridge.so ==="
echo "Source: $CPP_SOURCE"

python3 -c "
import os, sys, glob, shutil
from torch.utils.cpp_extension import load

cpp_source = '$CPP_SOURCE'
extra_ldflags = []

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
    name='ix_attn_bridge',
    sources=[cpp_source],
    extra_cflags=['-O2', '-std=c++17'],
    extra_ldflags=extra_ldflags,
    verbose=True,
)

import torch.utils.cpp_extension as ext
build_dir = ext._get_build_directory('ix_attn_bridge', verbose=False)

for f in glob.glob(os.path.join(build_dir, '*.so')):
    dst = '$PREBUILT_DIR/ix_attn_bridge.so'
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(f, dst)
    sz = os.path.getsize(dst)
    print(f'✓ ix_attn_bridge.so ({sz} bytes) → {dst}')
    break

fns = [x for x in dir(mod) if not x.startswith('_')]
print(f'Functions: {fns}')
print('=== Build SUCCESS ===')
"
