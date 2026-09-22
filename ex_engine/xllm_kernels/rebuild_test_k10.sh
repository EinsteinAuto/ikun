#!/bin/bash
# rebuild_test_k10.sh — Clean rebuild and test kernel 10 Config B
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="${SCRIPT_DIR}/cuda"

echo "=== Clean old builds ==="
rm -rf "${SCRIPT_DIR}/build/tmp_hgemm_warptiling"
rm -f "${SCRIPT_DIR}/build/hgemm_warptiling.so"

echo "=== Compile ==="
python3 -c "
import torch.utils.cpp_extension as ext
import os, shutil, glob

name = 'hgemm_warptiling'
build_dir = '${SCRIPT_DIR}/build/tmp_' + name
os.makedirs(build_dir, exist_ok=True)

mod = ext.load(
    name=name,
    sources=[
        '${CUDA_DIR}/hgemm_warptiling.cu',
        '${CUDA_DIR}/bindings/hgemm_warp_bind.cpp',
    ],
    extra_include_paths=['${CUDA_DIR}/headers'],
    extra_cflags=['-O2', '-std=c++17'],
    extra_cuda_cflags=['-O2'],
    build_directory=build_dir,
    verbose=True,
)
built = glob.glob(build_dir + '/' + name + '*.so')
if built:
    dst = '${SCRIPT_DIR}/build/' + name + '.so'
    shutil.copy2(built[0], dst)
    print(f'[build] SUCCESS: {dst}')
"

echo ""
echo "=== Test ==="
python3 << 'PYTEST'
import torch, sys, os, glob, time, importlib.util

build_dir = 'ex_engine/xllm_kernels/build'
so = glob.glob(f'{build_dir}/tmp_hgemm_warptiling/hgemm_warptiling*.so')
if not so:
    print("SKIP: .so not found")
    sys.exit(0)
spec = importlib.util.spec_from_file_location("hgemm_warptiling", so[0])
hw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hw)
print(f"Loaded: {so[0]}")

# Test 1: tiny
print("\n--- 16x16 @ 16x16 ---")
A = torch.eye(16, dtype=torch.float16, device='cuda')
B = torch.ones(16, 16, dtype=torch.float16, device='cuda')
C = hw.hgemm_warp(A, B)
diff = (C.float() - B.float()).abs().max().item()
print(f"  I @ ones = ones? diff={diff:.6f}")

# Test 2: 128x128
print("\n--- 128x128 @ 128x128 ---")
A = torch.randn(128, 128, dtype=torch.float16, device='cuda') * 0.1
B = torch.randn(128, 128, dtype=torch.float16, device='cuda') * 0.1
C_ref = torch.matmul(A.float(), B.float()).half()
C_k10 = hw.hgemm_warp(A, B)
diff = (C_ref.float() - C_k10.float()).abs().max().item()
print(f"  max_diff={diff:.6f}")
if diff > 2.0:
    # Debug: print a few values
    print(f"  C_ref[0,:5]  = {C_ref[0,:5].tolist()}")
    print(f"  C_k10[0,:5]  = {C_k10[0,:5].tolist()}")
    print(f"  C_ref[-1,-5:] = {C_ref[-1,-5:].tolist()}")
    print(f"  C_k10[-1,-5:] = {C_k10[-1,-5:].tolist()}")
    print("  FAIL")
else:
    print("  PASS")

# Test 3: MoE size
print("\n--- 256x4096 @ 4096x11008 ---")
A = torch.randn(256, 4096, dtype=torch.float16, device='cuda') * 0.01
B = torch.randn(4096, 11008, dtype=torch.float16, device='cuda') * 0.01
C_ref = torch.matmul(A.float(), B.float()).half()
C_k10 = hw.hgemm_warp(A, B)
diff = (C_ref.float() - C_k10.float()).abs().max().item()
rel = diff / (C_ref.float().abs().max().item() + 1e-8)
print(f"  max_diff={diff:.6f}, rel={rel:.6f}")
if diff > 2.0:
    print(f"  C_ref[0,:5]  = {C_ref[0,:5].tolist()}")
    print(f"  C_k10[0,:5]  = {C_k10[0,:5].tolist()}")
    print("  FAIL")
else:
    print("  PASS")

# Test 4: Performance
print("\n--- Performance 256x4096 @ 4096x11008 ---")
for _ in range(10):
    hw.hgemm_warp(A, B)
torch.cuda.synchronize()

t0 = time.time()
for _ in range(100):
    hw.hgemm_warp(A, B)
torch.cuda.synchronize()
ms_k10 = (time.time() - t0) / 100 * 1000

for _ in range(10):
    torch.matmul(A, B)
torch.cuda.synchronize()

t0 = time.time()
for _ in range(100):
    torch.matmul(A, B)
torch.cuda.synchronize()
ms_torch = (time.time() - t0) / 100 * 1000

print(f"  kernel 10: {ms_k10:.2f} ms")
print(f"  torch.matmul: {ms_torch:.2f} ms")
print(f"  ratio: {ms_k10/ms_torch:.2f}x")
PYTEST
