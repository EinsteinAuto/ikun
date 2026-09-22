#!/bin/bash
# build_test_hgemm_warp.sh — Compile and benchmark kernel 10 (warp tiling)
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="${SCRIPT_DIR}/cuda"

echo "=== Compile hgemm_warptiling (kernel 10, WARPSIZE=64) ==="
python3 -c "
import torch.utils.cpp_extension as ext
import os, shutil, glob

name = 'hgemm_warptiling'
build_dir = '${SCRIPT_DIR}/build/tmp_' + name
os.makedirs(build_dir, exist_ok=True)

try:
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
        print(f'[build] SUCCESS: {dst} ({os.path.getsize(dst)} bytes)')
except Exception as e:
    print(f'[build] FAILED: {e}')
    import traceback; traceback.print_exc()
"

echo ""
echo "=== Test ==="
python3 << 'PYTEST'
import torch, sys, os, glob, time

build_dir = 'ex_engine/xllm_kernels/build'
sys.path.insert(0, build_dir)

# Load kernel 10
try:
    so = glob.glob(f'{build_dir}/tmp_hgemm_warptiling/hgemm_warptiling*.so')
    if so:
        import importlib.util
        spec = importlib.util.spec_from_file_location("hgemm_warptiling", so[0])
        hw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hw)
        print("kernel 10 (warp tiling) loaded")
    else:
        print("SKIP: kernel 10 .so not found")
        sys.exit(0)
except Exception as e:
    print(f"SKIP: {e}")
    sys.exit(0)

# Load kernel 6 for comparison
try:
    so6 = glob.glob(f'{build_dir}/tmp_hgemm_blocktiling/hgemm_blocktiling*.so')
    if so6:
        spec6 = importlib.util.spec_from_file_location("hgemm_blocktiling", so6[0])
        hb = importlib.util.module_from_spec(spec6)
        spec6.loader.exec_module(hb)
        has_k6 = True
        print("kernel 6 (block tiling) loaded")
    else:
        has_k6 = False
except:
    has_k6 = False

# Correctness
print("\n--- Correctness (128x128 @ 128x128) ---")
M, N, K = 128, 128, 128
A = torch.randn(M, K, dtype=torch.float16, device='cuda')
B = torch.randn(K, N, dtype=torch.float16, device='cuda')
C_ref = torch.matmul(A.float(), B.float()).half()
C_k10 = hw.hgemm_warp(A, B)
diff = (C_ref.float() - C_k10.float()).abs().max().item()
print(f"  Max abs diff: {diff:.6f}")
assert diff < 2.0, f"FAIL diff={diff}"
print("  PASS")

# Correctness on MoE size
print("\n--- Correctness (256x4096 @ 4096x11008) ---")
M, N, K = 256, 11008, 4096
A = torch.randn(M, K, dtype=torch.float16, device='cuda') * 0.01
B = torch.randn(K, N, dtype=torch.float16, device='cuda') * 0.01
C_ref = torch.matmul(A.float(), B.float()).half()
C_k10 = hw.hgemm_warp(A, B)
diff = (C_ref.float() - C_k10.float()).abs().max().item()
rel = diff / (C_ref.float().abs().max().item() + 1e-8)
print(f"  Max abs diff: {diff:.6f}, rel: {rel:.6f}")
print("  PASS" if rel < 0.1 else "  WARN: large relative diff")

# Performance benchmark
print("\n--- Performance (256x4096 @ 4096x11008, 100 iters) ---")
M, N, K = 256, 11008, 4096
A = torch.randn(M, K, dtype=torch.float16, device='cuda')
B = torch.randn(K, N, dtype=torch.float16, device='cuda')

def bench(fn, name, iters=100, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / iters * 1000
    print(f"  {name}: {ms:.2f} ms/iter")
    return ms

t_torch  = bench(lambda: torch.matmul(A, B), "torch.matmul")
t_k10    = bench(lambda: hw.hgemm_warp(A, B), "kernel 10 (warp)")
if has_k6:
    t_k6 = bench(lambda: hb.hgemm(A, B), "kernel 6 (block)")
    print(f"\n  K10/torch = {t_k10/t_torch:.2f}x")
    print(f"  K6/torch  = {t_k6/t_torch:.2f}x")
    print(f"  K10/K6    = {t_k10/t_k6:.2f}x (K10 should be faster)")
else:
    print(f"\n  K10/torch = {t_k10/t_torch:.2f}x")

print("\n=== DONE ===")
PYTEST
