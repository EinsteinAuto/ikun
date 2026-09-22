#!/bin/bash
# build_test_hgemm.sh — Compile and test hgemm_blocktiling on BI-V100
#
# Usage: bash ex_engine/xllm_kernels/build_test_hgemm.sh
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="${SCRIPT_DIR}/cuda"

echo "=== 1. Compile hgemm_blocktiling ==="
python3 -c "
import torch.utils.cpp_extension as ext
import os, shutil, glob

name = 'hgemm_blocktiling'
build_dir = '${SCRIPT_DIR}/build/tmp_' + name
os.makedirs(build_dir, exist_ok=True)

try:
    mod = ext.load(
        name=name,
        sources=[
            '${CUDA_DIR}/hgemm_blocktiling.cu',
            '${CUDA_DIR}/bindings/hgemm_bind.cpp',
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
    else:
        print('[build] WARNING: .so not found')
except Exception as e:
    print(f'[build] FAILED: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "=== 2. Functional test ==="
python3 << 'PYTEST'
import torch
import sys, os, glob

# Find and load the .so
build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.', 
                         'ex_engine/xllm_kernels/build')
sys.path.insert(0, build_dir)

try:
    import hgemm_blocktiling as hg
    print("Module loaded successfully")
except ImportError:
    # Try loading from tmp build dir
    import importlib.util
    so_files = glob.glob('ex_engine/xllm_kernels/build/tmp_hgemm_blocktiling/hgemm_blocktiling*.so')
    if not so_files:
        print("SKIP: .so not found (need GPU machine)")
        sys.exit(0)
    spec = importlib.util.spec_from_file_location("hgemm_blocktiling", so_files[0])
    hg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hg)
    print(f"Module loaded from {so_files[0]}")

# Test 1: Small GEMM correctness
print("\n--- Test 1: Small GEMM (64x64 @ 64x64) ---")
M, N, K = 64, 64, 64
A = torch.randn(M, K, dtype=torch.float16, device='cuda')
B = torch.randn(K, N, dtype=torch.float16, device='cuda')

C_ref = torch.matmul(A.float(), B.float()).half()
C_our = hg.hgemm(A, B)

diff = (C_ref.float() - C_our.float()).abs().max().item()
print(f"  Max abs diff: {diff:.6f}")
assert diff < 1.0, f"FAILED: diff={diff} too large"
print(f"  PASS (diff < 1.0)")

# Test 2: Larger GEMM (typical MoE dimensions)
print("\n--- Test 2: MoE-sized GEMM (256x4096 @ 4096x11008) ---")
M, N, K = 256, 11008, 4096
A = torch.randn(M, K, dtype=torch.float16, device='cuda') * 0.01
B = torch.randn(K, N, dtype=torch.float16, device='cuda') * 0.01

C_ref = torch.matmul(A.float(), B.float()).half()
C_our = hg.hgemm(A, B)

diff = (C_ref.float() - C_our.float()).abs().max().item()
rel_diff = diff / (C_ref.float().abs().max().item() + 1e-8)
print(f"  Max abs diff: {diff:.6f}, rel: {rel_diff:.6f}")
assert rel_diff < 0.05, f"FAILED: rel_diff={rel_diff} too large"
print(f"  PASS")

# Test 3: MoE expert GEMM with variable counts
print("\n--- Test 3: MoE expert GEMM (8 experts, variable tokens) ---")
num_experts = 8
K_dim = 128
N_dim = 256
expert_counts = torch.tensor([32, 16, 0, 48, 8, 24, 4, 12], dtype=torch.int32)
total_tokens = expert_counts.sum().item()

input_tensor = torch.randn(total_tokens, K_dim, dtype=torch.float16, device='cuda') * 0.1
weights = torch.randn(num_experts, N_dim, K_dim, dtype=torch.float16, device='cuda') * 0.1

output = hg.moe_expert_gemm(input_tensor, weights, expert_counts.cuda())

# Verify against torch reference
offset = 0
for e in range(num_experts):
    cnt = expert_counts[e].item()
    if cnt == 0:
        continue
    inp_e = input_tensor[offset:offset+cnt]
    w_e = weights[e]  # (N, K)
    ref_e = torch.matmul(inp_e.float(), w_e.float().t()).half()
    out_e = output[offset:offset+cnt]
    diff_e = (ref_e.float() - out_e.float()).abs().max().item()
    print(f"  Expert {e} (tokens={cnt}): max_diff={diff_e:.6f}")
    offset += cnt
print(f"  PASS")

# Test 4: Performance benchmark
print("\n--- Test 4: Performance (256x4096 @ 4096x11008, 100 iters) ---")
M, N, K = 256, 11008, 4096
A = torch.randn(M, K, dtype=torch.float16, device='cuda')
B = torch.randn(K, N, dtype=torch.float16, device='cuda')

# Warmup
for _ in range(10):
    hg.hgemm(A, B)
torch.cuda.synchronize()

import time
start = time.time()
for _ in range(100):
    hg.hgemm(A, B)
torch.cuda.synchronize()
elapsed = time.time() - start
print(f"  Custom kernel: {elapsed*10:.2f} ms/iter")

start = time.time()
for _ in range(100):
    torch.matmul(A, B)
torch.cuda.synchronize()
elapsed2 = time.time() - start
print(f"  torch.matmul:  {elapsed2*10:.2f} ms/iter")
print(f"  Ratio: {elapsed/elapsed2:.2f}x")

print("\n=== ALL TESTS PASSED ===")
PYTEST
