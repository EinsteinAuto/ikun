#!/bin/bash
# build_test_moe_tcu.sh — Build and test moe_tcu_dispatch.cpp
set -eo pipefail

echo "=== Compile moe_tcu_dispatch ==="
python3 -c "
import torch.utils.cpp_extension as ext
import os, shutil, glob

name = 'moe_tcu_dispatch'
build_dir = 'ex_engine/csrc/build/tmp_' + name
os.makedirs(build_dir, exist_ok=True)

mod = ext.load(
    name=name,
    sources=['ex_engine/csrc/moe_tcu_dispatch.cpp'],
    extra_cflags=['-O2', '-std=c++17'],
    build_directory=build_dir,
    verbose=True,
)

built = glob.glob(build_dir + '/' + name + '*.so')
if built:
    dst = 'ex_engine/csrc/build/' + name + '.so'
    os.makedirs('ex_engine/csrc/build', exist_ok=True)
    shutil.copy2(built[0], dst)
    print(f'[build] SUCCESS: {dst}')
"

echo ""
echo "=== Test ==="
python3 << 'PYTEST'
import torch
import torch.nn.functional as F
import sys, os, glob, time, importlib.util

build_dir = 'ex_engine/csrc/build'
so = glob.glob(f'{build_dir}/tmp_moe_tcu_dispatch/moe_tcu_dispatch*.so')
if not so:
    print("SKIP: .so not found")
    sys.exit(0)
spec = importlib.util.spec_from_file_location("moe_tcu_dispatch", so[0])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print(f"Loaded: {so[0]}")

# ============================================================
# Test 1: moe_decode correctness
# ============================================================
print("\n--- moe_decode correctness ---")
K, I = 128, 256
E = 8
top_k = 4
hidden = torch.randn(1, K, dtype=torch.float16, device='cuda')
w13 = torch.randn(E, 2*I, K, dtype=torch.float16, device='cuda') * 0.01
w2 = torch.randn(E, K, I, dtype=torch.float16, device='cuda') * 0.01
expert_ids = torch.tensor([0, 3, 5, 7], dtype=torch.int64, device='cuda')
expert_weights = torch.tensor([0.3, 0.25, 0.25, 0.2], dtype=torch.float32, device='cuda')

# C++ result
out_cpp = mod.moe_decode(hidden, w13, w2, expert_ids, expert_weights)

# Python reference
out_py = torch.zeros_like(hidden)
for k in range(top_k):
    eid = expert_ids[k].item()
    w = expert_weights[k].item()
    gate_up = F.linear(hidden, w13[eid])
    gate = F.silu(gate_up[:, :I])
    up = gate_up[:, I:]
    act = gate * up
    expert_out = F.linear(act, w2[eid])
    out_py += w * expert_out

diff = (out_cpp.float() - out_py.float()).abs().max().item()
print(f"  max_diff={diff:.6f} {'PASS' if diff < 1.0 else 'FAIL'}")

# ============================================================
# Test 2: moe_expert_gemm_tcu correctness
# ============================================================
print("\n--- moe_expert_gemm_tcu correctness ---")
num_experts = 4
K, N = 128, 256
expert_counts = torch.tensor([8, 0, 16, 4], dtype=torch.int64, device='cuda')
total = expert_counts.sum().item()
inp = torch.randn(total, K, dtype=torch.float16, device='cuda') * 0.1
weights = torch.randn(num_experts, N, K, dtype=torch.float16, device='cuda') * 0.1

out_cpp = mod.moe_expert_gemm_tcu(inp, weights, expert_counts)

# Python reference
out_py = torch.zeros(total, N, dtype=torch.float16, device='cuda')
off = 0
for e in range(num_experts):
    cnt = expert_counts[e].item()
    if cnt == 0: continue
    out_py[off:off+cnt] = F.linear(inp[off:off+cnt], weights[e])
    off += cnt

diff = (out_cpp.float() - out_py.float()).abs().max().item()
print(f"  max_diff={diff:.6f} {'PASS' if diff < 0.5 else 'FAIL'}")

# ============================================================
# Test 3: Performance — Python loop vs C++ loop
# ============================================================
print("\n--- Performance: decode (1 token, 8 experts) ---")
K, I = 4096, 11008
E, top_k = 64, 8
hidden = torch.randn(1, K, dtype=torch.float16, device='cuda')
w13 = torch.randn(E, 2*I, K, dtype=torch.float16, device='cuda') * 0.001
w2 = torch.randn(E, K, I, dtype=torch.float16, device='cuda') * 0.001
expert_ids = torch.tensor([0,5,10,20,30,40,50,60], dtype=torch.int64, device='cuda')
expert_weights = torch.ones(top_k, dtype=torch.float32, device='cuda') / top_k

# Warmup
for _ in range(3):
    mod.moe_decode(hidden, w13, w2, expert_ids, expert_weights)
torch.cuda.synchronize()

# C++ loop
t0 = time.time()
for _ in range(100):
    mod.moe_decode(hidden, w13, w2, expert_ids, expert_weights)
torch.cuda.synchronize()
ms_cpp = (time.time() - t0) / 100 * 1000

# Python loop
for _ in range(3):
    out_py = torch.zeros_like(hidden)
    for k in range(top_k):
        eid = expert_ids[k].item()
        w = expert_weights[k].item()
        gate_up = F.linear(hidden, w13[eid])
        gate = F.silu(gate_up[:, :I])
        up = gate_up[:, I:]
        act = gate * up
        out_py += w * F.linear(act, w2[eid])
torch.cuda.synchronize()

t0 = time.time()
for _ in range(100):
    out_py = torch.zeros_like(hidden)
    for k in range(top_k):
        eid = expert_ids[k].item()
        w = expert_weights[k].item()
        gate_up = F.linear(hidden, w13[eid])
        gate = F.silu(gate_up[:, :I])
        up = gate_up[:, I:]
        act = gate * up
        out_py += w * F.linear(act, w2[eid])
torch.cuda.synchronize()
ms_py = (time.time() - t0) / 100 * 1000

print(f"  C++ loop:    {ms_cpp:.2f} ms")
print(f"  Python loop: {ms_py:.2f} ms")
print(f"  Speedup:     {ms_py/ms_cpp:.2f}x")
print(f"  Saved:       {ms_py-ms_cpp:.2f} ms per forward")

print("\n=== DONE ===")
PYTEST
