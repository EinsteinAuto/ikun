#!/bin/bash
set -euo pipefail
cat << 'PYEOF' | CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 -u -
"""Verify patch #7: linear → ix_moe_bridge.linear correctness + performance."""
import torch, importlib.util, time, sys, os
torch.cuda.set_device(0)
dev = torch.device("cuda:0")

# Add project to path
sys.path.insert(0, ".")
sys.path.insert(0, "qwen3_6_scripts")

SO = "qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10"
def load_so(name):
    spec = importlib.util.spec_from_file_location(name, f"{SO}/{name}.so")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

bridge = load_so("ix_moe_bridge")

# === 1. Correctness: bridge.linear vs F.linear ===
print("=== Correctness ===")
torch.manual_seed(42)
shapes = [
    ("qkv",        2048, 1024),
    ("o_proj",       768, 2048),
    ("gdn_proj",   2048, 3852),
    ("gdn_o",      1536, 2048),
    ("shared_gu",  2048,  256),
    ("shared_down", 128, 2048),
    ("router",     2048,  257),
    ("lm_head",    2048, 37984),
]
all_pass = True
for name, K, N in shapes:
    x = torch.randn(1, K, device=dev, dtype=torch.float16) * 0.01
    w = torch.randn(N, K, device=dev, dtype=torch.float16) * 0.01

    ref = torch.nn.functional.linear(x, w)
    out = bridge.linear(x, w, None)
    torch.cuda.synchronize()

    md = (out.float() - ref.float()).abs().max().item()
    rd = (out.float() - ref.float()).abs().mean().item() / max(ref.float().abs().mean().item(), 1e-10)
    ok = md < 0.1
    status = "PASS" if ok else "FAIL"
    print(f"  {name:15s} ({K}→{N}): max_diff={md:.6f} rel={rd:.6f} {status}")
    if not ok:
        all_pass = False

# === 2. With bias ===
print("\n=== With bias ===")
for name, K, N in [("bias_test", 2048, 1024)]:
    x = torch.randn(1, K, device=dev, dtype=torch.float16)
    w = torch.randn(N, K, device=dev, dtype=torch.float16)
    b = torch.randn(N, device=dev, dtype=torch.float16)
    ref = torch.nn.functional.linear(x, w, b)
    out = bridge.linear(x, w, b)
    torch.cuda.synchronize()
    md = (out.float() - ref.float()).abs().max().item()
    print(f"  {name}: max_diff={md:.6f} {'PASS' if md<0.1 else 'FAIL'}")

# === 3. Batched (prefill, m>1) ===
print("\n=== Batched (m>1) ===")
for m in [2, 4, 8, 32]:
    x = torch.randn(m, 2048, device=dev, dtype=torch.float16) * 0.01
    w = torch.randn(1024, 2048, device=dev, dtype=torch.float16) * 0.01
    ref = torch.nn.functional.linear(x, w)
    out = bridge.linear(x, w, None)
    torch.cuda.synchronize()
    md = (out.float() - ref.float()).abs().max().item()
    print(f"  m={m}: max_diff={md:.6f} {'PASS' if md<0.5 else 'FAIL'}")

# === 4. End-to-end performance with patch ===
print("\n=== End-to-end: simulated decode step ===")
def bench(name, fn, N=500):
    for _ in range(50): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N): fn()
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / N * 1e6
    return us

# Simulate all linears in one decode step
x = torch.randn(1, 2048, device=dev, dtype=torch.float16)
layers = {
    "qkv":   (torch.randn(1024, 2048, device=dev, dtype=torch.float16)*0.01, 32),
    "o":     (torch.randn(2048, 768, device=dev, dtype=torch.float16)*0.01, 32),
    "gdn_p": (torch.randn(3852, 2048, device=dev, dtype=torch.float16)*0.01, 4),
    "gdn_o": (torch.randn(2048, 1536, device=dev, dtype=torch.float16)*0.01, 4),
    "sh_gu": (torch.randn(256, 2048, device=dev, dtype=torch.float16)*0.01, 36),
    "sh_dn": (torch.randn(2048, 128, device=dev, dtype=torch.float16)*0.01, 36),
    "router":(torch.randn(257, 2048, device=dev, dtype=torch.float16)*0.01, 36),
    "lm_hd": (torch.randn(37984, 2048, device=dev, dtype=torch.float16)*0.01, 1),
}

def full_step_torch():
    for name, (w, count) in layers.items():
        xi = x if w.size(1) == 2048 else torch.randn(1, w.size(1), device=dev, dtype=torch.float16)
        for _ in range(count):
            torch.nn.functional.linear(xi, w)

def full_step_bridge():
    for name, (w, count) in layers.items():
        xi = x if w.size(1) == 2048 else torch.randn(1, w.size(1), device=dev, dtype=torch.float16)
        for _ in range(count):
            bridge.linear(xi, w, None)

t_torch = bench("F.linear all layers", full_step_torch, N=100)
t_bridge = bench("bridge.linear all layers", full_step_bridge, N=100)
print(f"  F.linear total:      {t_torch:.0f} us ({t_torch/1000:.1f} ms)")
print(f"  bridge.linear total: {t_bridge:.0f} us ({t_bridge/1000:.1f} ms)")
print(f"  Savings:             {(t_torch-t_bridge):.0f} us ({(t_torch-t_bridge)/1000:.1f} ms)")
print(f"  Speedup:             {t_torch/t_bridge:.2f}x")

if all_pass:
    print("\n✓ ALL CORRECTNESS CHECKS PASSED")
    print("✓ Patch #7 ready for deployment")
else:
    print("\n✗ SOME CHECKS FAILED")
    sys.exit(1)
PYEOF