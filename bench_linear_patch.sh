#!/bin/bash
set -euo pipefail
cat << 'PYEOF' | CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 -u -
import torch, importlib.util, time
torch.cuda.set_device(0)
dev = torch.device("cuda:0")

SO = "qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10"
def load_so(name):
    spec = importlib.util.spec_from_file_location(name, f"{SO}/{name}.so")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

bridge = load_so("ix_moe_bridge")
act_m = load_so("xllm_activation")

H = 2048

def bench(name, fn, N=1000):
    for _ in range(100): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N): fn()
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / N * 1e6
    print(f"  {name:50s}: {us:8.1f} us")
    return us

x = torch.randn(1, H, device=dev, dtype=torch.float16)

# Match upstream gemv_conditions: m <= 1, k % 32 == 0, n % 2 == 0, no bias
# Test ALL linear ops in qwen3_5.py decode path

print("=== Every linear op in one decode step (TP=4) ===")
print("--- Attention layer (32 layers) ---")

# QKV: (1,2048) @ (1024,2048)^T → (1,1024)  [heads*head_dim + 2*kv_heads*head_dim]
w_qkv = torch.randn(1024, H, device=dev, dtype=torch.float16) * 0.01
bench("qkv F.linear (1,2048)→(1,1024)", lambda: torch.nn.functional.linear(x, w_qkv))
bench("qkv bridge.linear", lambda: bridge.linear(x, w_qkv, None))

# O_proj: (1,768) @ (2048,768)^T → (1,2048)
x_o = torch.randn(1, 768, device=dev, dtype=torch.float16)
w_o = torch.randn(H, 768, device=dev, dtype=torch.float16) * 0.01
bench("o_proj F.linear (1,768)→(1,2048)", lambda: torch.nn.functional.linear(x_o, w_o))
bench("o_proj bridge.linear", lambda: bridge.linear(x_o, w_o, None))

print("\n--- GDN layer (4 layers) ---")
# GDN in_proj: (1,2048) @ (3852,2048)^T → (1,3852)
w_gdn = torch.randn(3852, H, device=dev, dtype=torch.float16) * 0.01
bench("gdn_proj F.linear (1,2048)→(1,3852)", lambda: torch.nn.functional.linear(x, w_gdn))
bench("gdn_proj bridge.linear", lambda: bridge.linear(x, w_gdn, None))

# GDN o_proj: (1,1536) @ (2048,1536)^T → (1,2048)
x_gdn_o = torch.randn(1, 1536, device=dev, dtype=torch.float16)
w_gdn_o = torch.randn(H, 1536, device=dev, dtype=torch.float16) * 0.01
bench("gdn_oproj F.linear (1,1536)→(1,2048)", lambda: torch.nn.functional.linear(x_gdn_o, w_gdn_o))
bench("gdn_oproj bridge.linear", lambda: bridge.linear(x_gdn_o, w_gdn_o, None))

print("\n--- MoE shared expert (36 layers) ---")
I_shared = 128
w_gu = torch.randn(2*I_shared, H, device=dev, dtype=torch.float16) * 0.01
w_down = torch.randn(H, I_shared, device=dev, dtype=torch.float16) * 0.01
bench("shared gate_up F.linear (1,2048)→(1,256)", lambda: torch.nn.functional.linear(x, w_gu))
bench("shared gate_up bridge.linear", lambda: bridge.linear(x, w_gu, None))
x_down = torch.randn(1, I_shared, device=dev, dtype=torch.float16)
bench("shared down F.linear (1,128)→(1,2048)", lambda: torch.nn.functional.linear(x_down, w_down))
bench("shared down bridge.linear", lambda: bridge.linear(x_down, w_down, None))

print("\n--- Router (36 layers) ---")
w_router = torch.randn(257, H, device=dev, dtype=torch.float16) * 0.01
bench("router F.linear (1,2048)→(1,257)", lambda: torch.nn.functional.linear(x, w_router))
bench("router bridge.linear", lambda: bridge.linear(x, w_router, None))

print("\n--- LM head (1x) ---")
w_lm = torch.randn(37984, H, device=dev, dtype=torch.float16) * 0.01
bench("lm_head F.linear (1,2048)→(1,37984)", lambda: torch.nn.functional.linear(x, w_lm))
bench("lm_head bridge.linear", lambda: bridge.linear(x, w_lm, None))

# === Total impact ===
print("\n=== Projected total decode step savings ===")
shapes = [
    ("attn_qkv",   32, (1024, H)),
    ("attn_o",     32, (H, 768)),
    ("gdn_proj",    4, (3852, H)),
    ("gdn_o",       4, (H, 1536)),
    ("shared_gu",  36, (2*I_shared, H)),
    ("shared_down",36, (H, I_shared)),
    ("router",     36, (257, H)),
    ("lm_head",     1, (37984, H)),
]
total_torch = 0
total_bridge = 0
for name, count, (N, K) in shapes:
    w = torch.randn(N, K, device=dev, dtype=torch.float16) * 0.01
    xi = torch.randn(1, K, device=dev, dtype=torch.float16)
    t_torch = bench(f"  {name} F.linear", lambda xi=xi, w=w: torch.nn.functional.linear(xi, w), N=500)
    t_bridge = bench(f"  {name} bridge", lambda xi=xi, w=w: bridge.linear(xi, w, None), N=500)
    total_torch += t_torch * count
    total_bridge += t_bridge * count
    speedup = t_torch / t_bridge if t_bridge > 0 else 0
    print(f"    → x{count}: {t_torch*count:.0f} → {t_bridge*count:.0f} us ({speedup:.1f}x)")

print(f"\n  TOTAL linear ops: {total_torch:.0f} → {total_bridge:.0f} us")
print(f"  Savings: {total_torch - total_bridge:.0f} us = {(total_torch-total_bridge)/1000:.1f} ms")
PYEOF