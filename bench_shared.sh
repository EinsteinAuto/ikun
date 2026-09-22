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
I_shared = 128

x = torch.randn(1, H, device=dev, dtype=torch.float16)
w_gu = torch.randn(2*I_shared, H, device=dev, dtype=torch.float16) * 0.01
w_down = torch.randn(H, I_shared, device=dev, dtype=torch.float16) * 0.01

def bench(name, fn, N=1000):
    for _ in range(100): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N): fn()
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / N * 1e6
    print(f"  {name:45s}: {us:8.1f} us")
    return us

print("=== ix_moe_bridge.linear probe ===")
linear_ok = False
for desc, args in [("(x,w)", (x, w_gu)),
                    ("(x,w,None)", (x, w_gu, None)),
                    ("(x,w,bias0)", (x, w_gu, torch.zeros(2*I_shared,device=dev,dtype=torch.float16)))]:
    try:
        out = bridge.linear(*args); torch.cuda.synchronize()
        print(f"  linear{desc}: OK shape={out.shape}")
        linear_ok = True; break
    except Exception as e:
        print(f"  linear{desc}: {str(e)[:80]}")

print("\n=== Shared expert benchmarks ===")
act_buf = torch.empty(1, I_shared, device=dev, dtype=torch.float16)

def shared_torch():
    gu = torch.nn.functional.linear(x, w_gu)
    g, u = gu.chunk(2, dim=-1)
    act = torch.sigmoid(g) * g * u
    return torch.nn.functional.linear(act, w_down)
bench("A: torch linear + torch silu", shared_torch)

def shared_xllm_silu():
    gu = torch.nn.functional.linear(x, w_gu)
    act_m.silu_and_mul(act_buf, gu)
    return torch.nn.functional.linear(act_buf, w_down)
bench("B: torch linear + xllm silu", shared_xllm_silu)

if linear_ok:
    try:
        _ = bridge.linear(x, w_gu)
        def shared_bridge():
            gu = bridge.linear(x, w_gu)
            act_m.silu_and_mul(act_buf, gu)
            return bridge.linear(act_buf, w_down)
        bench("C: bridge linear + xllm silu", shared_bridge)
    except:
        try:
            b_gu = torch.zeros(2*I_shared,device=dev,dtype=torch.float16)
            b_dn = torch.zeros(H,device=dev,dtype=torch.float16)
            def shared_bridge_b():
                gu = bridge.linear(x, w_gu, b_gu)
                act_m.silu_and_mul(act_buf, gu)
                return bridge.linear(act_buf, w_down, b_dn)
            bench("C: bridge linear(bias0) + xllm silu", shared_bridge_b)
        except Exception as e:
            print(f"  C failed: {e}")

print("\n=== Step breakdown ===")
bench("gate_up F.linear (1,2048)@(256,2048)^T", lambda: torch.nn.functional.linear(x, w_gu))
gu_t = torch.nn.functional.linear(x, w_gu)
bench("silu_and_mul", lambda: act_m.silu_and_mul(act_buf, gu_t))
bench("down F.linear (1,128)@(2048,128)^T", lambda: torch.nn.functional.linear(act_buf, w_down))

print("\n=== matmul vs F.linear ===")
bench("torch.mm(x, w_gu.T)", lambda: torch.mm(x, w_gu.t()))
bench("F.linear(x, w_gu)", lambda: torch.nn.functional.linear(x, w_gu))
PYEOF