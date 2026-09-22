#!/usr/bin/env python3
"""End-to-end MoE forward path verification on single BI-V100.

Simulates Qwen3.5 MoE dimensions:
  hidden_size=2048, num_experts=256, top_k=8, intermediate=128
  w13: (256, 256, 2048), w2: (256, 2048, 128)

Tests the full chain:
  1. topk_softmax kernel (router_logits → topk_weights, topk_ids)
  2. moe_compute_index kernel (topk_ids → sorted order)
  3. Per-expert GEMM (F.linear through sorted experts)
  4. Weighted combine (output)

Compares kernel-accelerated path vs pure PyTorch path.

Run: python3 verify_moe_e2e.py
"""

import sys
import os
import time
import importlib.util
import torch
import torch.nn.functional as F


def load_so(name, so_path):
    if not os.path.exists(so_path):
        return None
    try:
        spec = importlib.util.spec_from_file_location(name, so_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        print(f"[WARN] Failed to load {so_path}: {e}")
        return None


def pure_pytorch_moe(hidden_states, router_logits, w13, w2, top_k):
    """Exact copy of qwen3_5.py _pure_pytorch_experts prefill path."""
    T = hidden_states.shape[0]
    topk_logits, topk_ids = torch.topk(router_logits.float(), top_k, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1).to(hidden_states.dtype)

    out = torch.zeros_like(hidden_states)
    flat_eids = topk_ids.reshape(-1)
    order = torch.argsort(flat_eids, stable=True)
    sorted_tok_ids = torch.arange(
        T, device=topk_ids.device).repeat_interleave(top_k)[order]
    sorted_weights = topk_weights.reshape(-1)[order]
    expert_counts = torch.bincount(flat_eids, minlength=w13.shape[0]).tolist()

    start = 0
    for eid, count in enumerate(expert_counts):
        end = start + count
        if count == 0:
            start = end
            continue
        tok_ids = sorted_tok_ids[start:end]
        tokens = hidden_states[tok_ids]
        gate_up = F.linear(tokens, w13[eid])
        gate, up = gate_up.chunk(2, dim=-1)
        act = F.silu(gate) * up
        expert_out = F.linear(act, w2[eid])
        weights = sorted_weights[start:end].unsqueeze(-1)
        out.index_add_(0, tok_ids, (expert_out * weights).to(out.dtype))
        start = end
    return out


def kernel_moe(hidden_states, router_logits, w13, w2, top_k,
               topk_mod, index_mod):
    """Kernel-accelerated MoE path."""
    T = hidden_states.shape[0]

    # Step 1: topk_softmax kernel
    topk_weights, topk_ids = topk_mod.moe_topk_softmax(
        router_logits.float(), top_k, True)
    topk_ids = topk_ids.to(torch.int64)
    topk_weights = topk_weights.to(hidden_states.dtype)

    # Step 2: moe_compute_index kernel
    flat_eids = topk_ids.reshape(-1)
    src_dst, dst_src, expert_sizes = index_mod.moe_compute_index(
        flat_eids, w13.shape[0])
    sorted_tok_ids = torch.arange(
        T, device=topk_ids.device).repeat_interleave(top_k)[dst_src.long()]
    sorted_weights = topk_weights.reshape(-1)[dst_src.long()]
    expert_counts = expert_sizes.tolist()

    # Step 3: Per-expert GEMM (same as PyTorch — this is the bottleneck)
    out = torch.zeros_like(hidden_states)
    start = 0
    for eid, count in enumerate(expert_counts):
        end = start + count
        if count == 0:
            start = end
            continue
        tok_ids = sorted_tok_ids[start:end]
        tokens = hidden_states[tok_ids]
        gate_up = F.linear(tokens, w13[eid])
        gate, up = gate_up.chunk(2, dim=-1)
        act = F.silu(gate) * up
        expert_out = F.linear(act, w2[eid])
        weights = sorted_weights[start:end].unsqueeze(-1)
        out.index_add_(0, tok_ids, (expert_out * weights).to(out.dtype))
        start = end
    return out


def main():
    print("=" * 60)
    print("BI-V100 MoE end-to-end verification")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("FATAL: No CUDA device")
        return 1

    # Load kernels
    prebuilt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10")
    topk_mod = load_so("corex_moe_topk_softmax",
                       os.path.join(prebuilt, "corex_moe_topk_softmax.so"))
    index_mod = load_so("corex_moe_index_combine",
                        "/tmp/moe_test/corex_moe_index_combine.so")

    if topk_mod is None:
        print("[FAIL] Cannot load topk_softmax .so")
        return 1
    if index_mod is None:
        print("[FAIL] Cannot load index_combine .so — run verify_moe_index_combine.py first")
        return 1

    print(f"[OK] Both kernel modules loaded")

    # Qwen3.5 MoE dimensions (TP=4 sharded)
    hidden_size = 2048
    num_experts = 256
    top_k = 8
    inter_per_partition = 128  # moe_intermediate_size / tp_size

    torch.manual_seed(42)

    # --- Test 1: Single token (decode) ---
    print("\n--- Test 1: 1 token (decode path) ---")
    h = torch.randn(1, hidden_size, device="cuda", dtype=torch.float16)
    router = torch.randn(1, num_experts, device="cuda", dtype=torch.float16)
    w13 = torch.randn(num_experts, 2 * inter_per_partition, hidden_size,
                       device="cuda", dtype=torch.float16) * 0.01
    w2 = torch.randn(num_experts, hidden_size, inter_per_partition,
                      device="cuda", dtype=torch.float16) * 0.01

    ref_out = pure_pytorch_moe(h, router, w13, w2, top_k)
    kern_out = kernel_moe(h, router, w13, w2, top_k, topk_mod, index_mod)

    diff = (ref_out.float() - kern_out.float()).abs().max().item()
    print(f"  Max diff: {diff:.8f}")
    print(f"  Match: {diff < 0.01}")

    # --- Test 2: 32 tokens (prefill) ---
    print("\n--- Test 2: 32 tokens (prefill path) ---")
    h = torch.randn(32, hidden_size, device="cuda", dtype=torch.float16)
    router = torch.randn(32, num_experts, device="cuda", dtype=torch.float16)

    ref_out = pure_pytorch_moe(h, router, w13, w2, top_k)
    kern_out = kernel_moe(h, router, w13, w2, top_k, topk_mod, index_mod)

    diff = (ref_out.float() - kern_out.float()).abs().max().item()
    rel_diff = diff / (ref_out.float().abs().max().item() + 1e-8)
    print(f"  Max abs diff: {diff:.8f}")
    print(f"  Relative diff: {rel_diff:.8f}")
    print(f"  Match: {rel_diff < 0.01}")

    # --- Test 3: Performance comparison ---
    print("\n--- Performance: 32 tokens prefill ---")
    h = torch.randn(32, hidden_size, device="cuda", dtype=torch.float16)
    router = torch.randn(32, num_experts, device="cuda", dtype=torch.float16)

    # Warmup
    for _ in range(5):
        pure_pytorch_moe(h, router, w13, w2, top_k)
        kernel_moe(h, router, w13, w2, top_k, topk_mod, index_mod)
    torch.cuda.synchronize()

    N = 20
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        pure_pytorch_moe(h, router, w13, w2, top_k)
    torch.cuda.synchronize()
    pt_ms = (time.perf_counter() - t0) / N * 1000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        kernel_moe(h, router, w13, w2, top_k, topk_mod, index_mod)
    torch.cuda.synchronize()
    kern_ms = (time.perf_counter() - t0) / N * 1000

    print(f"  PyTorch: {pt_ms:.1f} ms")
    print(f"  Kernel:  {kern_ms:.1f} ms")
    print(f"  Speedup: {pt_ms/kern_ms:.2f}x")

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
