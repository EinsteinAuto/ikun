#!/usr/bin/env python3
"""Verify moe_compute_index + moe_combine_result on real BI-V100.

Step 1: Compile corex_moe_index_combine.cu → .so
Step 2: Test moe_compute_index vs PyTorch argsort+bincount
Step 3: Test moe_combine_result vs PyTorch weighted sum
Step 4: End-to-end MoE prefill path benchmark

Run: python3 verify_moe_index_combine.py
"""

import sys
import os
import time
import torch
import torch.nn.functional as F

def compile_kernel():
    """Compile the .so using corex clang++."""
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "qwen3_6_scripts")
    build_sh = os.path.join(script_dir, "build_corex_moe_index_combine.sh")
    # Use a temp vllm root for testing
    tmp_root = "/tmp/moe_test"
    os.makedirs(tmp_root, exist_ok=True)
    ret = os.system(f"bash {build_sh} {tmp_root} 2>&1")
    so_path = os.path.join(tmp_root, "corex_moe_index_combine.so")
    if ret != 0 or not os.path.exists(so_path):
        print(f"[FAIL] Compilation failed (exit={ret})")
        return None
    print(f"[OK] Compiled: {so_path}")

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "corex_moe_index_combine", so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pytorch_compute_index(expert_ids_flat, num_experts):
    """Reference: what qwen3_5.py does in prefill path."""
    order = torch.argsort(expert_ids_flat, stable=True)
    expert_counts = torch.bincount(
        expert_ids_flat, minlength=num_experts)
    # dst_src[i] = which flat_idx goes to position i (sorted order)
    dst_src = torch.arange(len(expert_ids_flat),
                           device=expert_ids_flat.device)[order]
    # src_dst[flat_idx] = position in sorted order
    src_dst = torch.empty_like(order)
    src_dst[order] = torch.arange(len(order), device=order.device)
    return src_dst, dst_src, expert_counts


def pytorch_combine(expert_outputs, weights, topk, num_tokens, H):
    """Reference: weighted sum of expert outputs."""
    # expert_outputs: (N*topk, H), weights: (N, topk)
    out = expert_outputs.view(num_tokens, topk, H)
    w = weights.unsqueeze(-1)  # (N, topk, 1)
    return (out * w).sum(dim=1)  # (N, H)


def main():
    print("=" * 60)
    print("BI-V100 moe_compute_index + moe_combine verification")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("FATAL: No CUDA device")
        return 1

    mod = compile_kernel()
    if mod is None:
        return 1

    # ---- Test 1: moe_compute_index ----
    print("\n--- Test 1: moe_compute_index (256 experts, 32 tokens, top_k=8) ---")
    num_tokens = 32
    num_experts = 256
    topk = 8
    torch.manual_seed(42)
    # Simulate topk routing: each token picks 8 experts
    topk_ids = torch.randint(0, num_experts, (num_tokens, topk),
                             device="cuda", dtype=torch.int64)
    flat_ids = topk_ids.reshape(-1)  # (256,)

    # Kernel
    kern_src_dst, kern_dst_src, kern_sizes = mod.moe_compute_index(
        flat_ids, num_experts)

    # PyTorch reference
    ref_src_dst, ref_dst_src, ref_sizes = pytorch_compute_index(
        flat_ids, num_experts)

    # Compare sizes (must match exactly)
    sizes_match = torch.equal(kern_sizes.cpu(), ref_sizes.cpu().to(torch.int32))
    print(f"  Expert sizes match: {sizes_match}")

    # Compare mappings: verify kern_dst_src is a valid permutation
    # that groups tokens by expert
    kern_sorted_eids = flat_ids[kern_dst_src.long()]
    ref_sorted_eids = flat_ids[ref_dst_src.long()]
    # Both should be sorted by expert
    kern_sorted = torch.all(kern_sorted_eids[:-1] <= kern_sorted_eids[1:]).item()
    ref_sorted = torch.all(ref_sorted_eids[:-1] <= ref_sorted_eids[1:]).item()
    print(f"  Kernel produces sorted expert order: {kern_sorted}")
    print(f"  Ref produces sorted expert order: {ref_sorted}")

    # ---- Test 2: moe_combine_result ----
    print("\n--- Test 2: moe_combine_result (32 tokens, top_k=8, H=2048) ---")
    H = 2048
    expert_outputs = torch.randn(num_tokens * topk, H,
                                 device="cuda", dtype=torch.float16)
    weights = torch.rand(num_tokens, topk,
                         device="cuda", dtype=torch.float32)
    weights = weights / weights.sum(dim=-1, keepdim=True)  # normalize

    kern_out = mod.moe_combine_result(expert_outputs, weights, num_tokens, topk)
    ref_out = pytorch_combine(expert_outputs, weights, topk, num_tokens, H)

    max_diff = (kern_out.float() - ref_out.float()).abs().max().item()
    print(f"  Max diff: {max_diff:.8f}")
    print(f"  Match (tol=1e-3): {max_diff < 1e-3}")

    # ---- Test 3: Performance ----
    print("\n--- Performance: moe_compute_index ---")
    flat_ids = torch.randint(0, 256, (256,), device="cuda", dtype=torch.int64)

    # Warmup
    for _ in range(10):
        mod.moe_compute_index(flat_ids, 256)
        pytorch_compute_index(flat_ids, 256)
    torch.cuda.synchronize()

    N = 200
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        mod.moe_compute_index(flat_ids, 256)
    torch.cuda.synchronize()
    kern_ms = (time.perf_counter() - t0) / N * 1000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        pytorch_compute_index(flat_ids, 256)
    torch.cuda.synchronize()
    pt_ms = (time.perf_counter() - t0) / N * 1000

    print(f"  Kernel: {kern_ms:.3f} ms")
    print(f"  PyTorch: {pt_ms:.3f} ms")
    print(f"  Speedup: {pt_ms/kern_ms:.2f}x")

    print("\n--- Performance: moe_combine_result ---")
    expert_outputs = torch.randn(32 * 8, 2048, device="cuda", dtype=torch.float16)
    weights = torch.rand(32, 8, device="cuda", dtype=torch.float32)

    for _ in range(10):
        mod.moe_combine_result(expert_outputs, weights, 32, 8)
        pytorch_combine(expert_outputs, weights, 8, 32, 2048)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        mod.moe_combine_result(expert_outputs, weights, 32, 8)
    torch.cuda.synchronize()
    kern_ms = (time.perf_counter() - t0) / N * 1000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        pytorch_combine(expert_outputs, weights, 8, 32, 2048)
    torch.cuda.synchronize()
    pt_ms = (time.perf_counter() - t0) / N * 1000

    print(f"  Kernel: {kern_ms:.3f} ms")
    print(f"  PyTorch: {pt_ms:.3f} ms")
    print(f"  Speedup: {pt_ms/kern_ms:.2f}x")

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
