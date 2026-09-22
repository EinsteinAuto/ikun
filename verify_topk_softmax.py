#!/usr/bin/env python3
"""Verify corex_moe_topk_softmax.so on real BI-V100 hardware.

Run on the machine with BI-V100 GPU:
  python3 verify_topk_softmax.py

Tests:
1. Load prebuilt .so
2. Compare kernel output vs PyTorch reference (same input)
3. Print warp size from device
"""

import sys
import os
import importlib
import torch

def pytorch_topk_softmax(router_logits_f32, topk, renormalize=True):
    """Reference implementation — this is what the PyTorch fallback does."""
    topk_logits, topk_ids = torch.topk(router_logits_f32, topk, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


def main():
    print("=" * 60)
    print("BI-V100 topk_softmax kernel verification")
    print("=" * 60)

    # Step 0: Device info
    if not torch.cuda.is_available():
        print("FATAL: No CUDA device")
        return 1
    props = torch.cuda.get_device_properties(0)
    print(f"Device: {props.name}")
    print(f"SM count: {props.multi_processor_count}")
    print(f"Warp size: {getattr(props, 'warp_size', 'N/A')}")
    print()

    # Step 1: Try to load the prebuilt .so
    so_candidates = [
        # In vllm install path (where patch_ops.sh copies it)
        None,  # will try importlib
        # In prebuilt dir
        os.path.join(os.path.dirname(__file__),
                     "qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10/"
                     "corex_moe_topk_softmax.so"),
    ]

    kernel_mod = None

    # Try 1: import from vllm namespace (how qwen3_5.py loads it)
    try:
        from vllm import corex_moe_topk_softmax as kernel_mod
        print(f"[OK] Loaded from vllm namespace")
    except Exception as e:
        print(f"[--] vllm import failed: {e}")

    # Try 2: direct load from prebuilt
    if kernel_mod is None:
        for so_path in so_candidates:
            if so_path is None:
                continue
            if not os.path.exists(so_path):
                print(f"[--] Not found: {so_path}")
                continue
            try:
                spec = importlib.util.spec_from_file_location(
                    "corex_moe_topk_softmax", so_path)
                kernel_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(kernel_mod)
                print(f"[OK] Loaded from {so_path}")
                break
            except Exception as e:
                print(f"[FAIL] Load {so_path}: {e}")

    # Try 3: torch.ops.load_library on the .so
    if kernel_mod is None:
        so_path = os.path.join(os.path.dirname(__file__),
                               "qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10/"
                               "corex_moe_topk_softmax.so")
        if os.path.exists(so_path):
            try:
                torch.ops.load_library(so_path)
                print(f"[OK] torch.ops.load_library succeeded")
            except Exception as e:
                print(f"[FAIL] torch.ops.load_library: {e}")

    if kernel_mod is None:
        print("\nCannot load kernel .so — trying to compile from source...")
        # Try compile from source
        try:
            from torch.utils.cpp_extension import load
            script_dir = os.path.join(os.path.dirname(__file__),
                                      "qwen3_6_scripts")
            kernel_mod = load(
                name="corex_moe_topk_softmax",
                sources=[os.path.join(script_dir, "corex_moe_topk_softmax.cu")],
                extra_include_paths=[script_dir],
                verbose=True,
            )
            print(f"[OK] Compiled from source")
        except Exception as e:
            print(f"[FAIL] Compile from source: {e}")
            print("\nCANNOT VERIFY KERNEL — no .so available")
            return 1

    # Step 2: Test with Qwen3.5 dimensions (256 experts, top_k=8)
    print("\n--- Test: 256 experts, top_k=8, 1 token (decode) ---")
    num_tokens = 1
    num_experts = 256
    topk = 8
    torch.manual_seed(42)
    router_logits = torch.randn(num_tokens, num_experts,
                                device="cuda", dtype=torch.float32)

    # PyTorch reference
    ref_weights, ref_ids = pytorch_topk_softmax(router_logits.clone(), topk)

    # Kernel
    try:
        kern_weights, kern_ids = kernel_mod.moe_topk_softmax(
            router_logits.clone(), topk, True)
    except Exception as e:
        print(f"[FAIL] Kernel call failed: {e}")
        return 1

    kern_ids_i64 = kern_ids.to(torch.int64)

    # Compare: same top-k expert IDs (order may differ)?
    ref_set = set(ref_ids[0].cpu().tolist())
    kern_set = set(kern_ids_i64[0].cpu().tolist())
    ids_match = ref_set == kern_set
    print(f"  Ref  expert IDs: {sorted(ref_set)}")
    print(f"  Kern expert IDs: {sorted(kern_set)}")
    print(f"  IDs match: {ids_match}")

    # Compare weights for matching experts
    if ids_match:
        # Reorder kernel weights to match ref order
        ref_order = ref_ids[0].cpu().tolist()
        kern_id_list = kern_ids_i64[0].cpu().tolist()
        kern_w_list = kern_weights[0].cpu().tolist()
        kern_map = dict(zip(kern_id_list, kern_w_list))
        kern_reordered = torch.tensor([kern_map[eid] for eid in ref_order])
        ref_w = ref_weights[0].cpu()
        max_diff = (kern_reordered - ref_w).abs().max().item()
        print(f"  Max weight diff: {max_diff:.8f}")
        print(f"  Weights match (tol=1e-5): {max_diff < 1e-5}")
    else:
        print(f"  [GARBLED] Expert IDs don't match — kernel output is wrong!")
        print(f"  Missing from kernel: {ref_set - kern_set}")
        print(f"  Extra in kernel: {kern_set - ref_set}")
        print(f"  Kernel weights: {kern_weights[0].cpu().tolist()}")
        print(f"  Ref weights: {ref_weights[0].cpu().tolist()}")

    # Step 3: Test with multiple tokens (prefill)
    print("\n--- Test: 256 experts, top_k=8, 32 tokens (prefill) ---")
    num_tokens = 32
    router_logits = torch.randn(num_tokens, num_experts,
                                device="cuda", dtype=torch.float32)
    ref_weights, ref_ids = pytorch_topk_softmax(router_logits.clone(), topk)
    try:
        kern_weights, kern_ids = kernel_mod.moe_topk_softmax(
            router_logits.clone(), topk, True)
    except Exception as e:
        print(f"[FAIL] Kernel call failed: {e}")
        return 1

    kern_ids_i64 = kern_ids.to(torch.int64)
    mismatch_count = 0
    max_weight_diff = 0.0
    for t in range(num_tokens):
        ref_set = set(ref_ids[t].cpu().tolist())
        kern_set = set(kern_ids_i64[t].cpu().tolist())
        if ref_set != kern_set:
            mismatch_count += 1
        else:
            ref_order = ref_ids[t].cpu().tolist()
            kern_id_list = kern_ids_i64[t].cpu().tolist()
            kern_w_list = kern_weights[t].cpu().tolist()
            kern_map = dict(zip(kern_id_list, kern_w_list))
            kern_reordered = torch.tensor([kern_map[eid] for eid in ref_order])
            diff = (kern_reordered - ref_weights[t].cpu()).abs().max().item()
            max_weight_diff = max(max_weight_diff, diff)

    print(f"  ID mismatches: {mismatch_count}/{num_tokens}")
    print(f"  Max weight diff (matching rows): {max_weight_diff:.8f}")
    if mismatch_count == 0 and max_weight_diff < 1e-5:
        print(f"  [PASS] Kernel output matches PyTorch reference")
    elif mismatch_count == 0 and max_weight_diff < 1e-3:
        print(f"  [WARN] Small numerical diff but IDs correct")
    else:
        print(f"  [FAIL] Kernel output does NOT match")

    # Step 4: Performance comparison
    print("\n--- Performance: 256 experts, top_k=8, 1 token ---")
    router_logits = torch.randn(1, 256, device="cuda", dtype=torch.float32)

    # Warmup
    for _ in range(10):
        pytorch_topk_softmax(router_logits, topk)
        kernel_mod.moe_topk_softmax(router_logits.clone(), topk, True)
    torch.cuda.synchronize()

    import time
    N = 100

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        pytorch_topk_softmax(router_logits, topk)
    torch.cuda.synchronize()
    pt_time = (time.perf_counter() - t0) / N * 1000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        kernel_mod.moe_topk_softmax(router_logits.clone(), topk, True)
    torch.cuda.synchronize()
    kern_time = (time.perf_counter() - t0) / N * 1000

    print(f"  PyTorch: {pt_time:.3f} ms/call")
    print(f"  Kernel:  {kern_time:.3f} ms/call")
    print(f"  Speedup: {pt_time/kern_time:.2f}x")

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
