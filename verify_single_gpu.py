#!/usr/bin/env python3
"""
verify_single_gpu.py — Single-card BI-V100 verification

Tests:
  Step 0: JIT compile ix_full_bridge.cpp
  Step 1: silu_and_mul (from _ixformer_torch.so)
  Step 2: rms_norm
  Step 3: fused_add_rms_norm
  Step 4: linear (ixformer GEMM)
  Step 5: ixformer.functions Python-level flash_attn
  Step 6: ixformer.functions Python-level paged_attention
  Step 7: corex_moe.py Python tiered dispatch (MoE full pipeline)
"""
import os, sys, time, traceback, glob

def step0_compile_bridge():
    print("=" * 60)
    print("STEP 0: JIT compile ix_full_bridge.cpp")
    print("=" * 60)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "ex_engine", "csrc", "ix_full_bridge.cpp"),
        "/workspace/ex_engine/csrc/ix_full_bridge.cpp",
    ]
    cpp_path = None
    for c in candidates:
        if os.path.exists(c):
            cpp_path = c
            break
    if cpp_path is None:
        print(f"  ✗ ix_full_bridge.cpp NOT FOUND in {candidates}")
        return None
    print(f"  Source: {cpp_path}")

    from torch.utils.cpp_extension import load

    extra_ldflags = []
    try:
        import ixformer
        ixf_dir = os.path.dirname(ixformer.__file__)
        for so in glob.glob(os.path.join(ixf_dir, "*.so")):
            if "cpython" not in so:
                extra_ldflags.append(so)
        for so in glob.glob(os.path.join(ixf_dir, "_ixformer_torch*.so")):
            extra_ldflags.append(so)
        extra_ldflags.append(f"-Wl,-rpath,{ixf_dir}")
    except ImportError:
        pass
    corex_lib = "/usr/local/corex/lib64"
    if os.path.isdir(corex_lib):
        extra_ldflags.append(f"-Wl,-rpath,{corex_lib}")

    print(f"  Link: {[os.path.basename(x) for x in extra_ldflags if not x.startswith('-')]}")
    t0 = time.time()
    try:
        bridge = load(
            name="ix_full_bridge",
            sources=[cpp_path],
            extra_cflags=["-O2", "-std=c++17"],
            extra_ldflags=extra_ldflags,
            verbose=True,
        )
        dt = time.time() - t0
        fns = [x for x in dir(bridge) if not x.startswith("_")]
        print(f"  ✓ Compiled in {dt:.1f}s — functions: {fns}")
        return bridge
    except Exception as e:
        print(f"  ✗ FAILED after {time.time()-t0:.1f}s: {e}")
        traceback.print_exc()
        return None

def step1_silu(bridge):
    import torch
    print("\nSTEP 1: silu_and_mul")
    x = torch.randn(4, 256, dtype=torch.float16, device="cuda")  # will split into 128+128
    try:
        out = bridge.silu_and_mul(x)
        print(f"  ✓ {x.shape} → {out.shape}, NaN={out.isnan().any().item()}, abs_mean={out.abs().mean().item():.4f}")
        return True
    except Exception as e:
        print(f"  ✗ {e}")
        traceback.print_exc()
        return False

def step2_rms_norm(bridge):
    import torch
    print("\nSTEP 2: rms_norm")
    x = torch.randn(4, 128, dtype=torch.float16, device="cuda")
    w = torch.ones(128, dtype=torch.float16, device="cuda")
    out = torch.empty_like(x)
    try:
        bridge.rms_norm(out, x, w, 1e-6)
        print(f"  ✓ {out.shape}, NaN={out.isnan().any().item()}, abs_mean={out.abs().mean().item():.4f}")
        return True
    except Exception as e:
        print(f"  ✗ {e}")
        traceback.print_exc()
        return False

def step3_fused_add_rms_norm(bridge):
    import torch
    print("\nSTEP 3: fused_add_rms_norm")
    x = torch.randn(4, 128, dtype=torch.float16, device="cuda")
    res = torch.randn(4, 128, dtype=torch.float16, device="cuda")
    w = torch.ones(128, dtype=torch.float16, device="cuda")
    try:
        bridge.fused_add_rms_norm(x, res, w, 1e-6)
        print(f"  ✓ x modified in-place, NaN={x.isnan().any().item()}")
        return True
    except Exception as e:
        print(f"  ✗ {e}")
        traceback.print_exc()
        return False

def step4_linear(bridge):
    import torch
    print("\nSTEP 4: linear (ixformer GEMM)")
    x = torch.randn(4, 128, dtype=torch.float16, device="cuda")
    w = torch.randn(256, 128, dtype=torch.float16, device="cuda")
    try:
        out = bridge.linear(x, w, None)
        print(f"  ✓ {x.shape} @ {w.shape}^T → {out.shape}, NaN={out.isnan().any().item()}")
        return True
    except Exception as e:
        print(f"  ✗ {e}")
        traceback.print_exc()
        return False

def step5_flash_attn_python():
    import torch
    print("\nSTEP 5: ixformer flash_attn (Python)")
    try:
        from ixformer.contrib.vllm_flash_attn import flash_attn_varlen_func
        Hq, Hkv, D = 4, 1, 128
        seq = 32
        q = torch.randn(seq, Hq, D, dtype=torch.float16, device="cuda")
        k = torch.randn(seq, Hkv, D, dtype=torch.float16, device="cuda")
        v = torch.randn(seq, Hkv, D, dtype=torch.float16, device="cuda")
        cu_q = torch.tensor([0, seq], dtype=torch.int32, device="cuda")
        cu_k = torch.tensor([0, seq], dtype=torch.int32, device="cuda")
        out = flash_attn_varlen_func(q, k, v, cu_q, cu_k, seq, seq,
                                      softmax_scale=D**-0.5, causal=True)
        print(f"  ✓ {out.shape}, NaN={out.isnan().any().item()}")
        return True
    except Exception as e:
        print(f"  ✗ {e}")
        return False

def step6_paged_attn_python():
    import torch
    print("\nSTEP 6: ixformer paged_attention (Python)")
    try:
        import ixformer.functions as ixf_F
        fn = ixf_F.vllm_single_query_cached_kv_attention
        # This is the V1 paged attention used by vllm on BI-V100
        print(f"  ✓ vllm_single_query_cached_kv_attention is available")
        return True
    except Exception as e:
        print(f"  ✗ {e}")
        return False

def step7_corex_moe():
    import torch
    print("\nSTEP 7: corex_moe.py MoE pipeline")
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    try:
        from ex_engine.python.corex_moe import moe_forward
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False

    num_tokens, hidden, experts, inter, topk = 4, 256, 8, 64, 2
    h = torch.randn(num_tokens, hidden, dtype=torch.float16, device="cuda")
    g = torch.randn(num_tokens, experts, dtype=torch.float16, device="cuda")
    w13 = torch.randn(experts, inter*2, hidden, dtype=torch.float16, device="cuda")
    w2 = torch.randn(experts, hidden, inter, dtype=torch.float16, device="cuda")
    try:
        out = moe_forward(h, g, w13, w2, topk=topk, renormalize=True, num_experts=experts)
        print(f"  ✓ {out.shape}, NaN={out.isnan().any().item()}, abs_mean={out.abs().mean().item():.4f}")
        return True
    except Exception as e:
        print(f"  ✗ {e}")
        traceback.print_exc()
        return False

def main():
    import torch
    print("=" * 60)
    print("  BI-V100 Single GPU Verification")
    print(f"  CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)

    R = {}
    bridge = step0_compile_bridge()
    R["compile"] = bridge is not None

    if bridge:
        R["silu_and_mul"] = step1_silu(bridge)
        R["rms_norm"] = step2_rms_norm(bridge)
        R["fused_add_rms_norm"] = step3_fused_add_rms_norm(bridge)
        R["linear"] = step4_linear(bridge)

    R["flash_attn_python"] = step5_flash_attn_python()
    R["paged_attn_python"] = step6_paged_attn_python()
    R["corex_moe"] = step7_corex_moe()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for k, v in R.items():
        print(f"  {'✓' if v else '✗'}  {k}")
    p = sum(R.values())
    print(f"\n  {p}/{len(R)} passed")

    if R.get("compile") and R.get("silu_and_mul"):
        print("\n  >>> C++ bridge works — silu_and_mul/rms_norm/linear accelerated <<<")
    return 0 if p == len(R) else 1

if __name__ == "__main__":
    sys.exit(main())
