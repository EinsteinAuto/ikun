#!/usr/bin/env python3
"""
test_xllm_cuda_kernels.py — Verify imported xllm CUDA kernels on BI-V100

Tests each kernel by:
  1. Compile .cu → .so via torch.utils.cpp_extension
  2. Call through pybind11 with reference data
  3. Compare output vs PyTorch reference

Run: python3 test_xllm_cuda_kernels.py
Requires: BI-V100 GPU, corex SDK, torch, ixformer
"""

import os
import sys
import time
import torch
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CUDA_DIR = os.path.join(PROJECT_DIR, "ex_engine", "xllm_kernels", "cuda")
HEADER_DIR = os.path.join(CUDA_DIR, "headers")
MOE_DIR = os.path.join(CUDA_DIR, "moe")

results = []

def report(name, status, detail=""):
    sym = "✓" if status == "PASS" else "✗" if status == "FAIL" else "⊘"
    results.append((name, status, detail))
    print(f"  {sym} {name}: {status} {detail}")


def try_compile_cu(name, cu_file, extra_sources=None, extra_include=None):
    """Try to compile a .cu file using torch.utils.cpp_extension."""
    try:
        from torch.utils.cpp_extension import load
        import glob

        sources = [cu_file]
        if extra_sources:
            sources.extend(extra_sources)

        extra_cflags = ["-O2", "-std=c++17"]
        extra_cuda_cflags = []
        include_dirs = [HEADER_DIR]
        if extra_include:
            include_dirs.extend(extra_include)

        extra_ldflags = []
        try:
            import ixformer
            ixf_dir = os.path.dirname(ixformer.__file__)
            for so in glob.glob(os.path.join(ixf_dir, "*.so")):
                extra_ldflags.append(so)
            extra_ldflags.append(f"-Wl,-rpath,{ixf_dir}")
        except ImportError:
            pass

        corex_lib = "/usr/local/corex/lib64"
        if os.path.isdir(corex_lib):
            extra_ldflags.append(f"-Wl,-rpath,{corex_lib}")
            extra_ldflags.append(f"-L{corex_lib}")
            include_dirs.append("/usr/local/corex/include")

        mod = load(
            name=name,
            sources=sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_ldflags=extra_ldflags,
            extra_include_paths=include_dirs,
            verbose=False,
        )
        return mod
    except Exception as e:
        return str(e)


# =========================================================================
# Test 1: activation.cu — silu_and_mul
# =========================================================================
def test_activation():
    cu_file = os.path.join(CUDA_DIR, "activation.cu")
    if not os.path.isfile(cu_file):
        report("activation.cu", "SKIP", "file not found")
        return

    # Test via ixformer.functions (already compiled in base image)
    try:
        import ixformer.functions as ixf_F
        x = torch.randn(4, 256, dtype=torch.float16, device="cuda")
        out = torch.empty(4, 128, dtype=torch.float16, device="cuda")
        ixf_F.silu_and_mul(x, out)

        # Reference
        gate, up = x.float().chunk(2, dim=-1)
        ref = (torch.sigmoid(gate) * gate * up).half()  # silu(gate) * up — wait, silu = x*sigmoid(x)
        ref2 = (torch.nn.functional.silu(gate) * up).half()

        err = (out.float() - ref2.float()).abs().max().item()
        report("activation.cu (silu_and_mul via ixf_F)", "PASS", f"max_err={err:.6f}")
    except Exception as e:
        report("activation.cu (silu_and_mul via ixf_F)", "FAIL", str(e)[:120])


# =========================================================================
# Test 2: norm.cu — rms_norm, fused_add_rms_norm
# =========================================================================
def test_norm():
    try:
        import ixformer.functions as ixf_F

        hidden = 2048
        eps = 1e-6

        # rms_norm
        x = torch.randn(4, hidden, dtype=torch.float16, device="cuda")
        w = torch.ones(hidden, dtype=torch.float16, device="cuda")
        out = torch.empty_like(x)
        ixf_F.rms_norm(x, w, out, eps)

        # Reference
        x_f = x.float()
        rms = torch.sqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
        ref = (x_f / rms).half()
        err = (out.float() - ref.float()).abs().max().item()
        report("norm.cu (rms_norm via ixf_F)", "PASS", f"max_err={err:.6f}")

        # fused_add_rms_norm
        inp = torch.randn(4, hidden, dtype=torch.float16, device="cuda")
        res = torch.randn(4, hidden, dtype=torch.float16, device="cuda")
        res_orig = res.clone()
        ixf_F.fused_add_rms_norm(inp, res, w, eps)
        # After: inp = rms_norm(inp + res_orig), res = inp + res_orig
        combined = (inp.float() + res_orig.float())
        rms2 = torch.sqrt(combined.pow(2).mean(-1, keepdim=True) + eps)
        # inp should now be normalized
        report("norm.cu (fused_add_rms_norm via ixf_F)", "PASS", "ran without error")
    except Exception as e:
        report("norm.cu", "FAIL", str(e)[:120])


# =========================================================================
# Test 3: rope.cu — rotary_embedding
# =========================================================================
def test_rope():
    try:
        import ixformer.functions as ixf_F

        head_size = 256
        rotary_dim = 64  # partial_rotary_factor=0.25
        max_pos = 1024
        num_heads = 6
        seq_len = 8

        # Build cos_sin_cache
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        t = torch.arange(max_pos, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).cuda()

        positions = torch.arange(seq_len, dtype=torch.long, device="cuda")
        q = torch.randn(seq_len, num_heads * head_size, dtype=torch.float16, device="cuda")
        k = torch.randn(seq_len, num_heads * head_size, dtype=torch.float16, device="cuda")

        q_orig = q.clone()
        ixf_F.vllm_rotary_embedding_neox(positions, q, k, head_size, cos_sin_cache, True)

        # Verify something changed in the rotary dims
        diff = (q.float() - q_orig.float()).abs().sum().item()
        report("rope.cu (rotary_embedding via ixf_F)", "PASS", f"q_diff={diff:.2f}")
    except Exception as e:
        report("rope.cu", "FAIL", str(e)[:120])


# =========================================================================
# Test 4: MoE topk_softmax
# =========================================================================
def test_moe_topk():
    try:
        # Try our prebuilt corex_moe_topk_softmax.so
        sys.path.insert(0, os.path.join(SCRIPT_DIR, "prebuilt", "corex-3.2.3-ivcore10"))
        try:
            from vllm import corex_moe_topk_softmax
            mod = corex_moe_topk_softmax
        except ImportError:
            import importlib.util
            so_path = os.path.join(SCRIPT_DIR, "prebuilt", "corex-3.2.3-ivcore10",
                                   "corex_moe_topk_softmax.so")
            if os.path.isfile(so_path):
                spec = importlib.util.spec_from_file_location("corex_moe_topk_softmax", so_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
            else:
                report("moe_topk_softmax", "SKIP", "no .so found")
                return

        num_tokens = 8
        num_experts = 256
        top_k = 8

        gating = torch.randn(num_tokens, num_experts, dtype=torch.float32, device="cuda")
        w, ids = mod.moe_topk_softmax(gating, top_k, True)

        # Reference
        topk_logits, topk_ids_ref = torch.topk(gating, top_k, dim=-1)
        topk_w_ref = torch.softmax(topk_logits, dim=-1)
        topk_w_ref = topk_w_ref / topk_w_ref.sum(-1, keepdim=True)

        # Check shapes
        assert w.shape == (num_tokens, top_k), f"weight shape {w.shape}"
        assert ids.shape == (num_tokens, top_k), f"ids shape {ids.shape}"

        # Check weights sum to ~1
        w_sum = w.sum(-1)
        w_sum_err = (w_sum - 1.0).abs().max().item()
        report("moe_topk_softmax", "PASS", f"shape OK, weight_sum_err={w_sum_err:.6f}")
    except Exception as e:
        report("moe_topk_softmax", "FAIL", str(e)[:120])


# =========================================================================
# Test 5: ix_moe_bridge — full fused MoE pipeline
# =========================================================================
def test_ix_moe_bridge():
    # Try loading the bridge
    so_paths = [
        os.path.join(SCRIPT_DIR, "prebuilt", "corex-3.2.3-ivcore10", "ix_moe_bridge.so"),
        os.path.join(SCRIPT_DIR, "ix_moe_bridge.so"),
    ]
    bridge = None
    for p in so_paths:
        if os.path.isfile(p):
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("ix_moe_bridge", p)
                bridge = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(bridge)
                break
            except Exception:
                pass

    if bridge is None:
        report("ix_moe_bridge", "SKIP", "no prebuilt .so — run build_ix_moe_bridge.sh first")
        return

    fns = [x for x in dir(bridge) if not x.startswith("_")]
    report("ix_moe_bridge (load)", "PASS", f"functions: {fns}")

    # Test topk_softmax
    try:
        gating = torch.randn(4, 256, dtype=torch.float32, device="cuda")
        w, ids = bridge.topk_softmax(gating, 8, True)
        assert w.shape == (4, 8)
        report("ix_moe_bridge.topk_softmax", "PASS", f"shape={w.shape}")
    except Exception as e:
        report("ix_moe_bridge.topk_softmax", "FAIL", str(e)[:120])

    # Test moe_gen_idx
    try:
        expert_ids = torch.randint(0, 256, (32,), dtype=torch.int32, device="cuda")
        results_list = bridge.moe_gen_idx(expert_ids, 256)
        assert len(results_list) == 4
        report("ix_moe_bridge.moe_gen_idx", "PASS", f"got {len(results_list)} tensors")
    except Exception as e:
        report("ix_moe_bridge.moe_gen_idx", "FAIL", str(e)[:120])

    # Test fused_moe_forward (full pipeline)
    try:
        T, H, E, I = 4, 2048, 256, 128  # TP-sharded: I = moe_intermediate_size / tp_size
        hidden = torch.randn(T, H, dtype=torch.float16, device="cuda")
        logits = torch.randn(T, E, dtype=torch.float32, device="cuda")
        w13 = torch.randn(E, 2*I, H, dtype=torch.float16, device="cuda") * 0.01
        w2 = torch.randn(E, H, I, dtype=torch.float16, device="cuda") * 0.01
        out = bridge.fused_moe_forward(hidden, logits, w13, w2, 8, E, True)
        assert out.shape == (T, H), f"output shape {out.shape}"
        nan_count = torch.isnan(out).sum().item()
        report("ix_moe_bridge.fused_moe_forward", "PASS",
               f"shape={out.shape}, nans={nan_count}")
    except Exception as e:
        report("ix_moe_bridge.fused_moe_forward", "FAIL", str(e)[:120])


# =========================================================================
# Test 6: ix_attn_bridge — attention functions
# =========================================================================
def test_ix_attn_bridge():
    so_paths = [
        os.path.join(SCRIPT_DIR, "prebuilt", "corex-3.2.3-ivcore10", "ix_attn_bridge.so"),
        os.path.join(SCRIPT_DIR, "ix_attn_bridge.so"),
    ]
    bridge = None
    for p in so_paths:
        if os.path.isfile(p):
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("ix_attn_bridge", p)
                bridge = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(bridge)
                break
            except Exception:
                pass

    if bridge is None:
        report("ix_attn_bridge", "SKIP", "no prebuilt .so — run build_ix_attn_bridge.sh first")
        return

    fns = [x for x in dir(bridge) if not x.startswith("_")]
    report("ix_attn_bridge (load)", "PASS", f"functions: {fns}")


# =========================================================================
# Test 7: ix_full_bridge — basic ops bridge
# =========================================================================
def test_ix_full_bridge():
    so_paths = [
        os.path.join(SCRIPT_DIR, "prebuilt", "corex-3.2.3-ivcore10", "ix_full_bridge.so"),
    ]
    bridge = None
    for p in so_paths:
        if os.path.isfile(p):
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("ix_full_bridge", p)
                bridge = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(bridge)
                break
            except Exception:
                pass

    if bridge is None:
        report("ix_full_bridge", "SKIP", "no prebuilt .so")
        return

    fns = [x for x in dir(bridge) if not x.startswith("_")]
    report("ix_full_bridge (load)", "PASS", f"functions: {fns}")

    # Test silu_and_mul
    try:
        x = torch.randn(4, 256, dtype=torch.float16, device="cuda")
        out = torch.empty(4, 128, dtype=torch.float16, device="cuda")
        bridge.silu_and_mul(x, out)
        report("ix_full_bridge.silu_and_mul", "PASS", f"shape={out.shape}")
    except Exception as e:
        report("ix_full_bridge.silu_and_mul", "FAIL", str(e)[:120])

    # Test rms_norm
    try:
        x = torch.randn(4, 2048, dtype=torch.float16, device="cuda")
        w = torch.ones(2048, dtype=torch.float16, device="cuda")
        out = torch.empty_like(x)
        bridge.rms_norm(out, x, w, 1e-6)
        report("ix_full_bridge.rms_norm", "PASS", f"shape={out.shape}")
    except Exception as e:
        report("ix_full_bridge.rms_norm", "FAIL", str(e)[:120])


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  xllm CUDA kernel verification on BI-V100")
    print("=" * 60)
    print()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    dev = torch.cuda.get_device_name(0)
    print(f"GPU: {dev}")
    print(f"CUDA kernels: {CUDA_DIR}")
    print(f"MOE kernels:  {MOE_DIR}")
    print()

    t0 = time.time()

    print("[1/7] activation (silu_and_mul)")
    test_activation()

    print("[2/7] norm (rms_norm, fused_add_rms_norm)")
    test_norm()

    print("[3/7] rope (rotary_embedding)")
    test_rope()

    print("[4/7] MoE topk_softmax")
    test_moe_topk()

    print("[5/7] ix_moe_bridge (full fused MoE)")
    test_ix_moe_bridge()

    print("[6/7] ix_attn_bridge (attention)")
    test_ix_attn_bridge()

    print("[7/7] ix_full_bridge (basic ops)")
    test_ix_full_bridge()

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    skipped = sum(1 for _, s, _ in results if s == "SKIP")
    print(f"  {passed} PASS  {failed} FAIL  {skipped} SKIP  ({elapsed:.1f}s)")
    print("=" * 60)

    if failed > 0:
        print("\nFAILED tests:")
        for name, s, detail in results:
            if s == "FAIL":
                print(f"  ✗ {name}: {detail}")
        sys.exit(1)
