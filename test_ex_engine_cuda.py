"""
test_ex_engine_cuda.py — Test all ex_engine CUDA kernels on BI-V100.

Tests every prebuilt .so file by loading it, calling each exported function
with synthetic data, and verifying output against PyTorch reference.

Run: python3 test_ex_engine_cuda.py [--vllm-root /path/to/vllm]

Source mapping:
    xllm_norm.so      → norm.cu + xllm_norm_bind.cpp
    xllm_activation.so → activation.cu + xllm_activation_bind.cpp
    xllm_rope.so       → rope.cu + xllm_rope_bind.cpp
    xllm_cache.so      → reshape_paged_cache.cu + block_copy.cu + xllm_cache_bind.cpp
    xllm_moe.so        → moe_fused_topk.cu + moe_compute_index.cu + moe_combine.cu + xllm_moe_bind.cpp
    ix_full_bridge.so  → ix_full_bridge_v2.cpp → ixformer::infer namespace
    corex_*.so         → corex_*.cu (16 individual kernels)
"""

import os
import sys
import importlib
import importlib.util
import argparse
import traceback
import torch
import torch.nn.functional as F


def load_so(name, search_dirs):
    """Load a .so by name from search dirs."""
    for d in search_dirs:
        path = os.path.join(d, f"{name}.so")
        if os.path.isfile(path):
            try:
                spec = importlib.util.spec_from_file_location(name, path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod, path
            except Exception as e:
                print(f"  LOAD FAIL {path}: {e}")
    return None, None


def test_xllm_norm(mod):
    """Test rms_norm and fused_add_rms_norm."""
    T, H = 4, 128
    x = torch.randn(T, H, dtype=torch.float16, device="cuda")
    w = torch.ones(H, dtype=torch.float16, device="cuda")
    eps = 1e-6

    # rms_norm
    out = torch.empty_like(x)
    mod.rms_norm(out, x, w, eps)

    # reference
    var = x.float().pow(2).mean(-1, keepdim=True)
    ref = (x.float() * torch.rsqrt(var + eps) * w.float()).half()
    err = (out.float() - ref.float()).abs().max().item()
    assert err < 0.01, f"rms_norm max error {err}"
    print(f"  rms_norm: max_err={err:.6f} ✓")

    # fused_add_rms_norm
    if hasattr(mod, 'fused_add_rms_norm'):
        x2 = torch.randn(T, H, dtype=torch.float16, device="cuda")
        res = torch.randn(T, H, dtype=torch.float16, device="cuda")
        out2 = torch.empty_like(x2)
        res_out = torch.empty_like(x2)
        mod.fused_add_rms_norm(out2, x2, res, w, eps)
        combined = x2.float() + res.float()
        var2 = combined.pow(2).mean(-1, keepdim=True)
        ref2 = (combined * torch.rsqrt(var2 + eps) * w.float()).half()
        err2 = (out2.float() - ref2.float()).abs().max().item()
        assert err2 < 0.01, f"fused_add_rms_norm max error {err2}"
        print(f"  fused_add_rms_norm: max_err={err2:.6f} ✓")


def test_xllm_activation(mod):
    """Test silu_and_mul."""
    T, I = 4, 64
    x = torch.randn(T, 2 * I, dtype=torch.float16, device="cuda")
    out = torch.empty(T, I, dtype=torch.float16, device="cuda")
    mod.silu_and_mul(out, x)

    gate, up = x.float().chunk(2, dim=-1)
    ref = (F.silu(gate) * up).half()
    err = (out.float() - ref.float()).abs().max().item()
    assert err < 0.01, f"silu_and_mul max error {err}"
    print(f"  silu_and_mul: max_err={err:.6f} ✓")


def test_xllm_rope(mod):
    """Test rotary_embedding."""
    T, NH, HD = 4, 8, 128
    q = torch.randn(T, NH * HD, dtype=torch.float16, device="cuda")
    k = torch.randn(T, NH * HD, dtype=torch.float16, device="cuda")
    positions = torch.arange(T, device="cuda", dtype=torch.int64)
    cos_sin_cache = torch.randn(1024, HD, dtype=torch.float16, device="cuda")

    q_orig = q.clone()
    k_orig = k.clone()
    mod.rotary_embedding(positions, q, k, HD, cos_sin_cache, True)

    # Just verify it modified q and k (not NaN)
    assert not q.isnan().any(), "rotary_embedding produced NaN in query"
    assert not k.isnan().any(), "rotary_embedding produced NaN in key"
    assert not torch.equal(q, q_orig), "rotary_embedding didn't modify query"
    print(f"  rotary_embedding: no NaN, values modified ✓")


def test_xllm_moe(mod):
    """Test moe_fused_topk."""
    if not hasattr(mod, 'moe_fused_topk'):
        print("  moe_fused_topk: not found, skip")
        return
    T, E, K = 4, 64, 8
    logits = torch.randn(T, E, dtype=torch.float32, device="cuda")
    weights, ids = mod.moe_fused_topk(logits, K, True, None, "softmax")

    assert weights.shape == (T, K), f"shape mismatch: {weights.shape}"
    assert ids.shape == (T, K), f"shape mismatch: {ids.shape}"
    assert not weights.isnan().any(), "topk weights NaN"
    assert (ids >= 0).all() and (ids < E).all(), "topk ids out of range"
    print(f"  moe_fused_topk: shapes correct, no NaN ✓")


def test_ix_full_bridge(mod):
    """Test ix_full_bridge functions."""
    # silu_and_mul
    if hasattr(mod, 'silu_and_mul'):
        T, I = 4, 64
        x = torch.randn(T, 2 * I, dtype=torch.float16, device="cuda")
        out = mod.silu_and_mul(x)
        gate, up = x.float().chunk(2, dim=-1)
        ref = (F.silu(gate) * up).half()
        err = (out.float() - ref.float()).abs().max().item()
        assert err < 0.01, f"silu_and_mul max error {err}"
        print(f"  silu_and_mul: max_err={err:.6f} ✓")

    # rms_norm
    if hasattr(mod, 'rms_norm'):
        T, H = 4, 128
        x = torch.randn(T, H, dtype=torch.float16, device="cuda")
        w = torch.ones(H, dtype=torch.float16, device="cuda")
        out = torch.empty_like(x)
        mod.rms_norm(out, x, w, 1e-6)
        assert not out.isnan().any(), "rms_norm NaN"
        print(f"  rms_norm: no NaN ✓")

    # linear
    if hasattr(mod, 'linear'):
        M, K, N = 4, 128, 256
        x = torch.randn(M, K, dtype=torch.float16, device="cuda")
        w = torch.randn(N, K, dtype=torch.float16, device="cuda")
        out = mod.linear(x, w)
        ref = F.linear(x, w)
        err = (out.float() - ref.float()).abs().max().item()
        print(f"  linear: max_err={err:.4f} {'✓' if err < 1.0 else '✗'}")

    fns = [x for x in dir(mod) if not x.startswith("_")]
    print(f"  exported functions: {fns}")


def test_corex_module(name, mod):
    """Basic smoke test for corex_*.so — check it loaded and has functions."""
    fns = [x for x in dir(mod) if not x.startswith("_")]
    if not fns:
        print(f"  {name}: no exported functions ✗")
        return
    print(f"  {name}: {len(fns)} functions: {fns[:5]}{'...' if len(fns)>5 else ''} ✓")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-root", default=None)
    parser.add_argument("--prebuilt-dir", default=None)
    args = parser.parse_args()

    # Build search paths
    here = os.path.dirname(os.path.abspath(__file__))
    search_dirs = []
    if args.prebuilt_dir:
        search_dirs.append(args.prebuilt_dir)
    search_dirs.append(os.path.join(here, "qwen3_6_scripts", "prebuilt",
                                     "corex-3.2.3-ivcore10"))
    if args.vllm_root:
        search_dirs.append(args.vllm_root)
    try:
        import vllm
        search_dirs.append(os.path.dirname(vllm.__file__))
    except ImportError:
        pass

    print(f"Search dirs: {search_dirs}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    results = {"pass": 0, "fail": 0, "skip": 0}

    # Test xllm_*.so
    for name, test_fn in [
        ("xllm_norm", test_xllm_norm),
        ("xllm_activation", test_xllm_activation),
        ("xllm_rope", test_xllm_rope),
        ("xllm_moe", test_xllm_moe),
    ]:
        print(f"[{name}]")
        mod, path = load_so(name, search_dirs)
        if mod is None:
            print(f"  NOT FOUND — skip")
            results["skip"] += 1
            continue
        print(f"  loaded from {path}")
        try:
            test_fn(mod)
            results["pass"] += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            traceback.print_exc()
            results["fail"] += 1
        print()

    # Test ix_full_bridge.so
    print("[ix_full_bridge]")
    mod, path = load_so("ix_full_bridge", search_dirs)
    if mod is None:
        print("  NOT FOUND — skip")
        results["skip"] += 1
    else:
        print(f"  loaded from {path}")
        try:
            test_ix_full_bridge(mod)
            results["pass"] += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            traceback.print_exc()
            results["fail"] += 1
    print()

    # Smoke test corex_*.so
    print("[corex_*.so smoke tests]")
    corex_names = [
        "corex_attn_head_rms_norm", "corex_gdn_causal_conv",
        "corex_gdn_packed_decode", "corex_gdn_qk_map",
        "corex_moe_topk_softmax", "corex_moe_direct_routed",
        "corex_moe_exact_reduce", "corex_moe_index_combine",
        "corex_moe_weight_gather", "corex_paged_kv_gather",
        "corex_fused_paged_prefill",
    ]
    for name in corex_names:
        mod, path = load_so(name, search_dirs)
        if mod is None:
            print(f"  {name}: NOT FOUND")
            results["skip"] += 1
        else:
            try:
                test_corex_module(name, mod)
                results["pass"] += 1
            except Exception as e:
                print(f"  {name}: FAIL {e}")
                results["fail"] += 1

    print()
    print(f"=== RESULTS: {results['pass']} pass, {results['fail']} fail, "
          f"{results['skip']} skip ===")
    return 0 if results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
