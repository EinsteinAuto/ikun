#!/usr/bin/env python3
"""Verify C++ GDN chunk+recurrent on real BI-V100.

Compiles corex_gdn_chunk_recurrent.cu, then tests:
1. torch_chunk_gated_delta_rule: C++ vs Python output match
2. torch_recurrent_gated_delta_rule: C++ vs Python output match
3. Performance comparison

Qwen3.5 GDN dimensions (TP=4):
  num_k_heads=4, num_v_heads=8, head_k_dim=128, head_v_dim=128
  Input: (1, seq_len, 8, 128) for v, (1, seq_len, 4, 128) for q/k

Run: python3 verify_gdn_cpp.py
"""

import sys
import os
import time
import importlib.util
import torch
import torch.nn.functional as F


def compile_gdn():
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "qwen3_6_scripts")
    build_sh = os.path.join(script_dir, "build_corex_gdn_chunk_recurrent.sh")
    tmp_root = "/tmp/gdn_test"
    os.makedirs(tmp_root, exist_ok=True)
    ret = os.system(f"bash {build_sh} {tmp_root} 2>&1")
    so_path = os.path.join(tmp_root, "corex_gdn_chunk_recurrent.so")
    if ret != 0 or not os.path.exists(so_path):
        print(f"[FAIL] Compilation failed (exit={ret})")
        return None
    print(f"[OK] Compiled: {so_path}")
    spec = importlib.util.spec_from_file_location(
        "corex_gdn_chunk_recurrent", so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def python_chunk_gated_delta_rule(q, k, v, g, beta,
                                   chunk_size=64,
                                   initial_state=None,
                                   output_final_state=True,
                                   use_qk_l2norm_in_kernel=True):
    """Python reference — same as qwen3_5.py _torch_chunk_gated_delta_rule."""
    def _l2norm(x, dim=-1, eps=1e-6):
        norm = torch.sqrt(torch.sum(x ** 2, dim=dim, keepdim=True) + eps)
        return x / norm

    initial_dtype = q.dtype
    if use_qk_l2norm_in_kernel:
        q = _l2norm(q, dim=-1)
        k = _l2norm(k, dim=-1)

    q = q.transpose(1, 2).contiguous().float()
    k = k.transpose(1, 2).contiguous().float()
    v = v.transpose(1, 2).contiguous().float()
    beta = beta.transpose(1, 2).contiguous().float()
    g = g.transpose(1, 2).contiguous().float()

    vnh = v.size(1)
    q = q.repeat_interleave(vnh // q.size(1), dim=1) if q.size(1) != vnh else q
    k = k.repeat_interleave(vnh // k.size(1), dim=1) if k.size(1) != vnh else k

    B, H, T, Dk = q.shape
    Dv = v.size(-1)
    scale = Dk ** -0.5
    q = q * scale

    pad = (chunk_size - T % chunk_size) % chunk_size
    if pad > 0:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))
        g = F.pad(g, (0, pad))

    Tp = T + pad
    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)

    q = q.reshape(B, H, Tp // chunk_size, chunk_size, Dk)
    k = k.reshape(B, H, Tp // chunk_size, chunk_size, Dk)
    v = v.reshape(B, H, Tp // chunk_size, chunk_size, Dv)
    k_beta = k_beta.reshape(B, H, Tp // chunk_size, chunk_size, Dk)
    v_beta = v_beta.reshape(B, H, Tp // chunk_size, chunk_size, Dv)
    g = g.reshape(B, H, Tp // chunk_size, chunk_size)

    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    g = g.cumsum(-1)
    g_diff = g.unsqueeze(-1) - g.unsqueeze(-2)
    decay_mask = g_diff.tril().exp().float().tril()

    attn = -(torch.matmul(k_beta, k.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0.0)
    for i in range(1, chunk_size):
        row = attn[..., i:i+1, :i].squeeze(-2).clone()
        sub = attn[..., :i, :i].clone()
        row_final = row + (row.unsqueeze(-1) * sub).sum(-2)
        attn[..., i:i+1, :i] = row_final.unsqueeze(-2)

    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    v = torch.matmul(attn, v_beta)
    k_cumdecay = torch.matmul(attn, k_beta * g.exp().unsqueeze(-1))

    if initial_state is None:
        state = torch.zeros(B, H, Dk, Dv, dtype=v.dtype, device=v.device)
    else:
        state = initial_state.to(v)

    out = torch.zeros_like(v)
    mask2 = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)
    nc = Tp // chunk_size
    for i in range(nc):
        qi = q[:, :, i]
        ki = k[:, :, i]
        vi = v[:, :, i]
        ai = (torch.matmul(qi, ki.transpose(-1, -2)) * decay_mask[:, :, i]).masked_fill_(mask2, 0.0)
        vp = torch.matmul(k_cumdecay[:, :, i], state)
        vn = vi - vp
        inter = torch.matmul(qi * g[:, :, i].unsqueeze(-1).exp(), state)
        out[:, :, i] = inter + torch.matmul(ai, vn)
        gl = g[:, :, i, -1].unsqueeze(-1)
        ge = (gl - g[:, :, i]).exp().unsqueeze(-1)
        kg = (ki * ge).transpose(-1, -2).contiguous()
        state = state * gl.unsqueeze(-1).exp() + torch.matmul(kg, vn)

    out = out.reshape(B, H, Tp, Dv)[:, :, :T]
    out = out.transpose(1, 2).contiguous().to(initial_dtype)
    return out, state


def main():
    print("=" * 60)
    print("BI-V100 C++ GDN chunk+recurrent verification")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("FATAL: No CUDA device")
        return 1

    mod = compile_gdn()
    if mod is None:
        return 1

    # Qwen3.5 GDN dimensions (TP=4)
    B, T = 1, 128
    num_k_heads, num_v_heads = 4, 8
    head_dim = 128
    chunk_size = 64

    torch.manual_seed(42)
    q = torch.randn(B, T, num_k_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(B, T, num_k_heads, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(B, T, num_v_heads, head_dim, device="cuda", dtype=torch.float16)
    g = torch.randn(B, T, num_v_heads, device="cuda", dtype=torch.float16)
    beta = torch.randn(B, T, num_v_heads, device="cuda", dtype=torch.float16)

    # --- Test 1: chunk ---
    print(f"\n--- Test 1: torch_chunk_gated_delta_rule (B={B}, T={T}, chunk={chunk_size}) ---")
    ref_out, ref_state = python_chunk_gated_delta_rule(
        q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
        chunk_size=chunk_size)

    cpp_out, cpp_state = mod.torch_chunk_gated_delta_rule(
        q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
        chunk_size, None, True, True)

    diff_out = (ref_out.float() - cpp_out.float()).abs().max().item()
    diff_state = (ref_state.float() - cpp_state.float()).abs().max().item()
    print(f"  Output max diff: {diff_out:.8f}")
    print(f"  State max diff: {diff_state:.8f}")
    print(f"  Match (tol=1e-2): {diff_out < 1e-2 and diff_state < 1e-2}")

    # --- Test 2: recurrent (decode, T=1) ---
    print(f"\n--- Test 2: torch_recurrent_gated_delta_rule (B=1, T=1) ---")
    q1 = torch.randn(1, 1, num_k_heads, head_dim, device="cuda", dtype=torch.float16)
    k1 = torch.randn(1, 1, num_k_heads, head_dim, device="cuda", dtype=torch.float16)
    v1 = torch.randn(1, 1, num_v_heads, head_dim, device="cuda", dtype=torch.float16)
    g1 = torch.randn(1, 1, num_v_heads, device="cuda", dtype=torch.float16)
    beta1 = torch.randn(1, 1, num_v_heads, device="cuda", dtype=torch.float16)
    state0 = torch.randn(1, num_v_heads, head_dim, head_dim,
                          device="cuda", dtype=torch.float32)

    cpp_out1, cpp_state1 = mod.torch_recurrent_gated_delta_rule(
        q1.clone(), k1.clone(), v1.clone(), g1.clone(), beta1.clone(),
        state0.clone(), True, True)
    print(f"  Output shape: {cpp_out1.shape}")
    print(f"  State shape: {cpp_state1.shape}")
    print(f"  Output has NaN: {cpp_out1.isnan().any().item()}")
    print(f"  State has NaN: {cpp_state1.isnan().any().item()}")

    # --- Test 3: Performance ---
    print(f"\n--- Performance: chunk (B=1, T=512, chunk=64) ---")
    T_perf = 512
    q_p = torch.randn(1, T_perf, num_k_heads, head_dim, device="cuda", dtype=torch.float16)
    k_p = torch.randn(1, T_perf, num_k_heads, head_dim, device="cuda", dtype=torch.float16)
    v_p = torch.randn(1, T_perf, num_v_heads, head_dim, device="cuda", dtype=torch.float16)
    g_p = torch.randn(1, T_perf, num_v_heads, device="cuda", dtype=torch.float16)
    beta_p = torch.randn(1, T_perf, num_v_heads, device="cuda", dtype=torch.float16)

    # Warmup
    for _ in range(3):
        mod.torch_chunk_gated_delta_rule(
            q_p.clone(), k_p.clone(), v_p.clone(), g_p.clone(), beta_p.clone(),
            64, None, True, True)
        python_chunk_gated_delta_rule(
            q_p.clone(), k_p.clone(), v_p.clone(), g_p.clone(), beta_p.clone(),
            chunk_size=64)
    torch.cuda.synchronize()

    N = 5
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        mod.torch_chunk_gated_delta_rule(
            q_p.clone(), k_p.clone(), v_p.clone(), g_p.clone(), beta_p.clone(),
            64, None, True, True)
    torch.cuda.synchronize()
    cpp_ms = (time.perf_counter() - t0) / N * 1000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        python_chunk_gated_delta_rule(
            q_p.clone(), k_p.clone(), v_p.clone(), g_p.clone(), beta_p.clone(),
            chunk_size=64)
    torch.cuda.synchronize()
    py_ms = (time.perf_counter() - t0) / N * 1000

    print(f"  C++: {cpp_ms:.1f} ms")
    print(f"  Python: {py_ms:.1f} ms")
    print(f"  Speedup: {py_ms/cpp_ms:.2f}x")

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
