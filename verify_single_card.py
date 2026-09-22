#!/usr/bin/env python3
"""
verify_single_card.py — Single BI-V100 验证全部关键组件
在真机上运行: python3 verify_single_card.py

验证清单:
1. MoE topk_softmax CUDA kernel 编译+正确性
2. GDN chunked delta rule 无 NaN
3. GDN decode recurrent 无 NaN
4. topk_softmax 性能对比 (CUDA kernel vs PyTorch)
5. ixformer 可用算子清单
"""
import sys
import os
import time
import torch
import torch.nn.functional as F

print("=" * 70)
print("  BI-V100 Single Card Verification")
print("=" * 70)

# ========== 0. Environment ==========
print("\n[0] Environment")
print(f"  torch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("  ERROR: No CUDA device!")
    sys.exit(1)

# ========== 1. MoE topk_softmax CUDA kernel ==========
print("\n[1] MoE topk_softmax CUDA kernel")

# 1a. Try precompiled
moe_ext = None
try:
    import moe_topk_softmax_v3 as ext
    moe_ext = ext
    print("  ✓ Loaded precompiled moe_topk_softmax_v3")
except ImportError:
    print("  ✗ Precompiled not found, trying JIT compile...")
    cu_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "ex_engine", "csrc", "moe_topk_softmax_v3.cu")
    if os.path.isfile(cu_path):
        try:
            from torch.utils.cpp_extension import load
            ext = load(
                name="moe_topk_softmax_v3",
                sources=[cu_path],
                extra_cuda_cflags=["-O3"],
                verbose=True,
            )
            moe_ext = ext
            print(f"  ✓ JIT compiled from {cu_path}")
        except Exception as e:
            print(f"  ✗ JIT compile FAILED: {e}")
    else:
        print(f"  ✗ Source not found: {cu_path}")

if moe_ext is not None:
    # Correctness test
    gating = torch.randn(16, 64, device='cuda', dtype=torch.float32)
    results = moe_ext.moe_topk_softmax(gating, 8, False)
    w, ids, src = results[0], results[1], results[2]
    print(f"  weights shape: {w.shape}, ids shape: {ids.shape}")
    print(f"  NaN in weights: {w.isnan().any().item()}")
    print(f"  weights sum per row: {w.sum(dim=-1)[:4].tolist()}")
    # Verify topk correctness against PyTorch
    probs_ref = torch.softmax(gating, dim=-1)
    tw_ref, ti_ref = torch.topk(probs_ref, 8, dim=-1)
    weight_diff = (w - tw_ref).abs().max().item()
    print(f"  Max weight diff vs PyTorch: {weight_diff:.6e}")
    if weight_diff < 1e-4:
        print("  ✓ CUDA kernel matches PyTorch reference")
    else:
        print(f"  ✗ MISMATCH (diff={weight_diff})")

    # Performance benchmark
    gating_big = torch.randn(256, 64, device='cuda', dtype=torch.float32)
    # Warmup
    for _ in range(10):
        moe_ext.moe_topk_softmax(gating_big, 8, False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    N = 1000
    for _ in range(N):
        moe_ext.moe_topk_softmax(gating_big, 8, False)
    torch.cuda.synchronize()
    cuda_time = (time.perf_counter() - t0) / N * 1e6

    # PyTorch reference timing
    for _ in range(10):
        p = torch.softmax(gating_big, dim=-1)
        torch.topk(p, 8, dim=-1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        p = torch.softmax(gating_big, dim=-1)
        torch.topk(p, 8, dim=-1)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - t0) / N * 1e6

    print(f"  CUDA kernel: {cuda_time:.1f} μs/call")
    print(f"  PyTorch:     {pytorch_time:.1f} μs/call")
    print(f"  Speedup:     {pytorch_time/cuda_time:.1f}x")
else:
    print("  ✗ No CUDA kernel available — PyTorch fallback only")

# ========== 2. GDN chunked delta rule (NaN test) ==========
print("\n[2] GDN chunked delta rule — NaN test")

# Import our implementation
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3_6_scripts"))

# Minimal test of _torch_chunk_gated_delta_rule
try:
    # Load the function directly from source
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "qwen3_5_mod",
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "qwen3_6_scripts", "qwen3_5.py"))
    # Can't import full module (vllm deps), so test the math directly
    print("  Testing GDN math directly (no vllm imports needed)...")

    B, L, H, K, V = 1, 128, 4, 64, 64  # batch, seq, heads, k_dim, v_dim
    chunk_size = 16

    query = torch.randn(B, L, H, K, device='cuda', dtype=torch.float32)
    key = torch.randn(B, L, H, K, device='cuda', dtype=torch.float32)
    value = torch.randn(B, L, H, V, device='cuda', dtype=torch.float32)
    g = torch.randn(B, L, H, device='cuda', dtype=torch.float32) * 0.5  # gate values
    beta = torch.randn(B, L, H, device='cuda', dtype=torch.float32).sigmoid()

    # L2 norm
    def l2norm(x, dim=-1, eps=1e-6):
        return x / (x.norm(dim=dim, keepdim=True) + eps)

    query = l2norm(query)
    key = l2norm(key)

    # Transpose to (B, H, L, dim)
    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()
    b = beta.transpose(1, 2).contiguous()
    g_t = g.transpose(1, 2).contiguous()

    scale = K ** -0.5
    q = q * scale

    v_beta = v * b.unsqueeze(-1)
    k_beta = k * b.unsqueeze(-1)

    # Reshape to chunks
    q = q.reshape(B, H, -1, chunk_size, K)
    k = k.reshape(B, H, -1, chunk_size, K)
    v = v.reshape(B, H, -1, chunk_size, V)
    k_beta = k_beta.reshape(B, H, -1, chunk_size, K)
    v_beta = v_beta.reshape(B, H, -1, chunk_size, V)
    g_c = g_t.reshape(B, H, -1, chunk_size)

    mask_upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device='cuda'), diagonal=0)

    # THE CRITICAL FIX: cumsum WITHOUT pre-clamp, then difference form
    g_cum = g_c.cumsum(dim=-1)
    g_diff = g_cum.unsqueeze(-1) - g_cum.unsqueeze(-2)
    decay_mask = g_diff.tril().exp().to(torch.float32).tril()

    print(f"  decay_mask NaN: {decay_mask.isnan().any().item()}")
    print(f"  decay_mask inf: {decay_mask.isinf().any().item()}")
    print(f"  decay_mask range: [{decay_mask.min().item():.4f}, {decay_mask.max().item():.4f}]")

    # Full attention computation
    attn = -(torch.matmul(k_beta, k.transpose(-1, -2)) * decay_mask).masked_fill(mask_upper, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device='cuda')

    value_out = torch.matmul(attn, v_beta)
    k_cumdecay = torch.matmul(attn, k_beta * g_cum.exp().unsqueeze(-1))

    print(f"  attn NaN: {attn.isnan().any().item()}")
    print(f"  value_out NaN: {value_out.isnan().any().item()}")
    print(f"  k_cumdecay NaN: {k_cumdecay.isnan().any().item()}")

    # State propagation
    num_chunks = L // chunk_size
    state = torch.zeros(B, H, K, V, device='cuda', dtype=torch.float32)
    core_out = torch.zeros_like(v)
    mask_upper2 = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device='cuda'), diagonal=1)

    any_nan = False
    for i in range(num_chunks):
        q_i = q[:, :, i]
        k_i = k[:, :, i]
        v_i = value_out[:, :, i]
        attn_i = (torch.matmul(q_i, k_i.transpose(-1, -2)) * decay_mask[:, :, i]).masked_fill_(mask_upper2, 0)
        v_prime = torch.matmul(k_cumdecay[:, :, i], state)
        v_new = v_i - v_prime
        attn_inter = torch.matmul(q_i * g_cum[:, :, i].unsqueeze(-1).exp(), state)
        core_out[:, :, i] = attn_inter + torch.matmul(attn_i, v_new)

        # State update — xllm reference
        g_i_last = g_cum[:, :, i, -1].unsqueeze(-1)
        g_exp_term = (g_i_last - g_cum[:, :, i]).exp().unsqueeze(-1)
        k_g_exp = (k_i * g_exp_term).transpose(-1, -2).contiguous()
        state = state * g_i_last.unsqueeze(-1).exp() + torch.matmul(k_g_exp, v_new)

        chunk_nan = core_out[:, :, i].isnan().any().item()
        state_nan = state.isnan().any().item()
        if chunk_nan or state_nan:
            any_nan = True
            print(f"  chunk {i}: output_nan={chunk_nan}, state_nan={state_nan}")

    if not any_nan:
        print(f"  ✓ All {num_chunks} chunks: ZERO NaN")
        nan_frac = core_out.isnan().float().mean().item()
        print(f"  ✓ Total NaN fraction: {nan_frac}")
    else:
        print(f"  ✗ NaN detected in GDN!")

except Exception as e:
    import traceback
    print(f"  ✗ GDN test failed: {e}")
    traceback.print_exc()

# ========== 3. GDN decode (single step) ==========
print("\n[3] GDN decode — single step recurrent")
try:
    B, H, K, V = 2, 4, 64, 64
    q = torch.randn(B, H, K, device='cuda').float() * (K ** -0.5)
    k = torch.randn(B, H, K, device='cuda').float()
    v = torch.randn(B, H, V, device='cuda').float()
    g = torch.randn(B, H, device='cuda').float().clamp(-20, 2).exp()
    beta = torch.randn(B, H, device='cuda').float().sigmoid()
    state = torch.randn(B, H, K, V, device='cuda').float() * 0.01

    # l2norm
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

    state = state * g[:, :, None, None]
    kv_mem = (state * k[:, :, :, None]).sum(-2)
    delta = (v - kv_mem) * beta[:, :, None]
    state = state + k[:, :, :, None] * delta[:, :, None, :]
    out = (state * q[:, :, :, None]).sum(-2)

    print(f"  output NaN: {out.isnan().any().item()}")
    print(f"  state NaN: {state.isnan().any().item()}")
    print(f"  output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
    if not out.isnan().any():
        print("  ✓ Decode step: ZERO NaN")
    else:
        print("  ✗ Decode step has NaN!")
except Exception as e:
    print(f"  ✗ Decode test failed: {e}")

# ========== 4. ixformer available ops ==========
print("\n[4] ixformer available ops")
try:
    import ixformer.functions as ixf_F
    moe_ops = [x for x in dir(ixf_F) if 'moe' in x.lower() or 'topk' in x.lower()]
    attn_ops = [x for x in dir(ixf_F) if 'attn' in x.lower() or 'attention' in x.lower()]
    vllm_ops = [x for x in dir(ixf_F) if 'vllm' in x.lower()]
    print(f"  MoE-related: {moe_ops or '(none)'}")
    print(f"  Attention: {attn_ops[:5]}{'...' if len(attn_ops)>5 else ''}")
    print(f"  vLLM ops ({len(vllm_ops)}): {vllm_ops[:8]}{'...' if len(vllm_ops)>8 else ''}")

    # Test key ops
    x = torch.randn(4, 128, device='cuda', dtype=torch.float16)
    out = torch.empty(4, 64, device='cuda', dtype=torch.float16)
    try:
        ixf_F.silu_and_mul(x, out)
        print("  ✓ silu_and_mul works")
    except Exception as e:
        print(f"  ✗ silu_and_mul: {e}")

    try:
        w = torch.ones(64, device='cuda', dtype=torch.float16)
        inp = torch.randn(4, 64, device='cuda', dtype=torch.float16)
        rms_out = torch.empty_like(inp)
        ixf_F.rms_norm(inp, w, rms_out, None, 1e-6)
        print("  ✓ rms_norm works")
    except Exception as e:
        print(f"  ✗ rms_norm: {e}")

except ImportError:
    print("  ✗ ixformer not available")

# ========== 5. nm -D symbol check ==========
print("\n[5] libixformer.so symbol check")
import subprocess
so_paths = [
    "/usr/local/corex/lib64/python3/dist-packages/ixformer/libixformer.so",
    "/usr/local/corex/lib64/python3/dist-packages/ixformer/_ixformer_torch.cpython-310-x86_64-linux-gnu.so",
    "/usr/local/corex/lib64/python3/dist-packages/ixformer/_C.cpython-310-x86_64-linux-gnu.so",
]
for so_path in so_paths:
    if os.path.isfile(so_path):
        name = os.path.basename(so_path)
        result = subprocess.run(
            f"nm -D {so_path} | grep -i 'topk_softmax\\|moe_compute_token\\|group_gemm\\|moe_expand\\|reduce_sum' | head -5",
            shell=True, capture_output=True, text=True)
        if result.stdout.strip():
            print(f"  {name}: MoE symbols FOUND")
            for line in result.stdout.strip().split('\n')[:3]:
                print(f"    {line.strip()}")
        else:
            total = subprocess.run(f"nm -D {so_path} | wc -l", shell=True, capture_output=True, text=True)
            print(f"  {name}: NO MoE symbols ({total.stdout.strip()} total symbols)")

# ========== Summary ==========
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
print(f"  CUDA kernel topk_softmax: {'✓ READY' if moe_ext else '✗ NOT AVAILABLE'}")
print(f"  GDN chunked (no NaN):     {'✓' if not any_nan else '✗ HAS NaN'}")
print(f"  GPU memory used: {torch.cuda.memory_allocated()/1024**2:.0f} MB")
print(f"  GPU memory free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1024**3:.1f} GB")
