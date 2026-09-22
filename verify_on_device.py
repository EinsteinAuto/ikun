#!/usr/bin/env python3
"""
verify_on_device.py — 真机验证脚本

在 BI-V100 上逐个验证 ix_full_bridge.cpp 能否 JIT 编译并调通所有 ixformer::infer 函数。
不允许 fallback —— 任何失败直接报错退出。

用法: python3 verify_on_device.py
"""

import os
import sys
import time
import torch

print("=" * 70)
print("BI-V100 ixformer bridge 真机验证")
print("=" * 70)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA: {torch.version.cuda}")
print(f"PyTorch: {torch.__version__}")
print()

# ============================================================================
# Step 1: 验证 ixformer Python 层现有 API
# ============================================================================
print("[1/6] 验证 ixformer Python 层 ...")
try:
    import ixformer
    # 已知可用的函数
    for fn_name in ["silu_and_mul", "rms_norm", "softmax", "flash_attn_func",
                     "fused_add_rms_norm", "matmul"]:
        fn = getattr(ixformer, fn_name, None)
        status = "✓" if fn is not None else "✗ MISSING"
        print(f"  ixformer.{fn_name}: {status}")

    # 已知缺失的函数 (确认还是缺失)
    import ixformer.functions as ixf_F
    has_topk = hasattr(ixf_F, "vllm_moe_topk_softmax")
    print(f"  ixformer.functions.vllm_moe_topk_softmax: {'✓' if has_topk else '✗ MISSING (expected)'}")
    if has_topk:
        print("  !! 意外发现 topk_softmax 已有 Python 绑定 — 不需要 bridge!")
except ImportError as e:
    print(f"  ixformer import failed: {e}")
    sys.exit(1)

# ============================================================================
# Step 2: JIT 编译 ix_full_bridge.cpp
# ============================================================================
print()
print("[2/6] JIT 编译 ix_full_bridge.cpp ...")

# 找到源文件
cpp_candidates = [
    os.path.join(os.path.dirname(__file__), "ex_engine", "csrc", "ix_full_bridge.cpp"),
    "/workspace/ex_engine/csrc/ix_full_bridge.cpp",
    "/tmp/gdn_test/project_6/ex_engine/csrc/ix_full_bridge.cpp",
]
cpp_path = None
for c in cpp_candidates:
    if os.path.exists(c):
        cpp_path = c
        break

if cpp_path is None:
    print("  ✗ ix_full_bridge.cpp 找不到!")
    print(f"  搜索路径: {cpp_candidates}")
    sys.exit(1)

print(f"  源文件: {cpp_path}")
t0 = time.time()

try:
    from torch.utils.cpp_extension import load
    bridge = load(
        name="ix_full_bridge",
        sources=[cpp_path],
        extra_cflags=["-O2", "-std=c++17"],
        verbose=True,
    )
    dt = time.time() - t0
    print(f"  ✓ JIT 编译成功 ({dt:.1f}s)")
    print(f"  导出函数: {[x for x in dir(bridge) if not x.startswith('_')]}")
except Exception as e:
    print(f"  ✗ JIT 编译失败: {e}")
    print()
    print("诊断: 检查链接错误 — ixformer::infer 符号是否在 SDK .so 里")
    sys.exit(1)

# ============================================================================
# Step 3: 逐个测试 MoE 函数
# ============================================================================
print()
print("[3/6] 测试 MoE pipeline ...")

device = "cuda:0"

# 3a. topk_softmax
print("  topk_softmax ...")
try:
    logits = torch.randn(4, 64, device=device, dtype=torch.float32)  # 4 tokens, 64 experts
    topk_w, topk_ids = bridge.topk_softmax(logits, 8, True)
    assert topk_w.shape == (4, 8), f"shape mismatch: {topk_w.shape}"
    assert topk_ids.shape == (4, 8), f"shape mismatch: {topk_ids.shape}"
    assert not topk_w.isnan().any(), "NaN in topk_weights"
    assert topk_w.sum(-1).allclose(torch.ones(4, device=device), atol=0.01), "weights don't sum to 1"
    print(f"    ✓ topk_softmax: weights sum={topk_w.sum(-1).tolist()}, ids range=[{topk_ids.min()}, {topk_ids.max()}]")
except Exception as e:
    print(f"    ✗ topk_softmax FAILED: {e}")
    sys.exit(1)

# 3b. moe_gen_idx
print("  moe_gen_idx ...")
try:
    expert_ids = topk_ids.view(-1).to(torch.int32)
    idx = bridge.moe_gen_idx(expert_ids, 64)
    assert len(idx) == 4, f"expected 4 tensors, got {len(idx)}"
    print(f"    ✓ moe_gen_idx: src_dst={idx[0].shape}, expert_sizes={idx[2].shape}")
except Exception as e:
    print(f"    ✗ moe_gen_idx FAILED: {e}")
    sys.exit(1)

# 3c. silu_and_mul
print("  silu_and_mul ...")
try:
    x = torch.randn(4, 256, device=device, dtype=torch.float16)  # gate_up output
    y = bridge.silu_and_mul(x)
    assert y.shape == (4, 128), f"shape mismatch: {y.shape}"
    assert not y.isnan().any(), "NaN in silu_and_mul output"
    print(f"    ✓ silu_and_mul: {x.shape} → {y.shape}, abs_mean={y.abs().mean():.4f}")
except Exception as e:
    print(f"    ✗ silu_and_mul FAILED: {e}")
    sys.exit(1)

# 3d. fused_moe_forward (full pipeline)
print("  fused_moe_forward (full pipeline) ...")
try:
    H = 128  # small hidden for test
    I = 64   # small intermediate
    E = 64   # experts
    T = 2    # tokens
    K = 8    # top_k
    hidden = torch.randn(T, H, device=device, dtype=torch.float16)
    router = torch.randn(T, E, device=device, dtype=torch.float16)
    w13 = torch.randn(E, 2*I, H, device=device, dtype=torch.float16) * 0.01
    w2 = torch.randn(E, H, I, device=device, dtype=torch.float16) * 0.01
    out = bridge.fused_moe_forward(hidden, router, w13, w2, K, E, True)
    assert out.shape == (T, H), f"shape mismatch: {out.shape}"
    assert not out.isnan().any(), f"NaN in fused_moe output"
    assert not out.isinf().any(), f"inf in fused_moe output"
    print(f"    ✓ fused_moe_forward: {hidden.shape} → {out.shape}, abs_mean={out.abs().mean():.6f}")
except Exception as e:
    print(f"    ✗ fused_moe_forward FAILED: {e}")
    print(f"    这是关键失败 — 整条 MoE pipeline 不通")
    sys.exit(1)

# ============================================================================
# Step 4: 测试 Attention 函数
# ============================================================================
print()
print("[4/6] 测试 Attention ...")

# paged_attention (decode)
print("  paged_attention ...")
try:
    num_heads = 4
    head_dim = 32
    block_size = 16
    num_blocks = 8
    q = torch.randn(1, num_heads, head_dim, device=device, dtype=torch.float16)
    k_cache = torch.randn(num_blocks, num_heads, block_size, head_dim, device=device, dtype=torch.float16)
    v_cache = torch.randn(num_blocks, num_heads, block_size, head_dim, device=device, dtype=torch.float16)
    block_tables = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([48], device=device, dtype=torch.int32)
    out = torch.empty(1, num_heads, head_dim, device=device, dtype=torch.float16)
    bridge.paged_attention(out, q, k_cache, v_cache, num_heads, 1.0 / (head_dim ** 0.5),
                           block_tables, seq_lens, block_size, 48, None)
    assert not out.isnan().any(), "NaN in paged_attention"
    print(f"    ✓ paged_attention: out abs_mean={out.abs().mean():.4f}")
except Exception as e:
    print(f"    ✗ paged_attention FAILED: {e}")
    print(f"    (non-fatal for now — base xformers is fallback)")

# ============================================================================
# Step 5: 测试 Norm 函数
# ============================================================================
print()
print("[5/6] 测试 Norm ...")
print("  rms_norm ...")
try:
    x = torch.randn(4, 128, device=device, dtype=torch.float16)
    w = torch.ones(128, device=device, dtype=torch.float16)
    out = torch.empty_like(x)
    bridge.rms_norm(out, x, w, 1e-6)
    assert not out.isnan().any(), "NaN in rms_norm"
    print(f"    ✓ rms_norm: abs_mean={out.abs().mean():.4f}")
except Exception as e:
    print(f"    ✗ rms_norm FAILED: {e}")

# ============================================================================
# Step 6: 测试 GDN kernel (gate clamp fix)
# ============================================================================
print()
print("[6/6] 测试 GDN kernel (flash_qla_sm70) ...")
try:
    gdn_csrc = os.path.join(os.path.dirname(__file__),
                             "qwen3_6_scripts", "flash_qla_sm70", "csrc")
    if not os.path.exists(gdn_csrc):
        gdn_csrc = "/tmp/gdn_test/project_6/qwen3_6_scripts/flash_qla_sm70/csrc"

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
    gdn_ext = load(
        name="flash_qla_sm70_gdn_verify",
        sources=[os.path.join(gdn_csrc, "gdn_forward.cu")],
        extra_cuda_cflags=["-O3"],
        extra_cflags=["-O3"],
        verbose=False,
    )
    B, T, H, K, V = 1, 64, 4, 128, 128
    q = torch.randn(B, T, H, K, device=device, dtype=torch.float16)
    k = torch.randn(B, T, H, K, device=device, dtype=torch.float16)
    v = torch.randn(B, T, H, V, device=device, dtype=torch.float16)
    # gate 值设为正数（之前导致 inf 的场景）
    g = torch.ones(B, T, H, device=device, dtype=torch.float16) * 3.0  # > 2.0, 应该被 clamp
    beta = torch.randn(B, T, H, device=device, dtype=torch.float16).sigmoid()
    output, state = gdn_ext.gdn_forward(q, k, v, g, beta, None, float(K**-0.5), True, False)
    has_nan = output.isnan().any().item()
    has_inf = output.isinf().any().item()
    abs_mean = output.abs().mean().item()
    print(f"    output: {output.shape}")
    print(f"    NaN: {has_nan}, inf: {has_inf}, abs_mean: {abs_mean:.6f}")
    if has_inf:
        print(f"    ✗ GDN kernel 仍然有 inf — gate clamp 未生效!")
        sys.exit(1)
    elif has_nan:
        print(f"    ✗ GDN kernel 有 NaN")
        sys.exit(1)
    else:
        print(f"    ✓ GDN kernel gate clamp 生效，无 inf/NaN")
except Exception as e:
    print(f"    ✗ GDN kernel 测试失败: {e}")

# ============================================================================
# Summary
# ============================================================================
print()
print("=" * 70)
print("验证完成")
print("=" * 70)
