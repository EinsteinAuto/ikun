# dlopen Dispatch Chain — BI-V100 Runtime .so Loading

## Source: comp 168 docker log (2d5232c5)

Two runs in `dockerrizhi.txt`:
- **07-23**: Competitor 168's Docker (working, full fused kernels)
- **08-07**: Our Docker (broken MoE, NaN in GDN)

## Competitor 168's Working AST Call Chain

```
HTTP Request → api_server.py → serving_chat.py
  → vLLM AsyncLLMEngine
    → model_runner.py:1074 (base image version, NOT our 1119)
      → qwen3_5.py (base image version with corex imports)
        │
        ├── Attention layers (32 of 36):
        │   → selector.py:115 → Using XFormers backend
        │   → ixf_F.vllm_single_query_cached_kv_attention  [ixformer .so — WORKS]
        │   → ixf_F.vllm_rotary_embedding_neox              [ixformer .so — WORKS]
        │
        ├── GDN layers (4 of 36):
        │   │
        │   ├── PREFILL:
        │   │   → corex_gdn.py:228 "Using fused CoreX GDN prefill operator"
        │   │   → corex_gdn.py:56  dlopen("/usr/local/corex/lib64/libcorex_gdn.so")
        │   │   → [chunked delta rule kernel — fp32 accumulate, NO NaN]
        │   │
        │   └── DECODE:
        │       → corex_gdn.py:138 "Using fused CoreX GDN decode operator"
        │       → [single-step recurrent kernel from libcorex_gdn.so]
        │
        ├── MoE layers (all 36):
        │   │
        │   ├── PREFILL (tokens=4096):
        │   │   → corex_moe.py:339 "Using CoreX fused MoE prefill: kernel=expert-grouped-wmma"
        │   │   → [topk routing — NOT via ixf_F, own implementation]
        │   │   → [expert GEMM via WMMA/cublas group_gemm]
        │   │   → ixf_F.silu_and_mul for activation
        │   │
        │   └── DECODE:
        │       → corex_moe.py:249 "Using CoreX fused MoE decode operator"
        │       → [same pipeline, fewer tokens]
        │
        └── Supporting ops (all via ixformer .so — confirmed working):
            → ixf_F.rms_norm
            → ixf_F.fused_add_rms_norm
            → ixf_F.vllm_cache_ops_reshape_and_cache
            → ixf_F.copy_blocks
            → ixf_F.swap_blocks
```

## Our 08-07 Docker — What Broke

```
HTTP Request → api_server.py → serving_chat.py
  → vLLM AsyncLLMEngine
    → model_runner.py:1119 (OUR version, +45 lines from base)
      → qwen3_5.py (OUR version — 1500+ lines)
        │
        ├── GDN layers: ✗ NaN (99.98%)
        │   → No corex_gdn.py found
        │   → FlashQLA SM70 disabled (abs_mean=inf in test)
        │   → Falls to _torch_chunk_gated_delta_rule (our PyTorch)
        │   → qwen3_5.py:445 "NaN in prefill GatedDeltaNet layer N"
        │   → nan_to_num(0) → garbage output → quality collapse
        │
        └── MoE layers: ✗ fallback to pure PyTorch
            → No corex_moe.py found
            → Tries ixf_F.vllm_moe_topk_softmax → AttributeError (NOT IN ixformer!)
            → _custom_ops.py:58 "Error in calling custom op topk_softmax"
            → qwen3_5.py:913 "falling back to pure PyTorch experts permanently"
            → Python for-loop over 64 experts × 8 topk = ~50x slower
```

## .so Files in Base Image

Available (confirmed by hardware probe):
```
/usr/local/corex/lib64/libcublas.so       ← used by torch.matmul
/usr/local/corex/lib64/libcublasLt.so     ← cublas lite
/usr/local/corex/lib64/libcuda.so         ← CUDA driver
/usr/local/corex/lib64/libcudart.so       ← CUDA runtime
/usr/local/corex/lib64/libcudnn.so        ← cuDNN
/usr/local/corex/lib64/libcutlass.so      ← CUTLASS
/usr/local/corex/lib64/libixattn.so       ← ixformer attention kernel
/usr/local/corex/lib64/libcuinfer.so      ← custom inference lib
/usr/local/corex/lib64/libixkninject.so   ← kernel injection
```

NOT available (must be built or bypassed):
```
/usr/local/corex/lib64/libcorex_gdn.so    ← GDN kernel (168 built this)
ixf_F.vllm_moe_topk_softmax              ← MoE routing (ABSENT from ixformer)
ixf_F.vllm_invoke_fused_moe_kernel       ← MoE GEMM (present but crashes)
```

## What We Need to Build

### Module 1: corex_gdn.py
**Location**: `$VLLM/model_executor/models/corex_gdn.py`
**Purpose**: GDN fused kernel dispatch
**Dispatch**:
1. FlashQLA .so (gdn_forward.cu compiled on BI-V100) — needs inf fix
2. PyTorch chunked delta rule with fp32 accumulation + clamping

### Module 2: corex_moe.py
**Location**: `$VLLM/model_executor/models/corex_moe.py`
**Purpose**: MoE fused pipeline (routing + expert GEMM + activation)
**Dispatch**:
1. PyTorch topk_softmax (replaces missing ixf_F.vllm_moe_topk_softmax)
2. Per-expert torch.matmul (goes to cublas via libcublas.so)
3. ixformer.silu_and_mul for activation (confirmed working)

### Integration: patch_ops.sh additions
```bash
# Add to patch_ops.sh after line 10 (deploy corex modules):
cp /workspace/ex_engine/python/corex_gdn.py $VLLM/model_executor/models/
cp /workspace/ex_engine/python/corex_moe.py $VLLM/model_executor/models/
```

## ixformer.functions — Confirmed API

### WORKS (no errors in any log):
```
ixf_F.silu_and_mul(x, out)
ixf_F.gelu_and_mul(x, out)
ixf_F.gelu_tanh_and_mul(x, out)
ixf_F.rms_norm(input, weight, out, epsilon)
ixf_F.fused_add_rms_norm(input, residual, weight, epsilon)
ixf_F.vllm_single_query_cached_kv_attention(...)  → paged_attn v1
ixf_F.vllm_rotary_embedding_neox(positions, query, key, ...)
ixf_F.vllm_batched_rotary_embedding(...)
ixf_F.vllm_cache_ops_reshape_and_cache(key, value, ...)
ixf_F.reshape_and_cache_flash(...)
ixf_F.paged_attention_cache_appended(...)
ixf_F.copy_blocks(key_caches, value_caches, block_mapping)
ixf_F.swap_blocks(src, dst, block_mapping)
ixf_F.advance_step_flashattn(...)
ixf_F.w8a8(a, b, scale_a, scale_b, bias, ...)
ixf_F.w8a16(x, qweight, scales, ...)
ixf_F.static_scaled_int8_quant(output, input, scale)
ixf_F.dynamic_scaled_int8_quant(output, input, input_scales)
ixf_F.vllm_gptq_shuffle(q_weight, q_perm)
ixf_F.quantized_linear(input, qweight, scales, ...)
ixf_F.quantized_weight_dequant(...)
```

### BROKEN/MISSING:
```
ixf_F.vllm_moe_topk_softmax        → AttributeError (doesn't exist)
ixf_F.vllm_invoke_fused_moe_kernel  → present but crashes (wrong BI-V100 config)
ixf_F.vllm_moe_align_block_size     → present, untested
```

## Version Differences

| Metric | 168's Docker (07-23) | Our Docker (08-07) |
|--------|---------------------|-------------------|
| model_runner.py line | :1074 | :1119 |
| Model weights | 17.35 GB | 16.23 GB |
| corex_gdn.py | ✓ (built + deployed) | ✗ (not found) |
| corex_moe.py | ✓ (built + deployed) | ✗ (not found) |
| GDN result | clean (no NaN) | 99.98% NaN |
| MoE result | fused WMMA kernel | PyTorch loop fallback |
| topk_softmax | own implementation | tries ixf_F (crashes) |

## CCCL Pattern Mapping

| Kernel | CCCL Algorithm | .so Target |
|--------|---------------|-----------|
| GDN prefill | `scan_by_key` (chunked lookback) | libcorex_gdn.so or PyTorch |
| GDN decode | `device_reduce` (single-tile) | libcorex_gdn.so or PyTorch |
| MoE topk | `device_select_if` (softmax + argmax) | PyTorch softmax + topk |
| MoE expert GEMM | `batch_memcpy` → `transform` (per-expert tile) | cublas via torch.matmul |
| MoE activation | `transform` (element-wise SiLU) | ixformer.silu_and_mul |
| MoE scatter-add | `reduce_by_key` (weighted accumulation) | PyTorch scatter |
| Attention | `reduce` (Q·K reduction) | ixf_F.vllm_single_query_cached_kv_attention |
| Softmax | `scan` (prefix sum for online softmax) | XFormers SDPA backend |
| RoPE | `transform` (element-wise rotation) | ixf_F.vllm_rotary_embedding_neox |
| RMSNorm | `reduce` + `transform` | ixf_F.rms_norm |
