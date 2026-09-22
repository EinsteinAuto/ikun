# MoE Execution Path Analysis

> Source: vllm/model_executor/layers/fused_moe/fused_moe.py + vllm/_custom_ops.py
> Read: 2026-08-04

## The Real Bottleneck

Qwen3.6-35B-A3B has 64 MoE layers, each with:
- 256 experts, top-8 routing
- Gate up projection (w1): hidden_dim → intermediate_dim
- SiLU activation
- Down projection (w2): intermediate_dim → hidden_dim
- Weighted sum of 8 expert outputs

### Per-decode-step kernel launches

| Operation | Count | Implementation |
|---|---|---|
| fused_moe_kernel (w1) | 64 | ixf_F.vllm_invoke_fused_moe_kernel |
| silu_and_mul | 64 | ixf_F.silu_and_mul |
| fused_moe_kernel (w2) | 64 | ixf_F.vllm_invoke_fused_moe_kernel |
| topk_softmax | 64 | ixf_F.vllm_moe_topk_softmax |
| moe_align_block_size | 64 | ixf_F.vllm_moe_align_block_size |
| torch.sum (expert merge) | 64 | PyTorch |
| paged_attention_v1 | 1 | ixf_F.vllm_single_query_cached_kv_attention |
| rms_norm | 128 | ixf_F.rms_norm |
| fused_add_rms_norm | 64 | ixf_F.fused_add_rms_norm |
| rotary_embedding | 64 | ixf_F.vllm_rotary_embedding_neox |
| **Total** | **~640+** | |

640+ kernel launches per decode step. At target Output TPS ≥ 395,
that's 395 × 640 = 253,000 kernel launches per second.

### Memory allocation per step

```python
# Inside fused_experts, called 64 times per step:
intermediate_cache1 = torch.empty((M, topk, N))        # 64 × alloc
intermediate_cache2 = torch.empty((M * topk, N // 2))   # 64 × alloc
intermediate_cache3 = torch.empty((M, topk, w2_shape[1]))  # 64 × alloc
```

192 torch.empty calls per decode step = 192 CUDA mallocs.
At 395 TPS = 75,840 mallocs/second.

### What we can actually change

1. **BLOCK_SIZE_M** (passed to ixformer): 16 for decode (numel=8, M=1×topk=8)
   - Already optimized: 16 for ≤16 tokens, 32 for ≤64, 64 for ≤1024
   - ixformer may or may not respect N/K/GROUP values

2. **Intermediate cache pre-allocation**: move torch.empty outside the layer loop
   - Allocate once, reuse across 64 layers
   - Saves 192 CUDA mallocs per decode step

3. **torch.sum → ixformer?**: the expert merge `torch.sum(dim=1)` is PyTorch,
   could potentially be fused into the second fused_moe_kernel call

4. **Chunk size**: VLLM_FUSED_MOE_CHUNK_SIZE controls batching.
   For decode M=1, chunking adds overhead for no benefit.
