# Qwen3.6-35B-A3B Bootstrap Issue

## Problem
vllm 0.6.3+corex.3.2.3 does not recognize `qwen3_5_moe` model type.

```
ValueError: The checkpoint you are trying to load has model type `qwen3_5_moe` 
but Transformers does not recognize this architecture.
```

## Root Cause
- Model `config.json` specifies `"model_type": "qwen3_5_moe"` and `"architectures": ["Qwen3_5MoeForCausalLM"]`
- Server transformers version: 4.51.3 (needs ≥ 4.57.1)
- Server vllm version: 0.6.3+corex.3.2.3

## Model Architecture (from config.json)
- **Type**: Qwen3_5MoeForCausalLM (MoE with linear attention)
- **Total params**: ~35B
- **Active params per token**: ~3B (8 of 256 experts)
- **Hidden size**: 2048
- **Layers**: 40 (30 linear_attention + 10 full_attention, every 4th is full)
- **Experts**: 256 total, 8 per token
- **Expert intermediate**: 512
- **Shared expert intermediate**: 512
- **Head dim**: 256
- **KV heads**: 2 (GQA ratio 8:1)
- **Max position**: 262144
- **Vocab**: 248320
- **Precision**: bfloat16
- **Linear attention**: conv kernel dim=4, 16 key heads (dim128), 32 value heads (dim128)
- **MTP**: 1 hidden layer (multi-token prediction)
- **Vision**: yes (patch16, depth27, hidden1152)

## Key Architecture Features
1. **Hybrid attention**: 3 linear_attention + 1 full_attention pattern (30+10=40 layers)
2. **MoE**: 256 experts, top-8 routing = very sparse
3. **Linear attention with conv**: NOT standard transformer — uses conv kernel dim=4
4. **Multi-token prediction (MTP)**: 1 extra hidden layer for speculative prediction
5. **Multimodal**: has vision encoder (but competition likely tests text only)

## Solution Paths
1. **EngineX route**: Check if the competition's enginex-vllm package already supports this model
   - The repo has `enginex-vllm-bi100-qwen36-main.zip` (96MB) — THIS is likely the answer
2. **Upgrade transformers**: `pip install transformers>=4.57.1` (may break corex compatibility)
3. **Custom model registration**: Register Qwen3_5MoeForCausalLM in vllm's model registry
