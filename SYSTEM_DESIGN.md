# System Design

## Architecture

```
Docker Image (FROM bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3)
│
├── /workspace/
│   ├── computility-run.yaml          # vLLM launch args
│   └── qwen3_6_scripts/
│       ├── patch_ops.sh              # Build-time: deploy all patches
│       ├── precompile_gdn.py         # Build-time: compile .cu → .so
│       ├── qwen3_5.py                # Model: GDN + MoE + Attention
│       ├── flash_qla_sm70/
│       │   ├── csrc/gdn_forward.cu   # SM70 fused GDN CUDA kernel (1919 lines)
│       │   ├── fused_fwd.py          # Python wrapper, loads .so
│       │   ├── naive_gdn.py          # PyTorch reference fallback
│       │   └── __init__.py
│       ├── serving_chat.py           # OpenAI API handler
│       ├── protocol.py               # Request/response models
│       ├── chat_utils.py             # Tool call handling
│       ├── api_server.py             # FastAPI app
│       ├── cli_args.py               # CLI argument extensions
│       ├── registry.py               # Model registry (adds Qwen3_5)
│       ├── paged_attn.py             # Paged attention PyTorch fallback
│       ├── mamba_cache.py            # GDN state cache manager
│       ├── sequence.py               # Token count fix
│       ├── scheduler.py              # Chunked prefill fix
│       ├── xformers.py               # SDPA fallback patches
│       ├── patch_xformers_*.py       # xformers monkey-patches
│       ├── patch_model_runner.py     # prefix_cache_hit fix
│       ├── patch_numerical_stability.py
│       ├── patch_transformers_qwen3_5.py
│       ├── patch_vllm_tool_parser.py
│       ├── qwen3coder_tool_parser.py # Tool call parser
│       └── tool_parsers_init.py
│
├── /usr/local/corex/                 # Base image SDK
│   ├── lib64/
│   │   ├── libcublas.so
│   │   ├── libcudart.so
│   │   ├── libcudnn.so
│   │   ├── libcutlass.so
│   │   ├── libixattn.so
│   │   └── clang/16/                # CUDA compiler
│   └── lib/python3/dist-packages/
│       ├── torch/
│       ├── vllm/                     # Base vLLM 0.6.3
│       └── ixformer/                 # Hardware acceleration ops
│
└── /model/                           # Qwen3.5-27B weights (16 shards)
```

## Build Pipeline

```
Dockerfile
    │
    ├── COPY qwen3_6_scripts/ → /workspace/qwen3_6_scripts/
    ├── COPY computility-run.yaml → /workspace/
    │
    └── RUN patch_ops.sh
         │
         ├── 1. Find vllm install path ($VLLM)
         ├── 2. apt install ninja-build
         ├── 3. pip install transformers==4.55.3
         ├── 4. Shell probe (ls corex .so, ls corex .py, ls native qwen3_5.py)
         ├── 5. Deploy qwen3_5.py → $VLLM/model_executor/models/
         ├── 6. Deploy registry.py (add Qwen3_5ForCausalLM)
         ├── 7. Deploy flash_qla_sm70/ → $VLLM/model_executor/models/
         ├── 8. Run precompile_gdn.py → flash_qla_sm70/build/*.so
         ├── 9. Deploy paged_attn.py, mamba_cache.py, sequence.py, scheduler.py
         ├── 10. Deploy xformers patches (monkey-patch SDPA)
         ├── 11. Deploy tool parser + reasoning parser
         ├── 12. Deploy serving_chat.py, protocol.py, api_server.py, chat_utils.py
         └── 13. Mirror all to $VLLM2 if second vllm install exists
```

## Runtime Data Flow

```
HTTP Request (OpenAI format)
    │
    ▼
api_server.py → serving_chat.py
    │
    ├── protocol.py: validate request, handle max_completion_tokens
    ├── chat_utils.py: format messages, handle tool_calls
    │
    ▼
vLLM AsyncLLMEngine
    │
    ├── scheduler.py → batch requests
    ├── model_runner.py → execute_model()
    │
    ▼
qwen3_5.py: Qwen3_5ForCausalLM.forward()
    │
    ├── Embedding → token embeddings
    │
    ├── 64 Decoder Layers (loop):
    │   │
    │   ├── Layers with GatedDeltaNet (4 of 36 attention layers):
    │   │   │
    │   │   ├── Projections: in_proj_qkv, in_proj_z, in_proj_b, in_proj_a
    │   │   ├── Conv1d (depthwise causal)
    │   │   ├── L2 normalize q, k
    │   │   │
    │   │   ├── DISPATCH:
    │   │   │   ├── 1st: CoreX fused kernel (if corex_gdn.py packaged)
    │   │   │   ├── 2nd: FlashQLA SM70 kernel (prefill only, gdn_forward.cu)
    │   │   │   └── 3rd: PyTorch _torch_chunk_gated_delta_rule (with NaN clamp)
    │   │   │
    │   │   ├── Gated RMSNorm
    │   │   └── out_proj
    │   │
    │   ├── Layers with Full Attention (32 of 36):
    │   │   └── xformers SDPA (patched fallback for BI-V100)
    │   │
    │   ├── MoE (all 36 layers):
    │   │   ├── Gate → router logits → topk
    │   │   ├── DISPATCH:
    │   │   │   ├── 1st: CoreX fused MoE (if corex_moe.py packaged)
    │   │   │   └── 2nd: PyTorch loop over experts
    │   │   ├── Shared expert (with sigmoid gate)
    │   │   └── All-reduce (TP)
    │   │
    │   └── RMSNorm (pre/post)
    │
    ├── Final RMSNorm
    ├── LM Head → logits
    └── Sampler → tokens
```

## GDN Kernel Dispatch Detail

```
GatedDeltaNet.forward(hidden_states, attn_metadata, conv_state, temporal_state)
    │
    ├── is_prefill? (attn_metadata.num_prefill_tokens > 0)
    │   │
    │   ├── YES (prefill):
    │   │   ├── Try FlashQLA SM70:
    │   │   │   ├── Project q,k,v,gate,beta
    │   │   │   ├── Conv1d
    │   │   │   ├── L2norm
    │   │   │   ├── Reshape to [1, L, H, 128]
    │   │   │   ├── chunk_gated_delta_rule_fwd_sm70(q,k,v,g,beta,state)
    │   │   │   │   └── gdn_forward.cu → flash_qla_sm70_gdn_strided.so
    │   │   │   ├── Update temporal_state
    │   │   │   ├── Gated RMSNorm + out_proj
    │   │   │   └── Return
    │   │   │
    │   │   └── Fallback: _torch_chunk_gated_delta_rule (PyTorch, chunked)
    │   │
    │   └── NO (decode):
    │       └── PyTorch single-step recurrent update
    │           ├── Conv1d state update
    │           ├── temporal_state decay + delta write
    │           ├── Query @ state → output
    │           └── Return
    │
    └── Both paths end with: Gated RMSNorm → out_proj → all_reduce
```

## computility-run.yaml Key Args

```yaml
max_model_len: 80000        # Must be < KV cache capacity (88112)
gpu_memory_utilization: 0.9
max_num_seqs: 1
tensor_parallel_size: 4
enforce_eager: true          # No CUDA graphs (BI-V100 compatibility)
enable_prefix_caching: true
max_seq_len_to_capture: 8192
tool_call_parser: qwen3_coder
reasoning_parser: qwen3
```

## File Dependencies

```
qwen3_5.py imports:
    ├── vllm.attention (Attention, AttentionMetadata)
    ├── vllm.model_executor.layers.* (linear, norm, sampler, etc.)
    ├── vllm.model_executor.models.mamba_cache (MambaCacheManager)
    ├── vllm.model_executor.models.flash_qla_sm70 (SM70 kernel)
    ├── ixformer (optional, hardware-accelerated ops)
    └── vllm.model_executor.models.corex_gdn (optional, if packaged)

flash_qla_sm70/fused_fwd.py imports:
    ├── torch.utils.cpp_extension.load (JIT compile .cu → .so)
    └── gdn_forward.cu (CUDA source, compiled to .so)

serving_chat.py imports:
    ├── vllm.entrypoints.openai.protocol (request validation)
    ├── vllm.entrypoints.chat_utils
    └── vllm engine client
```

## Scoring Modules (competition)

```
Module 1: functional_acceptance (52 tests)
    ├── d01-d10: basic, stream, tools, reasoning, multimodal, thinking
    ├── t1-t16: auth, n=2, max_tokens, stop, system, temperature, etc.
    └── 4 skipped: d08, t11a, t11b, t16b

Module 2: case_truncation
    └── Output truncation correctness

Module 3: replay_tencent
    └── 881 real requests, throughput scoring
    └── Output TPS weight: 83%

Module 4: opencompass
    └── Model quality benchmarks
```
