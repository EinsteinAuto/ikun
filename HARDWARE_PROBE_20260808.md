# BI-V100 Hardware Probe Results

Date: 2026-08-08
Machine: cc-b2042074-46c3-4222-9d14-49c0c3637086-0
GPU: Iluvatar BI-V100 32768MiB
IX-ML: 3.2.3 | Driver: 3.2.1 | CUDA: 10.2

## 1. corex .so files

```
find /usr/local/corex/ -name "libcorex_*.so" -ls 2>/dev/null
# (empty — zero results)

find / -name "libcorex_gdn*" -ls 2>/dev/null
# (empty — zero results)
```

## 2. corex Python modules

```
find / -name "corex_gdn.py" -ls 2>/dev/null
# (empty)

find / -name "corex_moe.py" -ls 2>/dev/null
# (empty)
```

## 3. vllm models directory

```
ls -la /usr/local/corex/lib/python3/dist-packages/vllm/model_executor/models/ | grep -i "corex\|qwen3_5"
# (empty — neither corex modules nor qwen3_5.py in base image)
```

## 4. All corex-named files in SDK

```
find /usr/local/corex/ -name "*corex*" -type f 2>/dev/null
/usr/local/corex/bin/corex-uninstaller
/usr/local/corex/lib64/clang/16/include/__clang_cuda_ivcorex_intrinsics.h
/usr/local/corex/lib64/python3/dist-packages/paddle/include/paddle/phi/core/corex.h
/usr/local/corex/lib64/python3/dist-packages/torch/__pycache__/corex.cpython-310.pyc
/usr/local/corex/lib64/python3/dist-packages/torch/corex.py
/usr/local/corex/release-corex.txt
```

## 5. Available .so libraries

```
find /usr/local/corex/lib64/ -name "*.so" 2>/dev/null | head -30
/usr/local/corex/lib64/clang/16/lib/x86_64-unknown-linux-gnu/libclang_rt.asan.so
/usr/local/corex/lib64/clang/16/lib/x86_64-unknown-linux-gnu/libclang_rt.dyndd.so
/usr/local/corex/lib64/clang/16/lib/x86_64-unknown-linux-gnu/libclang_rt.hwasan.so
/usr/local/corex/lib64/clang/16/lib/x86_64-unknown-linux-gnu/libclang_rt.hwasan_aliases.so
/usr/local/corex/lib64/clang/16/lib/x86_64-unknown-linux-gnu/libclang_rt.memprof.so
/usr/local/corex/lib64/clang/16/lib/x86_64-unknown-linux-gnu/libclang_rt.scudo_standalone.so
/usr/local/corex/lib64/clang/16/lib/x86_64-unknown-linux-gnu/libclang_rt.tsan.so
/usr/local/corex/lib64/clang/16/lib/x86_64-unknown-linux-gnu/libclang_rt.ubsan_minimal.so
/usr/local/corex/lib64/clang/16/lib/x86_64-unknown-linux-gnu/libclang_rt.ubsan_standalone.so
/usr/local/corex/lib64/libLTO.so
/usr/local/corex/lib64/libclang.so
/usr/local/corex/lib64/libRemarks.so
/usr/local/corex/lib64/libclang-cpp.so
/usr/local/corex/lib64/libcublas.so
/usr/local/corex/lib64/libcublasLt.so
/usr/local/corex/lib64/libcuda.so
/usr/local/corex/lib64/libcudart.so
/usr/local/corex/lib64/libcudnn.so
/usr/local/corex/lib64/libcufft.so
/usr/local/corex/lib64/libcufftw.so
/usr/local/corex/lib64/libcuinfer.so
/usr/local/corex/lib64/libcupti.so
/usr/local/corex/lib64/libcurand.so
/usr/local/corex/lib64/libcusolver.so
/usr/local/corex/lib64/libcusparse.so
/usr/local/corex/lib64/libcutlass.so
/usr/local/corex/lib64/libibverbs.so
/usr/local/corex/lib64/libixToolsExt.so
/usr/local/corex/lib64/libixattn.so
/usr/local/corex/lib64/libixkninject.so
```

## 6. qwen3_5.py in base image

```
find / -name "qwen3_5.py" -ls 2>/dev/null
# (empty — not in base image, must be deployed by us)
```

## 7. ixformer API

```python
import ixformer
# Full dir() output:
['AVG', 'AddFunction', 'Any', 'BnbDequantFunction', 'BnbDoubleQuantFunction',
 'BnbMmDequantFunction', 'BnbQGemmFunction', 'BnbQuantFunction',
 'BnbRowColAbsMaxFunction', 'ChatGLM', 'ChunkFunction', 'ConcatFunction',
 'ContextBase', 'Contiguous', 'Copy', 'CudaStream', 'DataType', 'Device',
 'DeviceType', 'GLM130B', 'GPT2', 'GeluFunction', 'GptAttention', 'LLaMa',
 'LLaMaPipeline', 'List', 'MAX', 'MIN', 'MatmulFunction',
 'MemoryAllocatorType', 'MemoryFormat', 'MulFunction', 'Optional', 'PROD',
 'ParallelGpt', 'Permute', 'ReduceOp', 'ReductionSum', 'Reshape', 'SUM',
 'SplitFunction', 'Stream', 'StreamContext', 'SubFunction', 'Tensor',
 'TensorBase', 'TensorLayout', 'TensorOptions', 'TensorParallelLlama',
 'ToDevice', 'Transpose', 'Tuple', 'UndefinedTensor', 'Union', 'View',
 '_C', '_ixformer_torch', '_tensor',
 'act_bias_mm', 'add', 'allocate_memory', 'as_subclass',
 'attention_kv_cache_concat', 'attention_masked_softmax', 'autograd',
 'bfloat16', 'bnb_dequant', 'bnb_double_quant', 'bnb_mm_dequant',
 'bnb_qgemm', 'bnb_quant', 'bnb_rowcol_absmax', 'bool', 'byte',
 'can_device_access_peer', 'cat', 'channels_last', 'channels_last3d',
 'char', 'chunk', 'concat', 'contiguous', 'contiguous_format', 'contrib',
 'conv2d', 'copy', 'cuda', 'current_device', 'current_stream',
 'default_stream', 'device', 'device_count', 'device_synchronize',
 'distributed', 'double', 'dtype', 'elementwise', 'empty', 'empty_like',
 'empty_memory_caching', 'enable_grad', 'fill',
 'flash_attn', 'flash_attn_func', 'flash_attn_lib',
 'flash_attn_padded_func', 'flash_attn_varlen_func',
 'float', 'float16', 'free_memory', 'from_data_ptr', 'from_numpy',
 'from_torch', 'full', 'full_like', 'functions',
 'fused_add_rms_norm', 'gather_last_token_logits', 'geglu', 'gelu',
 'gelu_and_mul', 'gemv', 'gen_rotary_emb_weight',
 'get_arch_list', 'get_default_dtype', 'get_device_capability',
 'get_device_name', 'get_device_properties', 'get_gencode_flags',
 'get_memory_allocator', 'get_memory_allocator_type', 'get_tensor_ref_obj',
 'glm', 'glm2_rotary_embedding', 'glm_multi_query_repeat_key_value',
 'glm_multi_query_split_qkv', 'glm_split_qkv', 'gpt_attention',
 'group_norm', 'groupnorm', 'half', 'init_ixformer_context',
 'init_ixformer_modules', 'int', 'int32', 'int4WeightCompression',
 'int4WeightExtractionHalf', 'int64', 'int8', 'int8WeightExtractionHalf',
 'ipc_collect', 'is_available', 'is_differentiable_type', 'is_grad_enabled',
 'is_tensor', 'ixdnn_flash_attn_pad', 'ixdnn_flash_attn_unpad',
 'ixformer', 'ixinfer_flash_attn_pad', 'ixinfer_flash_attn_unpad',
 'kCPU', 'kCUDA', 'kCaching', 'kCustom', 'kNumDeviceType',
 'kNumMemoryAllocatorType', 'kNumReduceOp', 'kRaw', 'kUnknown',
 'kv_cache_concat', 'layernorm', 'lightllm', 'lightllm_apply_penalty',
 'lightllm_destindex_copy_kv', 'lightllm_glm2_rope',
 'lightllm_tokenattention', 'linalg', 'linear', 'linear_allreduce',
 'linear_allreduce_sum', 'linear_i8w8o32', 'llama_rotary_embedding',
 'masked_softmax', 'matmul', 'mul', 'new_tensor', 'no_grad',
 'num_data_type', 'num_memory_format', 'num_tensor_layout', 'ones',
 'ones_like', 'os', 'parse_kwargs', 'permute', 'preserve_format', 'qint8',
 'quantized_linear', 'quantized_weight_dequant', 'quint8', 'reduction',
 'reshape', 'residual_bias', 'residual_bias_ln', 'rms_norm',
 'rotary_embedding', 'rotary_embedding_2d',
 'scaled_dot_product_attention', 'set_custom_memory_allocator',
 'set_default_dtype', 'set_device', 'set_grad_enabled',
 'set_memory_allocator', 'set_stream', 'set_tensor_ref_obj',
 'silu_and_mul', 'skip_layer_norm', 'softmax', 'solve', 'split', 'stream',
 'stream_synchronize', 'strided', 'sub', 'sum', 'synchronize',
 't5', 't5_split_qkv', 't5_split_qkv_update_kv_cache', 'tensor', 'tgi',
 'tgi_apply_rotary', 'tgi_apply_rotary_emb_torch', 'to', 'torch_lib',
 'transpose', 'trt_llm_gpt_attention', 'uint32', 'uint64', 'uint8',
 'utils', 'view', 'vllm',
 'vllm_cache_ops_reshape_and_cache', 'vllm_copy_cache', 'vllm_gptq_shuffle',
 'vllm_llama_mlp', 'vllm_rotary_embedding_neox',
 'vllm_single_query_cached_kv_attention',
 'vllm_single_query_cached_kv_attention_v2',
 'vllm_smooth_dequant', 'vllm_smooth_dequant_add_residual',
 'vllm_smooth_dequant_fused_add_rms_norm_quant',
 'vllm_smooth_dequant_rotary_embedding_neox',
 'vllm_smooth_dequant_silu_and_mul_quant',
 'vllm_smooth_fused_add_rms_norm_quant', 'vllm_smooth_quant',
 'vllm_smooth_rms_norm_quant', 'vllm_swap_blocks',
 'w8a16', 'zeros', 'zeros_like']
```

## 8. ixformer function signatures (confirmed)

```
flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, return_attn_probs=False)
flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, softmax_scale=None, causal=False, return_attn_probs=False, out=None)
conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
fused_add_rms_norm(input, residual, weight, eps=1e-05, scale=1.0)
silu_and_mul(input, output=None)
gemv(x, A)
matmul(input, other, *, out=None, transa=False, transb=False, alpha=1.0, beta=0.0)
rms_norm(input, weight, output=None, eps=1e-06)
softmax(input, dim=None, _stacklevel=3, dtype=None, output=None)
scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
```

## 9. ixformer.vllm submodule

```
['CF', 'CacheOpsReshapeCacheFunction', 'Function', 'FunctionCtx',
 'RotaryEmbeddingNeoxFunction', 'Union',
 'compatible_torch_function', 'ixformer', 'ixformer_torch_ops', 'torch',
 'vllm_cache_ops_reshape_and_cache', 'vllm_copy_cache', 'vllm_gptq_shuffle',
 'vllm_llama_mlp', 'vllm_rotary_embedding_neox',
 'vllm_single_query_cached_kv_attention',
 'vllm_single_query_cached_kv_attention_v2',
 'vllm_smooth_dequant', 'vllm_smooth_dequant_add_residual',
 'vllm_smooth_dequant_fused_add_rms_norm_quant',
 'vllm_smooth_dequant_rotary_embedding_neox',
 'vllm_smooth_dequant_silu_and_mul_quant',
 'vllm_smooth_fused_add_rms_norm_quant', 'vllm_smooth_quant',
 'vllm_smooth_rms_norm_quant', 'vllm_swap_blocks']
```

## 10. topk/moe/expert/gate related ops

```
# (empty — zero topk/moe/expert/gate ops in ixformer)
```

## 11. Compilation toolchain

```
/usr/local/corex/lib64/clang/16/  — CUDA/C++ compiler
libcublas.so, libcublasLt.so      — BLAS
libcuda.so, libcudart.so          — CUDA runtime
libcudnn.so                       — cuDNN
libcutlass.so                     — CUTLASS
libcufft.so, libcusolver.so       — math libs
libixattn.so                      — ixformer attention kernel
```
