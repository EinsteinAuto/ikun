from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from transformers import LlamaConfig

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.models.llama import LlamaDecoderLayer as VllmLlamaDecoderLayer

import vllm._custom_ops as ops
# from ..overlap_comm import DecoderLayerOverlapComm, get_overlap_linear_method


# This method is needed for support smoothquant no overlap forward
def forward_smoothquant(
    input_ids: Optional[torch.Tensor],
    positions: torch.Tensor,
    kv_caches: List[torch.Tensor],
    attn_metadata: AttentionMetadata,
    inputs_embeds: Optional[torch.Tensor] = None,
    self = None, # will be set by partial
) -> torch.Tensor:
    dtype = self.dtype
    
    def forward_smoothquant_mlp(self,x,scales):
        # gate_up_proj
        # Int8 Matrix multiply.
        bias = self.gate_up_proj.bias if not self.gate_up_proj.skip_bias_add else None
        gate_up = ops.w8a8(x, self.gate_up_proj.weight, scales, self.gate_up_proj.weight_scales, dtype)
        if bias:
            gate_up += bias
        
        # act_fun
        x, scales = ops.silu_and_mul_smoothquant(gate_up, self.down_proj.smooth_scales)
        
        # down_proj
        output_parallel = ops.w8a8(x, self.down_proj.weight, scales, self.down_proj.weight_scales, dtype)
        if self.down_proj.reduce_results and self.down_proj.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        if not self.down_proj.skip_bias_add:
            output = output + self.down_proj.bias if self.down_proj.bias is not None else output

        return output
    
    def forward_smoothquant_attn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        # qkv proj
        bias = self.qkv_proj.bias if not self.qkv_proj.skip_bias_add else None
        
        qkv = ops.w8a8(hidden_states, self.qkv_proj.weight, scales, self.qkv_proj.weight_scales, dtype)
        if bias:
            qkv += bias
        
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)  # TODO
        return output
    
    if inputs_embeds is not None:
        hidden_states = inputs_embeds
    else:
        hidden_states = self.get_input_embeddings(input_ids)
    residual = None
    for i in range(len(self.layers)):
        layer = self.layers[i]
        if residual is None:
            residual = hidden_states
            hidden_states, scales = ops.rms_norm_smoothquant(hidden_states,layer.input_layernorm.weight,layer.input_layernorm.variance_epsilon, layer.self_attn.qkv_proj.smooth_scales)
        else:
            hidden_states, residual, scales = ops.fused_add_rms_norm_smoothquant(hidden_states, residual, layer.input_layernorm.weight, layer.input_layernorm.variance_epsilon, layer.self_attn.qkv_proj.smooth_scales)
        
        hidden_states = forward_smoothquant_attn(
            layer.self_attn,
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_caches[i],
            attn_metadata=attn_metadata,
            scales=scales,
            )

        # Fully Connected
        hidden_states, residual, scales = ops.fused_add_rms_norm_smoothquant(hidden_states, residual, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.variance_epsilon, layer.mlp.gate_up_proj.smooth_scales)
        
        hidden_states = forward_smoothquant_mlp(layer.mlp, hidden_states, scales)

    hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states

