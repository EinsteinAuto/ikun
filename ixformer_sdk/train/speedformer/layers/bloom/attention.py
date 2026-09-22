import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input 
from ixformer.train.speedformer.models.bloom.modeling_bloom import BloomAttention, dropout_add
from ixformer.train.speedformer.models.bloom.configuration_bloom import BloomConfig

from apex.transformer.functional.fused_rope import fused_apply_rotary_pos_emb_cached
from apex.transformer.functional.fused_rope import FusedRoPEFunc


class FlashAttention(BloomAttention):
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        fused_qkv = self.query_key_value(hidden_states)
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)  # 3 x [batch_size, seq_length, num_heads, head_dim]
        batch_size, q_length, _, _ = query_layer.shape
        
        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key, key_layer), dim=1)
            value_layer = torch.cat((past_value, value_layer), dim=1)
            
        present = (key_layer, value_layer) if use_cache else None
        # if attention_mask is not None:
        if False:
            query_layer, key_layer, value_layer, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_layer, key_layer, value_layer, attention_mask, q_length
            )
            
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
            attn_output_unpad = flash_attn_varlen_func(
                query_layer,
                key_layer,
                value_layer,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=0.0,
                softmax_scale=None,
                causal=True,
                use_alibi=True,
            )
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, q_length)
        else:
            attn_output = flash_attn_func(
                query_layer, key_layer, value_layer, 0.0, softmax_scale=None, causal=True, use_alibi=True,
            )

        attn_output = attn_output.reshape(batch_size, q_length, attn_output.shape[2]*attn_output.shape[3]).contiguous()
        output_tensor = self.dense(attn_output)
        
        output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)
        
        outputs = (output_tensor, present, None)

        return outputs
        

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        
        def _get_unpad_data(attention_mask):
            seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
            max_seqlen_in_batch = seqlens_in_batch.max().item()
            cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
            return (
                indices,
                cu_seqlens,
                max_seqlen_in_batch,
            )
            
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
        
        
class BloomFlashAttention(FlashAttention):

    def __init__(self) -> None:
        raise NotImplementedError(
            "BloomAttention is not implemented as a physical class. "
            "It is meant to be used only with the from_native_module interface to Convert a native BloomAttention module to FlashAttention module provided above."
        )
        
    @staticmethod
    def from_native_module(module: nn.Module, *args, **kwargs) -> nn.Module:

        # try to get normalized_shape, eps, elementwise_affine from the module
        new_config = BloomConfig()
        new_config.pretraining_tp = module.pretraining_tp
        new_config.slow_but_exact = module.slow_but_exact
        new_config.hidden_size = module.hidden_size
        new_config.n_head = module.num_heads
        new_config.hidden_size = module.split_size
        new_config.hidden_dropout = module.hidden_dropout
        new_config.attention_dropout = module.attention_dropout.p

        attention = FlashAttention(
            config=new_config,
        )
    
        attention.query_key_value.weight = module.query_key_value.weight
        attention.query_key_value.bias = module.query_key_value.bias

        attention.dense.weight = module.dense.weight
        attention.dense.bias = module.dense.bias

        return attention