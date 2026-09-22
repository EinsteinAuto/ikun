import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input 
from ixformer.train.speedformer.models.baichuan.configuration_baichuan import BaichuanConfig
from ixformer.train.speedformer.models.baichuan.modeling_baichuan import Attention

from ixformer.train.functions.fused_rope import fused_apply_rotary_pos_emb
from ixformer.train.speedformer.layers.rotary_pos_embedding import RotaryEmbedding

from ixformer.train.speedformer.layers.lazy import LazyInitContext


class FlashAttention(Attention):
    # 这个类主要的改进包含：1. apply_rotary_pos_emb；2. flash-attn 代替 native attention
    def __init__(self, config: BaichuanConfig):
        super().__init__(config)
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        proj = self.W_pack(hidden_states)
        proj = proj.unflatten(-1, (3, self.hidden_size)).unsqueeze(0).transpose(0, -2).squeeze(-2)
        
        # fused_apply_rotary_pos_emb need qk to be in "sbhd", v stay in "bshd"
        query_states = proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(0, 1).contiguous()
        key_states = proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(0, 1).contiguous()
        value_states = proj[2].view(bsz, q_len, self.num_heads, self.head_dim)

        kv_seq_len = key_states.shape[0]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[0]
            
        # fused_apply_rotary_pos_emb need emb in float32
        emb = self.rotary_emb(kv_seq_len).to(dtype=torch.float32)
        query_states = fused_apply_rotary_pos_emb(query_states, emb)
        key_states = fused_apply_rotary_pos_emb(key_states, emb)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=0)
            value_states = torch.cat([past_key_value[1], value_states], dim=0)

        past_key_value = (key_states, value_states) if use_cache else None

        # after fused_apply_rotary_pos_emb, qk change to "bshd" for flashattn or "bhsd" for sdpa
        if attention_mask is None: # flash-attn
            query_states = query_states.transpose(0, 1).contiguous()
            key_states   = key_states.transpose(0, 1).contiguous()
        else:  # sdpa
            query_states = query_states.permute(1, 2, 0, 3).contiguous()
            key_states   = key_states.permute(1, 2, 0, 3).contiguous()
            value_states = value_states.transpose(1, 2).contiguous()

        '''
        if attention_mask is not None:
            batch_size = query_states.shape[0]   # bsz, q_len, self.num_heads, self.head_dim
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, q_len
            )
            
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=0.0,
                softmax_scale=None,
                causal=True,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, q_len)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, 0.0, softmax_scale=None, causal=True
            )
        '''
        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, q_len, attention_mask, dropout=0.0
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value
    
    def _flash_attention_forward(
        self, 
        query_states: torch.Tensor, 
        key_states: torch.Tensor, 
        value_states: torch.Tensor, 
        query_length: int, 
        attention_mask: Optional[torch.Tensor] = None, 
        dropout=0.0, 
        softmax_scale=None
    ):
        if attention_mask is not None:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=0.0,
                # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
                is_causal=query_length > 1,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
            
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=self.is_causal
            )

        return attn_output
        
class BaichuanAttention(FlashAttention):
    def __init__(self) -> None:
        raise NotImplementedError(
            "BaichuanAttention is not implemented as a physical class. "
            "It is meant to be used only with the from_native_module interface to Convert a native BaichuanAttention module to LlamaAttention module provided above."
        )
        
    @staticmethod
    def from_native_module(module: nn.Module, *args, **kwargs) -> nn.Module:

        LazyInitContext.materialize(module)

        # try to get normalized_shape, eps, elementwise_affine from the module
        config = getattr(module, "config")

        attention = FlashAttention(
            config=config,
        )
    
        attention.W_pack.weight = module.W_pack.weight
        attention.o_proj.weight = module.o_proj.weight

        return attention
