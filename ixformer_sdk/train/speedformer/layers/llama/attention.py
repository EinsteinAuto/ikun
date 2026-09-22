import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ixformer.train.speedformer.models.llama.configuration_llama import LlamaConfig
from ixformer.train.speedformer.models.llama.modeling_llama import LlamaFlashAttention2
from transformers import Cache
from transformers.utils import logging

from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

from ixformer.train.functions.fused_rope import fused_apply_rotary_pos_emb
from ixformer.train.speedformer.layers.rotary_pos_embedding import RotaryEmbedding


class BaseLlamaAttention(LlamaFlashAttention2):
    """ 
    加这个层的原因：1.当原模型中使用的是torch nvtive的attention，强制替换成flash_attn; 2.优化rope 
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.rope_scaling is None:
            self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # fused_apply_rotary_pos_emb need qk to be in "sbhd"
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 0).contiguous()
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 0).contiguous()
        value_states = value_states.view(
            bsz, q_len, self.num_heads, self.head_dim)

        kv_seq_len = key_states.shape[0]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[0]

        emb = self.rotary_emb(kv_seq_len).to(dtype=torch.float32)
        query_states = fused_apply_rotary_pos_emb(query_states, emb)
        key_states = fused_apply_rotary_pos_emb(key_states, emb)

        # kv cache staff
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=0)
            value_states = torch.cat([past_key_value[1], value_states], dim=0)
        past_key_value = (key_states, value_states) if use_cache else None

        dropout_rate = self.attention_dropout if self.training else 0.0

        # after fused_apply_rotary_pos_emb, qk change to "bshd" for flashattn or "bhsd" for sdpa
        if attention_mask is None:  # flash-attn
            query_states = query_states.transpose(0, 1).contiguous()
            key_states = key_states.transpose(0, 1).contiguous()
        else:  # sdpa
            query_states = query_states.permute(1, 2, 0, 3).contiguous()
            key_states = key_states.permute(1, 2, 0, 3).contiguous()
            value_states = value_states.transpose(1, 2).contiguous()
        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(
            bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        for now, if attention_mask is none, flash-attn has better performance than torch.nn.functional.scaled_dot_product_attention;
        if attention_mask is not none, torch.nn.functional.scaled_dot_product_attention works better
        so sdpa and flash-attn is perfered according to attention_mask

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        # Contains at least one padding token in the sequence
        # if attention_mask is not None:
        if attention_mask is not None:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
                is_causal=self.is_causal and attention_mask is None and query_length > 1,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()

        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=self.is_causal
            )

        return attn_output


class LlamaAttention(BaseLlamaAttention):
    def __init__(self) -> None:
        raise NotImplementedError(
            "LlamaAttention is not implemented as a physical class. "
            "It is meant to be used only with the from_native_module interface to Convert a native LlamaAttention module to LlamaAttention module provided above."
        )

    @staticmethod
    def from_native_module(module: nn.Module, *args, **kwargs) -> nn.Module:

        # LazyInitContext.materialize(module)

        # try to get normalized_shape, eps, elementwise_affine from the module
        config = getattr(module, "config")
        layer_idx = getattr(module, "layer_idx", None)

        attention = BaseLlamaAttention(
            config=config,
            layer_idx=layer_idx,
        )

        attention.q_proj.weight = module.q_proj.weight
        attention.k_proj.weight = module.k_proj.weight
        attention.v_proj.weight = module.v_proj.weight
        attention.o_proj.weight = module.o_proj.weight

        if config.attention_bias:
            attention.q_proj.bias = module.q_proj.bias
            attention.k_proj.bias = module.k_proj.bias
            attention.v_proj.bias = module.v_proj.bias
            attention.o_proj.bias = module.o_proj.bias
        return attention
