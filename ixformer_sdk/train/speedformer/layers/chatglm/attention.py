import math
import os
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ixformer.train.speedformer.models.chatglm.modeling_chatglm import (
    CoreAttention,
    SelfAttention,
    split_tensor_along_last_dim,
    apply_rotary_pos_emb
)
from ixformer.train.speedformer.models.chatglm.configuration_chatglm import ChatGLMConfig

from transformers.utils import is_flash_attn_2_available

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


class FlashCoreAttention(CoreAttention):

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        if int(os.environ.get("USE_FLASH_ATTN", 0)):
            query_layer, key_layer, value_layer = [
                k.permute(1, 0, 2, 3) for k in [query_layer, key_layer, value_layer]]
            batch_size, query_length, _, _ = query_layer.shape

            if attention_mask is not None:
                batch_size = query_layer.shape[0]
                query_layer, key_layer, value_layer, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                    query_layer, key_layer, value_layer, attention_mask, query_length
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
                )
                attn_output = pad_input(
                    attn_output_unpad, indices_q, batch_size, query_length)
                context_layer = attn_output.permute(1, 0, 2, 3)
            else:
                attn_output = flash_attn_func(
                    query_layer, key_layer, value_layer, 0.0, softmax_scale=None, causal=True
                )
                context_layer = attn_output.permute(1, 0, 2, 3)

        if attention_mask is not None:
            if query_layer.shape[2] != key_layer.shape[2]:
                num_group = query_layer.shape[2] // key_layer.shape[2]
                final_shape = (*key_layer.shape[:2], *query_layer.shape[2:])
                key_layer = key_layer.unsqueeze(-2)
                key_layer = key_layer.expand(
                    -1, -1, -1, num_group, -1
                )
                key_layer = key_layer.contiguous().view(
                    final_shape
                )
                value_layer = value_layer.unsqueeze(-2)
                value_layer = value_layer.expand(
                    -1, -1, -1, num_group, -1
                )
                value_layer = value_layer.contiguous().view(
                    final_shape
                )

            query_layer, key_layer, value_layer = [
                k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]]  # bhsd
            attention_mask = ~attention_mask
            context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                             attention_mask)
            context_layer = context_layer.permute(2, 0, 1, 3)

        else:
            query_layer, key_layer, value_layer = [
                k.permute(1, 0, 2, 3) for k in [query_layer, key_layer, value_layer]]  # bshd
            context_layer = flash_attn_func(
                query_layer, key_layer, value_layer, 0, softmax_scale=None, causal=True
            )  # bshd
            context_layer = context_layer.permute(1, 0, 2, 3)

        context_layer = context_layer.reshape(
            context_layer.size(0), context_layer.size(1), -1)

        return context_layer


class FlashSelfAttention(SelfAttention):

    def __init__(self, config: ChatGLMConfig, layer_number, device=None):
        super().__init__(config, layer_number, device=device)
        self.core_attention = FlashCoreAttention(config, self.layer_number)

    def forward(self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True):
        mixed_x_layer = self.query_key_value(hidden_states)
        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_layer = query_layer.view(
                query_layer.size()[
                    :-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            key_layer = key_layer.view(
                key_layer.size()[
                    :-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = value_layer.view(
                value_layer.size()[:-1]
                + (self.num_multi_query_groups_per_partition,
                   self.hidden_size_per_attention_head)
            )
        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(
                mixed_x_layer, 3)

        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        # adjust key and value for inference
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            key_layer = torch.cat((cache_k, key_layer), dim=0)
            value_layer = torch.cat((cache_v, value_layer), dim=0)
        if use_cache:
            kv_cache = (key_layer, value_layer)
        else:
            kv_cache = None

        # 这里省略了 kv "sbhd" -> "sb(h*num_multi-group)d" 的过程，因为flash-attn支持 MGA
        # ==================================
        # core attention computation
        # ==================================

        context_layer = self.core_attention(
            query_layer, key_layer, value_layer, attention_mask)

        # =================
        # Output. [sq, b, h]
        # =================

        output = self.dense(context_layer)

        return output, kv_cache


class ChatglmFlashAttention(FlashSelfAttention):

    def __init__(self) -> None:
        raise NotImplementedError(
            "BloomAttention is not implemented as a physical class. "
            "It is meant to be used only with the from_native_module interface to Convert a native BloomAttention module to FlashAttention module provided above."
        )

    @staticmethod
    def from_native_module(module: nn.Module, *args, **kwargs) -> nn.Module:
        # 这个原实现没有在类中保存config，所以需要初始化一个config
        layer_number = getattr(module, "layer_number")
        config = getattr(module, "config")
        attention = FlashSelfAttention(
            config=config,
            layer_number=layer_number,
        )

        attention.query_key_value.weight.data = module.query_key_value.weight.data
        attention.dense.weight.data = module.dense.weight.data
        if getattr(attention.query_key_value, "bias") is not None:
            attention.query_key_value.bias.data = module.query_key_value.bias.data
        if getattr(attention.dense, "bias") is not None:
            attention.dense.bias.data = module.dense.bias.data

        return attention
