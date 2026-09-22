import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ixformer.train.speedformer.models.chatglm.modeling_chatglm import ChatGLMModel


def ChatGLMModel_forward():
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.utils import logging, is_flash_attn_2_available

    def forward(
            self,
            input_ids,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.BoolTensor] = None,
            full_attention_mask: Optional[torch.BoolTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        def is_lower_triangular(mask):
            """
            ixdnn 虽然支持2种causal mask, 如下图:
            mode0:
                if seqlen_q < seqlen_k
                    1 0 0 0 0
                    1 1 0 0 0 
                if seqlen_k < seqlen_q
                    1 0
                    1 1
                    1 1
                    1 1
                    1 1
            mode1:
                if seqlen_q < seqlen_k
                    1 1 1 1 0
                    1 1 1 1 1
                if seqlen_k < seqlen_q
                    0 0
                    0 0
                    0 0
                    1 0
                    1 1

            但 flash-attn 目前只支持 mode1, 所以下面需要判断一下传入的mask是不是mode1这种模式
            """
            batch_size, _, rows, cols = mask.shape

            # 创建一个mode1的下三角矩阵
            if rows <= cols:
                part = torch.ones(rows, cols - rows,
                                  dtype=torch.bool, device=mask.device)
                gt = ~torch.triu(torch.ones(
                    rows, rows, dtype=torch.bool, device=mask.device), diagonal=1)
                gt = torch.cat((part, gt), dim=1)
            else:
                part = torch.zeros(
                    rows-cols, cols, dtype=torch.bool, device=mask.device)
                gt = ~torch.triu(torch.ones(
                    cols, cols, dtype=torch.bool, device=mask.device), diagonal=1)
                gt = torch.cat((part, gt), dim=0)
            gt = gt[None, None, :, :].expand(batch_size, -1, -1, -1)

            # 检查所有的元素是不是都一样
            check = (gt == mask).all()

            return check

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = input_ids.shape

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if self.pre_seq_len is not None:
            if past_key_values is None:
                past_key_values = self.get_prompt(batch_size=batch_size, device=input_ids.device,
                                                  dtype=inputs_embeds.dtype)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask.new_ones((batch_size, self.pre_seq_len)),
                                            attention_mask], dim=-1)

        if full_attention_mask is None:
            if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                full_attention_mask = self.get_masks(
                    input_ids, past_key_values, padding_mask=attention_mask)

        # Rotary positional embeddings
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

        # Run encoder.
        attn_mask = None
        if full_attention_mask is not None:
            if not is_lower_triangular(full_attention_mask):
                attn_mask = full_attention_mask
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds, attn_mask, rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values, use_cache=use_cache, output_hidden_states=output_hidden_states
        )

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    return forward
