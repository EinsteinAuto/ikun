import math

import ixformer.functions as ixf_F
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import logging

logger = logging.get_logger(__name__)


def mha(query, key, value, attention_mask):
    if attention_mask is None and query.shape[2] == key.shape[2]:
        context_layer = ixf_F.scaled_dot_product_attention(
            query.contiguous(), key.contiguous(), value.contiguous(), is_causal=True
        )
    else:
        # if attention_mask is not None:
        #     # attention_mask = attention_mask
        #     attention_mask = (~attention_mask).cuda().float()*(-10000)
        context_layer = ixf_F.scaled_dot_product_attention(
            query.contiguous(), key.contiguous(), value.contiguous(), attention_mask
        )

    # context_layer = context_layer.transpose(1, 2).contiguous()
    # res_shape = list(context_layer.shape)
    # res_shape = res_shape[:2] + [-1]
    # context_layer = context_layer.view(*res_shape)
    return context_layer
    # batch_size, head_num, seq_len, head_dim = query.shape
    # src_len = query.shape[-2]
    # tgt_len = key.shape[-2]

    # if attention_mask is None and src_len == tgt_len:
    #     attention_mask = ~torch.tril(torch.ones([src_len, tgt_len])).bool()
    # elif attention_mask is None:
    #     attention_mask = torch.zeros([src_len, tgt_len])
    # attention_mask = attention_mask.cuda().int()

    # attention_scores = ixf_F.act_bias_mm(
    #     query, key, scale=1 / math.sqrt(head_dim), trans_format="TN"
    # )
    # # softmax
    # # if tgt_len > 2048:
    # #     if not (attention_mask == 0).all():
    # #         attention_scores.masked_fill_(attention_mask.bool(), -10000.0)
    # #     dtype = attention_scores.dtype
    # #     attention_probs = F.softmax(attention_scores.float(), dim=-1)
    # #     attention_probs = attention_probs.type(dtype)
    # # else:
    # #     raise NotImplementedError()
    # attention_probs = ixf_F.attention_masked_softmax(
    #     attention_scores, attention_mask.int()
    # )
    # # s * v
    # # batch_size,head_num,seq_len,head_dim
    # context_layer = ixf_F.act_bias_mm(
    #     attention_probs, value, trans_format="NN")
    # context_layer = context_layer.transpose(1, 2).contiguous()
    # context_layer = context_layer.view(
    #     batch_size, seq_len, head_num * head_dim)


def mlp(mlp_input, ff1_weight, ff1_bias, ff2_weight):
    input_shape = list(mlp_input.shape)
    mlp_input = mlp_input.view(-1, input_shape[-1])
    mlp_output = ixf_F.act_bias_mm(
        mlp_input, ff1_weight, ff1_bias, scale=1, act_type="gelu", trans_format="TN"
    )
    mlp_output = ixf_F.linear(mlp_output, ff2_weight, None)
    input_shape[-1] = -1
    mlp_output = mlp_output.view(*input_shape)
    return mlp_output


def mlp_forward(self, hidden_states):
    # [s, b, 4hp]
    # intermediate_parallel = self.dense_h_to_4h(hidden_states)
    # intermediate_parallel = self.activation_func(intermediate_parallel)
    input_shape = list(hidden_states.shape)
    hidden_states = hidden_states.view(-1, input_shape[-1])
    mlp_output = ixf_F.act_bias_mm(
        hidden_states,
        self.c_fc.weight,
        self.c_fc.bias,
        scale=1,
        act_type="gelu",
        trans_format="TN",
    )
    if isinstance(self.c_proj, nn.Linear):
        output = ixf_F.linear(
            mlp_output,
            self.c_proj.weight,
            self.c_proj.bias,
        )
    else:
        output = self.c_proj(mlp_output)
    output = output.view(*input_shape)
    return output
