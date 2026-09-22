import math
import warnings
import inspect
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ixformer.train.speedformer.models.qwen2.configuration_qwen2 import Qwen2Config
from ixformer.train.speedformer.models.qwen2.modeling_qwen2 import Qwen2FlashAttention2
from transformers import Cache
from transformers.utils import logging

from flash_attn import flash_attn_func, flash_attn_varlen_func

from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

from ixformer.train.functions.fused_rope import fused_apply_rotary_pos_emb
from ixformer.train.speedformer.layers.rotary_pos_embedding import RotaryEmbedding

from ixformer.train.speedformer.layers.lazy import LazyInitContext

_flash_supports_window_size = "window_size" in list(
    inspect.signature(flash_attn_func).parameters)
logger = logging.get_logger(__name__)


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class BaseQwenAttention(Qwen2FlashAttention2):
    """ 
    加这个层的原因：1.当原模型中使用的是torch nvtive的attention，强制替换成flash_attn; 2.优化rope 
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        out_dim = self.num_heads * self.head_dim + \
            self.num_key_value_heads * self.head_dim * 2
        self.qkv_proj = nn.Linear(self.hidden_size, out_dim, bias=True)
        del self.q_proj, self.k_proj, self.v_proj
        self.rotary_emb = RotaryEmbedding(self.head_dim, self.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        bsz, q_len, _ = hidden_states.size()
        qkv = self.qkv_proj(hidden_states)
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_key_value_heads * self.head_dim
        query_states, key_states, value_states = torch.split(
            qkv, (q_dim, kv_dim, kv_dim), dim=-1)
        # fused_apply_rotary_pos_emb need qk to be in "sbhd", v stay "bshd"
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(0, 1).contiguous()
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(0, 1).contiguous()
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim)

        kv_seq_len = key_states.shape[0]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value[0].shape[0]

        emb = self.rotary_emb(kv_seq_len).to(dtype=torch.float32)
        query_states = fused_apply_rotary_pos_emb(query_states, emb)
        key_states = fused_apply_rotary_pos_emb(key_states, emb)
        use_sliding_windows = (
            _flash_supports_window_size
            and getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
            and self.config.use_sliding_window
        )

        if not _flash_supports_window_size:
            logger.warning_once(
                "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
                " make sure to upgrade flash-attn library."
            )

        # for now, attention with sliding_windows have not test, so if use_sliding_windows throw error
        if use_sliding_windows:
            raise KeyError("use_sliding_windows not support for now")

        # kv cache staff
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=0)
            value_states = torch.cat([past_key_value[1], value_states], dim=0)
        past_key_value = (key_states, value_states) if use_cache else None

        # if attention mask is None, use flashattn which support GQA
        if attention_mask is not None:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        dropout_rate = 0.0 if not self.training else self.attention_dropout
        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # after fused_apply_rotary_pos_emb, qk change to "bshd" for flashattn or "bhsd" for sdpa
        if attention_mask is None:  # flash-attn
            query_states = query_states.transpose(0, 1).contiguous()
            key_states = key_states.transpose(0, 1).contiguous()
        else:  # sdpa
            query_states = query_states.permute(1, 2, 0, 3).contiguous()
            key_states = key_states.permute(1, 2, 0, 3).contiguous()
            value_states = value_states.transpose(1, 2).contiguous()

        attn_output = self._attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            use_sliding_windows=use_sliding_windows,
        )

        attn_output = attn_output.reshape(
            bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

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
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        if attention_mask is not None:
            batch_size = query_states.shape[0]
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
                is_causal=causal,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

        return attn_output


class QwenAttention(BaseQwenAttention):
    def __init__(self) -> None:
        raise NotImplementedError(
            "LlamaAttention is not implemented as a physical class. "
            "It is meant to be used only with the from_native_module interface to Convert a native LlamaAttention module to LlamaAttention module provided above."
        )

    @staticmethod
    def from_native_module(module: nn.Module, *args, **kwargs) -> nn.Module:

        LazyInitContext.materialize(module)

        # try to get normalized_shape, eps, elementwise_affine from the module
        config = getattr(module, "config")
        layer_idx = getattr(module, "layer_idx", None)

        attention = BaseQwenAttention(
            config=config,
            layer_idx=layer_idx,
        )

        attention.qkv_proj.weight.data = torch.cat(
            (module.q_proj.weight.data, module.k_proj.weight.data, module.v_proj.weight.data), dim=0)
        attention.qkv_proj.bias.data = torch.cat(
            (module.q_proj.bias.data, module.k_proj.bias.data, module.v_proj.bias.data), dim=0)

        attention.o_proj.weight.data = module.o_proj.weight.data

        return attention
