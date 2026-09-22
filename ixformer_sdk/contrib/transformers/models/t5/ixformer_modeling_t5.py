import torch
from transformers.activations import NewGELUActivation

import ixformer


def self_attention_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_bias=None,
    layer_head_mask=None,
    past_key_value=None,
    use_cache=False,
    output_attentions=False,
):
    assert output_attentions is False
    assert layer_head_mask is None

    normed_hidden_states = self.layer_norm(hidden_states)

    if not hasattr(self, "qkv_weight"):
        self.qkv_weight = torch.cat(
            [
                self.SelfAttention.q.weight,
                self.SelfAttention.k.weight,
                self.SelfAttention.v.weight,
            ],
            dim=0,
        )
        self.qkv_bias = None

        del self.SelfAttention.q.weight
        del self.SelfAttention.k.weight
        del self.SelfAttention.v.weight

    batch_size, seq_length = hidden_states.shape[:2]
    real_seq_length = seq_length
    if past_key_value is not None:
        if len(past_key_value) != 2:
            raise ValueError(
                f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            )
        real_seq_length += past_key_value[0].shape[2]
    key_length = real_seq_length

    def unshape(states):
        """reshape"""
        return (
            states.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.SelfAttention.inner_dim)
        )

    qkv = ixformer.functions.linear(
        normed_hidden_states, self.qkv_weight, self.qkv_bias
    )

    if past_key_value is not None:
        pask_key, past_value = past_key_value
        (
            query_states,
            key_states,
            value_states,
        ) = ixformer.functions.t5_split_qkv_update_kv_cache(
            qkv,
            pask_key,
            past_value,
            self.SelfAttention.n_heads,
            self.SelfAttention.key_value_proj_dim,
        )
    else:
        query_states, key_states, value_states = ixformer.functions.t5_split_qkv(
            qkv, self.SelfAttention.n_heads, self.SelfAttention.key_value_proj_dim
        )

    if position_bias is None:
        if not self.SelfAttention.has_relative_attention_bias:
            position_bias = torch.zeros(
                (1, self.SelfAttention.n_heads, real_seq_length, key_length),
                device=query_states.device,
                dtype=query_states.dtype,
            )
        else:
            position_bias = self.SelfAttention.compute_bias(
                real_seq_length, key_length, device=query_states.device
            )

        # if key and values are already calculated
        # we want only the last query position bias
        if past_key_value is not None:
            position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

        if attention_mask is not None:
            # (batch_size, n_heads, seq_length, key_length)
            position_bias = position_bias + attention_mask

    if self.SelfAttention.pruned_heads:
        mask = torch.ones(position_bias.shape[1])
        mask[list(self.pruned_heads)] = 0
        position_bias_masked = position_bias[:, mask.bool()]
    else:
        position_bias_masked = position_bias

    attn_output = ixformer.functions.ixinfer_flash_attn_pad(
        query_states.contiguous(),
        key_states.contiguous(),
        value_states.contiguous(),
        mask=position_bias_masked.float().contiguous(),
        atten_scale=1,
    )
    attn_output = unshape(attn_output)

    attn_output = self.SelfAttention.o(attn_output)

    present_key_value_state = (
        (key_states, value_states)
        if (self.SelfAttention.is_decoder and use_cache)
        else None
    )
    outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

    if output_attentions:
        outputs = outputs + (None,)
    hidden_states = attn_output + hidden_states
    outputs = (hidden_states,) + outputs[1:]

    return outputs


def cross_attention_forward(
    self,
    hidden_states,
    key_value_states,
    attention_mask=None,
    position_bias=None,
    layer_head_mask=None,
    past_key_value=None,
    use_cache=False,
    query_length=None,
    output_attentions=False,
):

    assert output_attentions is False
    assert layer_head_mask is None

    def unshape(states):
        """reshape"""
        return (
            states.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.EncDecAttention.inner_dim)
        )

    normed_hidden_states = self.layer_norm(hidden_states)

    # cross attn need key_value_states
    assert key_value_states is not None
    batch_size, seq_length = hidden_states.shape[:2]
    real_seq_length = seq_length

    if past_key_value is not None:
        if len(past_key_value) != 2:
            raise ValueError(
                f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            )
        real_seq_length += (
            past_key_value[0].shape[2] if query_length is None else query_length
        )

    key_length = (
        real_seq_length if key_value_states is None else key_value_states.shape[1]
    )
    head_num, head_dim = (
        self.EncDecAttention.n_heads,
        self.EncDecAttention.key_value_proj_dim,
    )

    query_states = (
        self.EncDecAttention.q(normed_hidden_states)
        .view(batch_size, seq_length, head_num, head_dim)
        .transpose(1, 2)
        .contiguous()
    )

    if past_key_value is not None:
        if past_key_value[0].shape[2] != key_value_states.shape[1]:
            # checking that the `sequence_length` of the `past_key_value` is the same as
            # the provided `key_value_states` to support prefix tuning
            # cross-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            key_states = (
                self.EncDecAttention.k(key_value_states)
                .view(batch_size, key_length, head_num, head_dim)
                .transpose(1, 2)
            )
            value_states = (
                self.EncDecAttention.v(key_value_states)
                .view(batch_size, key_length, head_num, head_dim)
                .transpose(1, 2)
            )
        else:
            # cross-attn
            key_states = past_key_value[0]
            value_states = past_key_value[1]
    else:
        key_states = (
            self.EncDecAttention.k(key_value_states)
            .view(batch_size, key_length, head_num, head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.EncDecAttention.v(key_value_states)
            .view(batch_size, key_length, head_num, head_dim)
            .transpose(1, 2)
        )

    if not query_states.is_contiguous():
        query_states = query_states.contiguous()

    # TODO: fix this bug
    if not value_states.is_contiguous():
        new_value_states = query_states.new_empty(value_states.shape)
        new_value_states.copy_(value_states)
        value_states = new_value_states
    if not key_states.is_contiguous():
        numel = torch.numel(key_states)
        new_key_states = query_states.new_empty([numel * 2])[:numel].view(
            *list(key_states.shape)
        )
        new_key_states.copy_(key_states)
        key_states = new_key_states

    if position_bias is None:
        if not self.EncDecAttention.has_relative_attention_bias:
            position_bias = torch.zeros(
                (1, self.EncDecAttention.n_heads, real_seq_length, key_length),
                device=query_states.device,
                dtype=query_states.dtype,
            )
        else:
            position_bias = self.EncDecAttention.compute_bias(
                real_seq_length, key_length, device=query_states.device
            )

        # if key and values are already calculated
        # we want only the last query position bias
        if past_key_value is not None:
            position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

        if attention_mask is not None:
            # (batch_size, n_heads, seq_length, key_length)
            position_bias = position_bias + attention_mask

    if self.EncDecAttention.pruned_heads:
        mask = torch.ones(position_bias.shape[1])
        mask[list(self.pruned_heads)] = 0
        position_bias_masked = position_bias[:, mask.bool()]
    else:
        position_bias_masked = position_bias

    attn_output = ixformer.functions.ixinfer_flash_attn_pad(
        query_states,
        key_states.contiguous(),
        value_states.contiguous(),
        mask=position_bias_masked.float().contiguous(),
        atten_scale=1,
    )
    attn_output = unshape(attn_output)

    attn_output = self.EncDecAttention.o(attn_output)

    present_key_value_state = (
        (key_states, value_states)
        if (self.EncDecAttention.is_decoder and use_cache)
        else None
    )
    outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

    if output_attentions:
        outputs = outputs + (None,)
    hidden_states = attn_output + hidden_states
    outputs = (hidden_states,) + outputs[1:]

    return outputs


def dense_gated_act_dense_forward(self, hidden_states):
    if isinstance(self.act, NewGELUActivation):
        if not hasattr(self, "wi"):
            self.wi = torch.cat([self.wi_1.weight, self.wi_0.weight], dim=0)
            del self.wi_1
            del self.wi_0
        hidden_states = ixformer.functions.linear(hidden_states, self.wi, None)
        hidden_states = ixformer.functions.gelu_and_mul(hidden_states)
        hidden_states = ixformer.functions.linear(hidden_states, self.wo.weight, None)
    else:
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear

        hidden_states = self.dropout(hidden_states)

        # To make 8bit quantization work for google/flan-t5-xxl, self.wo is kept in float32.
        # See https://github.com/huggingface/transformers/issues/20287
        # we also make sure the weights are not in `int8` in case users will force `_keep_in_fp32_modules` to be `None``
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        hidden_states = self.wo(hidden_states)
    return hidden_states
