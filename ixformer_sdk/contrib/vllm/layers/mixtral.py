import functools
from typing import Dict, Optional, Tuple

import ixformer.inference.functions as ixf
import torch


def mixtral_decoder_layer_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
    residual: Optional[torch.Tensor],
) -> torch.Tensor:
    if self.use_int_w8a8:
        return w8a8_forward(
            self, positions, hidden_states, kv_cache, attn_metadata, residual
        )
    else:
        return original_forward(
            self, positions, hidden_states, kv_cache, attn_metadata, residual
        )


def original_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
    residual: Optional[torch.Tensor],
) -> torch.Tensor:
    # Self Attention
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    hidden_states = self.self_attn(
        positions=positions,
        hidden_states=hidden_states,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
    )

    # Fully Connected
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.block_sparse_moe(hidden_states)
    return hidden_states, residual


def dynamic_scaled_int8_quant(x):
    m, k = x.shape
    i8_x = x.new_empty([m, k], dtype=torch.int8, device="cuda")
    i8_scales = torch.empty([m], dtype=torch.float32, device="cuda")
    ixf.dynamic_scaled_int8_quant(i8_x, x, i8_scales)
    return i8_x, i8_scales


def dynamic_w8a8(x, i8_weight, weight_scale):
    i8_x, i8_scale = dynamic_scaled_int8_quant(x)
    m, k = x.shape
    k, n = i8_weight.shape
    output = x.new_empty([m, n], dtype=x.dtype, device="cuda")
    ixf.w8a8(
        i8_x,
        i8_weight.transpose(0, 1),
        i8_scale,
        weight_scale,
        output=output,
        out_dtype=x.dtype,
    )
    return output


def fused_rms_norm_quant_linear(
    self,
    hidden_states,
    ln_weight,
    eps,
    linear_weight,
    linear_weight_scale,
    residual=None,
):
    # lower rouge
    # if residual is None:
    #     residual = hidden_states
    #     i8_hidden_states, _, i8_scales = ixf.residual_rms_norm_dynamic_int8(
    #         input=hidden_states,
    #         weight=ln_weight,
    #         residual=None,
    #         eps=eps,
    #     )
    # else:
    #     i8_hidden_states, residual, i8_scales = ixf.residual_rms_norm_dynamic_int8(
    #         input=hidden_states,
    #         weight=ln_weight,
    #         residual=residual,
    #         eps=eps,
    #     )

    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    i8_hidden_states, i8_scales = dynamic_scaled_int8_quant(hidden_states)

    qkv = hidden_states.new_empty(hidden_states.shape[0], linear_weight.shape[1])
    ixf.w8a8(
        i8_hidden_states,
        linear_weight.transpose(0, 1),
        i8_scales,
        linear_weight_scale,
        output=qkv,
        out_dtype=hidden_states.dtype,
    )
    return qkv, residual


def attention(qkv, positions, kv_cache, attn_metadata, self_attn):
    q, k, v = qkv.split(
        [self_attn.q_size, self_attn.kv_size, self_attn.kv_size], dim=-1
    )
    q, k = self_attn.rotary_emb(positions, q, k)
    attn_output = self_attn.attn(q, k, v, kv_cache, attn_metadata)
    return attn_output


def fused_rms_norm_attention(
    self,
    hidden_states,
    ln_weight,
    eps,
    positions,
    kv_cache,
    attn_metadata,
    self_attn,
    residual=None,
):
    hidden_states, residual = fused_rms_norm_quant_linear(
        self,
        hidden_states,
        ln_weight,
        eps,
        self_attn.qkv_proj.weight,
        self_attn.qkv_proj.weight_scale,
        residual,
    )
    hidden_states = attention(
        hidden_states, positions, kv_cache, attn_metadata, self_attn
    )

    hidden_states = dynamic_w8a8(
        hidden_states, self_attn.o_proj.weight, self_attn.o_proj.weight_scale
    )
    # hidden_states,_ = self_attn.o_proj(hidden_states) # quant+linear+allreduce
    return hidden_states, residual


def w8a8_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
    residual: Optional[torch.Tensor],
) -> torch.Tensor:

    # qkv,_ = self.self_attn.qkv_proj(hidden_states)
    hidden_states, residual = fused_rms_norm_attention(
        self,
        hidden_states,
        self.input_layernorm.weight,
        self.input_layernorm.variance_epsilon,
        positions,
        kv_cache,
        attn_metadata,
        self.self_attn,
        residual,
    )

    # allreduce
    tp_size = self.block_sparse_moe.experts.tp_size
    if tp_size > 1:
        from vllm.distributed import tensor_model_parallel_all_reduce

        hidden_states = tensor_model_parallel_all_reduce(hidden_states)

    # rms norm
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    # moe
    hidden_states = fused_moe(
        hidden_states,
        self.block_sparse_moe.gate.weight,
        top_k=self.block_sparse_moe.experts.top_k,
        w1=self.block_sparse_moe.experts.w13_weight,
        w2=self.block_sparse_moe.experts.w2_weight,
        w1_scale=self.block_sparse_moe.experts.w13_weight_scale,
        w2_scale=self.block_sparse_moe.experts.w2_weight_scale,
    )

    # allreduce
    if tp_size > 1:
        from vllm.distributed import tensor_model_parallel_all_reduce

        hidden_states = tensor_model_parallel_all_reduce(hidden_states)

    return hidden_states, residual


def fused_experts(hidden_states, router_logits, top_k, w1, w2, w1_scale, w2_scale):

    """
    Args:
        hidden_states:          (num_tokens, k)                 dtype
        router_logits:          (num_tokens, num_experts)       torch.float32
        top_k                                                   int
        w1:                     (num_experts, 2n, k)            torch.int8
        w2:                     (num_experts, k, n)             torch.int8
        w1_scale:               (num_experts, 2n)               torch.float32
        w2_scale:               (num_experts, k)                torch.float32
    Returns
        final_hidden_states:    (num_tokens, k)                 dtype
    """

    # topk_weight: (num_tokens, top_k) torch.float32
    # topk_ids:    (num_tokens, top_k) torch.int32
    topk_weight, topk_ids = ixf.moe_topk_softmax(
        gating_output=router_logits,
        topk=top_k,
        renormalize=True,
    )

    dtype = hidden_states.dtype
    num_tokens, num_experts = router_logits.shape
    expand_tokens = num_tokens * top_k

    (
        src_to_dst,
        sorted_token_ids,
        expert_sizes_gpu,
        expert_sizes_cpu,
    ) = ixf.moe_compute_token_index(
        topk_ids=topk_ids,
        num_experts=num_experts,
    )
    expert_sizes_cpu = expert_sizes_gpu.cpu()

    # expand + reorder + quant
    # i8_hidden_states: (expand_tokens, k) torch.int8
    i8_hidden_states, a_scale = ixf.moe_expand_input_dynamic_scaled_int8(
        hidden_states=hidden_states,
        dst_to_src=sorted_token_ids,
        dst_tokens=expand_tokens,
        topk=top_k,
        src_to_dst=src_to_dst,
        topk_ids=None,  # use smooth quant
        smooth_scales=None,  # use smooth quant
    )

    # w8a8 group gemm 1
    # pt_output_1: (expand_tokens, 2n) dtype
    pt_output_1 = ixf.moe_w8a8_group_gemm(
        input=i8_hidden_states,
        weight=w1,
        i_scales=a_scale,
        w_scales=w1_scale,
        output_dtype=dtype,
        tokens_per_experts=expert_sizes_cpu,
        dst_to_src=None,
        format="TN",
    )

    # act + quant
    # pt_output_2: (expand_tokens, n) torch.int8
    pt_output_2, a2_scale = ixf.activation_dynamic_scaled_int8(
        input=pt_output_1,
        bias=None,  # add gemm bias
        smooth_scales=None,  # use smooth quant
        dst_to_src=sorted_token_ids,
        topk_ids=None,  # add gemm bias or use smooth quant
        act_type="swiglu",
    )

    # w8a8 group gemm 2 + reorder
    # pt_output_3: (expand_tokens, k) dtype
    pt_output_3 = ixf.moe_w8a8_group_gemm(
        input=pt_output_2,
        weight=w2,
        i_scales=a2_scale,
        w_scales=w2_scale,
        output_dtype=dtype,
        tokens_per_experts=expert_sizes_cpu,
        dst_to_src=sorted_token_ids,
        format="TN",
    )

    # mul + reduce_sum
    # final_hidden_states: (num_tokens, k)
    final_hidden_states = ixf.moe_output_reduce_sum(
        input=pt_output_3.view(num_tokens, top_k, -1),
        topk_weight=topk_weight,
    )

    return final_hidden_states


def fused_moe(hidden_states, gate_weight, top_k, w1, w2, w1_scale, w2_scale):
    orig_shape = hidden_states.shape
    hidden_size = hidden_states.shape[-1]

    hidden_states = hidden_states.view(-1, hidden_size)

    # router_logits: (num_tokens, n_experts)
    # gate_weight: fp16
    router_logits = ixf.linear(hidden_states, gate_weight)
    router_logits = router_logits.to(torch.float32)

    final_hidden_states = fused_experts(
        hidden_states,
        router_logits,
        top_k,
        w1,
        w2,
        w1_scale,
        w2_scale,
    )

    return final_hidden_states.view(orig_shape)
