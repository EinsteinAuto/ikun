import os
from typing import Optional, Tuple

import ixformer._C as ops
import torch

__all__ = [
    "ref_moe_output_reduce_sum",
    "moe_output_reduce_sum",
    "moe_expand_input",
    "ref_moe_expand_input",
    "moe_expand_input_dynamic_scaled_int8",
    "ref_moe_expand_input_dynamic_scaled_int8",
    "moe_compute_token_index",
    "moe_compute_token_index_ep",
    "ref_moe_compute_token_index_ep",
    "moe_topk_softmax",
    "ref_moe_topk_softmax",
    "moe_grouped_topk",
    "ref_moe_grouped_topk",
    "moe_align_token_index",
    "ref_moe_align_token_index",
    "ref_activation_dynamic_scaled_int8",
    "activation_dynamic_scaled_int8",
    "moe_w8a8_group_gemm",
    "ref_moe_w8a8_group_gemm",
    "moe_w4a8_group_gemm",
    "moe_w4a8_group_gemv",
    "ref_moe_w4a8_group_gemm",
    "quant_repack_int4",
    "moe_w4a16_group_gemm",
    "ref_moe_w4a16_group_gemm",
]


def ref_moe_output_reduce_sum(
    input: torch.Tensor,
    topk_weight: torch.Tensor = None,
    output: torch.Tensor = None,
    mask: torch.Tensor = None,
    extra_residual: torch.Tensor = None,
    scaling_factor: float = 1.0,
):
    if output is None:
        m, topk, k = input.shape
        output = torch.empty([m, k], dtype=input.dtype, device=input.device)
    temp = input.clone().to(torch.float32)
    if topk_weight is not None:
        temp *= topk_weight.unsqueeze(-1)
    if mask is not None:
        mask = mask.reshape(m, topk)
        mask_value = torch.where(mask, 0.0, 1.0)
        temp *= mask_value.unsqueeze(-1)

    temp = torch.sum(temp, dim=1)
    if extra_residual is not None:
        temp = temp * scaling_factor + extra_residual
    output.copy_(temp.to(input.dtype))
    return output


def moe_output_reduce_sum(
    input: torch.Tensor,
    topk_weight: torch.Tensor = None,
    output: torch.Tensor = None,
    mask: torch.Tensor = None,
    extra_residual: torch.Tensor = None,
    scaling_factor: float = 1.0,
):
    """
    Args:
        input:              (m, topk, k)   torch.float16, torch.bfloat16
        topk_weight:        (m, topk)      torch.float32
        mask:               (m * topk)     torch.bool
        extra_residual:     (m, k)         torch.float16, torch.bfloat16
        scaling_factor:                    float32
                            scaling factor for output
    Returns:
        output:             (m, k)         torch.float16, torch.bfloat16
    """
    if output is None:
        m, topk, k = input.shape
        output = torch.empty([m, k], dtype=input.dtype, device=input.device)
    ops.infer.moe_output_reduce_sum(
        output, input, topk_weight, mask, extra_residual, scaling_factor
    )
    return output


def ref_moe_expand_input(
    hidden_states: torch.Tensor,
    dst_to_src: torch.Tensor,
    dst_tokens: int,
    topk: int,
    src_to_dst: torch.Tensor = None,
    output: torch.Tensor = None,
):
    src_tokens, hidden_size = hidden_states.shape
    input_expand = (
        hidden_states.view(src_tokens, 1, hidden_size)
        .repeat(1, topk, 1)
        .reshape(-1, hidden_size)
    )
    if output is None:
        output = input_expand[dst_to_src]
    else:
        output.copy_(input_expand[dst_to_src])
    return output


def moe_expand_input(
    hidden_states: torch.Tensor,
    dst_to_src: torch.Tensor,
    dst_tokens: int,
    topk: int,
    src_to_dst: torch.Tensor = None,
    output: torch.Tensor = None,
):
    """
    Args:
        hidden_states:    (num_tokens, hidden_size)      torch.float16, torch.bfloat16
        dst_to_src:       (num_tokens*topk)              torch.int32
                          index of dst to src.
        dst_tokens:                                      int
                          the number of tokens after expansion.
        topk:                                            int
                          topk for moe
        src_to_dst:       (num_tokens*topk)              torch.int32
                          index of src to dst.
    Returns:
        output:           (dst_tokens, hidden_size)      hidden_states.dtype
    """
    src_tokens, hidden_size = hidden_states.shape
    if output is None:
        output = torch.empty(
            (dst_tokens, hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
    ops.infer.moe_expand_input(
        output,
        hidden_states,
        dst_to_src,
        src_to_dst,
        dst_tokens,
        topk,
    )
    return output


def ref_moe_expand_input_dynamic_scaled_int8(
    hidden_states: torch.Tensor,
    dst_to_src: torch.Tensor,
    dst_tokens: int,
    topk: int,
    src_to_dst: torch.Tensor = None,
    topk_ids: torch.Tensor = None,
    smooth_scales: torch.Tensor = None,
    i8_output: torch.Tensor = None,
    output_scales: torch.Tensor = None,
):
    import ixformer.functions as F

    src_tokens, hidden_size = hidden_states.shape
    expand_tokens = src_tokens * topk
    input_expand = (
        hidden_states.view(src_tokens, 1, hidden_size)
        .repeat(1, topk, 1)
        .reshape(-1, hidden_size)
    )

    if smooth_scales is not None and topk_ids is not None:
        input_expand = input_expand * smooth_scales[topk_ids.flatten()]
        input_expand = input_expand.to(hidden_states.dtype)

    intput_i8 = torch.empty(
        (expand_tokens, hidden_size), dtype=torch.int8, device=hidden_states.device
    )
    input_scales = torch.empty(
        expand_tokens, dtype=torch.float32, device=hidden_states.device
    )
    F.dynamic_scaled_int8_quant(intput_i8, input_expand, input_scales)

    if i8_output is None:
        i8_output = torch.zeros(
            (dst_tokens, hidden_size), dtype=torch.int8, device=hidden_states.device
        )
    if output_scales is None:
        output_scales = torch.zeros(
            dst_tokens, dtype=torch.float32, device=hidden_states.device
        )

    i8_output = intput_i8[dst_to_src]
    output_scales = input_scales[dst_to_src]

    return i8_output, output_scales


def moe_expand_input_dynamic_scaled_int8(
    hidden_states: torch.Tensor,
    dst_to_src: torch.Tensor,
    dst_tokens: int,
    topk: int,
    src_to_dst: torch.Tensor = None,
    topk_ids: torch.Tensor = None,
    smooth_scales: torch.Tensor = None,
    output_format: int = 0,
    i8_output: torch.Tensor = None,
    output_scales: torch.Tensor = None,
):
    """
    Args:
        hidden_states:    (num_tokens, hidden_size)      torch.float16, torch.bfloat16
        dst_to_src:       (num_tokens*topk)              torch.int32
                          index of dst to src.
        dst_tokens:                                      int
                          the number of tokens after expansion.
        topk:                                            int
                          topk for moe
        src_to_dst:       (num_tokens*topk)              torch.int32
                          index of src to dst.
        topk_ids:         (num_tokens, topk)             torch.int32
        smooth_scales:    (num_experts, hidden_size)     torch.float16, torch.bfloat16
        output_format:                                   int
                          specific output format for subsequent kernel
                              0 : origin output
                              1 : used for w4a8 group gemv
    Returns:
        i8_output:        (dst_tokens, hidden_size)      torch.int8
        output_scales:    (dst_tokens)                   torch.float32
    """
    hidden_size = hidden_states.shape[-1]
    if i8_output is None:
        i8_output = torch.empty(
            (dst_tokens, hidden_size), dtype=torch.int8, device=hidden_states.device
        )
    if output_scales is None:
        output_scales = torch.empty(
            dst_tokens, dtype=torch.float32, device=hidden_states.device
        )
    ops.infer.moe_expand_input_dynamic_scaled_int8(
        i8_output,
        output_scales,
        hidden_states.view(-1, hidden_size),
        dst_to_src,
        src_to_dst,
        topk_ids,
        smooth_scales,
        dst_tokens,
        topk,
        output_format,
    )

    return i8_output, output_scales


def moe_compute_token_index(
    topk_ids: torch.Tensor,
    num_experts: int,
    src_dst: torch.Tensor = None,
    dst_src: torch.Tensor = None,
    expert_sizes_gpu: torch.Tensor = None,
    expert_sizes_cpu: torch.Tensor = None,
):
    """
    Args:
        topk_ids:           (num_tokens, topk)              torch.int32
        num_experts:                                        int
    Returns:
        src_dst:            (num_tokens*topk)               torch.int32
                            index of src to dst, e.g.  src_tensor[i] = dst_tensor[src_dst[i]]
        dst_src:            (num_tokens*topk)               torch.int32
                            index of dst to src.
        expert_sizes_gpu:   (num_experts)                   torch.int32
                            the number of tokens allocated to each expert. (num_experts)
        expert_sizes_cpu:   (num_experts)                   torch.int32
                            the number of tokens allocated to each expert. (num_experts)
    """
    if src_dst is None:
        src_dst = topk_ids.new_empty([topk_ids.numel()])
    if dst_src is None:
        dst_src = torch.empty_like(src_dst)
    if expert_sizes_gpu is None:
        expert_sizes_gpu = topk_ids.new_empty([num_experts])

    ops.infer.moe_compute_token_index(
        topk_ids,
        src_dst,
        dst_src,
        expert_sizes_gpu,
        expert_sizes_cpu,
        None,
        0,
        num_experts,
        num_experts,
    )

    return src_dst, dst_src, expert_sizes_gpu, expert_sizes_cpu


def ref_moe_topk_softmax(
    gating_output: torch.Tensor,
    topk: int,
    topk_weight: torch.Tensor = None,
    topk_ids: torch.Tensor = None,
    renormalize: bool = True,
):
    score = torch.softmax(gating_output, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    if renormalize:
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
    return topk_weight, topk_ids.int()


def moe_topk_softmax(
    gating_output: torch.Tensor,
    topk: int,
    topk_weight: torch.Tensor = None,
    topk_ids: torch.Tensor = None,
    renormalize: bool = True,
):
    """
    Args:
        gating_output:    (num_tokens, num_experts)    torch.float32
        topk:                                          int
        renormalize:                                   bool
    Returns:
        topk_weight:      (num_tokens, topk)           torch.float32
        topk_ids:         (num_tokens, topk)           torch.int32
    """
    num_tokens, num_experts = gating_output.shape
    device = gating_output.device
    if topk_weight is None:
        topk_weight = torch.empty(
            [num_tokens, topk], dtype=torch.float32, device=device
        )
    if topk_ids is None:
        topk_ids = torch.empty([num_tokens, topk], dtype=torch.int32, device=device)
    token_expert_indicies = torch.empty(
        [num_tokens, topk], dtype=torch.int32, device="cuda"
    )  # not use

    ops.infer.moe_topk_softmax(
        topk_weight, topk_ids, token_expert_indicies, gating_output, renormalize
    )
    return topk_weight, topk_ids


def ref_moe_grouped_topk(
    gating_output: torch.Tensor,
    topk: int,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None,
    renormalize: bool = True,
):

    gating_output = gating_output.to(torch.float32)
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    if e_score_correction_bias is not None:
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)

    num_token = scores.shape[0]
    group_scores = (
        scores.view(num_token, num_expert_group, -1).max(dim=-1).values
    )  # [n, n_group]

    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]

    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def moe_grouped_topk(
    gating_output: torch.Tensor,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    scoring_func: str = "softmax",
    e_score_correction_bias: torch.Tensor = None,
    topk_weight: torch.Tensor = None,
    topk_ids: torch.Tensor = None,
    renormalize: bool = True,
):
    """
    Args:
        gating_output:    (num_tokens, num_experts)    torch.float32, torch.bfloat16
        topk:                                          int
        num_expert_group:                              int
        topk_group:                                    int
        scoring_func:                                  str
        e_score_correction_bias: (num_experts)         torch.float16, torch.bfloat16
        renormalize:                                   bool
    Returns:
        topk_weight:      (num_tokens, topk)           torch.float32
        topk_ids:         (num_tokens, topk)           torch.int32 torch.int64
    """
    num_tokens, num_experts = gating_output.shape
    device = gating_output.device
    if topk_weight is None:
        topk_weight = torch.empty(
            [num_tokens, topk], dtype=torch.float32, device=device
        )
    if topk_ids is None:
        topk_ids = torch.empty([num_tokens, topk], dtype=torch.int32, device=device)

    ops.infer.moe_grouped_topk(
        topk_weight,
        topk_ids,
        gating_output,
        e_score_correction_bias,
        num_expert_group,
        topk_group,
        scoring_func,
        renormalize,
    )
    return topk_weight, topk_ids


def ref_moe_align_token_index(
    topk_ids: torch.Tensor,
    num_experts: int,
    src_to_dst: torch.Tensor = None,
    dst_to_src: torch.Tensor = None,
    expert_sizes_gpu: torch.Tensor = None,
    expert_sizes_cpu: torch.Tensor = None,
):
    src_to_dst = []
    dst_to_src = [-1 for _ in range(topk_ids.numel())]
    expert_sizes_gpu = torch.empty([num_experts], dtype=torch.int32, device="cuda")

    for i in range(num_experts):
        expert_sizes_gpu[i] = (topk_ids == i).sum()

    expert_sizes_gpu_cu = torch.zeros(
        [num_experts + 1], dtype=torch.int32, device="cuda"
    )
    expert_sizes_gpu_cu[1:] = expert_sizes_gpu

    expert_sizes_gpu_cu = expert_sizes_gpu_cu.cumsum(dim=-1).cpu().tolist()

    topk_ids = topk_ids.view(-1).cpu().tolist()
    for i, expert_id in enumerate(topk_ids):
        dst_idx = expert_sizes_gpu_cu[expert_id]
        expert_sizes_gpu_cu[expert_id] += 1
        src_to_dst.append(dst_idx)
        dst_to_src[dst_idx] = i

    src_to_dst = torch.tensor(src_to_dst, dtype=torch.int32, device="cuda")
    dst_to_src = torch.tensor(dst_to_src, dtype=torch.int32, device="cuda")
    expert_sizes_cpu = expert_sizes_gpu.cpu()

    return src_to_dst, dst_to_src, expert_sizes_gpu, expert_sizes_cpu


def moe_align_token_index(
    topk_ids: torch.Tensor,
    num_experts: int,
    src_to_dst: torch.Tensor = None,
    dst_to_src: torch.Tensor = None,
    expert_sizes_gpu: torch.Tensor = None,
    expert_sizes_cpu: torch.Tensor = None,
):
    """
    Args:
        topk_ids:            (num_tokens, topk)           torch.int32
        num_experts:                                      int
    Returns:
        src_to_dst:          (num_tokens*topk)            torch.int32
                             index of src to dst, e.g.  src_tensor[i] = dst_tensor[src_to_dst[i]]
        dst_to_src:          (num_tokens*topk)            torch.int32
                             index of dst to src.
        expert_sizes_gpu:    (num_experts)                torch.int32
                             the number of tokens allocated to each expert. (num_experts)
        expert_sizes_cpu:    (num_experts)                torch.int32
                             the number of tokens allocated to each expert. (num_experts)
    """
    if src_to_dst is None:
        src_to_dst = topk_ids.new_empty([topk_ids.numel()])
    if dst_to_src is None:
        dst_to_src = torch.empty_like(src_to_dst)
    if expert_sizes_gpu is None:
        expert_sizes_gpu = topk_ids.new_empty([num_experts])

    ops.infer.moe_compute_token_index(
        topk_ids,
        src_to_dst,
        dst_to_src,
        expert_sizes_gpu,
        expert_sizes_cpu,
        None,
        0,
        num_experts,
        num_experts,
    )

    if expert_sizes_cpu is None:
        expert_sizes_cpu = expert_sizes_gpu.detach().cpu()

    return src_to_dst, dst_to_src, expert_sizes_gpu, expert_sizes_cpu


def ref_moe_compute_token_index_ep(
    topk_ids: torch.Tensor,
    num_experts: int,
    start_expert_id: int,
    end_expert_id: int,
    src_to_dst: torch.Tensor = None,
    dst_to_src: torch.Tensor = None,
    expert_sizes_gpu: torch.Tensor = None,
    expert_sizes_cpu: torch.Tensor = None,
):
    vaild_num_experts = end_expert_id - start_expert_id
    expert_sizes_gpu = torch.empty(
        [vaild_num_experts], dtype=torch.int32, device="cuda"
    )

    for i in range(vaild_num_experts):
        expert_sizes_gpu[i] = (topk_ids == (i + start_expert_id)).sum()

    expert_sizes_gpu_cu = torch.zeros(
        [vaild_num_experts + 1], dtype=torch.int32, device="cuda"
    )
    expert_sizes_gpu_cu[1:] = expert_sizes_gpu

    expert_sizes_gpu_cu = expert_sizes_gpu_cu.cumsum(dim=-1).cpu().tolist()
    expand_tokens = expert_sizes_gpu_cu[-1]
    topk_ids = topk_ids.view(-1).cpu().tolist()
    src_to_dst = []
    dst_to_src = [-1 for _ in range(expand_tokens)]
    for i, expert_id in enumerate(topk_ids):
        if expert_id >= start_expert_id and expert_id < end_expert_id:
            eid = expert_id - start_expert_id
            dst_idx = expert_sizes_gpu_cu[eid]
            expert_sizes_gpu_cu[eid] += 1
            src_to_dst.append(dst_idx)
            dst_to_src[dst_idx] = i
        else:
            src_to_dst.append(-1)
    src_to_dst = torch.tensor(src_to_dst, dtype=torch.int32, device="cuda")
    dst_to_src = torch.tensor(dst_to_src, dtype=torch.int32, device="cuda")
    expert_sizes_cpu = expert_sizes_gpu.cpu()

    return src_to_dst, dst_to_src, expert_sizes_gpu, expert_sizes_cpu, expand_tokens


def moe_compute_token_index_ep(
    topk_ids: torch.Tensor,
    num_experts: int,
    start_expert_id: int,
    end_expert_id: int,
    src_to_dst: torch.Tensor = None,
    dst_to_src: torch.Tensor = None,
    expert_sizes_gpu: torch.Tensor = None,
    expert_sizes_cpu: torch.Tensor = None,
):
    """
    Args:
        topk_ids:            (num_tokens, topk)                 torch.int32
        num_experts:                                            int
                             the number of tokens overall
        start_expert_id                                         int
                             start expert id of the vaild expert interval
        end_expert_id                                           int
                             end expert id of the vaild expert interval
    Returns:
        src_to_dst:          (num_tokens*topk)                  torch.int32
                             index of src to dst, e.g.  src_tensor[i] = dst_tensor[src_to_dst[i]]
        dst_to_src:          (expand_tokens_ep)                 torch.int32
                             index of dst to src.
        expert_sizes_gpu:    (expand_tokens_ep)                 torch.int32
                             the number of tokens allocated to each expert. (num_experts)
        expert_sizes_cpu:    (expand_tokens_ep)                 torch.int32
                             the number of tokens allocated to each expert. (num_experts)
        expand_tokens        the number of tokens which expert id in [start_expert_id, end_expert_id)
                                                                int
    """
    vaild_num_experts = end_expert_id - start_expert_id
    if src_to_dst is None:
        src_to_dst = topk_ids.new_empty([topk_ids.numel()])
    if dst_to_src is None:
        dst_to_src = torch.empty_like(src_to_dst)
    if expert_sizes_gpu is None:
        expert_sizes_gpu = topk_ids.new_empty([vaild_num_experts])
    expand_tokens_gpu = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    ops.infer.moe_compute_token_index(
        topk_ids,
        src_to_dst,
        dst_to_src,
        expert_sizes_gpu,
        expert_sizes_cpu,
        expand_tokens_gpu,
        start_expert_id,
        end_expert_id,
        num_experts,
    )

    if expert_sizes_cpu is None:
        expert_sizes_cpu = expert_sizes_gpu.detach().cpu()
    expand_tokens = expand_tokens_gpu.cpu().item()

    return (
        src_to_dst,
        dst_to_src[:expand_tokens],
        expert_sizes_gpu,
        expert_sizes_cpu,
        expand_tokens,
    )


def ref_activation_dynamic_scaled_int8(
    input: torch.Tensor,
    bias: torch.Tensor = None,
    smooth_scales: torch.Tensor = None,
    dst_to_src: torch.Tensor = None,
    topk_ids: torch.Tensor = None,
    act_type: str = "silu",
    i8_output: torch.Tensor = None,
    output_scales: torch.Tensor = None,
):
    if i8_output is None:
        output_shape = (
            input.shape[:-1] + (input.shape[-1] // 2,)
            if act_type == "swiglu"
            else input.shape
        )
        i8_output = torch.empty(output_shape, dtype=torch.int8, device=input.device)
    if output_scales is None:
        output_scales = torch.empty(
            input.shape[:-1], dtype=torch.float32, device=input.device
        )

    temp = input.clone().to(torch.float32)  # m, k

    # Add bias
    if bias is not None:
        if topk_ids is not None and dst_to_src is not None:
            # bias (num_experts, k)
            temp += bias[topk_ids.flatten()[dst_to_src]]
        else:
            # bias (k)
            temp += bias.view(1, -1)

    # Activation
    if act_type == "silu":
        temp = torch.nn.functional.silu(temp)
    elif act_type == "gelu":
        temp = torch.nn.functional.gelu(temp)
    elif act_type == "swiglu":
        x1, x2 = temp.chunk(chunks=2, dim=-1)
        temp = torch.nn.functional.silu(x1) * x2

    # Quant
    if smooth_scales is not None:
        assert len(smooth_scales.shape) <= 2
        # Multi smooth scale
        if len(smooth_scales.shape) == 2:
            temp *= smooth_scales[topk_ids.flatten()[dst_to_src]]
        else:
            temp *= smooth_scales.view(1, -1)

    amax_, _ = torch.max(torch.abs(temp), dim=-1)
    output_scales.copy_(amax_ / 127.0)
    output = temp / output_scales.view(-1, 1)
    output = torch.clamp(torch.round(output), -127, 127).to(torch.int8)
    i8_output.copy_(output)

    return i8_output, output_scales


def activation_dynamic_scaled_int8(
    input: torch.Tensor,
    bias: torch.Tensor = None,
    smooth_scales: torch.Tensor = None,
    dst_to_src: torch.Tensor = None,
    topk_ids: torch.Tensor = None,
    act_type: str = "silu",
    output_format: int = 0,
    i8_output: torch.Tensor = None,
    output_scales: torch.Tensor = None,
):
    """
    Args:
        input:              (m, k)                                     torch.float16, torch.bfloat16
        bias:               (num_experts, k)                           torch.float32
        smooth_scales:      (num_experts, k) or (num_experts, k//2)    torch.float16, torch.bfloat16
                            if act_type==swiglu, shape=(num_experts, k//2)
        dst_to_src:         (m)                                        torch.int32
                            index of dst to src.
        topk_ids:           (m)                                        torch.int32
        act_type:                                                      str activation type.
                            Options include gelu, silu, and swiglu.
        output_format:                                                 int
                            specific output format for subsequent kernel
                                0 : origin output
                                1 : used for w4a8 group gemv
    Returns:
        i8_output:          (m, k) or (m, k//2)                        torch.int8
                            if act_type==swiglu, shape=(m, k//2)
        output_scales:      (m)                                        torch.float32
    """
    if i8_output is None:
        output_shape = (
            input.shape[:-1] + (input.shape[-1] // 2,)
            if act_type == "swiglu"
            else input.shape
        )
        i8_output = torch.empty(output_shape, dtype=torch.int8, device=input.device)
    if output_scales is None:
        output_scales = torch.empty(
            input.shape[:-1], dtype=torch.float32, device=input.device
        )
    ops.infer.activation_dynamic_scaled_int8(
        i8_output,
        output_scales,
        input,
        smooth_scales,
        dst_to_src,
        topk_ids,
        act_type,
        bias,
        output_format,
    )

    return i8_output, output_scales


def ref_moe_w8a8_group_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    i_scales: torch.Tensor,
    w_scales: torch.Tensor,
    output_dtype: torch.dtype,
    tokens_per_experts: torch.Tensor,
    dst_to_src: torch.Tensor = None,
    format: str = "TN",
    output: torch.Tensor = None,
    group_size=-1,
):
    def get_align_size(format):
        input_format = format[1]
        return 1 if input_format == "N" else 64

    if output is None:
        if output_dtype is None:
            raise RuntimeError(
                "ref_moe_w8a8_group_gemm need output_dtype argument when output is none."
            )
        m = tokens_per_experts.sum()
        output = torch.empty(
            (m, w_scales.shape[1]),
            dtype=output_dtype,
            device=input.device,
        )
    assert format in ["NN", "TN", "TT", "NT"]
    prefix = 0
    out_prefix = 0
    align_size = get_align_size(format)
    for eid, n in enumerate(tokens_per_experts):
        start, end = prefix, prefix + n
        out_start, out_end = out_prefix, out_prefix + n
        cur_inputs = (
            input[start:end] if format[1] == "N" else input[:, start:end].T.contiguous()
        )
        cur_scales_i = i_scales[start:end].view(-1, 1)
        cur_weights = weight[eid] if format[0] == "T" else weight[eid].T.contiguous()
        cur_scales_w = w_scales[eid].view(1, -1)
        input_f32 = cur_inputs.to(torch.float32)
        weight_f32 = cur_weights.to(torch.float32)
        if group_size != -1:
            w_shape = weight_f32.shape
            weight_f32 = weight_f32.view(-1, group_size)
            weight_f32 = weight_f32 * cur_scales_w.view(-1, 1)
            weight_f32 = weight_f32.view(w_shape)
            output[out_start:out_end] = (
                torch.nn.functional.linear(input_f32, weight_f32) * cur_scales_i
            )
        else:
            output[out_start:out_end] = (
                torch.nn.functional.linear(input_f32, weight_f32)
                * cur_scales_i
                * cur_scales_w
            )
        prefix += (n + align_size - 1) // align_size * align_size
        out_prefix += n
    if dst_to_src is not None:
        tmp = output.clone()
        tmp[dst_to_src] = output
        output[:] = tmp[:]
    return output


def moe_w8a8_group_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    i_scales: torch.Tensor,
    w_scales: torch.Tensor,
    output_dtype: torch.dtype,
    tokens_per_experts: torch.Tensor,
    dst_to_src: torch.Tensor = None,
    format: str = "TN",
    output: torch.Tensor = None,
):
    """
    Args:
       input:              (m, k) if format[1]=="N" else (k, m)                          torch.int8
       weight:             (n_experts, n, k) if format[0]=="T" else (n_experts, k, n)    torch.int8
       i_scales:           (m)                                                           torch.float32
       w_scales:           (n_experts, n)                                                torch.float32
       output_dtype:                                                                     torch.dtype
                           support torch.float16 or torch.bfloat16
       tokens_per_experts: (n_experts)                                                   torch.int32
       dst_to_src:         (m)                                                           torch.int32
                           index of dst to src.
       format:                                                                           str
                           format of input and weight
    Returns:
       output:             (sum(tokens_per_experts), n)                                  output_dtype
                           input and i_scales may be padding when NT format
    """
    m = tokens_per_experts.sum()
    if output is None:
        if output_dtype is None:
            raise RuntimeError(
                "moe_w8a8_group_gemm need output_dtype argument when output is none."
            )
        output = torch.empty(
            (m, w_scales.shape[1]),
            dtype=output_dtype,
            device=input.device,
        )

    ops.infer.moe_w8a8_group_gemm(
        output,
        input,
        weight,
        i_scales,
        w_scales,
        tokens_per_experts,
        dst_to_src,
        format,
        0,
        m,
    )
    return output


def quant_repack_int4(x, group_size, version, format, isAsymQuant: bool = False):
    n_experts, n, k = x.shape
    if version == 1:
        assert not isAsymQuant

        if group_size == -1:
            max_x, _ = torch.max(torch.abs(x), dim=-1, keepdim=True)
            scales = torch.round(max_x / 7)
            scales[scales < 1e-6] = 1
            out = torch.round(x / scales).clamp(-8, 7).to(torch.int8)
        else:
            x = x.view(n_experts, -1, group_size)
            max_x, _ = torch.max(torch.abs(x), dim=-1, keepdim=True)
            scales = torch.round(max_x / 7)
            scales[scales < 1e-6] = 1
            out = torch.round(x / scales).clamp(-8, 7).to(torch.int8)

        out = out.view(n_experts, n, k)

        if format[0] == "N":
            out = out.transpose(-2, -1).contiguous()  # NT (num_experts, k , n)
            out = out.reshape(n_experts, k // 32, 2, 16, n // 32, 2, 16)
            out = out.view(n_experts, k // 32, 2, 16, n // 32, 2, 16)
            out = out.permute(0, 1, 5, 3, 4, 2, 6).contiguous().view(n_experts, k, n)

        ## rearange 32 token
        shape = out.shape
        out = out.view(shape[0], shape[1], shape[-1] // 32, 32)
        out_tmp = out.new_empty(shape[0], shape[1], shape[-1] // 32, 16)
        for i in range(16):
            sign_low_4bit = (out[:, :, :, i] < 0).to(torch.int8)
            low_4bit = sign_low_4bit * 8 + (out[:, :, :, i] & 0x07)
            high_4bit = out[:, :, :, i + 16] << 4
            out_tmp[:, :, :, i] = high_4bit + low_4bit
        out = out_tmp.view(shape[0], shape[1], shape[-1] // 2).contiguous()

        scales = (
            scales.view(n_experts, n, k // group_size).permute(0, 2, 1).contiguous()
            if group_size != -1
            else scales.view(n_experts, n)
        )

        return out, scales, None

    if version == 2:
        """
        For group_size == -1 (per-channel), the default scale factor is 18 since
        127 / 7 = 18, for quantization with clip, the scale can be set to 16, 17, etc.
        the alpha in ixinfer_gemm_helper need to be set to scale / 16.0, and the ixformer
        need to be rebuilt.
        """
        if group_size == -1:
            out = torch.round(x / 18).clamp(-8, 7).to(torch.int8)
        else:
            x = x.view(n_experts, -1, group_size)
            if isAsymQuant:
                max_x, _ = torch.max(x, dim=-1, keepdim=True)
                min_x, _ = torch.min(x, dim=-1, keepdim=True)
                scales = ((max_x.to(torch.float32) - min_x.to(torch.float32)) / 15).to(
                    torch.int8
                )
                zeros = (-min_x / scales - 8).to(
                    torch.int8
                )  # weight use int4 not uint4, and zero use int8
                out = (x / scales + zeros).clamp(-8, 7).to(torch.int8)
            else:
                max_x, _ = torch.max(torch.abs(x), dim=-1, keepdim=True)
                scales = torch.round(max_x / 7)
                scales[scales < 1e-6] = 1
                scales = scales.to(torch.int8)
                out = torch.round(x / scales).clamp(-8, 7).to(torch.int8)
            out = out.view(n_experts, n, k).contiguous()

        if format[0] == "N":
            out = out.transpose(-2, -1).contiguous()  # NT (num_experts, k , n)
            out = out.reshape(n_experts, k // 32, 2, 16, n // 32, 2, 16)
            out = out.view(n_experts, k // 32, 2, 16, n // 32, 2, 16)
            out = out.permute(0, 1, 5, 3, 4, 2, 6).contiguous().view(n_experts, k, n)

        ## rearange 32 token
        shape = out.shape
        out = out.view(shape[0], shape[1], shape[-1] // 32, 32)
        out_tmp = out.new_empty(shape[0], shape[1], shape[-1] // 32, 16)
        for i in range(16):
            sign_low_4bit = (out[:, :, :, i] < 0).to(torch.int8)
            low_4bit = sign_low_4bit * 8 + (out[:, :, :, i] & 0x07)
            high_4bit = out[:, :, :, i + 16] << 4
            out_tmp[:, :, :, i] = high_4bit + low_4bit
        out = out_tmp.view(shape[0], shape[1], shape[-1] // 2).contiguous()

        if group_size == -1:
            return out, None, None

        scales = scales.to(torch.uint8)
        scales_4i8pack = scales.clone().to(torch.int32)
        for i in range(3):
            scales_4i8pack <<= 8
            scales_4i8pack |= scales
        scales_4i8pack = (
            scales_4i8pack.view(n_experts, n, k // group_size)
            .permute(0, 2, 1)
            .contiguous()
        )

        if not isAsymQuant:
            return out, scales_4i8pack, None

        zeros = zeros.to(torch.uint8)
        zeros_4i8pack = zeros.clone().to(torch.int32)
        for i in range(3):
            zeros_4i8pack <<= 8
            zeros_4i8pack |= zeros
        zeros_4i8pack = (
            zeros_4i8pack.view(n_experts, n, k // group_size)
            .permute(0, 2, 1)
            .contiguous()
        )

        return out, scales_4i8pack, zeros_4i8pack


def _dequant_weight_int8(tensor, i8scales, i8zeros, group_size, version, format):
    """
    format == TN
        tensor: (num_experts, n, k/2)
        scales: (num_experts, n) if group_size == -1 else (num_experts, k // group_size, n)
    format == NT or NN
        tensor: (num_experts, k, n/2)
        scales: (num_experts, n) if group_size == -1 else (num_experts, k // group_size, n)
    output tensor is always k-major
    """
    dtype = torch.int8

    left = (tensor & 0xF0) >> 4
    right = tensor & 0x0F
    sign_bit = (tensor >> 3) & 1
    right = (right - (sign_bit * 16)).clamp(-8, 7)
    left, right = right, left

    shape = list(left.shape)
    left = left.reshape(
        shape[:-1] + [shape[-1] // 16, 16]
    )  # TN (num_experts, n, k/2/16, 16)
    right = right.reshape(
        shape[:-1] + [shape[-1] // 16, 16]
    )  # TN (num_experts, n, k/2/16, 16)
    ret = torch.cat((left, right), dim=-1)  # TN (num_experts, n, k/2/16, 32)
    ret = ret.reshape(
        shape[:-1] + [shape[-1] * 2]
    )  # TN (num_experts, n, k); NT (num_experts, k, n)

    ## NT 需要再次转换
    if format[0] == "T":
        n_experts, n, k = ret.shape
    else:
        n_experts, k, n = ret.shape
    if format[0] == "N":
        ret = ret.view(n_experts, k // 32, 2, 16, n // 32, 2, 16)
        ret = ret.permute(0, 1, 5, 3, 4, 2, 6).contiguous().view(n_experts, k, n)
        ret = ret.transpose(-2, -1).contiguous()  # (num_experts, n, k)

    ret_shape = ret.size()
    ret = ret.view(-1, group_size) if group_size != -1 else ret.view(-1, ret.shape[-1])

    if version == 2:
        if group_size == -1:
            ret *= 18  # same with quant_repack_int4
        else:
            scales = i8scales.to(torch.int8).transpose(-2, -1).contiguous().view(-1, 1)
            if i8zeros is not None:
                zeros = (
                    i8zeros.to(torch.int8).transpose(-2, -1).contiguous().view(-1, 1)
                )
                ret -= zeros
            ret = scales * ret

    ret = ret.reshape(ret_shape).to(dtype)
    return ret


def ref_moe_w4a8_group_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    i_scales: torch.Tensor,
    w_scales: torch.Tensor,
    output_dtype: torch.dtype,
    tokens_per_experts: torch.Tensor,
    w_i8scales: torch.Tensor = None,
    w_i8zeros: torch.Tensor = None,
    dst_to_src: torch.Tensor = None,
    format: int = 0,
    version: int = 2,
    group_size: int = -1,
    persistent: int = 0,
    output: torch.Tensor = None,
):
    assert format in ["NN", "NT", "TN"], f"w4a8 group gemm only support NN, NT, TN"

    weight_i8 = _dequant_weight_int8(
        weight, w_i8scales, w_i8zeros, group_size, version, format
    )

    if format[0] == "N":
        weight_i8 = weight_i8.permute(0, 2, 1).contiguous()
    if version == 1 and group_size != -1:
        w_scales = w_scales.permute(0, 2, 1).contiguous()

    output = ref_moe_w8a8_group_gemm(
        input,
        weight_i8,
        i_scales,
        w_scales,
        output_dtype,
        tokens_per_experts,
        dst_to_src,
        format,
        output,
        group_size=-1 if version == 2 else group_size,
    )
    return output


def moe_w4a8_group_gemv(
    input: torch.Tensor,
    weight: torch.Tensor,
    i_scales: torch.Tensor,
    w_scales: torch.Tensor,
    output_dtype: torch.dtype,
    tokens_per_experts: torch.Tensor,
    w_i8scales: torch.Tensor = None,
    w_i8zeros: torch.Tensor = None,
    dst_to_src: torch.Tensor = None,
    format: int = 0,
    group_size: int = -1,
    persistent: int = 0,
    output: torch.Tensor = None,
):
    """
    Args:
       input:              (m, k)                                                                       torch.int8
       weight:             (n_experts, n, k//2) if format & 0b10 else (n_experts, k, n//2)              torch.int8
       i_scales:           (m)                                                                          torch.float32
       w_scales:           (n_experts, n) if group_size == -1 else (n_experts, k // group_size, n)      torch.float32
       output_dtype:                                                                                    torch.float16, torch.bfloat16
       tokens_per_experts: (n_experts)                                                                  torch.int32
       w_i8scales          (n_experts, k // group_size, n) if gorup_size != -1 else (n_experts, n)      torch.int32
       w_i8zeros           (n_experts, k // group_size, n) if gorup_size != -1 else (n_experts, n)      torch.int32
       dst_to_src:         (sum(tokens_per_experts))                                                    torch.int32
       format:             [0(0b00, NN), 1(0b01, NT), 2(0b10, TN), 3(0b11, TT)]                         int
       group_size          version1: NN/NT:[-1,256,320,512], TN:[-1,256,512],
                           version2: NN:[-1,64], NT:[-1],    TN:[-1]                                    int
    Returns:
       output:             (m, n)                                                 output_dtype
    """
    assert format in [2], f"w4a8 group gemv only support 2(TN)"

    # unsupported EP
    # outout_m = tokens_per_experts.sum()

    outout_m = input.size(0)
    if output is None:
        assert output_dtype is not None, print(
            "moe_w4a8_group_gemv need output_dtype argument when output is none."
        )
        output = torch.empty(
            (outout_m, w_scales.shape[-1]),
            dtype=output_dtype,
            device=input.device,
        )

    ops.infer.moe_w4a8_group_gemv(
        output,
        input,
        weight,
        i_scales,
        w_scales,
        tokens_per_experts,
        w_i8scales,
        w_i8zeros,
        dst_to_src,
        format,
        group_size,
        persistent,
        outout_m,
    )
    return output


def moe_w4a8_group_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    i_scales: torch.Tensor,
    w_scales: torch.Tensor,
    output_dtype: torch.dtype,
    tokens_per_experts: torch.Tensor,
    w_i8scales: torch.Tensor = None,
    w_i8zeros: torch.Tensor = None,
    dst_to_src: torch.Tensor = None,
    format: int = 0,
    version: int = 2,
    group_size: int = -1,
    persistent: int = 0,
    output: torch.Tensor = None,
):
    """
    ColumnMajor matrix :(m, k) @ (k, n) -> (m, n)
    RowMajof matrix    :(k, m) @ (n, k) -> (n, m)
    C = A(weight) @ B(input)
    Args:
       input:              (n, k)                                                                       torch.int8
       weight:             (n_experts, m, k//2) if format & 0b10 else (n_experts, k, m//2)              torch.int8
       i_scales:           (n)                                                                          torch.float32
       w_scales:           (n_experts, m) if group_size == -1 else (n_experts, k // group_size, m)      torch.float32
       output_dtype:                                                                                    torch.float16, torch.bfloat16
       tokens_per_experts: (n_experts)                                                                  torch.int32
       w_i8scales          (n_experts, k // group_size, m) if gorup_size != -1 else (n_experts, m)      torch.int32
       w_i8zeros           (n_experts, k // group_size, m) if gorup_size != -1 else (n_experts, m)      torch.int32
       dst_to_src:         (sum(tokens_per_experts))                                                    torch.int32
       format:             [0(0b00, NN), 1(0b01, NT), 2(0b10, TN), 3(0b11, TT)]                         int
       version:            [1, 2]                                                                       int
       group_size          version1: NN/NT:[-1,256,320,512], TN:[-1,256,512],
                           version2: NN:[-1,64], NT:[-1],    TN:[-1]                                    int
    Returns:
       output:             (sum(tokens_per_experts), n)                                                 output_dtype
    """

    assert format in [0, 1, 2], f"w4a8 group gemm only support 0(NN),1(NT),2(TN)"

    outout_n = tokens_per_experts.sum()
    if output is None:
        assert output_dtype is not None, print(
            "moe_w4a8_group_gemm need output_dtype argument when output is none."
        )
        output = torch.empty(
            (outout_n, w_scales.shape[-1]),
            dtype=output_dtype,
            device=input.device,
        )

    ops.infer.moe_w4a8_group_gemm(
        output,
        input,
        weight,
        i_scales,
        w_scales,
        tokens_per_experts,
        w_i8scales,
        w_i8zeros,
        dst_to_src,
        format,
        version,
        group_size,
        outout_n,
        persistent,
    )
    return output


def ref_moe_w4a16_group_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    w_scales: torch.Tensor,
    quant_type: str,
    tokens_per_experts: torch.Tensor,
    w_zeros: torch.Tensor = None,
    group_size: int = -1,
    dst_to_src: torch.Tensor = None,
    format: str = "NN",
    output: torch.Tensor = None,
):
    assert quant_type in ["awq"]
    assert format in ["NN"]
    from .quantized_linear import ref_quantized_weight_dequant

    output_dtype = input.dtype

    def get_align_size(format):
        input_format = format[1]
        return 1 if input_format == "N" else 64

    if output is None:
        m = tokens_per_experts.sum()
        output = torch.empty(
            (m, w_scales.shape[2]),
            dtype=output_dtype,
            device=input.device,
        )

    prefix = 0
    out_prefix = 0
    align_size = get_align_size(format)
    for eid, n in enumerate(tokens_per_experts):
        if n == 0:
            continue
        start, end = prefix, prefix + n
        out_start, out_end = out_prefix, out_prefix + n
        cur_inputs = (
            input[start:end] if format[1] == "N" else input[:, start:end].T.contiguous()
        )

        cur_weights = weight[eid]
        cur_scales = w_scales[eid]
        cur_zeros = w_zeros[eid]

        input_f32 = cur_inputs.to(torch.float32)
        weight_f32 = ref_quantized_weight_dequant(
            qweights=cur_weights,
            scales=cur_scales,
            quant_type="awq",
            output_type=torch.float32,
            bits=4,
            qzeros=cur_zeros,
            group_size=group_size,
            g_idx=None,
        )
        output[out_start:out_end] = torch.matmul(input_f32, weight_f32)
        prefix += (n + align_size - 1) // align_size * align_size
        out_prefix += n

    if dst_to_src is not None:
        tmp = output.clone()
        tmp[dst_to_src] = output
        output[:] = tmp[:]
    return output


def moe_w4a16_group_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    w_scales: torch.Tensor,
    quant_type: str,
    tokens_per_experts: torch.Tensor,
    w_zeros: torch.Tensor = None,
    group_size: int = -1,
    dst_to_src: torch.Tensor = None,
    format: str = "NN",
    output: torch.Tensor = None,
    tokens_per_experts_gpu: torch.Tensor = None,
):
    """
    Args:
       input:              (m, k)                                                        torch.float16, torch.bfloat16
       weight:
                            awq:(n_experts, k, n/8)                                      torch.int32
       w_scales:
                            awq:(n_experts, k/group_size, n)                             input.dtype
       w_zeros:
                            awq:(n_experts, k/group_size, n/8)                           torch.int32
       quant_type:                                                                       str
                           quant type for weight, support [awq] now
       tokens_per_experts: (n_experts)                                                   torch.int32
       group_size:                                                                       int
       dst_to_src:         (m)                                                           torch.int32
                           index of dst to src.
       format:                                                                           str
                           format of input and weight
    Returns:
       output:             (sum(tokens_per_experts), n)                                  output_dtype
                           input and i_scales may be padding when NT format
    """
    assert quant_type in ["awq"]
    assert format in ["NN"]
    output_dtype = input.dtype
    m = tokens_per_experts.sum()
    if output is None:
        output = torch.empty(
            (m, w_scales.shape[-1]),
            dtype=output_dtype,
            device=input.device,
        )
    FLAG = int(os.getenv("ENABLE_MOE_GROUP_GEMV", 1))
    if FLAG == 1 and tokens_per_experts.max() <= 2:
        assert (
            tokens_per_experts_gpu is not None
        ), "moe group gemv must have tokens_per_experts_gpu!"
        ops.infer.moe_group_gemv(
            output,
            input,
            weight,
            w_scales,
            tokens_per_experts,
            tokens_per_experts_gpu,
            w_zeros,
            dst_to_src,
            quant_type,
            format,
            group_size,
            0,
            m,
        )
        return output

    ops.infer.moe_w4a16_group_gemm(
        output,
        input,
        weight,
        w_scales,
        tokens_per_experts,
        w_zeros,
        dst_to_src,
        quant_type,
        format,
        group_size,
        0,
        m,
    )
    return output
