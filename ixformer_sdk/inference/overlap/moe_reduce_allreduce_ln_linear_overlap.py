import dataclasses
import math
from typing import Optional, Tuple

import ixformer.distributed as ixfd
import ixformer.functions as F
import torch
import torch.distributed as dist
from ixformer.distributed.overlap_comm import SplitOverlapComm

from ixformer.core import config as ixff_config


@dataclasses.dataclass
class MoeReduceAllReduceLnQkvLinearParams:
    # ==============================
    # MOE Reduce Sum
    # ==============================

    # shape: [Batch * SeqLen, TopK, HiddenSize], dtype: float16 or bfloat16
    input: torch.Tensor

    # shape: [Batch * SeqLen, TopK], dtype: float32
    topk_weight: Optional[torch.Tensor]

    # ==============================
    # Ln
    # ==============================

    # shape: [Batch * SeqLen, HiddenSize], dtype: float16 or bfloat16
    residual: torch.Tensor

    # shape: [HiddenSize], dtype: float16 or bfloat16
    ln_weight: torch.Tensor

    # shape: [HiddenSize], dtype: float16 or bfloat16
    ln_bias: torch.Tensor
    ln_eps: float

    # ==============================
    # QkvLinear
    # ==============================

    # shape: [(NumHeads + 2 * NumKvHeads) * HeadDim / TP, HiddenSize], dtype: float16 or bfloat16
    qkv_weight: torch.Tensor

    # shape: [(NumHeads + 2 * NumKvHeads) * HeadDim / TP]
    qkv_weight_scale: torch.Tensor

    # shape: [Batch * SeqLen, (NumHeads + 2 * NumKvHeads) * HeadDim / TP], dtype: float16 or bfloat16
    qkv_out: torch.Tensor


class MoeReduceSumAllReduceLnQkvLinearOverlap(SplitOverlapComm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.allreduce_end_events = [torch.cuda.Event() for _ in range(self.num_chunks)]

    def start_qkv_linear(self, chunk_idx):
        compute_stream = self._compute_streams[chunk_idx % self.num_compute_streams]
        compute_stream.wait_event(self.allreduce_end_events[chunk_idx])

    def compute(self, params: MoeReduceAllReduceLnQkvLinearParams, split_ratio=0.5):
        input_chunk_sizes = [int(math.ceil(params.input.shape[0] * split_ratio))]
        input_chunk_sizes.append(params.input.shape[0] - input_chunk_sizes[0])

        input_chunks = list(
            torch.split_with_sizes(params.input, input_chunk_sizes, dim=0)
        )
        topk_weight_chunks = list(
            torch.split_with_sizes(params.topk_weight, input_chunk_sizes, dim=0)
        )
        out_chunks = []

        for chunk_idx in range(len(input_chunks)):
            with self.compute_stream_context(chunk_idx):
                out_chunks.append(
                    F.moe_output_reduce_sum(
                        input_chunks[chunk_idx],
                        topk_weight=topk_weight_chunks[chunk_idx],
                    )
                )

            self.start_comm(chunk_idx)

            ixfd.all_reduce(
                out_chunks[chunk_idx],
                async_op=True,
                group=self.comm_group,
                use_comm_stream=True,
            )

            self.allreduce_end_events[chunk_idx].record(self._comm_stream)

        residual_chunk_sizes = [int(params.residual.shape[0] * split_ratio)]
        residual_chunk_sizes.append(params.residual.shape[0] - residual_chunk_sizes[0])
        residual_chunks = torch.split_with_sizes(
            params.residual, residual_chunk_sizes, dim=0
        )

        qkv_out_chunk_sizes = [int(params.qkv_out.shape[0] * split_ratio)]
        qkv_out_chunk_sizes.append(params.qkv_out.shape[0] - qkv_out_chunk_sizes[0])
        qkv_out_chunks = torch.split_with_sizes(
            params.qkv_out, qkv_out_chunk_sizes, dim=0
        )

        for chunk_idx in range(len(input_chunks)):
            self.start_qkv_linear(chunk_idx)
            with self.compute_stream_context(chunk_idx):
                (
                    i8_hidden_states,
                    residual,
                    i_scales,
                ) = F.residual_layer_norm_dynamic_int8(
                    input=out_chunks[chunk_idx],
                    residual=residual_chunks[chunk_idx],
                    weight=params.ln_weight,
                    bias=params.ln_bias,
                    eps=params.ln_eps,
                )

                F.w8a8(
                    i8_hidden_states,
                    params.qkv_weight,
                    i_scales,
                    params.qkv_weight_scale,
                    output=qkv_out_chunks[chunk_idx],
                )

        return params.qkv_out, params.residual


_moe_reduce_with_allreduce_overlap = None


def moe_reduce_sum_allreduce_ln_qkv_linear(
    params: MoeReduceAllReduceLnQkvLinearParams,
    enable_overlap=False,
    comm_group=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MOE Reduce Sum + AllReduce + LayerNorm + QkvLinear

    Args:
        params: fused operator params
        enable_overlap: whether enable overlap
        comm_group: communication group
    Returns:
        QkvLinearOutput: [Batch * SeqLen, (NumHeads + 2 * NumKvHeads) * HeadDim / TP], dtype: float16 or bfloat16
        Residual: [Batch * SeqLen, HiddenSize], dtype: float16 or bfloat16
    """

    global _moe_reduce_with_allreduce_overlap
    if _moe_reduce_with_allreduce_overlap is None:
        _moe_reduce_with_allreduce_overlap = (
            MoeReduceSumAllReduceLnQkvLinearOverlap.dispatcher(
                num_chunks=2, comm_group=comm_group
            ).forward
        )

    if (
        enable_overlap
        and ixff_config.IXFORMER_ENABLE_OVERLAP_COMM
        and dist.is_initialized()
        and dist.get_world_size(comm_group) > 1
        and params.input.shape[0] > 1
    ):
        return _moe_reduce_with_allreduce_overlap(params, split_ratio=0.5)

    out = F.moe_output_reduce_sum(params.input, topk_weight=params.topk_weight)
    ixfd.all_reduce(out, async_op=True, group=comm_group)

    i8_hidden_states, residual, i_scales = F.residual_layer_norm_dynamic_int8(
        input=out,
        residual=params.residual,
        weight=params.ln_weight,
        bias=params.ln_bias,
        eps=params.ln_eps,
    )

    out = F.w8a8(
        i8_hidden_states,
        params.qkv_weight,
        i_scales,
        params.qkv_weight_scale,
        output=params.qkv_out,
    )

    return out, residual
