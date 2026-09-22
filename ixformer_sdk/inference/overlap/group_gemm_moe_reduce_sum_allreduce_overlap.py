import dataclasses
import math
from contextlib import contextmanager
from typing import List, Optional

import ixformer.distributed as ixfd
import ixformer.functions as F
import torch
import torch.distributed as dist
from ixformer.distributed.overlap_comm import SplitOverlapComm

from ixformer.core import config as ixff_config


@dataclasses.dataclass
class GroupGemmMoeReduceSumAllReduceParams:
    # M: NumTokens * TopK
    # K: InnerSize // TP
    # N: HiddenSize
    # NumTokens: M // TopK

    # =================================
    # group gemm
    # =================================

    # the top k of experts
    topk: int

    # shape: [M, K] if format[1]=="N" else [K, M], dtype: int8
    input: torch.Tensor

    # shape: [NumExperts, N, K] if format[0]=="T" else [NumExperts, K, N], dtype: int8
    weight: torch.Tensor

    # shape: [M], dtype: float32
    i_scales: torch.Tensor

    # shape: [NumExperts, N], dtype: float32
    w_scales: torch.Tensor

    # shape: [NumExperts], dtype: int32
    tokens_per_experts: torch.Tensor

    # the dtype of output, support float16 and bfloat16
    out_dtype: torch.dtype = None

    # index of dst to src, shape: [M], dtype: int32
    dst_to_src: torch.Tensor = None

    # only support TN now
    format: str = "TN"

    # =================================
    # moe reduce sum
    # =================================

    # shape: [M // TopK, TopK], dtype: torch.float16 or torch.bfloat16
    topk_weight: torch.Tensor = None

    # shape: [M // TopK, N], dtype: torch.float16 or torch.bfloat16
    output: torch.Tensor = None

    # overlap
    output_chunks: Optional[List[torch.Tensor]] = None

    @property
    def M(self):
        return self.input.shape[0]

    @property
    def N(self):
        if torch.is_tensor(self.weight):
            return self.weight.shape[1]
        return sum(t.shape[1] for t in self.weight)

    @property
    def K(self):
        return self.input.shape[-1]

    def prepare_overlap_params(
        self, num_chunks: int, split_ratio: Optional[float] = None
    ):
        if num_chunks == 2 and split_ratio not in [0, None]:
            return self.prepare_overla_params_with_ratio(split_ratio)
        return self.prepare_overlap_params_with_chunks(num_chunks)

    def prepare_overlap_params_with_chunks(self, num_chunks: int):
        if torch.is_tensor(self.weight):
            weight_chunks = torch.chunk(self.weight, num_chunks, dim=1)
            self.weight = list(weight_chunks)

        if torch.is_tensor(self.w_scales):
            weight_scale_chunks = torch.chunk(self.w_scales, num_chunks, dim=1)
            self.w_scales = list(weight_scale_chunks)

        if self.output is None:
            self.output = torch.empty(
                self.M // self.topk, self.N, dtype=self.out_dtype, device="cuda"
            )

        if torch.is_tensor(self.output):
            output_chunks = torch.chunk(self.output, num_chunks, dim=1)
            self.output_chunks = list(output_chunks)

    def prepare_overla_params_with_ratio(self, split_ratio: float):
        N = self.N
        n_chunks = [int(math.ceil(N * split_ratio))]
        n_chunks.append(N - n_chunks[0])

        if torch.is_tensor(self.weight):
            weight_chunks = torch.split(self.weight, n_chunks, dim=1)
            self.weight = list(weight_chunks)

        if torch.is_tensor(self.w_scales):
            weight_scale_chunks = torch.split(self.w_scales, n_chunks, dim=1)
            self.w_scales = list(weight_scale_chunks)

        if self.output is None:
            self.output = torch.empty(
                self.M // self.topk, self.N, dtype=self.out_dtype, device="cuda"
            )

        if torch.is_tensor(self.output):
            output_chunks = torch.split(self.output, n_chunks, dim=1)
            self.output_chunks = list(output_chunks)


class GroupGemmMoeReduceSumAllReduceSplitNOverlap(SplitOverlapComm):
    def compute(self, params: GroupGemmMoeReduceSumAllReduceParams):
        for chunk_idx, (weight, weight_scale) in enumerate(
            zip(params.weight, params.w_scales)
        ):
            with self.compute_stream_context(chunk_idx):
                out = F.moe_w8a8_group_gemm(
                    input=params.input,
                    weight=weight,
                    i_scales=params.i_scales,
                    w_scales=weight_scale,
                    output_dtype=params.out_dtype,
                    tokens_per_experts=params.tokens_per_experts,
                    dst_to_src=params.dst_to_src,
                    format=params.format,
                )
                out = out.reshape(-1, params.topk, out.shape[-1])
                out = F.moe_output_reduce_sum(
                    input=out,
                    topk_weight=params.topk_weight,
                    output=params.output_chunks[chunk_idx],
                )

            self.start_comm(chunk_idx)
            ixfd.all_reduce(
                out,
                async_op=True,
                group=self.comm_group,
                use_comm_stream=True,
                algo=ixfd.AllReduceAlgo.Stride,
            )

            # if chunk_idx == 0: torch.cuda.synchronize()

        return params.output


_group_gemm_moe_reduce_sum_all_reduce_overlap = None


def group_gemm_moe_reduce_sum_allreduce(
    params: GroupGemmMoeReduceSumAllReduceParams,
    enable_overlap: bool = False,
    comm_group=None,
    num_chunks=2,
    split_ratio: Optional[float] = None,
):
    if params.output is None and params.out_dtype is None:
        raise RuntimeError(
            "group_gemm_moe_reduce_sum_all_reduce need out_dtype argument when output is none."
        )

    if params.out_dtype is None:
        params.out_dtype = params.output.dtype

    if (
        enable_overlap
        and ixff_config.IXFORMER_ENABLE_OVERLAP_COMM
        and dist.is_initialized()
        and dist.get_world_size(comm_group) > 1
    ):
        global _group_gemm_moe_reduce_sum_all_reduce_overlap
        if _group_gemm_moe_reduce_sum_all_reduce_overlap is None:
            _group_gemm_moe_reduce_sum_all_reduce_overlap = (
                GroupGemmMoeReduceSumAllReduceSplitNOverlap.dispatcher(
                    num_chunks=num_chunks, comm_group=comm_group
                ).forward
            )

        params.prepare_overlap_params(num_chunks=num_chunks, split_ratio=split_ratio)
        return _group_gemm_moe_reduce_sum_all_reduce_overlap(params)

    out = F.moe_w8a8_group_gemm(
        input=params.input,
        weight=params.weight,
        i_scales=params.i_scales,
        w_scales=params.w_scales,
        output_dtype=params.out_dtype,
        tokens_per_experts=params.tokens_per_experts,
        dst_to_src=params.dst_to_src,
        format=params.format,
    )

    out = out.reshape(-1, params.topk, out.shape[-1])
    out = F.moe_output_reduce_sum(
        input=out, topk_weight=params.topk_weight, output=params.output
    )

    if dist.is_initialized() and dist.get_world_size(comm_group) > 1:
        ixfd.all_reduce(out, group=comm_group, async_op=True)

    return out
