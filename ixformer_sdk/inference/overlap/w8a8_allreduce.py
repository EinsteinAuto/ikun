import math
from typing import Optional

import ixformer.distributed as ixfd
import ixformer.functions as F
import torch
import torch.distributed as dist
from ixformer.distributed.overlap_comm import SplitOverlapComm

from ixformer.core import config as ixff_config

__all__ = ["w8a8_allreduce"]


class W8A8AllReduceOverlap(SplitOverlapComm):
    def compute(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
        format: str = "TN",
        out_dtype: torch.dtype = None,
        comm_group=None,
        split_ratio=0.5,
    ):
        # compute the chunk size of input
        input_chunk_sizes = [int(math.ceil(input.shape[0] * split_ratio))]
        input_chunk_sizes.append(input.shape[0] - input_chunk_sizes[0])

        # split input and input_scale
        input_chunks = list(torch.split_with_sizes(input, input_chunk_sizes, dim=0))
        input_scale_chunks = torch.split(
            input_scale,
            input_chunk_sizes,
        )

        # create output and split it
        if output is None:
            if out_dtype is None:
                raise RuntimeError(
                    "w8a8 gemm need out_dtype argument when output is none."
                )
            output = torch.empty(
                (input.shape[:-1] + (weight.shape[0],)),
                dtype=out_dtype,
                device=input.device,
            )
        out_chunks = torch.split(output, input_chunk_sizes)

        # overlap gemm and allreduce
        for chunk_idx in range(len(input_chunks)):
            # submit gemm kernel into compute stream
            with self.compute_stream_context(chunk_idx):
                F.w8a8(
                    input=input_chunks[chunk_idx],
                    weight=weight,
                    i_scales=input_scale_chunks[chunk_idx],
                    w_scales=weight_scale,
                    bias=bias,
                    output=out_chunks[chunk_idx],
                    format=format,
                    persistent=chunk_idx != 0,
                )

            # recode compute stream and wait gemm
            self.start_comm(chunk_idx)

            # submit allreduce kernel into communication stream by set use_comm_stream to true
            ixfd.all_reduce(
                out_chunks[chunk_idx],
                async_op=True,
                group=self.comm_group,
                use_comm_stream=True,
            )

        return output


_w8a8_allreduce_overlap = None


def w8a8_allreduce(
    enable_overlap: bool,
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    format: str = "TN",
    out_dtype: torch.dtype = None,
    comm_group=None,
    split_ratio=0.5,
) -> torch.Tensor:
    """
    Gemm(w8a8) + AllReduce

    Args:
        enable_overlap: whether enable gemm and allreduce overlap
        input:          shape: [M, K], dtype: int8,                linear input
        weight:         shape: [N, K], dtype: int8,                linear weight
        input_scale:    shape: [M],    dtype: float32,             quantized scale of input
        weight_scale:   shape: [N],    dtype: float32,             quantized scale of weight
        bias:           shape: [N],    dtype: float16 or bfloat16, linear bias
        output:         shape: [M, N], dtype: float16 or bfloat16, allreduce output
        format:         options include TN, NN and NT
        out_dtype:      use the argument to decide to the dtype of output when output is None
        comm_group:     communication group
        split_ratio:    split the ratio of input.shape[0] when using overlap, range: (0, 1),
                        it will affect area of the overlap for gemm and allreduce.
    Returns: output
    """
    if (
        enable_overlap
        and ixff_config.IXFORMER_ENABLE_OVERLAP_COMM
        and dist.is_initialized()
        and dist.get_world_size(comm_group) > 1
        and input.shape[0] > 1
    ):
        global _w8a8_allreduce_overlap
        if _w8a8_allreduce_overlap is None:
            _w8a8_allreduce_overlap = W8A8AllReduceOverlap.dispatcher(
                num_chunks=2, comm_group=comm_group
            ).forward
        return _w8a8_allreduce_overlap(
            input=input,
            weight=weight,
            input_scale=input_scale,
            weight_scale=weight_scale,
            bias=bias,
            output=output,
            format=format,
            out_dtype=out_dtype,
            split_ratio=split_ratio,
        )

    out = F.w8a8(
        input=input,
        weight=weight,
        i_scales=input_scale,
        w_scales=weight_scale,
        bias=bias,
        output=output,
        format=format,
        out_dtype=out_dtype,
    )

    if dist.get_world_size() > 1:
        ixfd.all_reduce(out, op=ixfd.ReduceOp.SUM, async_op=True, group=comm_group)

    return out
