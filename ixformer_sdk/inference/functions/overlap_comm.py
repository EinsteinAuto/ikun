import itertools
from functools import partial
from typing import Callable, Dict, Iterable, Tuple

import torch
import torch.distributed as dist

import ixformer.distributed as ixfd
from ixformer.core.dispatcher import Dispatcher
from ixformer.core.operator_autotuning import (
    OperatorPreBaseRangeAutotuning,
    sync_ranks_metric,
)
from ixformer.distributed import overlap_comm
from ixformer.inference.overlap.linear_mlp_overlap_comm import linear_mlp_overlap
from ixformer.distributed.overlap_comm import GemmMethod

__all__ = ["linear_allreduce_overlap", "linear_mlp_overlap"]


class LinearAllReducePreAutotuning(OperatorPreBaseRangeAutotuning, Dispatcher):
    def __init__(self, comm_group, *args, **kwargs):
        dist_barrier = True
        if "dist_barrier" in kwargs:
            dist_barrier = kwargs.pop("dist_barrier")

        super().__init__(dist_barrier=dist_barrier, *args, **kwargs)
        self._comm_group = comm_group
        self._world_size = ixfd.get_group_world_size(comm_group)

    @classmethod
    def dispatcher_key(cls, comm_group, *args, **kwargs):
        return (comm_group,)

    def operators(self):
        chunks = [2, 4]
        gemm_algos = [GemmMethod.kCUINFER, GemmMethod.kCUBLAS, GemmMethod.kLIMITED_GEMM]
        candidate_ops = [overlap_comm.GemmAllReduceSplitOverlapComm.native_forward]
        for num_chunks, algo in itertools.product(chunks, gemm_algos):
            candidate_ops.append(
                partial(
                    overlap_comm.linear_allreduce_overlap,
                    num_chunks=num_chunks,
                    gemm_method=algo,
                )
            )

        return candidate_ops

    @property
    def _gemm_shapes(self):
        basic_k = [4096, 6114, 8192]
        tp_k = [k // self._world_size for k in basic_k]
        basic_k = tp_k

        basic_m = (512, 1024, 2048, 4096, 8192)

        shapes = set(itertools.product(basic_m, basic_k))

        return shapes

    def get_operator_key(self, input, *args, **kwargs):
        ndim = input.ndim
        shape = input.shape

        if ndim == 1:
            return (1, shape[0])
        elif ndim == 2:
            return shape
        else:
            return (sum(shape[:-1]), shape[-1])

    def generate_operator_inputs(self) -> Iterable[Tuple[Tuple, Dict]]:
        for m, kn in self._gemm_shapes:
            input = torch.randn(m, kn, device="cuda", dtype=torch.half)
            weight = torch.randn(kn, kn, device="cuda", dtype=torch.half)
            yield (input, weight), {}

    def perf_operator_time(self, op: Callable, *args, **kwargs) -> float:
        op_time = super().perf_operator_time(op, *args, **kwargs)
        return sync_ranks_metric(op_time, group=self._comm_group)


linear_allreduce_overlap = overlap_comm.linear_allreduce_overlap
