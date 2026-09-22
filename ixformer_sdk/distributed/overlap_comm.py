import abc
import enum
import os
from contextlib import contextmanager, nullcontext
from typing import List, Optional

import torch.cuda
from ixformer.core.dispatcher import Dispatcher

from ixformer.core import config

from . import _distributed as ixfd


class SplitOverlapComm(Dispatcher):
    def __init__(self, num_chunks, num_compute_streams=None, comm_group=None):
        """
        Args:
            num_chunks: the number of chunks
            num_compute_streams: the number of compute streams, default: 1
            comm_group: communicator group
        """

        self._num_chunks = num_chunks
        self._num_compute_streams = num_compute_streams or 1
        self._comm_group = comm_group

        self._compute_streams: List[torch.cuda.Stream] = self.create_compute_streams()
        self._comm_stream: torch.cuda.Stream = torch.cuda.Stream(priority=-1)

        self._start_compute_event: torch.cuda.Event = torch.cuda.Event()
        self._stop_compute_event: torch.cuda.Event = torch.cuda.Event()

        self._start_comm_event: torch.cuda.Event = torch.cuda.Event()
        self._stop_comm_event: torch.cuda.Event = torch.cuda.Event()

        # keep origin state
        self._main_stream: Optional[torch.cuda.Stream] = None
        self._origin_ixf_comm_stream = None
        self._ixformer_streams = dict()

    @classmethod
    def dispatcher_key(
        cls, num_chunks, num_compute_streams=None, comm_group=None, *args, **kwargs
    ):
        """
        the key of SplitOverlapComm
        Args:
            num_chunks: the number of chunks
            num_compute_streams: the number of compute streams, default: 1
            comm_group: communicator group
        Returns: unique key
        """
        # warn: keey same function parameters with init
        return (cls.__name__, num_chunks, num_compute_streams, comm_group)

    @classmethod
    def enable(cls):
        return config.IXFORMER_ENABLE_OVERLAP_COMM

    @property
    def num_chunks(self):
        return self._num_chunks

    @property
    def num_compute_streams(self):
        return self._num_compute_streams

    @property
    def comm_group(self):
        return self._comm_group

    def create_compute_streams(self):
        streams = []
        for _ in range(self.num_compute_streams):
            streams.append(torch.cuda.Stream())
        return streams

    def start_overlap(self):
        self._main_stream = torch.cuda.current_stream()

        self._start_compute_event.record(torch.cuda.current_stream())
        for compute_stream in self._compute_streams:
            compute_stream.wait_event(self._start_compute_event)

        self._origin_ixf_comm_stream = ixfd.get_comm_group_stream(self._comm_group)
        ixfd.set_comm_group_stream(self._comm_stream.cuda_stream, self._comm_group)

    def stop_overlap(self):
        last_compute_stream_id = (
            self.num_chunks + self.num_compute_streams - 1
        ) % self.num_compute_streams
        self._stop_compute_event.record(self._compute_streams[last_compute_stream_id])
        self._stop_comm_event.record(self._comm_stream)
        torch.cuda.current_stream().wait_event(self._stop_compute_event)
        torch.cuda.current_stream().wait_event(self._stop_comm_event)

        ixfd.set_comm_group_stream(self._origin_ixf_comm_stream, self._comm_group)

    def start_comm(self, chunk_idx):
        """
        prepare communication stream and wait event.
        Args:
            chunk_idx: the index of chunk
        """

        self._start_comm_event.record(
            self._compute_streams[chunk_idx % self.num_compute_streams]
        )
        self._comm_stream.wait_event(self._start_comm_event)

    @contextmanager
    def compute_stream_context(self, chunk_idx):
        """
        open python context and switch to compute stream in torch context
        Args:
            chunk_idx: the index of chunk
        """

        stream = self._compute_streams[chunk_idx % self.num_compute_streams]

        # print("before stream:", torch.cuda.current_stream())
        torch.cuda.set_stream(stream)

        # print("after stream:", torch.cuda.current_stream(), ixformer.cuda.current_stream())
        yield stream

        torch.cuda.set_stream(self._main_stream)

    @contextmanager
    def stream_context(self, stream):
        # print("before stream:", torch.cuda.current_stream())
        torch.cuda.set_stream(stream)

        # print("after stream:", torch.cuda.current_stream(), ixformer.cuda.current_stream())
        yield stream

        torch.cuda.set_stream(self._main_stream)

    def forward(self, *args, **kwargs):
        self.start_overlap()
        out = self.compute(*args, **kwargs)
        self.stop_overlap()
        return out

    @abc.abstractmethod
    def compute(self, *args, **kwargs):
        """
        it is abstract method to execute compute and communication.
        """
        pass


class GemmMethod(enum.IntEnum):
    kCUINFER = 0
    kCUBLAS = 1
    kLIMITED_GEMM = 2


class GemmWithLimitedBlock:
    def __init__(self, limit_algo=0) -> None:
        self.limit_algo = limit_algo
        self.env_key = "PYTORCH_GEMM_BLOCK_LIMITATION"

    def __enter__(self) -> None:
        os.environ[self.env_key] = str(self.limit_algo)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del os.environ[self.env_key]


class IxFormerLimitedGemmContext:
    def __init__(self) -> None:
        self.env_key = "IXFORMER_ENABLE_PERSISTENT_GEMM"

    def __enter__(self) -> None:
        os.environ[self.env_key] = "1"

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        os.environ[self.env_key] = "0"


class GemmAllReduceSplitOverlapComm(SplitOverlapComm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gemm_method_env = config.IXFORMER_OVERLAP_GEMM_METHOD

        if self.gemm_method_env is None:
            if ixfd.get_world_size(self.comm_group) == 2:
                self.gemm_method_env = 0
            else:
                self.gemm_method_env = 2

        self.gemm_method = GemmMethod(int(self.gemm_method_env))
        self.limited_gemm_ctx = GemmWithLimitedBlock()
        self.ixf_limited_gemm_ctx = IxFormerLimitedGemmContext()
        self.split_ratio = config.IXFORMER_OVERLAP_SPLIT_RATIO

    @classmethod
    def compute_row_parallel_dims(cls, input):
        batch = 1
        if input.ndim == 2:
            seqlen = input.shape[0]
        else:
            batch = input.shape[0]
            seqlen = input.shape[1]

        parallel_dims = batch * seqlen
        return parallel_dims

    def compute(self, input, weight, bias=None, out=None, *args, **kwargs):
        """
        :param input: [Batch, SeqLen, Hidden]
        :param weight: [OutChannel, InChannel]
        :param bias: [OutChannel]
        """

        is_update_shape = input.ndim > 2
        batch = 1
        if input.ndim == 2:
            seqlen = input.shape[0]
        else:
            batch = input.shape[0]
            seqlen = input.shape[1]

        parallel_dims = batch * seqlen

        if is_update_shape:
            input = input.reshape(parallel_dims, -1)

        if out is None:
            out_shape = [parallel_dims, weight.shape[0]]
            out_dtype = kwargs["out_dtype"] if "out_dtype" in kwargs else input.dtype
            out = torch.empty(out_shape, dtype=out_dtype, device=input.device)

        if self.split_ratio is not None:
            round_multiples = 256 if parallel_dims >= 256 else parallel_dims
            first_chunk_size = (
                round((parallel_dims * float(self.split_ratio)) / round_multiples)
                * round_multiples
            )
            middle_chunk_size = (parallel_dims - first_chunk_size) // (
                self.num_chunks - 1
            )
            middle_chunk_size = (middle_chunk_size // round_multiples) * round_multiples
            last_chunk_size = (
                parallel_dims
                - first_chunk_size
                - middle_chunk_size * (self.num_chunks - 2)
            )

            chunk_sizes = (
                [first_chunk_size]
                + [middle_chunk_size] * (self.num_chunks - 2)
                + [last_chunk_size]
            )
            input_chunks = torch.split_with_sizes(input, chunk_sizes, dim=0)
            out_chunks = torch.split_with_sizes(out, chunk_sizes, dim=0)

            # print(first_chunk_size, middle_chunk_size, last_chunk_size, chunk_sizes)
        else:
            input_chunks = torch.chunk(input, self.num_chunks, dim=0)
            out_chunks = torch.chunk(out, self.num_chunks, dim=0)

        for chunk_idx in range(len(input_chunks)):
            with self.compute_stream_context(chunk_idx):
                chunk_out = self.gemm_dispatcher(
                    chunk_idx,
                    input_chunks[chunk_idx],
                    weight,
                    out_chunks[chunk_idx],
                    *args,
                    **kwargs,
                )

            self.start_comm(chunk_idx)

            ixfd.all_reduce(
                chunk_out, async_op=True, group=self.comm_group, use_comm_stream=True
            )

        if is_update_shape:
            out = out.reshape(batch, seqlen, -1)

        if bias is not None:
            out = out + bias

        return out

    def gemm_dispatcher(
        self,
        chunk_idx,
        chunk_input,
        weight,
        chunk_out=None,
        user_gemm_method=None,
        *args,
        **kwargs,
    ):
        if user_gemm_method is not None and callable(user_gemm_method):
            ctx = nullcontext() if chunk_idx == 0 else self.ixf_limited_gemm_ctx
            with ctx:
                return user_gemm_method(
                    chunk_input, weight, out=chunk_out, *args, **kwargs
                )

        if user_gemm_method is None:
            user_gemm_method = self.gemm_method

        if user_gemm_method == GemmMethod.kCUINFER:
            import ixformer.functions as ixff

            return ixff.linear(chunk_input, weight, output=chunk_out)
        elif user_gemm_method == GemmMethod.kCUBLAS:
            return torch.matmul(chunk_input, weight.T, out=chunk_out)
        elif user_gemm_method == GemmMethod.kLIMITED_GEMM:
            ctx = self.limited_gemm_ctx
            with ctx:
                return torch.matmul(chunk_input, weight.T, out=chunk_out)
        elif user_gemm_method == GemmMethod.kCUBLAS:
            return torch.matmul(chunk_input, weight.T, out=chunk_out)
        else:
            raise RuntimeError(f"Invalid gemm method, got {self.gemm_method}.")

    @classmethod
    def native_forward(
        cls,
        input,
        weight,
        bias=None,
        out=None,
        group=None,
        user_gemm_method=None,
        *args,
        **kwargs,
    ):
        if user_gemm_method is not None and callable(user_gemm_method):
            gemm_out = user_gemm_method(
                input, weight, bias=bias, out=out, *args, **kwargs
            )
            out = out if gemm_out is None else gemm_out
        else:
            import ixformer.functions as ixff

            # warning: 下面的两种 gemm 可能存在精度不一致
            # out = torch.matmul(input, weight.T, out=out)
            out = ixff.linear(input=input, weight=weight, bias=bias, output=out)
        ixfd.all_reduce(out, async_op=True, group=group)
        return out

    @classmethod
    def is_supported(cls, input, num_chunks, comm_group):
        if not cls.enable():
            return False

        ndim = input.ndim
        shape = input.shape

        if ndim == 1:
            m, k = 1, shape[0]
        elif ndim == 2:
            m, k = shape
        else:
            m, k = sum(shape[:-1]), shape[-1]

        return m >= 512


_DEFAULT_OVERLAP_GROUP = None
_DEFAULT_OVERLAP_COMM_N2 = None
_DEFAULT_OVERLAP_COMM_N4 = None
_DEFAULT_OVERLAP_CHUNKS = config.IXFORMER_OVERLAP_CHUNKS


def linear_allreduce_overlap(
    input, weight, bias=None, out=None, group=None, num_chunks=None, *args, **kwargs
):
    num_chunks = num_chunks or _DEFAULT_OVERLAP_CHUNKS

    # print("call overlap:", GemmAllReduceSplitOverlapComm.is_supported(input, num_chunks=num_chunks, comm_group=group), input.shape, weight.shape if torch.is_tensor(weight) else None, "WorldSize:", ixfd.get_group_world_size(group), ", NumChunks:", num_chunks)
    if not GemmAllReduceSplitOverlapComm.is_supported(
        input, num_chunks=num_chunks, comm_group=group
    ):
        return GemmAllReduceSplitOverlapComm.native_forward(
            input, weight, bias=bias, out=out, group=group, *args, **kwargs
        )

    global _DEFAULT_OVERLAP_GROUP
    global _DEFAULT_OVERLAP_COMM_N2
    global _DEFAULT_OVERLAP_COMM_N4

    if _DEFAULT_OVERLAP_GROUP is None:
        _DEFAULT_OVERLAP_GROUP = group

    if num_chunks == 2 and group == _DEFAULT_OVERLAP_GROUP:
        if _DEFAULT_OVERLAP_COMM_N2 is None:
            _DEFAULT_OVERLAP_COMM_N2 = GemmAllReduceSplitOverlapComm.dispatcher(
                num_chunks=num_chunks, comm_group=group
            )
        overlap_comm = _DEFAULT_OVERLAP_COMM_N2
    elif num_chunks == 4 and group == _DEFAULT_OVERLAP_GROUP:
        if _DEFAULT_OVERLAP_COMM_N4 is None:
            _DEFAULT_OVERLAP_COMM_N4 = GemmAllReduceSplitOverlapComm.dispatcher(
                num_chunks=num_chunks, comm_group=group
            )
        overlap_comm = _DEFAULT_OVERLAP_COMM_N4
    else:
        overlap_comm = GemmAllReduceSplitOverlapComm.dispatcher(
            num_chunks=num_chunks, comm_group=group
        )
    return overlap_comm.forward(input, weight, bias=bias, out=out, *args, **kwargs)
