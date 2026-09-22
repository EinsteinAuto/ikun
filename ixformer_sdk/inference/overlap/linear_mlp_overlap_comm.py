from contextlib import nullcontext
from typing import List

import torch.cuda

from ...distributed import _distributed as ixfd
from ...distributed import overlap_comm as base_overlap_comm
from ...distributed.overlap_comm import GemmAllReduceSplitOverlapComm
from .. import overlap as overlap_base


class LinearMLPOverlapCommHook:
    def on_mlp_linear2_finished(
        self,
        overlap_comm: "LinearMLPOverlapComm",
        num_chunks,
        chunk_idx,
        hidden_states_chunk,
        residual_chunk,
    ):
        pass

    def on_mlp_finished(
        self,
        overlap_comm: "LinearMLPOverlapComm",
        hidden_states_chunks,
        residual_chunks,
    ):
        pass


class LinearMLPOverlapComm(GemmAllReduceSplitOverlapComm):
    def __init__(self, *args, **kwargs):
        super().__init__(num_compute_streams=1, *args, **kwargs)

        self._mlp_linear1_start_events: List[torch.cuda.Event] = [
            torch.cuda.Event() for _ in range(self.num_chunks)
        ]
        self._mlp_linear1_end_events: List[torch.cuda.Event] = [
            torch.cuda.Event() for _ in range(self.num_chunks)
        ]

        self._mlp_linear1_stream: torch.cuda.Stream = torch.cuda.Stream()

    def stop_linear_comm(self, chunk_idx):
        event = self._mlp_linear1_start_events[chunk_idx]
        event.record(self._comm_stream)

    def start_mlp_linear1(self, chunk_idx):
        self._mlp_linear1_stream.wait_event(self._mlp_linear1_start_events[chunk_idx])

    def stop_mlp_linear1(self, chunk_idx):
        event = self._mlp_linear1_end_events[chunk_idx]
        event.record(self._mlp_linear1_stream)

    def start_mlp_linear2(self, chunk_idx):
        compute_stream = self._compute_streams[chunk_idx % self.num_compute_streams]
        compute_stream.wait_event(self._mlp_linear1_end_events[chunk_idx])

    def compute(
        self,
        protocol: "overlap_base.LlamaDecoderLayerOverlapDefault",
        attn_output,
        residual,
        *,
        mlp_linear2_finished_callback=None,
        mlp_finished_callback=None,
    ):
        """ """
        attn_output_shape = attn_output.shape
        residual_shape = None if residual is None else residual.shape

        is_update_shape = attn_output.ndim > 2
        batch = 1
        if attn_output.ndim == 2:
            seqlen = attn_output_shape[0]
        else:
            batch = attn_output_shape[0]
            seqlen = attn_output_shape[1]

        parallel_dims = batch * seqlen

        if is_update_shape:
            attn_output = attn_output.reshape(parallel_dims, -1)
            if residual is not None:
                residual = residual.reshape(-1, residual_shape[-1])

        attn_output_chunks, residual_chunks = protocol.split_mlp_inputs(
            attn_output, residual, self.num_chunks
        )

        out = protocol.create_mlp_output()
        out_chunks = protocol.split_mlp_output(out, self.num_chunks)

        res_chunks = []

        # 1. output project linear
        for chunk_idx, (attn_output_chunk, residual_chunk) in enumerate(
            zip(attn_output_chunks, residual_chunks)
        ):
            with self.compute_stream_context(chunk_idx):
                hidden_states = protocol.attn_output_proj_linear(
                    self.num_chunks,
                    chunk_idx,
                    attn_output_chunk,
                    use_limited_gemm=chunk_idx != 0,
                )

            self.start_comm(chunk_idx)
            ixfd.all_reduce(
                hidden_states,
                async_op=True,
                group=self.comm_group,
                use_comm_stream=True,
            )
            self.stop_linear_comm(chunk_idx)
            attn_output_chunks[chunk_idx] = hidden_states

        # 2. ln, mlp_linear1 and act
        for chunk_idx, (hidden_states, residual_chunk) in enumerate(
            zip(attn_output_chunks, residual_chunks)
        ):
            self.start_mlp_linear1(chunk_idx)
            with self.stream_context(self._mlp_linear1_stream):
                (
                    hidden_states,
                    residual_chunk,
                ) = protocol.attn_output_proj_linear_layer_norm(
                    self.num_chunks, chunk_idx, hidden_states, residual_chunk
                )

                hidden_states = protocol.mlp_linear1(
                    self.num_chunks, chunk_idx, hidden_states, use_limited_gemm=True
                )
                hidden_states = protocol.mlp_activation(hidden_states)

                attn_output_chunks[chunk_idx] = hidden_states
                res_chunks.append(residual_chunk)

            self.stop_mlp_linear1(chunk_idx)

        # 3. mlp_linear2
        for chunk_idx, hidden_states in enumerate(attn_output_chunks):
            self.start_mlp_linear2(chunk_idx)
            with self.compute_stream_context(chunk_idx):
                hidden_states = protocol.mlp_linear2(
                    self.num_chunks,
                    chunk_idx,
                    hidden_states,
                    out=out_chunks[chunk_idx],
                    use_limited_gemm=True,
                )

            self.start_comm(chunk_idx)
            ixfd.all_reduce(
                hidden_states,
                async_op=True,
                group=self.comm_group,
                use_comm_stream=True,
            )

            if mlp_linear2_finished_callback is not None:
                mlp_linear2_finished_callback(
                    self,
                    self.num_chunks,
                    chunk_idx,
                    hidden_states,
                    res_chunks[chunk_idx],
                )

        if mlp_finished_callback is not None:
            mlp_finished_callback(self, out_chunks, res_chunks)

        if is_update_shape:
            out = out.reshape(attn_output_shape)
            if residual is not None:
                residual = residual.reshape(residual_shape)

        return out, residual

    def gemm_dispatcher(
        self,
        chunk_idx,
        chunk_input,
        weight,
        chunk_out,
        use_limited_gemm=False,
        user_gemm_method=None,
        *args,
        **kwargs,
    ):
        if user_gemm_method is not None and callable(user_gemm_method):
            ctx = self.ixf_limited_gemm_ctx if use_limited_gemm else nullcontext()
            with ctx:
                return user_gemm_method(
                    chunk_input, weight, out=chunk_out, *args, **kwargs
                )

        ctx = self.limited_gemm_ctx if use_limited_gemm else nullcontext()
        with ctx:
            return torch.matmul(chunk_input, weight.T, out=chunk_out)

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

    @classmethod
    def native_forward(
        cls,
        attn_output,
        residual,
        linear_weight,
        ln_layer,
        mlp_weight1,
        mlp_weight2,
        mlp_activation,
        linear_method=None,
        mlp_linear1_method=None,
        mlp_linear2_method=None,
        group=None,
        *args,
        **kwargs,
    ):
        import ixformer.functions as ixff

        linear_method = linear_method or ixff.linear
        mlp_linear1_method = mlp_linear1_method or ixff.linear
        mlp_linear2_method = mlp_linear2_method or ixff.linear

        hidden_states = linear_method(attn_output, linear_weight)
        ixfd.all_reduce(hidden_states, async_op=True, group=group)

        if ln_layer is not None:
            hidden_states, residual = ln_layer(hidden_states, residual)

        hidden_states = mlp_linear1_method(hidden_states, mlp_weight1)
        hidden_states = mlp_activation(hidden_states)
        hidden_states = mlp_linear2_method(hidden_states, mlp_weight2)
        ixfd.all_reduce(hidden_states, async_op=True, group=group)

        return hidden_states, residual


_DEFAULT_OVERLAP_GROUP = None
_DEFAULT_OVERLAP_COMM_N2 = None
_DEFAULT_OVERLAP_COMM_N4 = None
_DEFAULT_OVERLAP_CHUNKS = base_overlap_comm._DEFAULT_OVERLAP_CHUNKS


def linear_mlp_overlap(
    protocol: "overlap_base.LlamaDecoderLayerOverlapProtocol",
    attn_output,
    residual,
    num_chunks=None,
    group=None,
    *,
    mlp_linear2_finished_callback=None,
    mlp_finished_callback=None,
):
    num_chunks = num_chunks or _DEFAULT_OVERLAP_CHUNKS

    global _DEFAULT_OVERLAP_GROUP
    global _DEFAULT_OVERLAP_COMM_N2
    global _DEFAULT_OVERLAP_COMM_N4

    if _DEFAULT_OVERLAP_GROUP is None:
        _DEFAULT_OVERLAP_GROUP = group

    if num_chunks == 2 and group == _DEFAULT_OVERLAP_GROUP:
        if _DEFAULT_OVERLAP_COMM_N2 is None:
            _DEFAULT_OVERLAP_COMM_N2 = LinearMLPOverlapComm.dispatcher(
                num_chunks=num_chunks, comm_group=group
            )
        overlap_comm = _DEFAULT_OVERLAP_COMM_N2
    elif num_chunks == 4 and group == _DEFAULT_OVERLAP_GROUP:
        if _DEFAULT_OVERLAP_COMM_N4 is None:
            _DEFAULT_OVERLAP_COMM_N4 = LinearMLPOverlapComm.dispatcher(
                num_chunks=num_chunks, comm_group=group
            )
        overlap_comm = _DEFAULT_OVERLAP_COMM_N4
    else:
        overlap_comm = LinearMLPOverlapComm.dispatcher(
            num_chunks=num_chunks, comm_group=group
        )

    return overlap_comm.forward(
        protocol,
        attn_output,
        residual,
        mlp_linear2_finished_callback=mlp_linear2_finished_callback,
        mlp_finished_callback=mlp_finished_callback,
    )
