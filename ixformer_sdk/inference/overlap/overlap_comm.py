import enum
from typing import Optional

import ixformer.functions as ixff
import torch
from ixformer.inference.overlap.linear_mlp_overlap_comm import (
    LinearMLPOverlapComm,
    LinearMLPOverlapCommHook,
)


def get_overlap_linear_method(layer):
    if hasattr(layer, "_overlap_comm_gemm_fn"):
        return layer._overlap_comm_gemm_fn

    if layer.linear_weights["weight"].itemsize == 2:
        layer._overlap_comm_gemm_fn = None
        return None

    def overlap_linear_fn(input, weight, bias=None, out: torch.Tensor = None, **kwargs):
        return layer.linear_method.apply_weights(
            layer.linear_weights, input, output=out
        )

    layer._overlap_comm_gemm_fn = overlap_linear_fn
    return overlap_linear_fn


class DecoderLayerOverlapComm(LinearMLPOverlapCommHook):
    class HookStage(enum.IntEnum):
        kExited = 0
        kTracing = 1

    class HookState:
        def __init__(self, max_num_chunks):
            self.max_num_chunks = max_num_chunks
            self.stage = DecoderLayerOverlapComm.HookStage.kExited

            self.mlp_linaer2_end_events = [
                torch.cuda.Event() for _ in range(max_num_chunks)
            ]
            self.ln_attn_end_event = torch.cuda.Event()

            self.overlap_comm: Optional[LinearMLPOverlapComm] = None

        def is_tracing_stage(self):
            return self.stage == DecoderLayerOverlapComm.HookStage.kTracing

        def enter(self, overlap_comm, chunk_idx):
            self.stage = DecoderLayerOverlapComm.HookStage.kTracing

            self.overlap_comm = overlap_comm
            self.mlp_linaer2_end_events[chunk_idx].record(overlap_comm._comm_stream)

        def exit(self):
            self.overlap_comm = None
            self.stage = DecoderLayerOverlapComm.HookStage.kExited

        def __str__(self):
            return f"HookState(overlap_comm={self.overlap_comm}, stage={self.stage})"

        def __repr__(self):
            return self.__str__()

    _overlap_comm_hook_state = dict()

    def __init__(self, model_id, layer_idx, max_num_chunks: int = 4):
        """
        DecoderLayer 的流程：
        ln_qkv: InputLayerNorm(hidden_states, [residual]) -> qkv_proj(hidden_states) -> q, k, v = split(hidden_states) -> Attention(q, k, v)
        linear_mlp: AttentionOutputProj(hidden_states) -> PostLayerNorm(hidden_states) -> MLPLinear1 -> MLPActivation -> MLPLinear2

        其中：AttentionOutputProj 和 MLPLinear2 之后如果使用 TP，那么需要进行 AllReduce

        通过上述流程，该类的目的是将 MLPLinear2 后的 AllReduce 和 DecoderLayer 最开始的 ln_qkv 进行 Overlap。
        其中，第一层 DecoderLayer 不进行 ln_qkv 的 Overlap，因为在第一层之前没有通讯。
        我们需要将第 i 层 MLPLinear2 后的通讯 和 第 i + 1 层的 ln_qkv 进行 Overlap。

        为了管理当前的状态和获取前一层的状态，从而设计了 DecoderLayerOverlapComm 类。
        该类需要 model_id 来推断当前正在运行的模型，用 layer_idx 来标记每一层的开始和结束，
        以及通过 layer_idx 去获取前一层的状态。

        注：
            - 在 call_ln_qkv_overlap 中对 Tensor 进行切分时，
              需要保持和 linear_mlp 切分的大小是一致的，否则会出现 Tensor 的数据不对应；
            - 如果需要使用 ln_qkv 进行 Overlap，那么必须使用该类的 linear_mlp 去替换 linear_mlp_overlap

        :param model_id: 模型的 id，可以使用 id(model) 去设置
        :param layer_idx: layer 的索引，注意，需要从 0 到 NumLayers 的顺序去完成构造
        :param max_num_chunks: 最大能进行切分的次数
        """
        self._model_id = model_id
        self._layer_idx = layer_idx
        self._max_num_chunks = max_num_chunks

        self._state = self.HookState(max_num_chunks)
        self._overlap_comm_hook_state[(model_id, layer_idx)] = self._state
        self._prev_layer_state = (
            None
            if layer_idx == 0
            else self._overlap_comm_hook_state[(model_id, layer_idx - 1)]
        )

    @property
    def model_id(self):
        return self._model_id

    @property
    def layer_idx(self):
        return self._layer_idx

    @property
    def max_num_chunks(self):
        return self._max_num_chunks

    @property
    def state(self) -> "DecoderLayerOverlapComm.HookState":
        return self._state

    @property
    def prev_layer_state(self) -> "DecoderLayerOverlapComm.HookState":
        return self._prev_layer_state

    def is_ln_qkv_overlap(self):
        return not (
            self.layer_idx == 0
            or not self.prev_layer_state.is_tracing_stage()
            or self.prev_layer_state.overlap_comm is None
        )

    def ln_qkv(self, hidden_states, residual, ln_layer, qkv_layer, out_last_dim):
        """
        :param hidden_state: shape[Batch * SeqLen, HiddenSize]
        :param residual: shape[Batch * SeqLen, HiddenSize]
        :param ln_layer: torch.nn.Module or Function(hidden_state, residual=None)
        :param qkv_layer: vllm.QKVParallelLinear
        :param out_last_dim: qkv_layer 输出 Tensor 的最后一个维度
        :return: qkv, residual
        """
        if self.is_ln_qkv_overlap():
            qkv, residual = self.call_ln_qkv_overlap(
                hidden_states, residual, ln_layer, qkv_layer, out_last_dim
            )
        else:
            qkv, residual = self.call_ln_qkv(
                hidden_states, residual, ln_layer, qkv_layer
            )

        return qkv, residual

    def call_ln_qkv_overlap(
        self, hidden_states, residual, ln_layer, qkv_layer, out_last_dim
    ):
        if hidden_states.ndim != 2:
            raise RuntimeError(
                f"Expected 2-dim for hidden state, but got {hidden_states.ndim}."
            )

        num_chunks = self.prev_layer_state.overlap_comm.num_chunks
        overlap_comm: LinearMLPOverlapComm = self.prev_layer_state.overlap_comm

        if num_chunks > self.max_num_chunks:
            raise RuntimeError(
                f"The layer is not support more than {self.max_num_chunks}, got {num_chunks}."
            )

        hidden_state_chunks = list(torch.chunk(hidden_states, num_chunks, dim=0))
        if residual is None:
            residual = hidden_states
            residual_chunks = [None] * num_chunks
        else:
            residual_chunks = torch.chunk(residual, num_chunks, dim=0)

        out = torch.empty(
            (hidden_states.shape[0], out_last_dim),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        out_chunks = list(torch.chunk(out, num_chunks, dim=0))

        for chunk_idx, (hidden_state_chunk, residual_chunk, out_chunk) in enumerate(
            zip(hidden_state_chunks, residual_chunks, out_chunks)
        ):
            overlap_comm._compute_streams[
                chunk_idx % overlap_comm.num_compute_streams
            ].wait_event(self.prev_layer_state.mlp_linaer2_end_events[chunk_idx])
            with overlap_comm.compute_stream_context(chunk_idx):
                self.call_ln_qkv(
                    hidden_state_chunk,
                    residual_chunk,
                    ln_layer,
                    qkv_layer,
                    chunk_idx,
                    use_limited_gemm=chunk_idx != (num_chunks - 1),
                    out=out_chunk,
                    overlap_comm=overlap_comm,
                )

        self.prev_layer_state.exit()
        overlap_comm.stop_overlap()
        return out, residual

    def call_ln_qkv(
        self,
        hidden_state,
        residual,
        ln_layer,
        qkv_layer,
        chunk_idx=0,
        use_limited_gemm=False,
        out=None,
        overlap_comm: LinearMLPOverlapComm = None,
    ):
        if residual is None:
            residual = hidden_state
            if ln_layer is not None:
                hidden_state = ln_layer(hidden_state)
        else:
            hidden_state, residual = ln_layer(hidden_state, residual)

        if out is None:
            qkv, _ = qkv_layer(hidden_state)
        else:
            gemm_method = get_overlap_linear_method(qkv_layer)
            qkv = overlap_comm.gemm_dispatcher(
                chunk_idx=chunk_idx,
                chunk_input=hidden_state,
                weight=qkv_layer.linear_weights["weight"],
                chunk_out=out,
                user_gemm_method=gemm_method,
                use_limited_gemm=use_limited_gemm,
            )

        return qkv, residual

    def linear_mlp(self, *args, **kwargs):
        """ref: linear_mlp_overlap"""
        return ixff.linear_mlp_overlap(
            *args, **kwargs, mlp_linear2_finished_callback=self.on_mlp_linear2_finished
        )

    def on_mlp_linear2_finished(
        self,
        overlap_comm: LinearMLPOverlapComm,
        num_chunks,
        chunk_idx,
        hidden_states_chunk,
        residual_chunk,
    ):
        self.state.enter(overlap_comm, chunk_idx)
