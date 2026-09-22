import enum
import typing
from abc import abstractmethod
from typing import Any, Callable, List, Optional, Tuple

import ixformer._C.infer as ops
import ixformer.functions as ixff
import torch
import torch.distributed as dist

import ixformer.distributed as ixfd
from ixformer.core import config

from ...distributed.overlap_comm import GemmWithLimitedBlock
from .linear_mlp_overlap_comm import LinearMLPOverlapComm
from .overlap_comm import DecoderLayerOverlapComm

KVCache = Tuple[torch.Tensor, torch.Tensor]

OptionalTensor = typing.Union[None, torch.Tensor]


class LlamaDecoderLayerParams:
    def __init__(
        self,
        *,
        pre_input_layer_norm_weight: OptionalTensor,
        qkv_linear_weight: torch.Tensor,
        qkv_linear_bias: OptionalTensor,
        attn_output_proj_linear_weight: torch.Tensor,
        attn_output_proj_linear_bias: OptionalTensor,
        attn_output_proj_linear_layer_norm_weight: torch.Tensor,
        mlp_linear1_weight: torch.Tensor,
        mlp_linear1_bias: OptionalTensor,
        mlp_linear2_weight: torch.Tensor,
        mlp_linear2_bias: OptionalTensor,
        mlp_activation: Callable[[torch.Tensor, OptionalTensor], Any],
        layer_norm_eps: float = 1e-5,
        quant_mode: Optional[str] = None,
        **kwargs,
    ):
        self.pre_input_layer_norm_weight = pre_input_layer_norm_weight

        self.qkv_linear_weight = qkv_linear_weight
        self.qkv_linear_bias = qkv_linear_bias

        self.attn_output_proj_linear_weight = attn_output_proj_linear_weight
        self.attn_output_proj_linear_bias = attn_output_proj_linear_bias
        self.attn_output_proj_linear_layer_norm_weight = (
            attn_output_proj_linear_layer_norm_weight
        )

        self.mlp_linear1_weight = mlp_linear1_weight
        self.mlp_linear1_bias = mlp_linear1_bias

        self.mlp_linear2_weight = mlp_linear2_weight
        self.mlp_linear2_bias = mlp_linear2_bias

        self.mlp_activation = mlp_activation

        self.layer_norm_eps = layer_norm_eps

        self.quant_mode = quant_mode

        for k, v in kwargs:
            setattr(self, k, v)

    @classmethod
    def create_from(cls, params: "LlamaDecoderLayerParams", **extra_params):
        param_attrs = dict(**params.__dict__)
        param_attrs.update(**extra_params)

        return cls(**param_attrs)


class LlamaDecoderLayerOverlapProtocol:
    GLOBAL_ENABLE_OVERLAP_CACHE = False

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

    def __init__(
        self,
        model_id: int,
        layer_idx: int,
        attn_q_size: int,
        attn_kv_size: int,
        params: LlamaDecoderLayerParams = None,
        max_num_chunks=4,
    ):
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

        if layer_idx < 0:
            raise RuntimeError(f"Invalid layer_idx, got {layer_idx}.")
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

        self.params: LlamaDecoderLayerParams = params

        self.attn_q_size = attn_q_size
        self.attn_kv_size = attn_kv_size

        self.limited_blas_gemm_ctx = GemmWithLimitedBlock()

        self._current_comm_group = None

    def set_params(self, params: LlamaDecoderLayerParams):
        self.params = params

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

    @property
    def out_last_dim(self):
        return self.attn_q_size + 2 * self.attn_kv_size

    @classmethod
    def is_supported(cls, input, num_chunks, comm_group):
        return (
            config.IXFORMER_ENABLE_OVERLAP_COMM
            and LinearMLPOverlapComm.is_supported(input, num_chunks, comm_group)
        )

    def forward(
        self,
        self_attn: Callable,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        *,
        num_chunks=None,
        group=None,
        residual: Optional[torch.Tensor],
        self_attn_kwargs: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        等价于 DecoderLayer.forward

        :param self_attn: Function(Q, K, V, **self_attn_kwargs), 计算 SelfAttention 的输出，
                          Q, K, V 会在 self.attention 中根据 q_size 和 kv_size 进行分割得到，
                          如果需要额外的参数，可以使用 self_attn_kwargs 进行传递.
        :param positions: 如果使用 RotaryEmbedding，那么会被传入到该函数中
        :param hidden_states: [Batch * SeqLen, HiddenSize], 前一层的输出
        :param num_chunks: 在 Overlap 时分块数量
        :param group: 通讯组
        :param residual: 前一层的残差
        :param self_attn_kwargs: self_attn 函数的额外参数
        :return: DecoderLayerOut[Batch * SeqLen, HiddenSize], Residual[Batch * SeqLen, HiddenSize]
        """

        self._current_comm_group = group

        # 仅在第一层 Layer 去判断是否使用 Overlap，
        # 如果第一层 Layer 启用，那么后面的所有 Layer 也都会使用 Overlap
        if self.layer_idx == 0:
            self.__class__.GLOBAL_ENABLE_OVERLAP_CACHE = self.is_supported(
                hidden_states, num_chunks, comm_group=group
            )

        enable_overlap = self.__class__.GLOBAL_ENABLE_OVERLAP_CACHE

        qkv, residual = self.ln_qkv(
            hidden_states, residual, enable_overlap=enable_overlap
        )

        attn_output = self.attention(
            qkv, residual, self_attn=self_attn, positions=positions, **self_attn_kwargs
        )

        attn_output, residual = self.linear_mlp(
            attn_output,
            residual,
            num_chunks=num_chunks,
            group=group,
            enable_overlap=enable_overlap,
        )

        return attn_output, residual

    def is_ln_qkv_overlap(self):
        return not (
            self.layer_idx == 0
            or not self.prev_layer_state.is_tracing_stage()
            or self.prev_layer_state.overlap_comm is None
        )

    def ln_qkv(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        enable_overlap: bool = False,
    ) -> Tuple[torch.Tensor, OptionalTensor]:
        """
        LayerNorm -> QKVLinear

        :param hidden_states: shape[Batch * SeqLen, HiddenSize]
        :param residual: shape[Batch * SeqLen, HiddenSize]
        :return: qkv, residual
        """
        if self.is_ln_qkv_overlap() and enable_overlap:
            qkv, residual = self.call_ln_qkv_overlap(hidden_states, residual)
        else:
            qkv, residual = self.call_ln_qkv(hidden_states, residual)

        return qkv, residual

    def call_ln_qkv_overlap(self, hidden_states, residual):
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
            (hidden_states.shape[0], self.out_last_dim),
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
                    num_chunks,
                    chunk_idx,
                    use_limited_gemm=chunk_idx != (num_chunks - 1),
                    out=out_chunk,
                )

        self.prev_layer_state.exit()
        overlap_comm.stop_overlap()

        out, residual = self.ln_qkv_linear_callback(out, residual)

        return out, residual

    def call_ln_qkv(
        self,
        hidden_state,
        residual,
        num_chunks=1,
        chunk_idx=0,
        use_limited_gemm=False,
        out=None,
    ):
        pre_hidden_state = hidden_state
        hidden_state, residual = self.pre_input_layer_norm(
            num_chunks, chunk_idx, hidden_state, residual
        )
        if residual is None:
            residual = pre_hidden_state

        qkv = self.qkv_linear(
            num_chunks,
            chunk_idx,
            hidden_state,
            out=out,
            use_limited_gemm=use_limited_gemm,
        )

        return qkv, residual

    def linear_mlp(
        self,
        attn_output: torch.Tensor,
        residual: torch.Tensor,
        num_chunks: int,
        group=None,
        enable_overlap: bool = False,
    ) -> Tuple[torch.Tensor, OptionalTensor]:
        """
        OProj -> AllReduce -> PostLayerNorm -> Linear1 -> Activation -> Linear2 -> AllReduce

        :param attn_output: Attention 的输出
        :param residual: 残差
        :param num_chunks: 在 Overlap 时，需要分为多少块
        :param group: 通讯组
        :param enable_overlap: 是否启用 overlap
        :return: DecoderLayerOut[Batch * SeqLen, HiddenSize], Residual[Batch * SeqLen, HiddenSize]
        """

        if enable_overlap:
            return ixff.linear_mlp_overlap(
                self,
                attn_output,
                residual,
                num_chunks=num_chunks,
                group=group,
                mlp_linear2_finished_callback=self.on_mlp_linear2_finished,
            )

        hidden_states = self.attn_output_proj_linear(
            num_chunks=1,
            chunk_idx=0,
            attn_out_chunk=attn_output,
            use_limited_gemm=False,
        )
        ixfd.all_reduce(hidden_states, async_op=True, group=self._current_comm_group)

        hidden_states, residual = self.attn_output_proj_linear_layer_norm(
            num_chunks=1, chunk_idx=0, hidden_states=hidden_states, residual=residual
        )

        hidden_states = self.mlp_linear1(
            num_chunks=1,
            chunk_idx=0,
            hidden_states=hidden_states,
            use_limited_gemm=False,
        )

        hidden_states = self.mlp_activation(hidden_states)

        hidden_states = self.mlp_linear2(
            num_chunks=1,
            chunk_idx=0,
            hidden_states=hidden_states,
            out=None,
            use_limited_gemm=False,
        )
        ixfd.all_reduce(hidden_states, async_op=True, group=self._current_comm_group)

        return hidden_states, residual

    def on_mlp_linear2_finished(
        self,
        overlap_comm: LinearMLPOverlapComm,
        num_chunks,
        chunk_idx,
        hidden_states_chunk,
        residual_chunk,
    ):
        """
        这是一个 LinearMLPOverlapComm 的回调函数，在 MLP Linear2 后面的通讯结束时被执行，
        在这里是为了将 MLP Linear2 后面的通讯和 LnQKV 进行 Overlap，需要进行 cuda event 的同步。
        """
        self.state.enter(overlap_comm, chunk_idx)

    def _gemm_dispatcher(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: OptionalTensor = None,
        input_scales: OptionalTensor = None,
        smooth_scales: OptionalTensor = None,
        weight_scales: OptionalTensor = None,
        out: OptionalTensor = None,
        use_limited_gemm: bool = False,
        quant_group_size: int = -1,
        out_dtype: Optional[torch.dtype] = None,
    ):
        """
        调用不同精度的 gemm，已验证 fp16，w8a8
        """

        # 下面判断的顺序不能随意改变

        # float
        if input_scales is None and weight_scales is None:
            if out is not None and out.dtype not in [
                torch.half,
                torch.bfloat16,
                torch.float,
            ]:
                raise RuntimeError(
                    f"linear is supported half or float, but got {out.dtype}."
                )

            # if use_limited_gemm:
            #     with self.limited_blas_gemm_ctx:
            #         out = ops.cublas_linear(
            #             input, weight, bias=bias, out=out, persistent=use_limited_gemm
            #         )
            # else:
            out = ixff.linear(input, weight, bias=bias, output=out, persistent=use_limited_gemm)
            return out

        elif weight_scales is None:
            raise RuntimeError(f"got invalid quantized weight scales, got none.")

        # smmoth quant with w8a8
        elif (input_scales is None and smooth_scales is not None) or self.params.quant_mode in ["smoothquant", "compressed_tensors"]:
            x_shape = input.shape
            dtype = input.dtype
            if self.params.quant_mode == "compressed_tensors":
                x, x_scales = ixff.scaled_int8_quant(input, smooth_scales)
            else:
                x, x_scales = ixff.dynamic_scaled_quant_dynamic_int8(input, smooth_scales)

            x = ixff.w8a8(
                input=x,
                weight=weight,
                i_scales=x_scales,
                w_scales=weight_scales,
                output=out,
                persistent=use_limited_gemm,
                out_dtype=dtype,
            )
            out = x.view(*x_shape[:-1], -1)

        # w8a16
        elif input_scales is None and weight_scales is not None:
            if out is not None and out.dtype not in [
                torch.half,
                torch.bfloat16,
                torch.float,
            ]:
                raise RuntimeError(
                    f"w8a16 is supported half or float, but got {out.dtype}."
                )

            out = ixff.w8a16(
                input, weight, weight_scales, output=out, group_size=quant_group_size, persistent=int(use_limited_gemm)
            )

        # w8a8
        elif input_scales is not None and weight_scales is not None:
            if out is not None and out.dtype not in [
                torch.half,
                torch.bfloat16,
                torch.float,
            ]:
                raise RuntimeError(
                    f"w8a8 is supported half or float, but got {out.dtype}."
                )

            out = ixff.w8a8(
                input,
                weight,
                input_scales,
                weight_scales,
                output=out,
                persistent=use_limited_gemm,
                out_dtype=out_dtype,
            )

        else:
            raise RuntimeError("dispatcher gemm fail.")

        if bias is None:
            return out
        return out + bias

    @abstractmethod
    def split_ln_qkv_input(
        self, hidden_states: torch.Tensor, residual: OptionalTensor, num_chunks: int
    ) -> Tuple[List[torch.Tensor], List[OptionalTensor]]:
        """
        对 ln_qkv 的输入进行切分，仅被用在 overlap 时
        :param hidden_states: 对 hidden_states 进行切分
        :param residual: 对 residual 进行切分，如果 residual 为 None，那么应该返回 [None] * num_chunks
        :param num_chunks: 分块数量
        :return: HiddenStatesChunks, ResidualChunks
        """
        raise NotImplementedError()

    @abstractmethod
    def create_ln_qkv_output(self, num_chunks: int) -> torch.Tensor:
        """
        创建 ln_qkv 的输出，仅被用在 overlap 时
        :param num_chunks: 分块数量
        :return: Tensor
        """
        raise NotImplementedError()

    @abstractmethod
    def split_ln_qkv_output(
        self, out: torch.Tensor, num_chunks: int
    ) -> List[torch.Tensor]:
        """
        对上面创建的输出 Tensor 进行切分，仅被用在 overlap 时
        :param out: 上面创建的 Tensor
        :param num_chunks: 分块数量
        :return: OutChunks
        """
        raise NotImplementedError()

    @abstractmethod
    def pre_input_layer_norm(
        self,
        num_chunks: int,
        chunk_idx: int,
        hidden_states: torch.Tensor,
        residual: OptionalTensor,
    ) -> Tuple[torch.Tensor, OptionalTensor]:
        """
        DecoderLayer 中的第一个 LayerNorm
        :param num_chunks: 分块数量
        :param chunk_idx: 分块的索引
        :param hidden_states: 分块后的输入
        :param residual: 分块后的残差
        :return: HiddenStates，Residual
        """
        raise NotImplementedError()

    @abstractmethod
    def qkv_linear(
        self,
        num_chunks: int,
        chunk_idx: int,
        hidden_states: torch.Tensor,
        out: OptionalTensor,
        use_limited_gemm=False,
    ) -> torch.Tensor:
        """
        DecoderLayer 中的 QkvLinear

        :param num_chunks: 分块数量
        :param chunk_idx: 分块的索引
        :param hidden_states: linear 的输入
        :param out: linear 的输出
        :param use_limited_gemm: 是否限制 gemm 的计算资源
        :return: Out
        """
        raise NotImplementedError()

    @abstractmethod
    def ln_qkv_linear_callback(
        self, qkv: torch.Tensor, residual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return qkv, residual

    @abstractmethod
    def attention(
        self,
        qkv: torch.Tensor,
        residual: torch.Tensor,
        *,
        self_attn: Callable,
        positions: OptionalTensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        计算 Attention
        :param qkv: QkvLinear 的输出
        :param residual: PreLayerNorm 输出的残差
        :param self_attn: 计算 SelfAttention 的函数
        :param positions: 位置编码，被用在 RotaryEmbedding 中
        :param kwargs: self_attn 的额外参数
        :return: Attention 的输出
        """
        raise NotImplementedError()

    @abstractmethod
    def create_mlp_output(self) -> torch.Tensor:
        """
        创建 MLP 的输出，仅被用在 overlap 时
        """

        raise NotImplementedError()

    @abstractmethod
    def split_mlp_output(
        self, out: torch.Tensor, num_chunks: int
    ) -> List[torch.Tensor]:
        """
        对上面创建的输出进行分块，仅被用在 overlap 时
        :param out: 上面函数的输出
        :param num_chunks: 分块数量
        :return: OutChunks
        """
        raise NotImplementedError()

    @abstractmethod
    def split_mlp_inputs(
        self, attn_out: torch.Tensor, residual: torch.Tensor, num_chunks: int
    ) -> Tuple[List[torch.Tensor], List[OptionalTensor]]:
        """
        对 MLP 的输入进行分块，仅被用在 overlap 时
        :param attn_out: Attention 的输出
        :param residual: 残差
        :param num_chunks: 分块数量
        :return: AttnOutChunks, ResidualChunks
        """
        raise NotImplementedError()

    @abstractmethod
    def attn_output_proj_linear(
        self,
        num_chunks: int,
        chunk_idx: int,
        attn_out_chunk: torch.Tensor,
        use_limited_gemm: bool = False,
    ) -> torch.Tensor:
        """
        Attention 后面的 o_proj Linear
        :param num_chunks: 分块数量
        :param chunk_idx: 分块的索引
        :param attn_out_chunk: Linear 的输入
        :param use_limited_gemm: 是否限制 gemm 的计算资源
        :return: Out
        """
        raise NotImplementedError()

    @abstractmethod
    def attn_output_proj_linear_layer_norm(
        self,
        num_chunks: int,
        chunk_idx: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, OptionalTensor]:
        """
        attention output projection linear 后面的 LayerNorm
        :param num_chunks: 分块数量
        :param chunk_idx: 分块索引
        :param hidden_states: LN 的输入
        :param residual: 残差
        :return: HiddenStates，Residual
        """
        raise NotImplementedError()

    @abstractmethod
    def mlp_linear1(
        self,
        num_chunks: int,
        chunk_idx: int,
        hidden_states: torch.Tensor,
        use_limited_gemm: bool = False,
    ) -> torch.Tensor:
        """
        MLP 中的第一个 Linear
        :param num_chunks: 分块数量
        :param chunk_idx: 分块索引
        :param hidden_states: Linear 的输入
        :param use_limited_gemm: 是否限制 gemm 的计算资源
        :return: Out
        """
        raise NotImplementedError()

    @abstractmethod
    def mlp_activation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        MLP 中的激活函数
        """
        raise NotImplementedError()

    @abstractmethod
    def mlp_linear2(
        self,
        num_chunks: int,
        chunk_idx: int,
        hidden_states: torch.Tensor,
        out: OptionalTensor = None,
        use_limited_gemm: bool = False,
    ) -> torch.Tensor:
        """
        MLP 中的第二个 Linear
        :param num_chunks: 分块数量
        :param chunk_idx: 分块索引
        :param hidden_states: Linear 的输入
        :param out: Linear 的输出
        :param use_limited_gemm: 是否限制 gemm 的计算资源
        :return: Out
        """
        raise NotImplementedError()

    @abstractmethod
    def mlp_callback(self, mlp_out: torch.Tensor, residual: OptionalTensor):
        return mlp_out, residual


class LlamaDecoderLayerOverlapDefault(LlamaDecoderLayerOverlapProtocol):
    def split_ln_qkv_input(
        self, hidden_states: torch.Tensor, residual: OptionalTensor, num_chunks: int
    ) -> Tuple[List[torch.Tensor], List[OptionalTensor]]:
        self.input_hidden_states_shape = list(hidden_states.shape)
        self.input_hidden_states_device = hidden_states.device
        self.input_hidden_states_dtype = hidden_states.dtype

        hidden_states_chunks = list(torch.chunk(hidden_states, num_chunks, dim=0))
        if residual is None:
            residual_chunks = [None] * num_chunks
        else:
            residual_chunks = list(torch.chunk(residual, num_chunks, dim=0))

        return hidden_states_chunks, residual_chunks

    def create_ln_qkv_output(self, num_chunks: int) -> torch.Tensor:
        return torch.empty(
            (self.input_hidden_states_shape[0], self.out_last_dim),
            device=self.input_hidden_states_device,
            dtype=self.input_hidden_states_dtype,
        )

    def split_ln_qkv_output(
        self, out: torch.Tensor, num_chunks: int
    ) -> List[torch.Tensor]:
        return list(torch.chunk(out, num_chunks, dim=0))

    def pre_input_layer_norm(
        self,
        num_chunks: int,
        chunk_idx: int,
        hidden_states: torch.Tensor,
        residual: OptionalTensor,
    ) -> Tuple[torch.Tensor, OptionalTensor]:
        if residual is None:
            return (
                ixff.rms_norm(
                    hidden_states,
                    self.params.pre_input_layer_norm_weight,
                    eps=self.params.layer_norm_eps,
                ),
                None,
            )
        else:
            ixff.residual_rms_norm(
                hidden_states,
                residual,
                self.params.pre_input_layer_norm_weight,
                eps=self.params.layer_norm_eps,
            )
            return hidden_states, residual

    def qkv_linear(
        self,
        num_chunks: int,
        chunk_idx: int,
        hidden_states: torch.Tensor,
        out: OptionalTensor,
        use_limited_gemm=False,
    ) -> torch.Tensor:
        return self._gemm_dispatcher(
            input=hidden_states,
            weight=self.params.qkv_linear_weight,
            bias=self.params.qkv_linear_bias,
            out=out,
            use_limited_gemm=use_limited_gemm,
        )

    def attention(
        self,
        qkv: torch.Tensor,
        residual: torch.Tensor,
        *,
        self_attn: Callable,
        positions: OptionalTensor,
        rotary_embedding: Callable = None,
        **kwargs,
    ):
        q, k, v = qkv.split(
            [self.attn_q_size, self.attn_kv_size, self.attn_kv_size],
            dim=-1,
        )
        if rotary_embedding is not None:
            q, k = rotary_embedding(positions, q, k)
        attn_output = self_attn(q, k, v, **kwargs)

        self.attn_output_shape = attn_output.shape
        self.attn_output_device = attn_output.device
        self.attn_output_dtype = attn_output.dtype

        return attn_output

    def create_mlp_output(self) -> torch.Tensor:
        return torch.empty(
            [self.attn_output_shape[0], self.params.mlp_linear2_weight.shape[0]],
            device=self.attn_output_device,
            dtype=self.attn_output_dtype,
        )

    def split_mlp_output(
        self, out: torch.Tensor, num_chunks: int
    ) -> List[torch.Tensor]:
        return list(torch.chunk(out, num_chunks, dim=0))

    def split_mlp_inputs(
        self, attn_out: torch.Tensor, residual: torch.Tensor, num_chunks: int
    ) -> Tuple[List[torch.Tensor], List[OptionalTensor]]:
        attn_output_chunks = list(torch.chunk(attn_out, num_chunks, dim=0))
        if residual is None:
            residual_chunks = [None] * num_chunks
        else:
            residual_chunks = torch.chunk(residual, num_chunks, dim=0)

        return attn_output_chunks, residual_chunks

    def attn_output_proj_linear(
        self,
        num_chunks: int,
        chunk_idx: int,
        attn_out_chunk: torch.Tensor,
        use_limited_gemm: bool = False,
    ) -> torch.Tensor:
        return self._gemm_dispatcher(
            input=attn_out_chunk,
            weight=self.params.attn_output_proj_linear_weight,
            bias=self.params.attn_output_proj_linear_bias,
            use_limited_gemm=use_limited_gemm,
        )

    def attn_output_proj_linear_layer_norm(
        self,
        num_chunks: int,
        chunk_idx: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, OptionalTensor]:
        if residual is None:
            return (
                ixff.rms_norm(
                    hidden_states,
                    self.params.attn_output_proj_linear_layer_norm_weight,
                    eps=self.params.layer_norm_eps,
                ),
                None,
            )
        else:
            return ixff.residual_rms_norm(
                hidden_states,
                residual,
                self.params.attn_output_proj_linear_layer_norm_weight,
                self.params.layer_norm_eps,
            )

    @abstractmethod
    def mlp_linear1(
        self,
        num_chunks: int,
        chunk_idx: int,
        hidden_states: torch.Tensor,
        use_limited_gemm: bool = False,
    ) -> torch.Tensor:
        return self._gemm_dispatcher(
            input=hidden_states,
            weight=self.params.mlp_linear1_weight,
            bias=self.params.mlp_linear1_bias,
            use_limited_gemm=use_limited_gemm,
        )

    def mlp_activation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.params.mlp_activation(hidden_states)

    def mlp_linear2(
        self,
        num_chunks: int,
        chunk_idx: int,
        hidden_states: torch.Tensor,
        out: OptionalTensor = None,
        use_limited_gemm: bool = False,
    ) -> torch.Tensor:
        return self._gemm_dispatcher(
            input=hidden_states,
            weight=self.params.mlp_linear2_weight,
            bias=self.params.mlp_linear2_bias,
            out=out,
            use_limited_gemm=use_limited_gemm,
        )


class LlamaDecoderLayerParamsQuant(LlamaDecoderLayerParams):
    def __init__(
        self,
        *,
        # qkv
        qkv_linear_smooth_scales: OptionalTensor,
        qkv_linear_weight_scales: OptionalTensor,
        qkv_linear_quant_group_size: Optional[int] = -1,
        # attn_output_proj
        attn_output_proj_linear_smooth_scales: OptionalTensor,
        attn_output_proj_linear_weight_scales: OptionalTensor,
        attn_output_proj_linear_quant_group_size: Optional[int] = -1,
        # mlp_linear1
        mlp_linear1_smooth_scales: OptionalTensor,
        mlp_linear1_weight_scales: OptionalTensor,
        mlp_linear1_quant_group_size: Optional[int] = -1,
        # mlp_linear2
        mlp_linear2_smooth_scales: OptionalTensor,
        mlp_linear2_weight_scales: OptionalTensor,
        mlp_linear2_quant_group_size: Optional[int] = -1,
        # other
        activation_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.qkv_linear_smooth_scales = qkv_linear_smooth_scales
        self.qkv_linear_weight_scales = qkv_linear_weight_scales
        self.qkv_linear_quant_group_size = qkv_linear_quant_group_size

        self.attn_output_proj_linear_smooth_scales = (
            attn_output_proj_linear_smooth_scales
        )
        self.attn_output_proj_linear_weight_scales = (
            attn_output_proj_linear_weight_scales
        )
        self.attn_output_proj_linear_quant_group_size = (
            attn_output_proj_linear_quant_group_size
        )

        self.mlp_linear1_smooth_scales = mlp_linear1_smooth_scales
        self.mlp_linear1_weight_scales = mlp_linear1_weight_scales
        self.mlp_linear1_quant_group_size = mlp_linear1_quant_group_size

        self.mlp_linear2_smooth_scales = mlp_linear2_smooth_scales
        self.mlp_linear2_weight_scales = mlp_linear2_weight_scales
        self.mlp_linear2_quant_group_size = mlp_linear2_quant_group_size

        self.activation_dtype = activation_dtype


class LlamaDecoderLayerOverlapQuant(LlamaDecoderLayerOverlapDefault):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.qkv_linear_input_scales = [None] * self.max_num_chunks
        self.attn_output_proj_linear_input_scales = [None] * self.max_num_chunks
        self.mlp_linear1_input_scales = [None] * self.max_num_chunks
        self.mlp_linear2_input_scales = [None] * self.max_num_chunks

    def _dispatch_layer_norm(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        residual: OptionalTensor,
        smooth_scales: OptionalTensor,
    ) -> typing.Union[
        Tuple[torch.Tensor, OptionalTensor],
        Tuple[torch.Tensor, OptionalTensor, torch.Tensor],
    ]:
        if smooth_scales is None:
            if residual is None:
                return (
                    ixff.rms_norm(input, weight, eps=self.params.layer_norm_eps),
                    None,
                )
            else:
                ixff.residual_rms_norm(
                    input, residual, weight, eps=self.params.layer_norm_eps
                )
                return input, residual

        elif smooth_scales is not None:
            if residual is None:
                hidden_states, scales = ixff.residual_rms_norm_dynamic_int8(
                    input=input,
                    weight=weight,
                    smooth_scales=smooth_scales,
                    eps=self.params.layer_norm_eps,
                )
                return hidden_states, None, scales
            else:
                hidden_states, residual, scales = ixff.residual_rms_norm_dynamic_int8(
                    input=input,
                    residual=residual,
                    weight=weight,
                    smooth_scales=smooth_scales,
                    eps=self.params.layer_norm_eps,
                )
                return hidden_states, residual, scales
        else:
            raise RuntimeError("dispatcher layer norm fail.")

    def pre_input_layer_norm(
        self,
        num_chunks: int,
        chunk_idx: int,
        hidden_states: torch.Tensor,
        residual: OptionalTensor,
    ) -> Tuple[torch.Tensor, OptionalTensor]:
        self.params: LlamaDecoderLayerParamsQuant
        outs = self._dispatch_layer_norm(
            hidden_states,
            self.params.pre_input_layer_norm_weight,
            residual,
            self.params.qkv_linear_smooth_scales,
        )

        self.qkv_linear_input_scales[chunk_idx] = None
        if len(outs) == 3:
            self.qkv_linear_input_scales[chunk_idx] = outs[2]

        return outs[:2]

    def qkv_linear(
        self,
        num_chunks: int,
        chunk_idx: int,
        hidden_states: torch.Tensor,
        out: OptionalTensor,
        use_limited_gemm=False,
    ) -> torch.Tensor:
        self.params: LlamaDecoderLayerParamsQuant
        return self._gemm_dispatcher(
            input=hidden_states,
            weight=self.params.qkv_linear_weight,
            input_scales=self.qkv_linear_input_scales[chunk_idx],
            weight_scales=self.params.qkv_linear_weight_scales,
            out=out,
            use_limited_gemm=use_limited_gemm,
            quant_group_size=self.params.qkv_linear_quant_group_size,
            out_dtype=self.params.activation_dtype,
        )

    def attn_output_proj_linear(
        self,
        num_chunks: int,
        chunk_idx: int,
        attn_out_chunk: torch.Tensor,
        use_limited_gemm: bool = False,
    ) -> torch.Tensor:
        self.params: LlamaDecoderLayerParamsQuant
        return self._gemm_dispatcher(
            input=attn_out_chunk,
            weight=self.params.attn_output_proj_linear_weight,
            input_scales=self.attn_output_proj_linear_input_scales[chunk_idx],
            smooth_scales=self.params.attn_output_proj_linear_smooth_scales,
            weight_scales=self.params.attn_output_proj_linear_weight_scales,
            out=None,
            use_limited_gemm=use_limited_gemm,
            quant_group_size=self.params.attn_output_proj_linear_quant_group_size,
            out_dtype=self.params.activation_dtype,
        )

    def attn_output_proj_linear_layer_norm(
        self,
        num_chunks: int,
        chunk_idx: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, OptionalTensor]:
        self.params: LlamaDecoderLayerParamsQuant
        outs = self._dispatch_layer_norm(
            hidden_states,
            self.params.attn_output_proj_linear_layer_norm_weight,
            residual,
            self.params.mlp_linear1_smooth_scales,
        )

        self.mlp_linear1_input_scales[chunk_idx] = None
        if len(outs) == 3:
            self.mlp_linear1_input_scales[chunk_idx] = outs[2]

        return outs[:2]

    def mlp_linear1(
        self,
        num_chunks: int,
        chunk_idx: int,
        hidden_states: torch.Tensor,
        use_limited_gemm: bool = False,
    ) -> torch.Tensor:
        self.params: LlamaDecoderLayerParamsQuant
        self.mlp_linear1_num_chuns = num_chunks
        self.mlp_linear1_chunk_idx = chunk_idx
        return self._gemm_dispatcher(
            input=hidden_states,
            weight=self.params.mlp_linear1_weight,
            input_scales=self.mlp_linear1_input_scales[chunk_idx],
            weight_scales=self.params.mlp_linear1_weight_scales,
            out=None,
            use_limited_gemm=use_limited_gemm,
            quant_group_size=self.params.mlp_linear1_quant_group_size,
            out_dtype=self.params.activation_dtype,
        )

    def mlp_activation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.params: LlamaDecoderLayerParamsQuant
        if self.params.mlp_linear2_smooth_scales is None:
            self.mlp_linear2_input_scales[self.mlp_linear1_chunk_idx] = None
            return self.params.mlp_activation(hidden_states)

        out, scales = self.params.mlp_activation(
            hidden_states, self.params.mlp_linear2_smooth_scales
        )
        self.mlp_linear2_input_scales[self.mlp_linear1_chunk_idx] = scales
        return out

    def mlp_linear2(
        self,
        num_chunks: int,
        chunk_idx: int,
        hidden_states: torch.Tensor,
        out: OptionalTensor = None,
        use_limited_gemm: bool = False,
    ) -> torch.Tensor:
        self.params: LlamaDecoderLayerParamsQuant
        return self._gemm_dispatcher(
            input=hidden_states,
            weight=self.params.mlp_linear2_weight,
            input_scales=self.mlp_linear2_input_scales[chunk_idx],
            weight_scales=self.params.mlp_linear2_weight_scales,
            out=out,
            use_limited_gemm=use_limited_gemm,
            quant_group_size=self.params.mlp_linear2_quant_group_size,
            out_dtype=self.params.activation_dtype,
        )


def is_vllm_supported_quant_mode(quant_mode: Optional[str]):
    return quant_mode in [None, "smoothquant", "compressed_tensors"]


def get_vllm_llama_decoder_layer_protocol_cls(
    layer: torch.nn.Module, quant_mode: Optional[str]
):
    if quant_mode is None:
        return LlamaDecoderLayerOverlapDefault

    elif quant_mode in ["smoothquant", "compressed_tensors"]:
        return LlamaDecoderLayerOverlapQuant

    raise RuntimeError(f"got unsupported quantized mode: {quant_mode}.")


def create_vllm_llama_decoder_layer_params(
    layer: torch.nn.Module,
    quant_mode: Optional[str],
    activation_dtype: Optional[torch.dtype] = None,
) -> LlamaDecoderLayerParams:

    # 在 vllm 的 w8a8 中，input 的排布为 [M, K], weight 的排布为 [K, N] (通过是否是 contiguous 来判断)
    # 目前实现的 w8a8 不是该格式，需要将 weight 转置为 [N, K]
    def transpose_weight(weight: torch.Tensor):
        if not weight.is_contiguous() and weight.ndim == 2:
            return weight.transpose(0, 1)
        return weight

    T = transpose_weight

    def create_params_default():
        return LlamaDecoderLayerParams(
            pre_input_layer_norm_weight=layer.input_layernorm.weight,
            # qkv
            qkv_linear_weight=T(layer.self_attn.qkv_proj.weight),
            qkv_linear_bias=getattr(layer.self_attn.qkv_proj, "bias", None),
            # attn_output_proj
            attn_output_proj_linear_weight=T(layer.self_attn.o_proj.weight),
            attn_output_proj_linear_bias=layer.self_attn.o_proj.bias,
            # post layer norm
            attn_output_proj_linear_layer_norm_weight=layer.post_attention_layernorm.weight,
            # mlp linear1
            mlp_linear1_weight=T(layer.mlp.gate_up_proj.weight),
            mlp_linear1_bias=getattr(layer.mlp.gate_up_proj, "bias", None),
            # mlp linear2
            mlp_linear2_weight=T(layer.mlp.down_proj.weight),
            mlp_linear2_bias=getattr(layer.mlp.down_proj, "bias", None),
            mlp_activation=layer.mlp.act_fn,
            layer_norm_eps=layer.input_layernorm.variance_epsilon,
        )

    if quant_mode is None:
        return create_params_default()

    elif quant_mode in ["smoothquant", "compressed_tensors"]:
        if activation_dtype is None:
            raise RuntimeError(
                "The smooth quantization need activation dtype as the output of gemm_w8a8."
            )

        weight_scales_key = "weight_scales" if quant_mode == "smoothquant" else "weight_scale"
        smooth_scales_key = "smooth_scales" if quant_mode == "smoothquant" else "input_scale"

        model_params = create_params_default()
        params = LlamaDecoderLayerParamsQuant.create_from(
            model_params,
            # qkv
            qkv_linear_smooth_scales=getattr(
                layer.self_attn.qkv_proj, smooth_scales_key, None
            ),
            qkv_linear_weight_scales=getattr(
                layer.self_attn.qkv_proj, weight_scales_key, None
            ),
            # attn_output_proj
            attn_output_proj_linear_smooth_scales=getattr(
                layer.self_attn.o_proj, smooth_scales_key, None
            ),
            attn_output_proj_linear_weight_scales=getattr(
                layer.self_attn.o_proj, weight_scales_key, None
            ),
            # mlp_linear1
            mlp_linear1_smooth_scales=getattr(
                layer.mlp.gate_up_proj, smooth_scales_key, None
            ),
            mlp_linear1_weight_scales=getattr(
                layer.mlp.gate_up_proj, weight_scales_key, None
            ),
            # mlp_linear2
            mlp_linear2_smooth_scales=getattr(
                layer.mlp.down_proj, smooth_scales_key, None
            ),
            mlp_linear2_weight_scales=getattr(
                layer.mlp.down_proj, weight_scales_key, None
            ),
            # other
            activation_dtype=activation_dtype,
            quant_mode=quant_mode
        )

        if params.mlp_linear2_smooth_scales is not None:
            params.mlp_activation = ixff.silu_and_mul_smoothquant

        return params

    raise RuntimeError(f"unsupported quantized mode, got {quant_mode}.")


def create_vllm_llama_decoder_layer(
    layer: torch.nn.Module,
    model_id,
    layer_idx,
    enable_overlap=True,
    group=None,
    quant_config=None,
    activation_dtype: Optional[torch.dtype] = None,
) -> torch.nn.Module:
    """
    :param layer: vLLM 中 LLaMa 的 DecoderLayer
    :param model_id: 模型的 id，可以使用 id(model) 获取
    :param layer_idx: layer 的索引
    :param enable_overlap: 是否其中 overlap
    :param group: Communication group
    :param quant_config: vLLM 中量化的配置
    :param activation_dtype: 在使用 w8a8 的 gemm 时，需要通过该参数去决定输出的类型
    :return: vLLM 中 LLaMa 的 DecoderLayer
    """
    # 1. 如果是 overlap 不支持的量化类型 或者 不启用 overlap，那么直接返回 layer
    quant_mode = None if quant_config is None else quant_config.get_name()
    is_supported_quant_mode = is_vllm_supported_quant_mode(quant_mode=quant_mode)

    if (
        not enable_overlap
        or not config.IXFORMER_ENABLE_OVERLAP_COMM
        or not is_supported_quant_mode
    ):
        return layer

    # 2. 如果不是不是多卡推理，那么直接返回 layer
    if hasattr(group, "device_group"):
        group = group.device_group

    if not dist.is_initialized() or ixfd.get_world_size(group) < 2:
        return layer

    # 3. 是否使用 rotary enmedding
    rotary_embedding = None
    if (
        hasattr(layer.self_attn, "postion_embedding")
        and layer.self_attn.postion_embedding != "ALIBI"
    ) or (not hasattr(layer.self_attn, "postion_embedding")):
        rotary_embedding = layer.self_attn.rotary_emb

    # 4. 创建 Overlap 的 DecoderLayer
    protocol_cls = get_vllm_llama_decoder_layer_protocol_cls(
        layer=layer, quant_mode=quant_mode
    )

    params = create_vllm_llama_decoder_layer_params(
        layer=layer, quant_mode=quant_mode, activation_dtype=activation_dtype
    )

    num_chunks = 2
    overlap_layer = protocol_cls(
        model_id=model_id,
        layer_idx=layer_idx,
        attn_q_size=layer.self_attn.q_size,
        attn_kv_size=layer.self_attn.kv_size,
        params=params,
        max_num_chunks=num_chunks,
    )

    # 5. 替换 layer 的 forward
    def forward(
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, OptionalTensor]:
        """
        :param positions: 会被用在 RotaryEmbedding 中
        :param hidden_states: Shape[Batch * SeqLen, HiddenSize]
        :param kv_cache: 会被使用在 Attention 中
        :param input_metadata: 会被使用在 Attention 中
        :param residual: 残差
        :return: HiddenStates, Residual
        """
        return overlap_layer.forward(
            self_attn=layer.self_attn.attn,
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
            num_chunks=num_chunks,
            group=group,
            self_attn_kwargs={
                "kv_cache": kv_cache,
                "rotary_embedding": rotary_embedding,
                "attn_metadata": input_metadata,
            },
        )

    layer.forward = forward
    return layer
