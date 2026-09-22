import dataclasses
from typing import Optional, Tuple

import ixformer.distributed as ixfd
import ixformer.functions as F
import torch
import torch.distributed as dist
from ixformer.contrib.vllm_flash_attn import flash_attn_varlen_func
from ixformer.distributed.overlap_comm import SplitOverlapComm

from ixformer.core import config as ixff_config


@dataclasses.dataclass
class FmhaOProjAllReduceLnGatingParams:
    # ==============================
    # attention
    # ==============================

    # shape: [Batch * SeqLen, NumHeads / TP, HeadDim]
    q: torch.Tensor

    # shape: [Batch * SeqLen, NumHeads / TP, HeadDim]
    k: torch.Tensor

    # shape: [Batch * SeqLen, NumHeads / TP, HeadDim]
    v: torch.Tensor

    # shape [Batch + 1], dtype torch.int32. The cumulative sequence lengths
    # of the sequences in the batch, used to index into q.
    cu_seqlens_q: torch.Tensor

    # shape: [Batch + 1], dtype torch.int32. The cumulative sequence lengths
    # of the sequences in the batch, used to index into kv.
    cu_seqlens_k: torch.Tensor

    # Maximum query sequence length in the batch.
    max_seqlen_q: int

    # Maximum key sequence length in the batch.
    max_seqlen_k: int

    # ==============================
    # o_proj
    # ==============================

    # shape: [HiddenSize, NumHeads * HeadDim / TP], dtype: int8
    o_proj_weight: torch.Tensor

    # shape: [HiddenSize], dtype: float32
    o_proj_weight_scale: torch.Tensor

    # shape: [HiddenSize]
    o_proj_bias: torch.Tensor

    # shape: [NumHeads * HeadDim / TP], dtype: float16 or bfloat16
    o_proj_smooth_scale: torch.Tensor

    # ==============================
    # ln
    # ==============================

    # shape: [Batch * SeqLen, HiddenSize], dtype: float16 or bfloat16
    residual: torch.Tensor

    # shape: [HiddenSize], dtype: float16 or bfloat16
    ln_weight: torch.Tensor

    # shape: [HiddenSize], dtype: float16 or bfloat16
    ln_bias: torch.Tensor

    # ==============================
    # gating linear
    # ==============================

    # shape: [TopK, HiddenSize], dtype: float16 or bfloat16
    gating_weight: torch.Tensor

    # shape: [SeqLen, TopK], dtype: float16 or bfloat16
    out: Optional[torch.Tensor] = None

    # ==============================
    # default parameters
    # ==============================

    # the seqlens of q for per chunk when using overlap,
    # the parameter can be initiated by params.prepare_overlap_params(),
    # and only need to initialize once during the model's forward.
    cu_seqlens_q_chunks = None
    cu_seqlens_k_chunks = None

    softmax_scale: Optional[float] = None
    ln_eps: float = 1e-5

    @property
    def batch(self):
        return len(self.cu_seqlens_q) - 1

    @property
    def seqlen(self):
        return self.q.shape[0]

    @property
    def topk(self):
        return self.gating_weight.shape[0]

    def prepare_overlap_params(self):
        """compute the cu_seqlens qk of chunk when using overlap"""
        first_chunk_size = int(self.q.shape[0] // 2)

        if not hasattr(self.cu_seqlens_q, "q_chunks"):
            first_q_chunks_cu_seqlens = self.cu_seqlens_q.clone()
            first_q_chunks_cu_seqlens[-1] = first_chunk_size

            self.cu_seqlens_q.q_chunks = [
                first_q_chunks_cu_seqlens,
                first_q_chunks_cu_seqlens,
            ]

        self.cu_seqlens_q_chunks = self.cu_seqlens_q.q_chunks

        if not hasattr(self.cu_seqlens_k, "k_chunks"):
            first_chunk = self.cu_seqlens_k.clone()
            last_chunk = self.cu_seqlens_k
            first_chunk[-1] = first_chunk_size
            self.cu_seqlens_k.k_chunks = [first_chunk, last_chunk]

        self.cu_seqlens_k_chunks = self.cu_seqlens_k.k_chunks

        return self


class FmhaOProjAllreduceLnGatingOverlap(SplitOverlapComm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.num_chunks != 2:
            raise RuntimeError(
                f"Overlap only support num_chunks == 2, but got {self.num_chunks}."
            )

        self.allreduce_end_events = [torch.cuda.Event() for _ in range(self.num_chunks)]

    def start_ln_gating(self, chunk_idx):
        compute_stream = self._compute_streams[chunk_idx % self.num_compute_streams]
        compute_stream.wait_event(self.allreduce_end_events[chunk_idx])

    def compute(self, params: FmhaOProjAllReduceLnGatingParams):
        if params.out is None:
            params.out = torch.empty(
                [params.seqlen, params.topk], device="cuda", dtype=torch.float
            )

        seqlen_chunks = [int(params.q.shape[0] // 2)]
        seqlen_chunks.append(params.q.shape[0] - seqlen_chunks[0])

        q_chunks = torch.split_with_sizes(params.q, seqlen_chunks, dim=0)

        ar_out_chunks = []
        for chunk_idx in range(len(seqlen_chunks)):
            with self.compute_stream_context(chunk_idx):
                hidden_states = flash_attn_varlen_func(
                    q=q_chunks[chunk_idx],
                    k=params.k,
                    v=params.v,
                    cu_seqlens_q=params.cu_seqlens_q_chunks[chunk_idx],
                    cu_seqlens_k=params.cu_seqlens_k_chunks[chunk_idx],
                    max_seqlen_q=seqlen_chunks[chunk_idx],
                    max_seqlen_k=seqlen_chunks[0]
                    if chunk_idx == 0
                    else params.max_seqlen_k,
                    softmax_scale=params.softmax_scale,
                    causal=True,
                    window_size=(-1, -1),
                    alibi_slopes=None,
                    softcap=0,
                )

                hidden_states = hidden_states.view(hidden_states.shape[0], -1)
                hidden_states, i_scales = F.dynamic_scaled_quant_dynamic_int8(
                    hidden_states, params.o_proj_smooth_scale
                )

                out_chunk = F.w8a8(
                    hidden_states,
                    params.o_proj_weight,
                    i_scales,
                    params.o_proj_weight_scale,
                    bias=params.o_proj_bias,
                    out_dtype=params.residual.dtype,
                    output=None,
                    persistent=True,
                )

            self.start_comm(chunk_idx)

            ixfd.all_reduce(
                out_chunk, async_op=True, group=self.comm_group, use_comm_stream=True
            )
            ar_out_chunks.append(out_chunk)

            self.allreduce_end_events[chunk_idx].record(self._comm_stream)

        ln_out = torch.empty_like(params.residual)
        ln_out_chunks = ln_out.chunk(2, dim=0)
        residual_chunks = torch.split_with_sizes(params.residual, seqlen_chunks, dim=0)

        if params.out is None:
            params.out = torch.empty(
                [params.seqlen, params.topk], dtype=params.residual.dtype, device="cuda"
            )

        out_chunks = list(torch.split_with_sizes(params.out, seqlen_chunks, dim=0))

        for chunk_idx in range(len(seqlen_chunks)):
            self.start_ln_gating(chunk_idx)
            with self.compute_stream_context(chunk_idx):
                ln_out_chunk, residual_chunk = F.residual_layer_norm(
                    input=ar_out_chunks[chunk_idx],
                    weight=params.ln_weight,
                    bias=params.ln_bias,
                    residual=residual_chunks[chunk_idx].reshape(
                        ar_out_chunks[chunk_idx].shape
                    ),
                    eps=params.ln_eps,
                    output=ln_out_chunks[chunk_idx],
                )
                if ln_out_chunk.dtype == params.gating_weight.dtype:
                    F.linear(
                        ln_out_chunk, params.gating_weight, output=out_chunks[chunk_idx]
                    )
                else:
                    F.mixed_type_linear(
                        ln_out_chunk, params.gating_weight, output=out_chunks[chunk_idx]
                    )

        return (
            params.residual.reshape(params.batch, params.seqlen, -1),
            ln_out.reshape(params.batch, params.seqlen, -1),
            params.out,
        )


_fa_o_proj_allreduce_ln_gating_overlap = None


def fmha_oproj_allreduce_ln_gating(
    params: FmhaOProjAllReduceLnGatingParams,
    enable_overlap: bool = False,
    comm_group=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    FMHA + OProjLinear + AllReduce + LayerNorm + GatingLinear

    Args:
        params: fused operator params
        enable_overlap: whether enable overlap
        comm_group: communication group
    Returns:
        Residual: shape: [Batch * SeqLen, HiddenSize], dtype: float16 or bfloat16
        HiddenStates: shape: [Batch * SeqLen, HiddenSize], dtype: float16 or bfloat16
        GatingLinearOutput: shape: [SeqLen, TopK], dtype: float16 or bfloat16
    """

    global _fa_o_proj_allreduce_ln_gating_overlap
    if _fa_o_proj_allreduce_ln_gating_overlap is None:
        _fa_o_proj_allreduce_ln_gating_overlap = (
            FmhaOProjAllreduceLnGatingOverlap.dispatcher(
                num_chunks=2, comm_group=comm_group
            ).forward
        )

    if (
        enable_overlap
        and ixff_config.IXFORMER_ENABLE_OVERLAP_COMM
        and dist.is_initialized()
        and dist.get_world_size(comm_group) > 1
        and params.batch == 1
        and params.seqlen > 1
    ):
        if params.cu_seqlens_q_chunks is None:
            params = params.prepare_overlap_params()
        return _fa_o_proj_allreduce_ln_gating_overlap(params)

    hidden_states = flash_attn_varlen_func(
        q=params.q,
        k=params.k,
        v=params.v,
        cu_seqlens_q=params.cu_seqlens_q,
        cu_seqlens_k=params.cu_seqlens_k,
        max_seqlen_q=params.max_seqlen_q,
        max_seqlen_k=params.max_seqlen_k,
        softmax_scale=params.softmax_scale,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        softcap=0,
    )

    input = hidden_states.view(hidden_states.shape[0], -1)
    input, i_scales = F.dynamic_scaled_quant_smoothquant(
        input, params.o_proj_smooth_scale
    )

    hidden_states = F.w8a8(
        input,
        params.o_proj_weight,
        i_scales,
        params.o_proj_weight_scale,
        bias=params.o_proj_bias,
        out_dtype=params.residual.dtype,
        output=None,
    )
    ixfd.all_reduce(hidden_states, async_op=True, group=comm_group)

    hidden_states, residual = F.residual_layer_norm(
        input=hidden_states,
        weight=params.ln_weight,
        bias=params.ln_bias,
        residual=params.residual.reshape(hidden_states.shape),
        eps=params.ln_eps,
    )
    if hidden_states.dtype == params.gating_weight.dtype:
        out = F.linear(hidden_states, params.gating_weight, output=params.out)
    else:
        out = F.mixed_type_linear(
            hidden_states, params.gating_weight, output=params.out
        )
    return (
        residual.reshape(params.batch, params.seqlen, -1),
        hidden_states.reshape(params.batch, params.seqlen, -1),
        out,
    )
