# SPDX-License-Identifier: Apache-2.0
"""Attention layer with xFormers and PagedAttention.
BI-V100 adaptation: ixformer imports + SDPA fallback for head_dim > 128.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
# from xformers import ops as xops
from ixformer.contrib.xformers import ops as xops
from xformers.ops.fmha.attn_bias import (AttentionBias,
                                         BlockDiagonalMask,)
from ixformer.contrib.xformers.ops.fmha.attn_bias import (BlockDiagonalCausalMask,
                                                          LowerTriangularMaskWithTensorBias)

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata, AttentionType)
from vllm.attention.backends.utils import (
    CommonAttentionState, CommonMetadataBuilder,
    get_num_prefill_decode_query_kv_tokens, get_seq_len_block_table_args,
    is_all_cross_attn_metadata_set, is_all_encoder_attn_metadata_set)
from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)
from vllm.bi100_profile import bi100_timer
from vllm.logger import init_logger

logger = init_logger(__name__)


class XFormersBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "XFORMERS"

    @staticmethod
    def get_impl_cls() -> Type["XFormersImpl"]:
        return XFormersImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return XFormersMetadata

    @staticmethod
    def get_builder_cls() -> Type["XFormersMetadataBuilder"]:
        return XFormersMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                 num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)


@dataclass
class XFormersMetadata(AttentionMetadata, PagedAttentionMetadata):
    """Metadata for XFormersbackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """

    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ----------------------|
    #                                   |-- query_len ---|

    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # FIXME: It is for flash attn.
    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool

    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]] = None

    # FIXME: It is for flash attn.
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor] = None

    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor] = None

    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int] = None

    # Max number of query tokens among request in the batch.
    max_decode_query_len: Optional[int] = None

    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor] = None

    # Self-attention prefill/decode metadata cache
    _cached_prefill_metadata: Optional["XFormersMetadata"] = None
    _cached_decode_metadata: Optional["XFormersMetadata"] = None

    # Begin encoder attn & enc/dec cross-attn fields...

    # Encoder sequence lengths representation
    encoder_seq_lens: Optional[List[int]] = None
    encoder_seq_lens_tensor: Optional[torch.Tensor] = None
    # FIXME: It is for flash attn.
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    encoder_seq_start_loc: Optional[torch.Tensor] = None

    # Maximum sequence length among encoder sequences
    max_encoder_seq_len: Optional[int] = None

    # Number of tokens input to encoder
    num_encoder_tokens: Optional[int] = None

    # Cross-attention memory-mapping data structures: slot mapping
    # and block tables
    cross_slot_mapping: Optional[torch.Tensor] = None
    cross_block_tables: Optional[torch.Tensor] = None

    def __post_init__(self):
        # Set during the execution of the first attention op.
        # It is a list because it is needed to set per prompt
        # when alibi slopes is used. It is because of the limitation
        # from xformer API.
        # will not appear in the __repr__ and __init__
        self.attn_bias: Optional[List[AttentionBias]] = None
        self.encoder_attn_bias: Optional[List[AttentionBias]] = None
        self.cross_attn_bias: Optional[List[AttentionBias]] = None

    @property
    def is_all_encoder_attn_metadata_set(self):
        '''
        All attention metadata required for encoder attention is set.
        '''
        return is_all_encoder_attn_metadata_set(self)

    @property
    def is_all_cross_attn_metadata_set(self):
        '''
        All attention metadata required for enc/dec cross-attention is set.

        Superset of encoder attention required metadata.
        '''
        return is_all_cross_attn_metadata_set(self)

    @property
    def prefill_metadata(self) -> Optional["XFormersMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata

        assert ((self.seq_lens is not None)
                or (self.encoder_seq_lens is not None))
        assert ((self.seq_lens_tensor is not None)
                or (self.encoder_seq_lens_tensor is not None))

        # Compute some attn_metadata fields which default to None
        query_start_loc = (None if self.query_start_loc is None else
                           self.query_start_loc[:self.num_prefills + 1])
        seq_start_loc = (None if self.seq_start_loc is None else
                         self.seq_start_loc[:self.num_prefills + 1])
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[:self.num_prefill_tokens])
        seq_lens = (None if self.seq_lens is None else
                    self.seq_lens[:self.num_prefills])
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[:self.num_prefills])
        context_lens_tensor = (None if self.context_lens_tensor is None else
                               self.context_lens_tensor[:self.num_prefills])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[:self.num_prefills])

        self._cached_prefill_metadata = XFormersMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=self.
            multi_modal_placeholder_index_maps,
            enable_kv_scales_calculation=self.enable_kv_scales_calculation,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables)
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["XFormersMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata
        assert ((self.seq_lens_tensor is not None)
                or (self.encoder_seq_lens_tensor is not None))

        # Compute some attn_metadata fields which default to None
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[self.num_prefill_tokens:])
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[self.num_prefills:])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[self.num_prefills:])

        self._cached_decode_metadata = XFormersMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=True,
            seq_lens_tensor=seq_lens_tensor,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            block_tables=block_tables,
            use_cuda_graph=self.use_cuda_graph,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables)

        # Batch may be composed of prefill|decodes, adjust query start indices
        # to refer to the start of decodes when the two are split apart.
        # E.g. in tokens:[3 prefills|6 decodes], query_start_loc=[3,9] => [0,6].
        if self._cached_decode_metadata.query_start_loc is not None:
            qs = self._cached_decode_metadata.query_start_loc
            self._cached_decode_metadata.query_start_loc = qs - qs[0]
        return self._cached_decode_metadata


def _get_attn_bias(
    attn_metadata: XFormersMetadata,
    attn_type: str,
) -> Optional[AttentionBias]:
    '''
    Extract appropriate attention bias from attention metadata
    according to attention type.
    '''

    if (attn_type == AttentionType.DECODER
            or attn_type == AttentionType.ENCODER_ONLY):
        return attn_metadata.attn_bias
    elif attn_type == AttentionType.ENCODER:
        return attn_metadata.encoder_attn_bias
    elif attn_type == AttentionType.ENCODER_DECODER:
        return attn_metadata.cross_attn_bias
    else:
        raise AttributeError(f"Invalid attention type {str(attn_type)}")


def _set_attn_bias(
    attn_metadata: XFormersMetadata,
    attn_bias: List[Optional[AttentionBias]],
    attn_type: str,
) -> None:
    '''
    Update appropriate attention bias field of attention metadata,
    according to attention type.
    '''

    if (attn_type == AttentionType.DECODER
            or attn_type == AttentionType.ENCODER_ONLY):
        attn_metadata.attn_bias = attn_bias
    elif attn_type == AttentionType.ENCODER:
        attn_metadata.encoder_attn_bias = attn_bias
    elif attn_type == AttentionType.ENCODER_DECODER:
        attn_metadata.cross_attn_bias = attn_bias
    else:
        raise AttributeError(f"Invalid attention type {str(attn_type)}")


class XFormersMetadataBuilder(CommonMetadataBuilder[XFormersMetadata]):

    _metadata_cls = XFormersMetadata


class XFormersImpl(AttentionImpl[XFormersMetadata]):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prefill_tokens ----------------->|
    |<--prefill_0-->|<--prefill_1-->|...|<--prefill_N-1--->|

    Otherwise, the layout is as follows:
    |<----------------- num_decode_tokens ------------------>|
    |<--decode_0-->|..........|<--decode_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.

    If chunked prefill is enabled, prefill tokens and decode tokens can be
    batched together in a flattened 1D query.

    |<----- num_prefill_tokens ---->|<------- num_decode_tokens --------->|
    |<-prefill_0->|...|<-prefill_N-1->|<--decode_0-->|...|<--decode_M-1-->|

    Currently, cuda graph is disabled for chunked prefill, meaning there's no
    padding between prefill and decode tokens.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "XFormers does not support block-sparse attention.")
        if logits_soft_cap is not None:
            logger.warning_once("XFormers does not support logits soft cap. "
                                "Outputs may be slightly off.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        suppored_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in suppored_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {suppored_head_sizes}.")

        self.attn_type = attn_type

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        kv_cache: torch.Tensor,
        kv_cache_scale: torch.Tensor,
        attn_metadata: "XFormersMetadata",
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with xFormers and PagedAttention."""
        # [DEBUG] Check if kv_cache is properly bound
        _xf_diag = getattr(self, '_xf_diag_cnt', 0)
        if _xf_diag < 3:
            self._xf_diag_cnt = _xf_diag + 1
            logger.info(
                "[DEBUG_XFORMERS] call=%d kv_cache.shape=%s numel=%d "
                "num_prefills=%s num_decode=%s",
                _xf_diag,
                tuple(kv_cache.shape) if kv_cache.numel() > 0 else "(empty)",
                kv_cache.numel(),
                getattr(attn_metadata, 'num_prefills', '?'),
                getattr(attn_metadata, 'num_decode_tokens', '?'))
        attn_type = self.attn_type
        # Check that appropriate attention metadata attributes are
        # selected for the desired attention type
        if (attn_type == AttentionType.ENCODER
                and (not attn_metadata.is_all_encoder_attn_metadata_set)):
            raise AttributeError("Encoder attention requires setting "
                                 "encoder metadata attributes.")

        elif (attn_type == AttentionType.ENCODER_DECODER
              and (not attn_metadata.is_all_cross_attn_metadata_set)):
            raise AttributeError("Encoder/decoder cross-attention "
                                 "requires setting cross-attention "
                                 "metadata attributes.")

        query = query.view(-1, self.num_heads, self.head_size)
        if key is not None:
            assert value is not None
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
        else:
            assert value is None

        if (attn_type != AttentionType.ENCODER and kv_cache.numel() > 0):
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)

            if (key is not None) and (value is not None):

                if attn_type == AttentionType.ENCODER_DECODER:
                    updated_slot_mapping = attn_metadata.cross_slot_mapping
                else:
                    updated_slot_mapping = attn_metadata.slot_mapping

                with bi100_timer("xformers.kv_write"):
                    PagedAttention.write_to_paged_cache(
                        key, value, key_cache, value_cache,
                        updated_slot_mapping, self.kv_cache_dtype,
                        layer._k_scale, layer._v_scale)

        (num_prefill_query_tokens, num_prefill_kv_tokens,
        num_decode_query_tokens) = \
            get_num_prefill_decode_query_kv_tokens(attn_metadata, attn_type)

        output = torch.empty_like(query)
        # Query for decode. KV is not needed because it is already cached.
        decode_query = query[num_prefill_query_tokens:]
        # QKV for prefill.
        query = query[:num_prefill_query_tokens]
        if key is not None and value is not None:
            key = key[:num_prefill_kv_tokens]
            value = value[:num_prefill_kv_tokens]

        assert query.shape[0] == num_prefill_query_tokens
        assert decode_query.shape[0] == num_decode_query_tokens

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            if kv_cache.numel() == 0 or prefill_meta.block_tables.numel() == 0:
                # normal attention.
                with bi100_timer("xformers.dense_prefill"):
                    out = self._run_memory_efficient_xformers_forward(
                        query, key, value, prefill_meta, attn_type=attn_type)
                assert out.shape == output[:num_prefill_query_tokens].shape
                output[:num_prefill_query_tokens] = out
            else:
                assert attn_type != AttentionType.ENCODER_ONLY, (
                    "Encoder-only models should not have prefix attention.")

                assert prefill_meta.query_start_loc is not None
                assert prefill_meta.max_query_len is not None

                with bi100_timer("xformers.paged_prefill"):
                    out = PagedAttention.forward_prefix(
                        query,
                        key,
                        value,
                        self.kv_cache_dtype,
                        key_cache,
                        value_cache,
                        prefill_meta.block_tables,
                        prefill_meta.query_start_loc,
                        prefill_meta.seq_lens_tensor,
                        prefill_meta.max_query_len,
                        self.alibi_slopes,
                        self.sliding_window,
                        layer._k_scale,
                        layer._v_scale,
                    )
                assert output[:num_prefill_query_tokens].shape == out.shape
                output[:num_prefill_query_tokens] = out

        if decode_meta := attn_metadata.decode_metadata:
            assert attn_type != AttentionType.ENCODER_ONLY, (
                "Encoder-only models should not have decode metadata.")

            _dec_diag = getattr(self, '_dec_diag', 0)
            if _dec_diag < 3:
                self._dec_diag = _dec_diag + 1
                logger.info(
                    "[DEBUG_DECODE] call=%d kv_numel=%d "
                    "seq_lens=%s block_tables_shape=%s "
                    "num_prefill_q=%d num_decode_q=%d",
                    _dec_diag, kv_cache.numel(),
                    decode_meta.seq_lens_tensor[:3].tolist()
                        if decode_meta.seq_lens_tensor is not None else None,
                    tuple(decode_meta.block_tables.shape)
                        if decode_meta.block_tables is not None else None,
                    num_prefill_query_tokens, num_decode_query_tokens)

            (
                seq_lens_arg,
                max_seq_len_arg,
                block_tables_arg,
            ) = get_seq_len_block_table_args(decode_meta, False, attn_type)

            output[num_prefill_query_tokens:] = PagedAttention.forward_decode(
                decode_query,
                key_cache,
                value_cache,
                block_tables_arg,
                seq_lens_arg,
                max_seq_len_arg,
                self.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
                layer._k_scale,
                layer._v_scale,
            )

        # Reshape the output tensor.
        return output.view(-1, self.num_heads * self.head_size)

    def _run_sdpa_fallback(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: "XFormersMetadata",
    ) -> torch.Tensor:
        """Pure-math causal attention fallback with Q-tiling for head_size > 128.

        ixformer flash attention doesn't support head_dim > 128.
        Qwen3.6 uses head_dim=256 -> must use SDPA fallback.

        GQA broadcast matmul avoids materializing expanded KV
        (6x memory savings for Qwen3.6 ratio=6).
        Q-tiling: split Q into 256-token chunks to keep peak memory
        at O(chunk x seq_len) instead of O(seq_len^2).

        Memory: O(_Q_CHUNK x seq_len) instead of O(seq_len^2).
        """
        _Q_CHUNK = 256

        assert attn_metadata.seq_lens is not None
        orig_dtype = query.dtype
        num_seqs = len(attn_metadata.seq_lens)

        if (attn_metadata.query_start_loc is not None
                and len(attn_metadata.query_start_loc) == num_seqs + 1):
            q_lens = [
                int(attn_metadata.query_start_loc[i + 1].item()) -
                int(attn_metadata.query_start_loc[i].item())
                for i in range(num_seqs)
            ]
        else:
            q_lens = list(attn_metadata.seq_lens)

        q_flat = query.squeeze(0)
        k_flat = key.squeeze(0)
        v_flat = value.squeeze(0)

        output = torch.empty_like(q_flat)
        seq_start = 0
        for q_len in q_lens:
            seq_end = seq_start + q_len

            k_s = k_flat[seq_start:seq_end].permute(1, 0, 2).float()
            v_s = v_flat[seq_start:seq_end].permute(1, 0, 2).float()

            gqa_ratio = self.num_heads // k_s.shape[0]
            if gqa_ratio > 1:
                k_s = k_s.unsqueeze(1)
                v_s = v_s.unsqueeze(1)
                use_gqa_broadcast = True
            else:
                use_gqa_broadcast = False

            k_pos = torch.arange(q_len, device=query.device)
            _max_chunk = min(_Q_CHUNK, q_len)
            _qc_q_pos_base = torch.arange(_max_chunk, device=query.device)
            _num_chunks = (q_len + _Q_CHUNK - 1) // _Q_CHUNK

            for qc_idx in range(_num_chunks):
                qc_start = qc_idx * _Q_CHUNK
                qc_end = min(qc_start + _Q_CHUNK, q_len)
                chunk_len = qc_end - qc_start
                qc_q_pos = _qc_q_pos_base[:chunk_len] + qc_start

                if use_gqa_broadcast:
                    q_c = (q_flat[seq_start + qc_start:seq_start + qc_end]
                           .float()
                           .view(-1, self.num_kv_heads, gqa_ratio,
                                 self.head_size)
                           .permute(1, 2, 0, 3))

                    attn_w = torch.matmul(
                        q_c, k_s.transpose(-2, -1)) * self.scale

                    mask = k_pos.unsqueeze(0) > qc_q_pos.unsqueeze(1)
                    attn_w = attn_w.masked_fill(
                        mask.unsqueeze(0).unsqueeze(0), float("-inf"))

                    attn_w = torch.softmax(attn_w, dim=-1)
                    out_c = torch.matmul(attn_w, v_s).to(orig_dtype)
                    out_c = (out_c.permute(2, 0, 1, 3)
                             .contiguous()
                             .view(-1, self.num_heads, self.head_size))
                    output[seq_start + qc_start:seq_start + qc_end] = out_c
                else:
                    q_c = q_flat[seq_start + qc_start:seq_start + qc_end] \
                          .permute(1, 0, 2).float()

                    attn_w = torch.matmul(
                        q_c, k_s.transpose(-2, -1)) * self.scale

                    mask = k_pos.unsqueeze(0) > qc_q_pos.unsqueeze(1)
                    attn_w = attn_w.masked_fill(
                        mask.unsqueeze(0), float("-inf"))

                    attn_w = torch.softmax(attn_w, dim=-1)
                    out_c = torch.matmul(attn_w, v_s).to(orig_dtype)

                    output[seq_start + qc_start:seq_start + qc_end] = (
                        out_c.permute(1, 0, 2))

            seq_start = seq_end

        return output.unsqueeze(0)

    def _run_memory_efficient_xformers_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: XFormersMetadata,
        attn_type: str = AttentionType.DECODER,
    ) -> torch.Tensor:
        """Attention for 1D query of multiple prompts. Multiple prompt
        tokens are flattened in to `query` input.

        See https://facebookresearch.github.io/xformers/components/ops.html
        for API spec.

        Args:
            output: shape = [num_prefill_tokens, num_heads, head_size]
            query: shape = [num_prefill_tokens, num_heads, head_size]
            key: shape = [num_prefill_tokens, num_kv_heads, head_size]
            value: shape = [num_prefill_tokens, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
            attn_type: Select attention type, between encoder attention,
                       decoder self-attention, or encoder/decoder cross-
                       attention. Defaults to decoder self-attention,
                       which is the vLLM default generally
        """

        original_query = query
        # GQA/MQA is NOT reshaped here for BI-V100 because the
        # _run_sdpa_fallback handles GQA internally with broadcast matmul.
        # The standard xformers path (head_size <= 128) also does not need
        # GQA reshape because ixformer handles it natively.

        # Set attention bias if not provided. This typically happens at
        # the very attention layer of every iteration.
        # FIXME(woosuk): This is a hack.
        attn_bias = _get_attn_bias(attn_metadata, attn_type)
        if attn_bias is None:
            if self.alibi_slopes is None:

                # Cross attention block of decoder branch of encoder-decoder
                # model uses seq_lens for dec / encoder_seq_lens for enc
                if (attn_type == AttentionType.ENCODER_DECODER):
                    assert attn_metadata.seq_lens is not None
                    assert attn_metadata.encoder_seq_lens is not None

                    # Cross-attention mask is non-causal
                    attn_bias = BlockDiagonalMask.from_seqlens(
                        attn_metadata.seq_lens, attn_metadata.encoder_seq_lens)

                # Encoder branch of encoder-decoder model uses
                # attn_metadata.encoder_seq_lens
                elif attn_type == AttentionType.ENCODER:

                    assert attn_metadata.encoder_seq_lens is not None

                    # Encoder self-attention mask is non-causal
                    attn_bias = BlockDiagonalMask.from_seqlens(
                        attn_metadata.encoder_seq_lens)

                # Self-attention block of encoder-only model just
                # uses the seq_lens directly.
                elif attn_type == AttentionType.ENCODER_ONLY:
                    assert attn_metadata.seq_lens is not None

                    # Encoder self-attention mask is non-causal
                    attn_bias = BlockDiagonalMask.from_seqlens(
                        attn_metadata.seq_lens)

                # Self-attention block of decoder branch just
                # uses the seq_lens directly
                elif attn_type == AttentionType.DECODER:
                    assert attn_metadata.seq_lens is not None

                    # Decoder self-attention mask is causal
                    attn_bias = BlockDiagonalCausalMask.from_seqlens(
                        attn_metadata.seq_lens)
                else:
                    raise ValueError("Unknown AttentionType: %s", attn_type)

                if self.sliding_window is not None:
                    attn_bias = attn_bias.make_local_attention(
                        self.sliding_window)
                attn_bias = [attn_bias]
            else:
                assert attn_type == AttentionType.DECODER
                assert attn_metadata.seq_lens is not None
                attn_bias = _make_alibi_bias(self.alibi_slopes,
                                             self.num_kv_heads, query.dtype,
                                             attn_metadata.seq_lens)

            _set_attn_bias(attn_metadata, attn_bias, attn_type)

        # No alibi slopes.
        # TODO(woosuk): Too many view operations. Let's try to reduce
        # them in the future for code readability.
        self.attn_op = xops.fmha.flash.FwOp()
        if self.alibi_slopes is None:
            # Add the batch dimension.
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            # BI-V100: ixformer flash attention doesn't support head_dim > 128.
            # Qwen3.6 uses head_dim=256 -> must use SDPA fallback.
            if self.head_size > 128:
                out = self._run_sdpa_fallback(
                    query, key, value, attn_metadata)
            else:
                out = xops.memory_efficient_attention_forward(
                    query,
                    key,
                    value,
                    attn_bias=attn_bias[0],
                    p=0.0,
                    scale=self.scale,
                    op=self.attn_op
                    )
            return out.view_as(original_query)

        # Attention with alibi slopes.
        # FIXME(woosuk): Because xformers does not support dynamic sequence
        # lengths with custom attention bias, we process each prompt one by
        # one. This is inefficient, especially when we have many short prompts.
        assert attn_metadata.seq_lens is not None
        output = torch.empty_like(original_query)
        start = 0
        for i, seq_len in enumerate(attn_metadata.seq_lens):
            end = start + seq_len
            out = xops.memory_efficient_attention_forward(
                query[None, start:end],
                key[None, start:end],
                value[None, start:end],
                attn_bias=attn_bias[i],
                p=0.0,
                scale=self.scale,
                )
            # TODO(woosuk): Unnecessary copy. Optimize.
            output[start:end].copy_(out.view_as(original_query[start:end]))
            start += seq_len
        return output


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    dtype: torch.dtype,
    seq_lens: List[int],
) -> List[AttentionBias]:
    attn_biases: List[AttentionBias] = []
    for seq_len in seq_lens:
        bias = torch.arange(seq_len, dtype=dtype)
        # NOTE(zhuohan): HF uses
        #     `bias = bias[None, :].repeat(seq_len, 1)`
        # here. We find that both biases give the same results, but
        # the bias below more accurately follows the original ALiBi
        # paper.
        # Calculate a matrix where each element represents ith element- jth
        # element.
        bias = bias[None, :] - bias[:, None]

        padded_len = (seq_len + 7) // 8 * 8
        num_heads = alibi_slopes.shape[0]
        bias = torch.empty(
            1,  # batch size
            num_heads,
            seq_len,
            padded_len,
            device=alibi_slopes.device,
            dtype=dtype,
        )[:, :, :, :seq_len].copy_(bias)
        bias.mul_(alibi_slopes[:, None, None])
        attn_biases.append(LowerTriangularMaskWithTensorBias(bias))

    return attn_biases
