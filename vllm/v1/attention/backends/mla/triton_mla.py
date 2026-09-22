# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

import torch

from vllm.attention.backends.abstract import (AttentionType,
                                              is_quantized_kv_cache)
from vllm.attention.ops.triton_decode_attention import decode_attention_fwd
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import (MLACommonBackend,
                                                   MLACommonImpl,
                                                   MLACommonMetadata)
import ixformer.inference.functions as ixf_ops
import vllm.envs as envs
from vllm import _custom_ops as ops

logger = init_logger(__name__)


class TritonMLABackend(MLACommonBackend):

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["TritonMLAImpl"]:
        return TritonMLAImpl


class TritonMLAImpl(MLACommonImpl[MLACommonMetadata]):

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[list[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            blocksparse_params: Optional[dict[str, Any]],
            logits_soft_cap: Optional[float],
            attn_type: str,
            # MLA Specific Arguments
            **mla_args) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         blocksparse_params, logits_soft_cap, attn_type,
                         **mla_args)

        unsupported_features = [
            alibi_slopes, sliding_window, blocksparse_params, logits_soft_cap
        ]
        if any(unsupported_features):
            raise NotImplementedError(
                "TritonMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, blocksparse_params, "
                "logits_soft_cap")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "TritonMLAImpl")

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "TritonMLA V1 with FP8 KV cache not yet supported")
        self._k_scale = torch.tensor(1.0, dtype=torch.float32)

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        kv_c_and_k_pe_cache_scale: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_c_normed: torch.Tensor=None,
        k_pe: torch.Tensor=None,
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError("FP8 Triton MLA not yet supported")

        B = q_nope.shape[0]
        q = torch.cat([q_nope, q_pe], dim=-1)

        o = torch.empty(B,
                        self.num_heads,
                        self.kv_lora_rank,
                        dtype=q_nope.dtype,
                        device=q_nope.device)

        # num_kv_splits = 4  # TODO: heuristic

        # # TODO(lucas) Allocate ahead of time
        # attn_logits = torch.empty(
        #     (
        #         B,
        #         self.num_heads,
        #         num_kv_splits,
        #         # NOTE(lucas) idk why the +1 is here but sglang has it so we
        #         # just mirror that
        #         self.kv_lora_rank + 1,
        #     ),
        #     dtype=torch.float32,
        #     device=q.device,
        # )

        # # Add a head dim of 1
        # kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)
        # kv_c_cache = kv_c_and_k_pe_cache[..., :self.kv_lora_rank]
        # PAGE_SIZE = kv_c_and_k_pe_cache.size(1)

        # # Run MQA
        # decode_attention_fwd(q, kv_c_and_k_pe_cache, kv_c_cache, o,
        #                      attn_metadata.decode.block_table,
        #                      attn_metadata.decode.seq_lens, attn_logits,
        #                      num_kv_splits, self.scale, PAGE_SIZE)
        if envs.VLLM_USE_INT8_MLA:
            q_int8, q_scale = ops.quant_kv(q)
            ixf_ops.vllm_paged_attention_mla_int8(
                o,
                q_int8,
                q_scale,
                kv_c_and_k_pe_cache,
                kv_c_and_k_pe_cache_scale, 
                self.scale,
                attn_metadata.decode.block_table,
                attn_metadata.decode.seq_lens,
                attn_metadata.decode.max_decode_seq_len,
                attn_metadata.decode.use_cuda_graph
            )
            
        else:
            # fused q concat & cache write
            ixf_ops.vllm_paged_attention_mla_fused(
                output=o,
                q_nope=q_nope,
                q_pe=q_pe.contiguous(),
                kv_cache=kv_c_and_k_pe_cache,
                scale=self.scale,
                block_tables=attn_metadata.decode.block_table,
                context_lens=attn_metadata.decode.seq_lens,
                max_context_len=attn_metadata.decode.max_decode_seq_len,
                k_c_normed=k_c_normed,
                k_pe=k_pe,
                use_cuda_graph=attn_metadata.decode.use_cuda_graph
            )
        return self._v_up_proj_and_o_proj(o)
