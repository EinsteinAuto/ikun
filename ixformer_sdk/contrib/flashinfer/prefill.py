import math
from typing import Optional, Tuple, Union

import ixformer._C as ops
import torch


class BatchPrefillWithRaggedKVCacheWrapper:
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
    ):
        pass

    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        # custom_mask: Optional[torch.Tensor] = None,
        # packed_custom_mask: Optional[torch.Tensor] = None,
        causal: bool = True,
        # pos_encoding_mode: str = "NONE",
        # allow_fp16_qk_reduction: bool = False,
        # window_left: int = -1,
        # logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        # rope_scale: Optional[float] = None,
        # rope_theta: Optional[float] = None,
        # q_data_type: str = "float16",
    ) -> None:
        batch_size = len(qo_indptr) - 1
        if len(kv_indptr) != batch_size + 1:
            raise ValueError(
                "The kv_indptr length should be equal to qk_indptr length."
            )
        self._causal = causal
        self._sm_scale = sm_scale
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(head_dim)

        self.cu_seqlens_q = qo_indptr
        self.cu_seqlens_k = kv_indptr
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k

    begin_forward = plan

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = True,
        # pos_encoding_mode: str = "NONE",
        # allow_fp16_qk_reduction: bool = False,
        # window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        # rope_scale: Optional[float] = None,
        # rope_theta: Optional[float] = None,
    ) -> torch.Tensor:
        r"""Warning: This function is deprecated, please use :meth:`run` instead."""

        q = q.view(-1, self.num_qo_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        out = torch.empty_like(q)

        assert causal
        assert (
            logits_soft_cap is None or logits_soft_cap == 0
        ), f"logits_soft_cap not supported, but got logits_soft_cap={logits_soft_cap}"

        ops.infer.ixinfer_flash_attn_unpad(
            q,
            k,
            v,
            out,
            self.cu_seqlens_q,
            self.cu_seqlens_k,
            self.max_seqlen_q,
            self.max_seqlen_k,
            causal,
            False,  # need_lse =False
            sm_scale,
            False,
            None,
        )
        return out

    def end_forward(self) -> None:
        r"""Warning: this function is deprecated and has no effect."""
        pass


class BatchPrefillWithPagedKVCacheWrapper:
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
    ) -> None:
        pass
