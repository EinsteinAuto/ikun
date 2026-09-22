import math
from typing import Optional, Tuple, Union

import ixformer.inference.functions as ops
import torch


def _grouped_size_compiled_for_decode_kernels(
    num_qo_heads: int, num_kv_heads: int
) -> bool:
    return (num_qo_heads // num_kv_heads) in [1, 2, 4, 8]


class BatchDecodeWithPagedKVCacheWrapper:
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        use_tensor_cores: bool = False,
    ) -> None:
        pass

    def plan(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        # pos_encoding_mode: str = "NONE",
        # window_left: int = -1,
        # logits_soft_cap: Optional[float] = None,
        data_type: Union[str, torch.dtype] = "float16",
        q_data_type: Optional[Union[str, torch.dtype]] = None,
        sm_scale: Optional[float] = None,
        # rope_scale: Optional[float] = None,
        # rope_theta: Optional[float] = None,
        max_seqlen_q: int = None,
        max_seqlen_k: int = None,
    ) -> None:
        self.indptr = indptr
        self.indices = indices
        self.last_page_len = last_page_len
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        assert page_size == 1

        self.cu_seqlens_q = torch.ones_like(indptr)
        self.cu_seqlens_q[0] = 0
        self.cu_seqlens_q = torch.cumsum(self.cu_seqlens_q, dim=0).int()

        self.cu_seqlens_k = indptr
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(head_dim)

            self.sm_scale = sm_scale
            self.max_seqlen_q = max_seqlen_q
            self.max_seqlen_k = max_seqlen_k

    begin_forward = plan

    def forward(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        pos_encoding_mode: str = "NONE",
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> torch.Tensor:
        k_cache, v_cache = paged_kv_cache

        out = torch.empty_like(q)

        ops.paged_attention_flashinfer(
            output=out,
            query=q,
            paged_kv_data=(k_cache.unsqueeze(1), v_cache.unsqueeze(1)),
            paged_kv_indptr=self.indptr,
            paged_kv_indices=self.indices,
            paged_kv_last_page_len=self.last_page_len,
            scale=self.sm_scale,
            max_seq_len=self.max_seqlen_k,
            kv_cache_format="NHD",
        )

        return out

    def end_forward(self) -> None:
        r"""Warning: this function is deprecated and has no effect."""
        pass
