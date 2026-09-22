"""Multi-head attention."""
import os
enable_infer_paged_attn = os.getenv("ENABLE_INFER_PAGED_ATTN",None)
from typing import List, Optional

import importlib
import torch
import torch.nn as nn
from ixformer.contrib.xformers import ops as xops
from ixformer.contrib.xformers.ops.fmha.attn_bias import (BlockDiagonalCausalMask,
                                                          LowerTriangularMaskWithTensorBias)

from vllm._C import ops
from vllm._C import cache_ops
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.triton_kernel.prefix_prefill import (
    context_attention_fwd)
from vllm.utils import is_hip

# _SUPPORTED_HEAD_SIZES = [64, 80, 96, 112, 128, 256]
# # Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
# _PARTITION_SIZE = 512
# ═══════════════════════════════════════════════════════════════════════
# BI-V100 constants derived from CCCL source code analysis:
#
# head_size support: Qwen3.6 uses head_dim=128 for attention heads.
# EngineX base only supported [64, 128, 256]. Adding back the sizes
# that vllm's paged_attention_v2_launcher compiles for (the .so must
# have been compiled with these sizes for ops.paged_attention_v2 to work).
# If the precompiled .so only has [64, 128, 256], extra sizes are harmless
# (they'll hit the fallback xformers path instead of crashing).
#
# PARTITION_SIZE rationale (from CCCL dispatch_reduce.cuh + grid_even_share.cuh):
#   dispatch_reduce.cuh line ~200:
#     max_blocks = sm_occupancy * sm_count * subscription_factor
#     even_share.DispatchInit(num_items, max_blocks, tile_size)
#
#   BI-V100: sm_count=16, sm_occupancy=2, subscription_factor=5
#   → max_blocks = 160
#
#   GridEvenShare assigns "big" and "normal" shares:
#     big_shares = total_tiles - (avg_tiles_per_block * grid_size)
#     → first `big_shares` blocks get one extra tile
#
#   For V2 paged attention, PARTITION_SIZE = tile_size.
#   With PARTITION_SIZE=256 and max_seq_len=100K:
#     total_tiles = ceil(100000/256) = 391 partitions
#     grid_size = min(391, 160) = 160 CTAs
#     → 231 partitions are serialized (each CTA handles ~2.4 partitions)
#     → Phase 2 merge kernel processes 160 partial results
#
#   With PARTITION_SIZE=512:
#     total_tiles = ceil(100000/512) = 196 partitions
#     grid_size = min(196, 160) = 160 CTAs
#     → 36 extra partitions, better balanced
#     → Phase 2 merge processes fewer partitions → lower merge overhead
#
#   But the precompiled .so expects PARTITION_SIZE=256 (EngineX default).
#   Changing this without recompiling the .so will cause wrong results.
#   Keep 256 for now; document the CCCL-optimal value for rebuild.
# ═══════════════════════════════════════════════════════════════════════
_SUPPORTED_HEAD_SIZES = [64, 80, 96, 112, 120, 128, 192, 256]
# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
# CCCL-optimal for BI-V100 would be 512 (see rationale above),
# but must match the precompiled .so.
_PARTITION_SIZE = 256

# BI-V100 hardware profile (from CCCL grid_even_share.cuh + hardware.cuh)
_BI100_SM_COUNT = 16
_BI100_MAX_GRID = _BI100_SM_COUNT * 2 * 5  # sm_occupancy=2, subscription=5 → 160


class PagedAttention(nn.Module):
    """MHA/MQA/GQA layer with PagedAttention.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Reshape and store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention using either
        xformers or the PagedAttention custom op.
    3. Return the output tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.register_buffer("alibi_slopes", alibi_slopes, persistent=False)

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if self.head_size not in _SUPPORTED_HEAD_SIZES:
            raise ValueError(f"head_size ({self.head_size}) is not supported. "
                             f"Supported head sizes: {_SUPPORTED_HEAD_SIZES}.")

        self.use_ref_attention = self.check_use_ref_attention()

        # TODO align vllm do not need those
        self.attn_op = xops.fmha.flash.FwOp()        
        head_mapping = torch.repeat_interleave(
            torch.arange(self.num_kv_heads, dtype=torch.int32),
            self.num_queries_per_kv)
        self.register_buffer("head_mapping", head_mapping, persistent=False)

    def check_use_ref_attention(self) -> bool:
        if not is_hip():
            return False
        # For ROCm, check whether flash attention is installed or not.
        # if not, use_ref_attention needs to be True
        return importlib.util.find_spec("flash_attn") is None

    def ref_masked_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        seq_len, _, _ = query.shape
        attn_mask = torch.triu(torch.ones(seq_len,
                                          seq_len,
                                          dtype=query.dtype,
                                          device=query.device),
                               diagonal=1)
        attn_mask = attn_mask * torch.finfo(query.dtype).min

        attn_weights = self.scale * torch.einsum("qhd,khd->hqk", query,
                                                 key).float()
        attn_weights = attn_weights + attn_mask.float()
        attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
        out = torch.einsum("hqk,khd->qhd", attn_weights, value)
        return out

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        """PagedAttention forward pass.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for the inputs.
            cache_event: event to wait for the cache operations to finish.
        Returns:
            shape = [batch_size, seq_len, num_heads * head_size]
        """
        num_tokens, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
        slot_mapping = input_metadata.slot_mapping

        # Reshape the keys and values and store them in the cache.
        # If key_cache and value_cache are not provided, the new key and value
        # vectors will not be cached. This happens during the initial memory
        # profiling run.
        if key_cache is not None and value_cache is not None:
            cache_ops.reshape_and_cache(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
            )

        if input_metadata.is_prompt:
            # normal attention
            if (key_cache is None or value_cache is None
                    or input_metadata.block_tables.numel() == 0):
                if input_metadata.attn_bias is None:
                    if self.alibi_slopes is None:
                        attn_bias = BlockDiagonalCausalMask.from_seqlens(input_metadata.prompt_lens)
                        if self.sliding_window is not None:
                            attn_bias = attn_bias.make_local_attention(
                                self.sliding_window)
                        input_metadata.attn_bias = attn_bias
                    else:
                        attn_bias = BlockDiagonalCausalMask.from_seqlens(input_metadata.prompt_lens)
                        input_metadata.attn_bias = attn_bias
                
                if self.use_ref_attention:
                    output = self.ref_masked_attention(
                        query,
                        key,
                        value,
                    )
                    # Using view got RuntimeError: view size is not compatible with input tensor's size and stride
                    # (at least one dimension spans across two contiguous subspaces). Use reshape instead
                    return output.reshape(num_tokens, hidden_size)

                # TODO(woosuk): Too many view operations. Let's try to reduce
                # them in the future for code readability.
                query = query.unsqueeze(0)
                key = key.unsqueeze(0)
                value = value.unsqueeze(0)

                out = xops.memory_efficient_attention_forward(
                    query,
                    key,
                    value,
                    attn_bias=input_metadata.attn_bias,
                    p=0.0,
                    scale=self.scale,
                    op=self.attn_op,
                    alibi_slopes=self.alibi_slopes
                )
                output = out.view_as(query)  
            else:
                # prefix-enabled attention
                output = torch.empty_like(query)
                context_attention_fwd(
                    query,
                    key,
                    value,
                    output,
                    key_cache,
                    value_cache,
                    input_metadata.block_tables,  # [BS, max_block_per_request]
                    input_metadata.start_loc,
                    input_metadata.prompt_lens,
                    input_metadata.context_lens,
                    input_metadata.max_seq_len,
                    getattr(self, "alibi_slopes", None),
                )
        else:
            # Decoding run.
            output = _paged_attention(
                query,
                key_cache,
                value_cache,
                input_metadata,
                self.head_mapping, # self.num_kv_heads
                self.scale,
                self.alibi_slopes,
            )

        # Reshape the output tensor.
        return output.view(num_tokens, hidden_size)
    # TODO align
    """
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        PagedAttention forward pass.

        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for the inputs.
        Returns:
            shape = [batch_size, seq_len, num_heads * head_size]
        
        batch_size, seq_len, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        # Reshape the keys and values and store them in the cache.
        # If key_cache and value_cache are not provided, the new key and value
        # vectors will not be cached. This happens during the initial memory
        # profiling run.
        if key_cache is not None and value_cache is not None:
            cache_ops.reshape_and_cache(
                key,
                value,
                key_cache,
                value_cache,
                input_metadata.slot_mapping.flatten(),
                input_metadata.kv_cache_dtype,
            )

        if input_metadata.is_prompt:
            # normal attention
            if (key_cache is None or value_cache is None
                    or input_metadata.block_tables.numel() == 0):
                if self.num_kv_heads != self.num_heads:
                    # As of Nov 2023, xformers only supports MHA. For MQA/GQA,
                    # project the key and value tensors to the desired number of
                    # heads.
                    # TODO(woosuk): Use MQA/GQA kernels for higher performance.
                    query = query.view(query.shape[0], self.num_kv_heads,
                                       self.num_queries_per_kv,
                                       query.shape[-1])
                    key = key[:, :,
                              None, :].expand(key.shape[0], self.num_kv_heads,
                                              self.num_queries_per_kv,
                                              key.shape[-1])
                    value = value[:, :,
                                  None, :].expand(value.shape[0],
                                                  self.num_kv_heads,
                                                  self.num_queries_per_kv,
                                                  value.shape[-1])

                # Set attention bias if not provided. This typically happens at
                # the very attention layer of every iteration.
                # FIXME(woosuk): This is a hack.
                if input_metadata.attn_bias is None:
                    if self.alibi_slopes is None:
                        attn_bias = BlockDiagonalCausalMask.from_seqlens(
                            [seq_len] * batch_size)
                        if self.sliding_window is not None:
                            attn_bias = attn_bias.make_local_attention(
                                self.sliding_window)
                        input_metadata.attn_bias = attn_bias
                    else:
                        input_metadata.attn_bias = _make_alibi_bias(
                            self.alibi_slopes, self.num_kv_heads, batch_size,
                            seq_len, query.dtype)

                if self.use_ref_attention:
                    output = self.ref_masked_attention(
                        query,
                        key,
                        value,
                    )
                    # Using view got RuntimeError: view size is not compatible with input tensor's size and stride
                    # (at least one dimension spans across two contiguous subspaces). Use reshape instead
                    return output.reshape(batch_size, seq_len, hidden_size)

                # TODO(woosuk): Too many view operations. Let's try to reduce
                # them in the future for code readability.
                if self.alibi_slopes is None:
                    query = query.unsqueeze(0)
                    key = key.unsqueeze(0)
                    value = value.unsqueeze(0)
                else:
                    query = query.unflatten(0, (batch_size, seq_len))
                    key = key.unflatten(0, (batch_size, seq_len))
                    value = value.unflatten(0, (batch_size, seq_len))

                out = xops.memory_efficient_attention_forward(
                    query,
                    key,
                    value,
                    attn_bias=input_metadata.attn_bias,
                    p=0.0,
                    scale=self.scale,
                    op=xops.fmha.MemoryEfficientAttentionFlashAttentionOp[0] if
                    (is_hip()) else None,
                )
                output = out.view_as(query)
            else:
                # prefix-enabled attention
                output = torch.empty_like(query)
                context_attention_fwd(
                    query,
                    key,
                    value,
                    output,
                    key_cache,
                    value_cache,
                    input_metadata.block_tables,  # [BS, max_block_per_request]
                    input_metadata.start_loc,
                    input_metadata.prompt_lens,
                    input_metadata.context_lens,
                    input_metadata.max_seq_len,
                    getattr(self, "alibi_slopes", None),
                )

        else:
            # Decoding run.
            output = _paged_attention(
                query,
                key_cache,
                value_cache,
                input_metadata,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
            )

        # Reshape the output tensor.
        return output.view(batch_size, seq_len, hidden_size)
    """


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
) -> LowerTriangularMaskWithTensorBias:
    bias = torch.arange(seq_len, dtype=dtype)
    # NOTE(zhuohan): HF uses
    #     `bias = bias[None, :].repeat(prompt_len, 1)`
    # here. We find that both biases give the same results, but
    # the bias below more accurately follows the original ALiBi
    # paper.
    bias = bias[None, :] - bias[:, None]

    # When using custom attention bias, xformers requires the bias to
    # be sliced from a tensor whose length is a multiple of 8.
    padded_len = (seq_len + 7) // 8 * 8
    num_heads = alibi_slopes.shape[0]
    bias = torch.empty(
        batch_size,
        num_heads,
        seq_len,
        padded_len,
        device=alibi_slopes.device,
        dtype=dtype,
    )[:, :, :, :seq_len].copy_(bias)
    bias.mul_(alibi_slopes[:, None, None])
    if num_heads != num_kv_heads:
        bias = bias.unflatten(1, (num_kv_heads, num_heads // num_kv_heads))
    attn_bias = LowerTriangularMaskWithTensorBias(bias)
    return attn_bias


def _paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    input_metadata: InputMetadata,
    head_mapping: torch.Tensor, # num_kv_heads: int,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
    use_sqrt_alibi: bool = False
) -> torch.Tensor:
    output = torch.empty_like(query)

    # ═══════════════════════════════════════════════════════════════
    # CCCL dispatch_reduce.cuh single-tile vs two-phase decision:
    #
    # kernel_reduce.cuh DeviceReduceSingleTileKernel:
    #   Single CTA → ConsumeRange(0, num_items) → output
    #   No temp buffer, no Phase 2 merge, no cross-CTA synchronization
    #
    # kernel_reduce.cuh DeviceReduceKernel:
    #   Multiple CTAs → GridEvenShare → each CTA writes partial result
    #   → Phase 2: single CTA merges all partials
    #   OR (StableReductionOrder=false): atomic_ref::fetch_add
    #
    # dispatch_reduce.cuh Invoke():
    #   if (num_items <= threads_per_block * items_per_thread):
    #       InvokeSingleTile()    # one CTA, no overhead
    #   else:
    #       InvokePasses()        # multi-CTA + merge
    #
    # For paged attention:
    #   V1 = SingleTile: one CTA handles entire sequence
    #   V2 = TwoPasses: sequence partitioned across CTAs + merge
    #
    # Decision: V1 when sequence fits in one partition (no merge needed).
    # The original condition `key_cache.dim() == 4` is unrelated to this
    # decision — it checks tensor layout, not problem size.
    #
    # BI-V100 specifics:
    #   16 SMs → max ~160 CTAs → V2's Phase 2 merge is cheap
    #   But for short sequences (decode tokens 1→512), V1 avoids
    #   the 3-5μs overhead of tmp_output allocation + merge kernel launch
    # ═══════════════════════════════════════════════════════════════
    max_num_partitions_check = (
        (input_metadata.max_context_len + _PARTITION_SIZE - 1) //
        _PARTITION_SIZE)
    # V1 when single partition (CCCL InvokeSingleTile equivalent)
    # V2 when multi-partition (CCCL InvokePasses equivalent)
    # env override preserved for EngineX compatibility
    use_v1 = (enable_infer_paged_attn is not None
              or max_num_partitions_check <= 1)
    if use_v1:
        block_size = value_cache.shape[3]
        # Run PagedAttention V1.
        ops.paged_attention_v1(
            output,
            query,
            key_cache,
            value_cache,
            head_mapping, # num_kv_heads
            scale,
            input_metadata.block_tables,
            input_metadata.context_lens,
            block_size,
            input_metadata.max_context_len,
            alibi_slopes,
            input_metadata.kv_cache_dtype,
        )
    else:
        # Run PagedAttention V2.
        block_size = value_cache.shape[2]
        num_seqs, num_heads, head_size = query.shape
        max_num_partitions = (
            (input_metadata.max_context_len + _PARTITION_SIZE - 1) //
            _PARTITION_SIZE)
        # ═══════════════════════════════════════════════════════════════
        # CCCL agent_merge_sort.cuh union _TempStorage pattern:
        # Cache temp tensors across decode steps. During autoregressive
        # generation, num_seqs and num_heads are stable (only seq_len grows,
        # which increases max_num_partitions gradually). Reuse the allocation
        # when shapes haven't changed, avoiding cudaMalloc overhead per step.
        #
        # dispatch_reduce.cuh does the same: d_block_reductions is allocated
        # once based on max_blocks, then reused across Invoke() calls.
        #
        # For BI-V100 with 16 SMs, the V2 merge kernel (Phase 2) processes
        # at most max_num_partitions partial results. Caching eliminates
        # ~3-5μs of allocation overhead per decode step.
        # ═══════════════════════════════════════════════════════════════
        _v2_key = (num_seqs, num_heads, max_num_partitions,
                   head_size, output.dtype, str(output.device))
        _v2 = getattr(_paged_attention, '_v2_cache', {}).get(_v2_key)
        if _v2 is not None:
            tmp_output, exp_sums, max_logits = _v2
        else:
            tmp_output = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions, head_size),
                dtype=output.dtype,
                device=output.device,
            )
            exp_sums = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions),
                dtype=torch.float32,
                device=output.device,
            )
            max_logits = torch.empty_like(exp_sums)
            if not hasattr(_paged_attention, '_v2_cache'):
                _paged_attention._v2_cache = {}
            _paged_attention._v2_cache[_v2_key] = (
                tmp_output, exp_sums, max_logits)
        ops.paged_attention_v2(
            output,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            head_mapping, # num_kv_heads
            scale,
            input_metadata.block_tables,
            input_metadata.context_lens,
            block_size,
            input_metadata.max_context_len,
            alibi_slopes,
            input_metadata.kv_cache_dtype,
        )
    return output


# ↓ add for smoothquant
class DequantPagedAttention(PagedAttention):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
        quant_kv_cache: bool = False,
        kv_quant_params: torch.Tensor = None,
        quant_scale: float = 1.0,
        use_per_token_quant: bool = True,
    ) -> None:
        super().__init__(num_heads,
                         head_size,
                         scale,
                         num_kv_heads,
                         alibi_slopes,
                         sliding_window)
        self.register_parameter(
            "quant_scale",
            torch.nn.Parameter(
                torch.tensor(quant_scale, dtype=torch.float32,requires_grad=False))
           )
        self.use_per_token_quant = use_per_token_quant

    def _apply(self, fn):
        super()._apply(fn)
        self.quant_scale.data = self.quant_scale.cpu()
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.quant_scale.data = self.quant_scale.to(*args, **kwargs)
        self.quant_scale.data = self.quant_scale.to(torch.float32)
        return self

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        out = super().forward(
            query,
            key,
            value,
            key_cache,
            value_cache,
            input_metadata,
        )
        quant_out = torch.empty_like(out, dtype=torch.int8)
        if self.use_per_token_quant:
            scale = torch.empty(out.numel() // out.shape[-1],
                                dtype=torch.float32,
                                device=out.device)
            ops.quant(quant_out, out, scale)
            return quant_out, scale
        else:
            ops.quant(quant_out, out, self.quant_scale.item())
            return (quant_out, )
