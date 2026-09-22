"""EP-aware FusedMoE override for vllm 0.6.3.

Ported from tpu-inference upstream (xllm):
  - 04077875  PR #2577: Enable correct DP attention for hybrid models
  - df7f5b35  PR #2679: Optimize attn DP + MoE EP ReduceScatter
  - 57987c2   PR #3435: Reduce shared-expert output over in-group TP under attn DP

Architecture:
  4 GPUs, --tensor-parallel-size 4, VLLM_ENABLE_EXPERT_PARALLEL=1
  → TP group = EP group = WORLD = {rank0, rank1, rank2, rank3}
  → Routed experts: each rank holds 64/256 experts, tp_size=1 (full intermediate)
  → Shared expert: TP=4 sharded (uses default tp_size=world_size)

Design:
  1. __init__: allocate only local experts (64 per rank), tp_size=1
  2. weight_loader: skip non-local experts, remap expert_id to local
  3. forward (in qwen3_5.py): EP mask + per-expert compute + reduction

Reduction strategy (from upstream PR #2679 / #3435):
  - routed_out: partial across EP ranks (each rank computed local experts only)
  - shared_out: partial across TP ranks (RowParallelLinear reduce_results=False)
  - Since EP group == TP group == WORLD: single all-reduce on (routed + shared)
  - When DP > 1: can optimize to reduce-scatter (see dp_forward_moe_wrapper)

EP expert dispatch (from upstream fused_moe_gmm.py):
  - topk routing produces global expert ids on ALL ranks (identical)
  - Each rank masks non-local experts (weight=0), remaps to local [0, E_local)
  - Non-local (token,expert) pairs distributed via mod to avoid expert 0 overload
  - Per-expert loop skips experts with count=0 → no wasted GEMM for empty experts
  - Remaining waste: tokens routed to non-local experts still enter local experts
    with weight=0. Upstream uses ragged_gather to skip these entirely.
    TODO: implement token-level skip for further speedup.
"""

import os
import torch
import torch.distributed as dist

from vllm.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Environment / runtime helpers
# ---------------------------------------------------------------------------

def _is_ep_requested():
    return bool(int(os.environ.get("VLLM_ENABLE_EXPERT_PARALLEL", "0")))


def _get_ep_runtime_config():
    """Return (ep_size, ep_rank). EP group = WORLD group."""
    if dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


# ---------------------------------------------------------------------------
# EP-aware expert mask + remap (called from qwen3_5.py _pure_pytorch_experts)
# ---------------------------------------------------------------------------
# Ported from tpu-inference fused_moe_gmm.py:
#   _process_tokens_locally() → valid_rows_mask + ragged_gather
# Our CUDA version: mask weights to 0, distribute non-local ids via mod.

def ep_mask_and_remap(topk_ids, topk_weights, start_expert_id,
                      num_experts_per_rank, ep_rank):
    """Mask non-local experts and remap global ids to local.

    Args:
        topk_ids:    [T, top_k] global expert ids
        topk_weights: [T, top_k] routing weights
        start_expert_id: first global expert id on this rank
        num_experts_per_rank: number of local experts (E_local)
        ep_rank: this rank's EP index

    Returns:
        topk_ids:    [T, top_k] local expert ids in [0, E_local)
        topk_weights: [T, top_k] with non-local entries zeroed
        local_mask:  [T, top_k] bool, True for locally-routed pairs
    """
    E_local = num_experts_per_rank
    end_expert_id = start_expert_id + E_local

    # Identify local (token, expert) pairs
    local_mask = (topk_ids >= start_expert_id) & (topk_ids < end_expert_id)

    # Zero non-local weights (these contribute nothing to output)
    topk_weights = topk_weights * local_mask.to(topk_weights.dtype)

    # Remap global → local ids
    # Local experts: id - start → [0, E_local)
    # Non-local: distribute via mod to avoid inflating expert 0 count
    # (upstream uses ragged_gather to skip non-local tokens entirely;
    #  this mod trick achieves balanced distribution with weight=0)
    local_ids = topk_ids - start_expert_id
    non_local = ~local_mask
    if non_local.any():
        local_ids[non_local] = topk_ids[non_local] % E_local

    return local_ids, topk_weights, local_mask


# ---------------------------------------------------------------------------
# EP-aware reduction (called from qwen3_5.py forward())
# ---------------------------------------------------------------------------
# Ported from tpu-inference:
#   custom_ops/fused_moe.py  VllmMoERunner._maybe_reduce_shared_expert_output
#   custom_ops/fused_moe.py  VllmMoERunner._maybe_reduce_final_output
#   fused_moe_gmm.py         scatter_results / defer_all_reduce logic

def ep_reduce_output(routed_out, shared_out):
    """Reduce EP + TP partial outputs into final MoE output.

    In our topology (TP group == EP group == WORLD):
      routed_out: [T, H] partial — each rank only computed local experts
      shared_out: [T, H] partial — RowParallelLinear TP shard (reduce_results=False)

    IMPORTANT: use tensor_model_parallel_all_reduce (goes through ixformer
    optimized path on BI-V100) instead of raw dist.all_reduce (raw NCCL,
    40x slower on this hardware). This is valid because TP group == WORLD.

    Returns: [T, H] fully-reduced output.
    """
    from vllm.distributed import tensor_model_parallel_all_reduce
    out = routed_out + shared_out
    out = tensor_model_parallel_all_reduce(out)
    return out


def ep_reduce_scatter_output(routed_out, shared_out, dp_size, dp_rank):
    """Reduce-scatter variant for DP > 1 (upstream PR #2679 optimization).

    When DP attention is active, the MoE forward receives all-gathered tokens
    from all DP ranks (T_global = T_local * dp_size). Instead of:
      all_reduce(T_global) → slice(T_local)     # 2 steps
    we use:
      reduce_scatter(T_global) → T_local          # 1 step, half bandwidth

    Equivalent to upstream:
      jax.lax.psum_scatter(out, axis_name=scatter_axes,
                           scatter_dimension=0, tiled=True)

    Returns: [T_local, H] reduced output for this DP rank only.
    """
    out = routed_out + shared_out
    if not dist.is_initialized() or dp_size <= 1:
        return out

    T_global, H = out.shape
    assert T_global % dp_size == 0, (
        f"reduce_scatter: T_global={T_global} not divisible by dp_size={dp_size}")

    output = torch.empty(T_global // dp_size, H,
                         dtype=out.dtype, device=out.device)
    dist.reduce_scatter_tensor(output, out, op=dist.ReduceOp.SUM)
    return output


# ---------------------------------------------------------------------------
# Monkey-patch FusedMoE
# ---------------------------------------------------------------------------

def patch_fused_moe_for_ep():
    """Monkey-patch FusedMoE to support Expert Parallelism.

    Only patches __init__ and weight_loader. Forward is NOT patched —
    qwen3_5.py forward() handles EP dispatch via:
      ep_mask_and_remap()  for expert routing
      ep_reduce_output()   for reduction
    """
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    if not _is_ep_requested():
        logger.info("[EP] not enabled (VLLM_ENABLE_EXPERT_PARALLEL!=1), skipping")
        return

    logger.info("[EP] installing FusedMoE __init__ + weight_loader patches")

    _orig_init = FusedMoE.__init__

    def _ep_init(self, num_experts, top_k, hidden_size, intermediate_size,
                 params_dtype=None, reduce_results=False, renormalize=True,
                 use_grouped_topk=False, num_expert_group=None,
                 topk_group=None, quant_config=None, tp_size=None,
                 prefix="", custom_routing_function=None):
        """EP-aware __init__: allocate only local experts, full intermediate."""

        ep_size, ep_rank = _get_ep_runtime_config()

        num_experts_per_rank = num_experts // ep_size
        start_expert_id = ep_rank * num_experts_per_rank
        assert num_experts % ep_size == 0, (
            f"num_experts={num_experts} not divisible by ep_size={ep_size}")

        logger.info(
            "[EP] FusedMoE init: global=%d ep_size=%d ep_rank=%d "
            "local=%d (experts %d..%d) tp_size=1 (full intermediate)",
            num_experts, ep_size, ep_rank, num_experts_per_rank,
            start_expert_id, start_expert_id + num_experts_per_rank - 1)

        # tp_size=1: each rank holds full intermediate for its local experts.
        # Why: with EP+TP on same group, a single all-reduce sums across ranks.
        # If weights were TP-sharded (I/tp), each expert would only have 1/tp
        # of intermediate on one rank — all-reduce sums DIFFERENT experts (EP)
        # but does NOT reconstruct TP shards of the SAME expert.
        # With tp_size=1: each rank holds full intermediate, all-reduce sums
        # ep_size ranks × (num_experts/ep_size) experts = all experts. Correct.
        _orig_init(
            self,
            num_experts=num_experts_per_rank,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=params_dtype,
            reduce_results=False,   # we reduce in forward()
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            quant_config=quant_config,
            tp_size=1,
            prefix=prefix,
            custom_routing_function=custom_routing_function,
        )

        # Override num_experts back to global for routing (topk over 256)
        self.num_experts = num_experts

        # EP attributes — set AFTER _orig_init (super().__init__ wipes __dict__)
        self._ep_enabled = True
        self._ep_size = ep_size
        self._ep_rank = ep_rank
        self._global_num_experts = num_experts
        self._num_experts_per_rank = num_experts_per_rank
        self._start_expert_id = start_expert_id

    _orig_weight_loader = FusedMoE.weight_loader

    def _ep_weight_loader(self, param, loaded_weight, weight_name,
                          shard_id, expert_id):
        """EP-aware weight_loader: skip non-local experts, remap id.

        Since tp_size=1, we force tp_rank=0 during loading so narrow()
        reads offset 0 (full-size weight, no TP split to index into).
        """
        if not getattr(self, '_ep_enabled', False):
            return _orig_weight_loader(self, param, loaded_weight,
                                       weight_name, shard_id, expert_id)

        start = self._start_expert_id
        end = start + self._num_experts_per_rank

        if expert_id < start or expert_id >= end:
            return  # skip non-local expert

        local_expert_id = expert_id - start

        # Mock tp_rank=0 in layer.py's LOCAL namespace.
        # layer.py does `from vllm.distributed import get_tensor_model_parallel_rank`
        # so we must patch the name in layer.py's module, not in vllm.distributed.
        import vllm.model_executor.layers.fused_moe.layer as _layer_mod
        _real = _layer_mod.get_tensor_model_parallel_rank
        _layer_mod.get_tensor_model_parallel_rank = lambda: 0
        try:
            _orig_weight_loader(self, param, loaded_weight, weight_name,
                                shard_id, local_expert_id)
        finally:
            _layer_mod.get_tensor_model_parallel_rank = _real

    FusedMoE.__init__ = _ep_init
    FusedMoE.weight_loader = _ep_weight_loader

    logger.info("[EP] FusedMoE patches installed (__init__, weight_loader)")