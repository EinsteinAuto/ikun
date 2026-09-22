"""Parallel-layout tests for the Qwen3.5 Python model (DP/EP).

Ported from xLLM upstream commit 78aa2a85 (PR #2258)
  Original: tests/python/test_deepseek_v32_parallel.py
  Adapted:  Qwen3.5 MoeSparseBlock with 256 routed experts + shared expert

Test coverage maps to the issue's test cases:
  TC-01  MoE expert routing DP isolation
  TC-02  CUDA/ACL graph capture per DP rank
  TC-03  DP broadcast/gather correctness
  TC-04  Executor DP initialization
  TC-05  End-to-end DP parallel test suite
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Insert repo root so "python.*" resolves to project_6/python/*, not stdlib
_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import pytest
import torch
import torch.nn as nn

# Block the repo-root __init__.py from pulling in vllm
sys.modules.setdefault("project_6", MagicMock())

from python.layers.fused_moe import DPAwareMoEMixin  # noqa: E402
from python.models.qwen3_5 import (  # noqa: E402
    _dp_all_gather,
    _dp_all_gather_variable,
    configure_dp,
    dp_forward_moe_wrapper,
)
from python.model_executor.runners.decode_cuda_graph import (  # noqa: E402
    DecodeCudaGraphRunner,
    _decode_bucket,
    _decode_graph_buckets,
)
from python.model_executor.runners.decode_acl_graph import (  # noqa: E402
    DecodeAclGraphRunner,
)
from python.model_executor.executor import ModelExecutor  # noqa: E402
from python.attention.backend import (  # noqa: E402
    DPBackendConfig, create_attention_backend, register_backend,
)


# ---------------------------------------------------------------------------
# Mock MoeSparseBlock for testing
# ---------------------------------------------------------------------------


class MockMoeSparseBlock(nn.Module):
    """Minimal mock of Qwen3_5MoeSparseBlock for DP testing."""

    def __init__(self, hidden_size: int = 64, num_experts: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.dp_size = 1
        self.dp_rank = 0
        # Dummy weight so next(model.parameters()) works
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Identity forward — just returns input (MoE logic mocked)."""
        return hidden_states


def _make_mock_moe(dp_size: int = 1, dp_rank: int = 0) -> MockMoeSparseBlock:
    moe = MockMoeSparseBlock()
    configure_dp(moe, dp_size, dp_rank)
    return moe


def _mock_metadata(
    dp_token_counts=(4,),
    is_prefill=False,
    is_chunked_prefill=False,
    dp_is_decode=None,
    execution_state=None,
):
    metadata = SimpleNamespace(
        dp_token_counts=dp_token_counts,
        is_prefill=is_prefill,
        is_chunked_prefill=is_chunked_prefill,
        execution_state=execution_state,
    )
    if dp_is_decode is not None:
        metadata.dp_is_decode = dp_is_decode
    return metadata


# ---------------------------------------------------------------------------
# TC-01: MoE expert routing DP isolation
# ---------------------------------------------------------------------------


class TestMoEDPIsolation:
    """Verify that DP replicas route experts independently."""

    def test_dp1_no_gather(self):
        """With dp_size=1, no all-gather should occur."""
        moe = _make_mock_moe(dp_size=1)
        hidden = torch.randn(4, 64)
        metadata = _mock_metadata(dp_token_counts=(4,))

        result = dp_forward_moe_wrapper(
            moe, hidden, moe.forward, metadata
        )
        assert result.shape == hidden.shape

    def test_dp2_calls_gather(self):
        """With dp_size=2, all-gather should be invoked."""
        moe = _make_mock_moe(dp_size=2, dp_rank=0)
        hidden = torch.randn(3, 64)
        metadata = _mock_metadata(
            dp_token_counts=(3, 4), execution_state="graph"
        )

        with patch(
            "python.models.qwen3_5._dp_all_gather"
        ) as mock_gather:
            mock_gather.side_effect = lambda x, **kw: x.repeat(
                kw.get("world_size", 1), *([1] * (x.dim() - 1))
            )
            result = dp_forward_moe_wrapper(
                moe, hidden, moe.forward, metadata
            )
            mock_gather.assert_called_once()
            call_kwargs = mock_gather.call_args[1]
            assert call_kwargs["dim"] == 0
            assert call_kwargs["world_size"] == 2
            assert call_kwargs["group_name"] == "dp"

    def test_dp2_different_inputs_route_independently(self):
        """Different DP replicas with different inputs produce different
        routing decisions (verified by output shapes being correct)."""
        moe_r0 = _make_mock_moe(dp_size=2, dp_rank=0)
        moe_r1 = _make_mock_moe(dp_size=2, dp_rank=1)

        hidden_r0 = torch.randn(3, 64)
        hidden_r1 = torch.randn(4, 64)

        metadata = _mock_metadata(
            dp_token_counts=(3, 4), dp_is_decode=(1, 1)
        )

        # Rank 0
        result_r0 = dp_forward_moe_wrapper(
            moe_r0, hidden_r0, moe_r0.forward, metadata
        )
        assert result_r0.shape[0] == 3

        # Rank 1
        result_r1 = dp_forward_moe_wrapper(
            moe_r1, hidden_r1, moe_r1.forward, metadata
        )
        assert result_r1.shape[0] == 4


# ---------------------------------------------------------------------------
# TC-02: CUDA/ACL graph capture per DP rank
# ---------------------------------------------------------------------------


class TestCudaGraphDPCapture:
    """CUDA graph bucket computation and graph key with DP."""

    def test_bucket_dp1(self):
        buckets = _decode_graph_buckets(32, dp_size=1)
        assert 1 in buckets
        assert buckets[-1] == 32

    def test_bucket_dp2_halves_capacity(self):
        buckets = _decode_graph_buckets(32, dp_size=2)
        # max_local_batch = ceil(32/2) = 16
        assert buckets[-1] <= 16

    def test_graph_key_dp1(self):
        runner = DecodeCudaGraphRunner(
            nn.Linear(1, 1), torch.device("cpu"), max_batch=16
        )
        input_ids = torch.zeros(4, dtype=torch.int32)
        key = runner.graph_key(input_ids)
        assert key is not None
        padded, counts = key
        assert padded == _decode_bucket(4)
        assert counts == (padded,)

    def test_graph_key_dp2(self):
        runner = DecodeCudaGraphRunner(
            nn.Linear(1, 1), torch.device("cpu"), max_batch=32,
            dp_size=2, dp_rank=0
        )
        input_ids = torch.zeros(4, dtype=torch.int32)
        key = runner.graph_key(
            input_ids,
            dp_token_counts=(4, 3),
            dp_is_decode=(1, 1),
        )
        assert key is not None
        padded, counts = key
        assert len(counts) == 2
        assert counts[0] == counts[1] == padded

    def test_graph_key_dp2_exceeds_capacity_returns_none(self):
        runner = DecodeCudaGraphRunner(
            nn.Linear(1, 1), torch.device("cpu"), max_batch=8,
            dp_size=2, dp_rank=0
        )
        input_ids = torch.zeros(100, dtype=torch.int32)
        key = runner.graph_key(
            input_ids,
            dp_token_counts=(100, 100),
            dp_is_decode=(1, 1),
        )
        assert key is None

    def test_graph_key_dp_mismatched_counts_raises(self):
        runner = DecodeCudaGraphRunner(
            nn.Linear(1, 1), torch.device("cpu"), max_batch=32,
            dp_size=2, dp_rank=0
        )
        input_ids = torch.zeros(4, dtype=torch.int32)
        with pytest.raises(RuntimeError, match="dp_token_counts"):
            runner.graph_key(input_ids, dp_token_counts=(4,))


class TestAclGraphDPCapture:
    """ACL graph runner DP validation."""

    def test_acl_dp2_init(self):
        runner = DecodeAclGraphRunner(
            nn.Linear(1, 1), torch.device("cpu"), max_batch=32,
            dp_size=2, dp_rank=0
        )
        assert runner.dp_size == 2
        assert runner.dp_rank == 0
        assert runner.max_batch == 16  # ceil(32/2)

    def test_acl_dp_validate_wrong_counts_raises(self):
        runner = DecodeAclGraphRunner(
            nn.Linear(1, 1), torch.device("cpu"), max_batch=32,
            dp_size=2, dp_rank=0
        )
        with pytest.raises(RuntimeError, match="dp_token_counts"):
            runner._validate_dp_token_counts((4,))

    def test_acl_dp1_validate_accepts_none(self):
        runner = DecodeAclGraphRunner(
            nn.Linear(1, 1), torch.device("cpu"), max_batch=32,
            dp_size=1, dp_rank=0
        )
        # dp_size=1: validation is a no-op
        runner._validate_dp_token_counts(None)


# ---------------------------------------------------------------------------
# TC-03: DP broadcast/gather correctness
# ---------------------------------------------------------------------------


class TestDPBroadcastGather:
    """Verify that gather + scatter preserves local token identity."""

    def test_padded_gather_rank0_recovers_local(self):
        """dp_rank=0: after padded gather → slice, output shape matches input."""
        moe = _make_mock_moe(dp_size=2, dp_rank=0)
        hidden = torch.randn(3, 64)
        metadata = _mock_metadata(
            dp_token_counts=(3, 4), execution_state="graph"
        )

        with patch("python.models.qwen3_5._dp_all_gather") as mock_g:
            # Simulate all_gather: pad to 4, gather 2 replicas → [8, 64]
            mock_g.side_effect = lambda x, **kw: x.repeat(
                kw.get("world_size", 1), *([1] * (x.dim() - 1))
            )
            result = dp_forward_moe_wrapper(
                moe, hidden, moe.forward, metadata
            )
        # dp_rank=0, padded=4: narrow(0, 0, 3) → [3, 64]
        assert result.shape[0] == 3

    def test_padded_gather_rank1_recovers_local(self):
        """dp_rank=1: output is sliced from the second replica's region."""
        moe = _make_mock_moe(dp_size=2, dp_rank=1)
        hidden = torch.randn(4, 64)
        metadata = _mock_metadata(
            dp_token_counts=(3, 4), execution_state="graph"
        )

        with patch("python.models.qwen3_5._dp_all_gather") as mock_g:
            mock_g.side_effect = lambda x, **kw: x.repeat(
                kw.get("world_size", 1), *([1] * (x.dim() - 1))
            )
            result = dp_forward_moe_wrapper(
                moe, hidden, moe.forward, metadata
            )
        # dp_rank=1, padded=4: narrow(0, 4, 4) → [4, 64]
        assert result.shape[0] == 4

    def test_compact_gather_rank0(self):
        """Eager decode path uses compact gather; rank 0 gets first slice."""
        moe = _make_mock_moe(dp_size=2, dp_rank=0)
        hidden = torch.randn(3, 64)
        compact = torch.randn(7, 64)  # 3 + 4 tokens
        metadata = _mock_metadata(
            dp_token_counts=(3, 4), dp_is_decode=(1, 1)
        )

        with patch("python.models.qwen3_5._dp_all_gather_variable") as mock_gv:
            mock_gv.return_value = compact
            result = dp_forward_moe_wrapper(
                moe, hidden, moe.forward, metadata
            )
            mock_gv.assert_called_once()
        # dp_rank=0: offset=0, narrow(0, 0, 3)
        assert result.shape[0] == 3

    def test_compact_gather_rank1(self):
        """Eager decode path: rank 1 gets second slice."""
        moe = _make_mock_moe(dp_size=2, dp_rank=1)
        hidden = torch.randn(4, 64)
        compact = torch.randn(7, 64)
        metadata = _mock_metadata(
            dp_token_counts=(3, 4), dp_is_decode=(1, 1)
        )

        with patch("python.models.qwen3_5._dp_all_gather_variable") as mock_gv:
            mock_gv.return_value = compact
            result = dp_forward_moe_wrapper(
                moe, hidden, moe.forward, metadata
            )
        # dp_rank=1: offset=sum([3])=3, narrow(0, 3, 4)
        assert result.shape[0] == 4


# ---------------------------------------------------------------------------
# TC-04: Executor DP initialization
# ---------------------------------------------------------------------------


class TestExecutorDPInit:
    """Verify ModelExecutor reads and validates DP config."""

    def _make_model(self):
        return nn.Linear(10, 10)

    def test_dp_size_stored(self):
        model = self._make_model()
        executor = ModelExecutor(model, {"dp_size": 2, "dp_rank": 0}, 32)
        assert executor.dp_size == 2

    def test_dp1_default(self):
        model = self._make_model()
        executor = ModelExecutor(model, {}, 32)
        assert executor.dp_size == 1

    def test_dp_with_unsupported_backend_raises(self):
        model = self._make_model()
        with pytest.raises(NotImplementedError, match="data parallel"):
            ModelExecutor(
                model,
                {"dp_size": 2, "dp_rank": 0, "python_graph_backend": "inductor"},
                32,
            )

    def test_dp_with_cudagraphs_accepted(self):
        model = self._make_model()
        executor = ModelExecutor(
            model,
            {"dp_size": 2, "dp_rank": 0, "python_graph_backend": "cudagraphs"},
            32,
        )
        assert executor.decode_graph_runner is not None

    def test_dp_with_aclgraph_accepted(self):
        model = self._make_model()
        executor = ModelExecutor(
            model,
            {"dp_size": 2, "dp_rank": 0, "python_graph_backend": "aclgraph"},
            32,
        )
        assert executor.decode_graph_runner is not None


# ---------------------------------------------------------------------------
# TC-05: End-to-end DP parallel test suite
# ---------------------------------------------------------------------------


class TestEndToEndDP:
    """Integration tests combining executor + MoE + graph runner."""

    def test_2way_dp_moe_shapes(self):
        """2-way DP: both ranks produce correct output shapes."""
        for rank in (0, 1):
            moe = _make_mock_moe(dp_size=2, dp_rank=rank)
            local_tokens = 3 if rank == 0 else 4
            hidden = torch.randn(local_tokens, 64)
            metadata = _mock_metadata(
                dp_token_counts=(3, 4), dp_is_decode=(1, 1)
            )
            result = dp_forward_moe_wrapper(
                moe, hidden, moe.forward, metadata
            )
            assert result.shape[0] == local_tokens

    def test_4way_dp_moe_shapes(self):
        """4-way DP: all ranks produce correct output shapes."""
        counts = (2, 3, 4, 5)
        for rank in range(4):
            moe = _make_mock_moe(dp_size=4, dp_rank=rank)
            hidden = torch.randn(counts[rank], 64)
            metadata = _mock_metadata(
                dp_token_counts=counts, dp_is_decode=(1, 1, 1, 1)
            )
            result = dp_forward_moe_wrapper(
                moe, hidden, moe.forward, metadata
            )
            assert result.shape[0] == counts[rank]

    def test_dp_plus_tp_cuda_graph_buckets(self):
        """DP=2, TP=4 on 8 devices: graph buckets respect DP-reduced capacity."""
        # max_batch=64 across 2 DP replicas → 32 per replica
        buckets = _decode_graph_buckets(64, dp_size=2)
        assert buckets[-1] <= 32

    def test_varying_batch_sizes_padded_path(self):
        """Different batch sizes per DP rank use padded gather."""
        moe = _make_mock_moe(dp_size=2, dp_rank=0)
        for local_size in (1, 5, 8, 16):
            other_size = local_size + 2
            hidden = torch.randn(local_size, 64)
            metadata = _mock_metadata(
                dp_token_counts=(local_size, other_size),
                execution_state="graph",
            )
            with patch("python.models.qwen3_5._dp_all_gather") as mock_g:
                mock_g.side_effect = lambda x, **kw: x.repeat(
                    kw.get("world_size", 1), *([1] * (x.dim() - 1))
                )
                result = dp_forward_moe_wrapper(
                    moe, hidden, moe.forward, metadata
                )
            assert result.shape[0] == local_size


# ---------------------------------------------------------------------------
# TC-07: Existing tests unbroken (sanity)
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """Verify dp_size=1 (default) doesn't change existing behaviour."""

    def test_dp1_identity(self):
        moe = _make_mock_moe(dp_size=1)
        hidden = torch.randn(8, 64)
        metadata = _mock_metadata(dp_token_counts=(8,))
        result = dp_forward_moe_wrapper(
            moe, hidden, moe.forward, metadata
        )
        assert torch.equal(result, hidden)

    def test_executor_dp1_no_graph_runner(self):
        model = nn.Linear(10, 10)
        executor = ModelExecutor(model, {}, 32)
        assert executor.decode_graph_runner is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
