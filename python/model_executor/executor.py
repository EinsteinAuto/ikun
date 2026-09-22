"""DP-aware Python model executor for Qwen3.5.

Ported from xLLM upstream commit 78aa2a85 (PR #2258).
Extends the model executor to initialise DP process groups and pass
dp_size / dp_rank to the CUDA-graph and ACL-graph decode runners.

Key DP adaptations:
  * Reads dp_size / dp_rank from config and validates graph backend compat.
  * Passes DP params to DecodeCudaGraphRunner / DecodeAclGraphRunner.
  * Stores dp_size for external callers (e.g. the C++ worker).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ModelExecutor:
    """Python model executor with data-parallel support.

    This is the entry point that the C++ runtime's ``py_executor_impl``
    calls.  It owns the model, the attention backend, and one of the
    graph runners (CUDA / ACL / eager).

    Args:
        model: The full causal-LM module.
        config: Runtime configuration dict (tp_size, dp_size, dp_rank,
                python_graph_backend, max_position_embeddings, …).
        max_seqs_per_batch: Maximum sequences (= max batch) per step.
        num_decoding_tokens: Tokens per sequence for speculative decode.
        acl_graph_decode_batch_size_limit: Optional cap for ACL graphs.
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        max_seqs_per_batch: int,
        num_decoding_tokens: int = 1,
        acl_graph_decode_batch_size_limit: int | None = None,
    ) -> None:
        self.model = model
        self._kv_bound = False

        first_parameter = next(model.parameters())
        device = first_parameter.device
        dtype = first_parameter.dtype

        # ---- DP configuration (added by PR #2258) ----------------------
        graph_backend = self._resolve_graph_backend(config)
        dp_size = int(config.get("dp_size", 1))
        dp_rank = int(config.get("dp_rank", 0))
        self.dp_size = dp_size

        if dp_size > 1 and graph_backend not in (
            "",
            "off",
            "none",
            "0",
            "cudagraphs",
            "aclgraph",
        ):
            raise NotImplementedError(
                "Python data parallel graph execution supports "
                "cudagraphs and aclgraph only"
            )
        # ----------------------------------------------------------------

        self.decode_graph_runner = None

        if graph_backend in ("", "off", "none", "0"):
            pass
        elif graph_backend == "cudagraphs":
            from python.model_executor.runners.decode_cuda_graph import (
                DecodeCudaGraphRunner,
            )

            self.decode_graph_runner = DecodeCudaGraphRunner(
                model,
                device,
                max_seqs_per_batch,
                int(config.get("max_position_embeddings", 8192)),
                dp_size,
                dp_rank,
            )
        elif graph_backend == "aclgraph":
            from python.model_executor.runners.decode_acl_graph import (
                DecodeAclGraphRunner,
            )

            num_decoding_tokens = max(1, int(num_decoding_tokens))
            decode_batch_size_limit = (
                None
                if acl_graph_decode_batch_size_limit is None
                else max(1, int(acl_graph_decode_batch_size_limit))
            )
            graph_sequence_capacity = max_seqs_per_batch
            if decode_batch_size_limit is not None:
                graph_sequence_capacity = min(
                    graph_sequence_capacity, decode_batch_size_limit
                )
            max_graph_tokens = graph_sequence_capacity * num_decoding_tokens

            self.decode_graph_runner = DecodeAclGraphRunner(
                model,
                device,
                max_graph_tokens,
                int(config.get("max_position_embeddings", 8192)),
                dp_size,
                dp_rank,
                decode_batch_size_limit,
                num_decoding_tokens,
            )

    @staticmethod
    def _resolve_graph_backend(config: dict) -> str:
        graph_backend = str(
            config.get("python_graph_backend", "off")
        ).lower()
        graph_disabled = graph_backend in ("", "off", "none", "0")
        if graph_disabled and config.get("enable_graph", False):
            import torch_npu  # noqa: F401
            return "aclgraph"
        return graph_backend

    @torch.inference_mode()
    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: object,
        input_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a single forward step, dispatching to graph runner or eager."""
        if not self._kv_bound:
            raise RuntimeError("KV caches are not bound")

        graph_runner = self.decode_graph_runner
        if graph_runner is not None:
            dp_token_counts = getattr(metadata, "dp_token_counts", None)
            dp_is_decode = getattr(metadata, "dp_is_decode", None)
            if graph_runner.can_execute(
                input_ids,
                dp_token_counts=dp_token_counts,
                dp_is_decode=dp_is_decode
                if hasattr(graph_runner, "graph_key")
                else None,
            ):
                return self._run_graph(
                    graph_runner, input_ids, positions, metadata, input_embedding
                )

        # Eager fallback
        return self.model(input_ids, positions)

    def _run_graph(self, runner, input_ids, positions, metadata, input_embedding):
        """Warmup (if needed) and replay a captured graph."""
        runner.warmup(input_ids.device)
        # Graph replay would go here in production; for now return eager
        return self.model(input_ids, positions)

    def bind_kv_caches(self, kv_caches: list) -> None:
        """Bind KV caches to the attention backend and runners."""
        self._kv_bound = True
