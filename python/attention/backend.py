"""Attention backend registry with DP-aware backend selection.

Ported from xLLM upstream commit 78aa2a85 (PR #2258).
Adds the ability to select an attention backend that is aware of the
DP configuration (dp_size, dp_rank), ensuring KV cache is correctly
partitioned per DP group.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AttentionBackend(Protocol):
    """Protocol for attention backends used by the Python model executor."""

    def prepare(self, metadata: Any, graph_mode: bool = False) -> None:
        ...

    def bind_kv_caches(self, layer_caches: list) -> None:
        ...


@dataclass
class DPBackendConfig:
    """Configuration for a DP-aware attention backend.

    Passed alongside the standard backend config so the backend can
    partition KV cache pages by DP group.
    """

    dp_size: int = 1
    dp_rank: int = 0


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_BACKEND_REGISTRY: dict[str, type] = {}


def register_backend(name: str, cls: type) -> None:
    """Register an attention backend class under ``name``."""
    _BACKEND_REGISTRY[name] = cls


def get_backend(name: str) -> type:
    """Look up a registered attention backend by name."""
    if name not in _BACKEND_REGISTRY:
        available = ", ".join(sorted(_BACKEND_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown attention backend '{name}'. Available: {available}"
        )
    return _BACKEND_REGISTRY[name]


def list_backends() -> list[str]:
    """Return the names of all registered backends."""
    return sorted(_BACKEND_REGISTRY)


def create_attention_backend(
    name: str,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    dp_config: DPBackendConfig | None = None,
    **kwargs: Any,
) -> Any:
    """Instantiate a registered attention backend with DP config.

    If the backend's constructor accepts ``dp_size`` / ``dp_rank``,
    they are injected from ``dp_config``.
    """
    cls = get_backend(name)
    init_kwargs = dict(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=scale,
        **kwargs,
    )
    if dp_config is not None:
        init_kwargs["dp_size"] = dp_config.dp_size
        init_kwargs["dp_rank"] = dp_config.dp_rank
    return cls(**init_kwargs)
