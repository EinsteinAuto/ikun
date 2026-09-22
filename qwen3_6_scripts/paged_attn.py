# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import List, Optional, Tuple
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import torch
import traceback
from vllm import _custom_ops as ops
from vllm.bi100_env import env_bool, env_int
from vllm.bi100_profile import bi100_profile_count, bi100_timer

try:
    from vllm import corex_paged_kv_gather as _corex_paged_kv_gather
except ImportError:
    _corex_paged_kv_gather = None

try:
    from vllm import corex_fused_paged_prefill as _corex_fused_paged_prefill
except ImportError:
    _corex_fused_paged_prefill = None

# from vllm.attention.ops.prefix_prefill import context_attention_fwd
# NOTE: context_attention_fwd (Triton kernel from prefix_prefill.py) is NOT
# imported here.  On Iluvatar BI-V100 that kernel hangs the GPU card
# permanently.  Chunked-prefill / prefix-caching attention is handled by
# _forward_prefix_pytorch below (pure PyTorch, no Triton dependency).

# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
_PARTITION_SIZE = 512
_PYTORCH_DECODE_THRESHOLD = env_int(
    "BI100_PYTORCH_DECODE_THRESHOLD", 32768, 1, 262144)
_PREFIX_BLOCKS_PER_TILE = env_int(
    "BI100_PREFIX_BLOCKS_PER_TILE", 32, 1, 1024)
_FORCE_PAGED_ATTN_V2 = env_bool("BI100_FORCE_PAGED_ATTN_V2", False)
_PAGED_ATTN_DIAGNOSTICS = env_bool(
    "BI100_PAGED_ATTN_DIAGNOSTICS", False)
_USE_COREX_PAGED_KV_GATHER = (
    _corex_paged_kv_gather is not None
    and env_bool("BI100_ATTN_COREX_PAGED_GATHER", True))
_ENABLE_COREX_FUSED_PAGED_PREFILL = env_bool(
    "BI100_ATTN_COREX_FUSED_PREFILL", False)
_FUSED_PREFILL_DIAGNOSTICS = env_bool(
    "BI100_ATTN_COREX_FUSED_PREFILL_DIAGNOSTICS", False)


def _env_choice(name: str, default: str, choices: Tuple[str, ...]) -> str:
    value = os.environ.get(name, default)
    if value not in choices:
        raise RuntimeError(
            f"{name} must be one of {', '.join(choices)}, got {value!r}")
    return value


_FUSED_PREFILL_SHADOW = env_bool(
    "BI100_ATTN_COREX_FUSED_PREFILL_SHADOW", False)
_FUSED_PREFILL_SHADOW_MAX_CALLS_PER_CONTEXT = env_int(
    "BI100_ATTN_COREX_FUSED_PREFILL_SHADOW_MAX_CALLS_PER_CONTEXT",
    2, 1, 8)
_FUSED_PREFILL_SHADOW_NUMERIC_MODE = _env_choice(
    "BI100_ATTN_COREX_FUSED_PREFILL_SHADOW_NUMERIC_MODE",
    "legacy",
    ("legacy", "calibrated"),
)
_FUSED_PREFILL_SHADOW_FAILURE_ACTION = _env_choice(
    "BI100_ATTN_COREX_FUSED_PREFILL_SHADOW_FAILURE_ACTION",
    "raise",
    ("raise", "record"),
)
_USE_COREX_FUSED_PAGED_PREFILL = (
    _corex_fused_paged_prefill is not None
    and _ENABLE_COREX_FUSED_PAGED_PREFILL)
_DECODE_LOG_INTERVAL = 8192 if _PAGED_ATTN_DIAGNOSTICS else 0
_DECODE_DISPATCH_LOGGED = set()
_PREFIX_DISPATCH_LOGGED = set()
_FUSED_PREFILL_DIAGNOSTICS_LOGGED = set()
_CACHE_WRITE_LOGGED = False
_FUSED_PREFILL_SHADOW_RELATIVE_L2_LIMIT = 1.0e-5
_FUSED_PREFILL_SHADOW_MAX_ABS_LIMIT = 1.0e-3
_FUSED_PREFILL_SHADOW_ERROR_MULTIPLIER = 2.0
_FUSED_PREFILL_SHADOW_RATIO_FLOOR = 1.0e-12
_FUSED_PREFILL_SHADOW_STATE = {
    "pid": None,
    "records": [],
}
_ACTIVATION_CAPTURE_ENABLED = env_bool(
    "BI100_ATTN_CAPTURE_REPLAY", False)
_ACTIVATION_CAPTURE_ATTESTATION = (
    "synthetic-exact-prompt-v1")
_ACTIVATION_CAPTURE_STATE = {
    "pid": None,
    "seen_by_bucket": {},
    "records": [],
}


def _parse_fused_prefill_shadow_contexts(raw: str) -> Tuple[int, ...]:
    """Parse fixed lower-bound buckets used by the diagnostic shadow."""
    values = []
    for field in raw.split(","):
        field = field.strip()
        if not field:
            raise RuntimeError(
                "BI100_ATTN_COREX_FUSED_PREFILL_SHADOW_CONTEXTS "
                "contains an empty field")
        try:
            value = int(field)
        except ValueError as exc:
            raise RuntimeError(
                "BI100_ATTN_COREX_FUSED_PREFILL_SHADOW_CONTEXTS must "
                "contain integers") from exc
        if value < 0 or value > 262144:
            raise RuntimeError(
                "BI100_ATTN_COREX_FUSED_PREFILL_SHADOW_CONTEXTS value "
                "is outside [0, 262144]")
        values.append(value)
    if not values or values != sorted(set(values)):
        raise RuntimeError(
            "BI100_ATTN_COREX_FUSED_PREFILL_SHADOW_CONTEXTS must be "
            "strictly increasing and unique")
    return tuple(values)


_FUSED_PREFILL_SHADOW_CONTEXTS = _parse_fused_prefill_shadow_contexts(
    os.environ.get(
        "BI100_ATTN_COREX_FUSED_PREFILL_SHADOW_CONTEXTS",
        "49152,114688"))


def _validate_fused_prefill_shadow_configuration(
    enabled: bool,
    fused_enabled: bool,
    report_dir: Optional[str],
    run_id: Optional[str],
) -> Optional[Path]:
    """Validate that the diagnostic cannot silently write ambiguous data."""
    if not enabled:
        return None
    if not fused_enabled:
        raise RuntimeError(
            "fused-prefill shadow requires the fused-prefill path")
    if (
        _FUSED_PREFILL_SHADOW_FAILURE_ACTION == "record"
        and _FUSED_PREFILL_SHADOW_NUMERIC_MODE != "calibrated"
    ):
        raise RuntimeError(
            "record-only shadow failures require calibrated numeric mode")
    if not report_dir:
        raise RuntimeError(
            "BI100_ATTN_COREX_FUSED_PREFILL_SHADOW_REPORT_DIR is required")
    path = Path(report_dir).expanduser()
    tmp_root = Path("/tmp").resolve()
    try:
        path = path.resolve(strict=False)
    except OSError as exc:
        raise RuntimeError(
            "fused-prefill shadow report directory cannot be resolved") \
            from exc
    if (
        not path.is_absolute()
        or path == tmp_root
        or not path.is_relative_to(tmp_root)
    ):
        raise RuntimeError(
            "fused-prefill shadow report directory must be under /tmp")
    if (
        not run_id
        or len(run_id) > 96
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id) is None
    ):
        raise RuntimeError(
            "BI100_ATTN_COREX_FUSED_PREFILL_SHADOW_RUN_ID is invalid")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.resolve(strict=True).is_relative_to(tmp_root):
        raise RuntimeError(
            "fused-prefill shadow report directory escaped /tmp")
    try:
        path.chmod(0o700)
    except OSError:
        pass
    return path


_FUSED_PREFILL_SHADOW_RUN_ID = os.environ.get(
    "BI100_ATTN_COREX_FUSED_PREFILL_SHADOW_RUN_ID")
_FUSED_PREFILL_SHADOW_REPORT_DIR = (
    _validate_fused_prefill_shadow_configuration(
        _FUSED_PREFILL_SHADOW,
        _USE_COREX_FUSED_PAGED_PREFILL,
        os.environ.get(
            "BI100_ATTN_COREX_FUSED_PREFILL_SHADOW_REPORT_DIR"),
        _FUSED_PREFILL_SHADOW_RUN_ID,
    ))


def _parse_strict_int_tuple(
    raw: str,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> Tuple[int, ...]:
    values = []
    for field in raw.split(","):
        field = field.strip()
        if not field:
            raise RuntimeError(f"{name} contains an empty field")
        try:
            value = int(field)
        except ValueError as exc:
            raise RuntimeError(f"{name} must contain integers") from exc
        if value < minimum or value > maximum:
            raise RuntimeError(
                f"{name} value is outside [{minimum}, {maximum}]")
        values.append(value)
    if not values or values != sorted(set(values)):
        raise RuntimeError(f"{name} must be strictly increasing and unique")
    return tuple(values)


_ACTIVATION_CAPTURE_CONTEXTS = _parse_strict_int_tuple(
    os.environ.get(
        "BI100_ATTN_CAPTURE_REPLAY_CONTEXTS",
        "24576,57344,122880",
    ),
    name="BI100_ATTN_CAPTURE_REPLAY_CONTEXTS",
    minimum=0,
    maximum=262144,
)
_ACTIVATION_CAPTURE_CALL_ORDINALS = _parse_strict_int_tuple(
    os.environ.get(
        "BI100_ATTN_CAPTURE_REPLAY_CALL_ORDINALS",
        "0,4,9",
    ),
    name="BI100_ATTN_CAPTURE_REPLAY_CALL_ORDINALS",
    minimum=0,
    maximum=63,
)


def _validate_activation_capture_configuration(
    enabled: bool,
    fused_enabled: bool,
    report_dir: Optional[str],
    run_id: Optional[str],
    source_revision: Optional[str],
    runtime_identity: Optional[str],
    attestation: Optional[str],
) -> Optional[Path]:
    if not enabled:
        return None
    if fused_enabled:
        raise RuntimeError(
            "activation capture requires the baseline PyTorch fallback")
    if attestation != _ACTIVATION_CAPTURE_ATTESTATION:
        raise RuntimeError(
            "activation capture requires the synthetic prompt attestation")
    if (
        not source_revision
        or re.fullmatch(r"[0-9a-f]{40,64}", source_revision) is None
    ):
        raise RuntimeError(
            "BI100_ATTN_CAPTURE_REPLAY_SOURCE_REVISION is invalid")
    if (
        not runtime_identity
        or len(runtime_identity) > 160
        or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.:/-]*", runtime_identity) is None
    ):
        raise RuntimeError(
            "BI100_ATTN_CAPTURE_REPLAY_RUNTIME_IDENTITY is invalid")
    if (
        not run_id
        or len(run_id) > 96
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id) is None
    ):
        raise RuntimeError(
            "BI100_ATTN_CAPTURE_REPLAY_RUN_ID is invalid")
    if not report_dir:
        raise RuntimeError(
            "BI100_ATTN_CAPTURE_REPLAY_DIR is required")
    path = Path(report_dir).expanduser()
    tmp_root = Path("/tmp").resolve()
    try:
        path = path.resolve(strict=False)
    except OSError as exc:
        raise RuntimeError(
            "activation capture directory cannot be resolved") from exc
    if (
        not path.is_absolute()
        or path == tmp_root
        or not path.is_relative_to(tmp_root)
    ):
        raise RuntimeError(
            "activation capture directory must be under /tmp")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.resolve(strict=True).is_relative_to(tmp_root):
        raise RuntimeError(
            "activation capture directory escaped /tmp")
    try:
        path.chmod(0o700)
    except OSError:
        pass
    return path


_ACTIVATION_CAPTURE_RUN_ID = os.environ.get(
    "BI100_ATTN_CAPTURE_REPLAY_RUN_ID")
_ACTIVATION_CAPTURE_SOURCE_REVISION = os.environ.get(
    "BI100_ATTN_CAPTURE_REPLAY_SOURCE_REVISION")
_ACTIVATION_CAPTURE_RUNTIME_IDENTITY = os.environ.get(
    "BI100_ATTN_CAPTURE_REPLAY_RUNTIME_IDENTITY")
_ACTIVATION_CAPTURE_DIR = _validate_activation_capture_configuration(
    _ACTIVATION_CAPTURE_ENABLED,
    _ENABLE_COREX_FUSED_PAGED_PREFILL,
    os.environ.get("BI100_ATTN_CAPTURE_REPLAY_DIR"),
    _ACTIVATION_CAPTURE_RUN_ID,
    _ACTIVATION_CAPTURE_SOURCE_REVISION,
    _ACTIVATION_CAPTURE_RUNTIME_IDENTITY,
    os.environ.get("BI100_ATTN_CAPTURE_REPLAY_SYNTHETIC_ATTESTATION"),
)


def _log_corex_fused_prefill_diagnostic(stage: str, **fields) -> None:
    """Emit one privacy-safe guard snapshot per stage and worker."""
    if not _FUSED_PREFILL_DIAGNOSTICS:
        return
    key = (os.getpid(), stage)
    if key in _FUSED_PREFILL_DIAGNOSTICS_LOGGED:
        return
    details = " ".join(f"{name}={value}" for name, value in fields.items())
    print(
        "[BI100 PAGED_ATTN] fused_prefill_guard "
        f"pid={os.getpid()} rank={os.environ.get('RANK', '?')} "
        f"local_rank={os.environ.get('LOCAL_RANK', '?')} stage={stage} "
        f"{details}",
        file=sys.stderr,
        flush=True,
    )
    _FUSED_PREFILL_DIAGNOSTICS_LOGGED.add(key)


def _fused_prefill_shadow_rank() -> int:
    distributed = getattr(torch, "distributed", None)
    if distributed is not None:
        try:
            if (
                distributed.is_available()
                and distributed.is_initialized()
            ):
                rank = int(distributed.get_rank())
                if rank >= 0:
                    return rank
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    for name in ("RANK", "LOCAL_RANK"):
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value >= 0:
            return value
    cuda = getattr(torch, "cuda", None)
    if cuda is not None:
        try:
            if cuda.is_available():
                device = int(cuda.current_device())
                if device >= 0:
                    return device
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    return -1


def _activation_capture_process_state() -> dict:
    pid = os.getpid()
    if _ACTIVATION_CAPTURE_STATE["pid"] != pid:
        _ACTIVATION_CAPTURE_STATE["pid"] = pid
        _ACTIVATION_CAPTURE_STATE["seen_by_bucket"] = {}
        _ACTIVATION_CAPTURE_STATE["records"] = []
    return _ACTIVATION_CAPTURE_STATE


def _activation_capture_bucket(context_tokens: int) -> Optional[int]:
    for index, lower_bound in enumerate(_ACTIVATION_CAPTURE_CONTEXTS):
        upper_bound = (
            _ACTIVATION_CAPTURE_CONTEXTS[index + 1]
            if index + 1 < len(_ACTIVATION_CAPTURE_CONTEXTS)
            else 262145
        )
        if lower_bound <= context_tokens < upper_bound:
            return lower_bound
    return None


def _atomic_write_activation_manifest(records: list) -> None:
    if _ACTIVATION_CAPTURE_DIR is None:
        raise RuntimeError("activation capture directory is unset")
    rank = _fused_prefill_shadow_rank()
    if rank < 0:
        raise RuntimeError("activation capture cannot determine TP rank")
    value = {
        "schema": "bi100-fused-prefill-activation-bank-v1",
        "version": 1,
        "run_id": _ACTIVATION_CAPTURE_RUN_ID,
        "rank": rank,
        "source_revision": _ACTIVATION_CAPTURE_SOURCE_REVISION,
        "runtime_identity": _ACTIVATION_CAPTURE_RUNTIME_IDENTITY,
        "producer": "baseline-pytorch-fallback",
        "synthetic_prompt_attestation": (
            _ACTIVATION_CAPTURE_ATTESTATION),
        "selection": {
            "context_buckets": list(_ACTIVATION_CAPTURE_CONTEXTS),
            "full_attention_call_ordinals": list(
                _ACTIVATION_CAPTURE_CALL_ORDINALS),
        },
        "record_count": len(records),
        "records": records,
        "privacy": {
            "raw_activation_files_private": True,
            "raw_activation_files_may_be_committed": False,
            "contains_prompts": False,
            "contains_model_outputs": False,
            "contains_token_ids": False,
            "contains_credentials": False,
        },
    }
    destination = _ACTIVATION_CAPTURE_DIR / f"rank-{rank}.manifest.json"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            json.dump(value, stream, ensure_ascii=True, indent=2,
                      sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _reserve_activation_capture(
    context_tokens: int,
) -> Optional[Tuple[int, int]]:
    if not _ACTIVATION_CAPTURE_ENABLED:
        return None
    bucket = _activation_capture_bucket(context_tokens)
    if bucket is None:
        return None
    state = _activation_capture_process_state()
    ordinal = int(state["seen_by_bucket"].get(bucket, 0))
    state["seen_by_bucket"][bucket] = ordinal + 1
    if ordinal not in _ACTIVATION_CAPTURE_CALL_ORDINALS:
        return None
    return bucket, ordinal


def _tensor_shape_dtype(tensor: torch.Tensor) -> dict:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture_fused_prefill_activation(
    reservation: Tuple[int, int],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    active_block_table: torch.Tensor,
    context_tokens: int,
    scale: float,
) -> None:
    if _ACTIVATION_CAPTURE_DIR is None:
        raise RuntimeError("activation capture directory is unset")
    bucket, ordinal = reservation
    rank = _fused_prefill_shadow_rank()
    if rank < 0:
        raise RuntimeError("activation capture cannot determine TP rank")

    active_ids = [
        int(value) for value in active_block_table.detach().cpu().tolist()
    ]
    identity_to_compact = {}
    unique_ids = []
    compact_table = []
    for physical_id in active_ids:
        if physical_id < 0 or physical_id >= key_cache.shape[0]:
            raise RuntimeError(
                "activation capture block table is outside the KV cache")
        compact_id = identity_to_compact.get(physical_id)
        if compact_id is None:
            compact_id = len(unique_ids)
            identity_to_compact[physical_id] = compact_id
            unique_ids.append(physical_id)
        compact_table.append(compact_id)

    if unique_ids:
        physical = torch.tensor(
            unique_ids,
            dtype=torch.long,
            device=key_cache.device,
        )
        compact_key_cache = (
            key_cache.index_select(0, physical).detach().cpu().contiguous())
        compact_value_cache = (
            value_cache.index_select(0, physical).detach().cpu().contiguous())
    else:
        compact_key_cache = key_cache[:0].detach().cpu().contiguous()
        compact_value_cache = value_cache[:0].detach().cpu().contiguous()
    compact_block_table = torch.tensor(
        compact_table,
        dtype=torch.int32,
    )
    tensors = {
        "query": query.detach().cpu().contiguous(),
        "key": key.detach().cpu().contiguous(),
        "value": value.detach().cpu().contiguous(),
        "key_cache": compact_key_cache,
        "value_cache": compact_value_cache,
        "block_table": compact_block_table,
    }
    filename = (
        f"rank-{rank}.bucket-{bucket}.ordinal-{ordinal}."
        f"ctx-{context_tokens}.q-{query.shape[0]}.pt"
    )
    destination = _ACTIVATION_CAPTURE_DIR / filename
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{filename}.", suffix=".tmp",
        dir=destination.parent)
    os.close(descriptor)
    try:
        torch.save({
            "schema": "bi100-fused-prefill-activation-case-v1",
            "version": 1,
            "context_tokens": context_tokens,
            "scale": float(scale),
            "rank": rank,
            "bucket": bucket,
            "call_ordinal": ordinal,
            "tensors": tensors,
        }, temporary)
        os.chmod(temporary, 0o600)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass

    state = _activation_capture_process_state()
    state["records"].append({
        "bucket_min_context_tokens": bucket,
        "call_ordinal": ordinal,
        "context_tokens": context_tokens,
        "query_length": int(query.shape[0]),
        "file": filename,
        "sha256": _sha256_file(destination),
        "size_bytes": destination.stat().st_size,
        "compact_physical_blocks": len(unique_ids),
        "logical_blocks": len(compact_table),
        "tensors": {
            name: _tensor_shape_dtype(tensor)
            for name, tensor in tensors.items()
        },
    })
    _atomic_write_activation_manifest(state["records"])


def _fused_prefill_shadow_process_state() -> dict:
    pid = os.getpid()
    if _FUSED_PREFILL_SHADOW_STATE["pid"] != pid:
        _FUSED_PREFILL_SHADOW_STATE["pid"] = pid
        _FUSED_PREFILL_SHADOW_STATE["records"] = []
    return _FUSED_PREFILL_SHADOW_STATE


def _fused_prefill_shadow_report_path() -> Path:
    if _FUSED_PREFILL_SHADOW_REPORT_DIR is None:
        raise RuntimeError("fused-prefill shadow report directory is unset")
    rank = _fused_prefill_shadow_rank()
    rank_label = str(rank) if rank >= 0 else "unknown"
    return _FUSED_PREFILL_SHADOW_REPORT_DIR / (
        f"rank-{rank_label}-pid-{os.getpid()}.json")


def _atomic_write_fused_prefill_shadow_report(value: dict) -> None:
    path = _fused_prefill_shadow_report_path()
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=True, indent=2,
                      sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _build_fused_prefill_shadow_report(records: list) -> dict:
    expected = (
        len(_FUSED_PREFILL_SHADOW_CONTEXTS)
        * _FUSED_PREFILL_SHADOW_MAX_CALLS_PER_CONTEXT)
    completed = [
        record for record in records
        if record["status"] in {"pass", "fail", "invalid"}
    ]
    failures = [record for record in completed if record["status"] == "fail"]
    invalid = [record for record in completed if record["status"] == "invalid"]
    pending = [record for record in records if record["status"] == "pending"]
    relative_l2_values = [
        record["relative_l2"] for record in completed
        if isinstance(record.get("relative_l2"), float)
    ]
    max_abs_values = [
        record["max_abs"] for record in completed
        if isinstance(record.get("max_abs"), float)
    ]
    if invalid:
        status = "invalid"
    elif failures:
        status = "fail"
    elif len(completed) == expected and not pending:
        status = "pass"
    else:
        status = "collecting"
    report = {
        "schema": "bi100-fused-prefill-real-activation-shadow-v1",
        "version": 1,
        "run_id": _FUSED_PREFILL_SHADOW_RUN_ID,
        "pid": os.getpid(),
        "rank": _fused_prefill_shadow_rank(),
        "status": status,
        "selection": {
            "minimum_context_tokens": list(
                _FUSED_PREFILL_SHADOW_CONTEXTS),
            "max_calls_per_context": (
                _FUSED_PREFILL_SHADOW_MAX_CALLS_PER_CONTEXT),
        },
        "thresholds": {
            "require_finite_candidate": True,
            "require_finite_reference": True,
            "maximum_relative_l2": (
                _FUSED_PREFILL_SHADOW_RELATIVE_L2_LIMIT),
            "maximum_absolute_error": (
                _FUSED_PREFILL_SHADOW_MAX_ABS_LIMIT),
        },
        "observations": {
            "expected": expected,
            "reserved": len(records),
            "completed": len(completed),
            "passed": sum(record["status"] == "pass" for record in records),
            "failed": len(failures),
            "invalid": len(invalid),
            "pending": len(pending),
            "maximum_relative_l2": (
                max(relative_l2_values) if relative_l2_values else None),
            "maximum_absolute_error": (
                max(max_abs_values) if max_abs_values else None),
        },
        "records": records,
        "privacy": {
            "contains_prompts": False,
            "contains_model_outputs": False,
            "contains_tensor_values": False,
            "contains_token_ids": False,
            "contains_credentials": False,
        },
    }
    if _FUSED_PREFILL_SHADOW_NUMERIC_MODE == "calibrated":
        report["schema"] = (
            "bi100-fused-prefill-real-activation-calibrated-shadow-v1")
        report["thresholds"] = {
            "require_finite_candidate": True,
            "require_finite_reference": True,
            "maximum_candidate_vs_rounded_relative_l2": (
                _FUSED_PREFILL_SHADOW_RELATIVE_L2_LIMIT),
            "maximum_error_multiple_over_fp16_rounding": (
                _FUSED_PREFILL_SHADOW_ERROR_MULTIPLIER),
            "ratio_denominator_floor": (
                _FUSED_PREFILL_SHADOW_RATIO_FLOOR),
            "fixed_max_abs_role": "diagnostic_only",
            "finite_failure_action": (
                _FUSED_PREFILL_SHADOW_FAILURE_ACTION),
        }
        calibrated_metrics = (
            "candidate_to_fp32_relative_l2",
            "candidate_to_fp32_max_abs",
            "rounded_to_fp32_relative_l2",
            "rounded_to_fp32_max_abs",
            "relative_l2_baseline_ratio",
            "max_abs_baseline_ratio",
        )
        for name in calibrated_metrics:
            values = [
                record[name] for record in completed
                if isinstance(record.get(name), float)
            ]
            report["observations"][f"maximum_{name}"] = (
                max(values) if values else None)
    return report


def _reserve_fused_prefill_shadow(
    query: torch.Tensor,
    block_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> Optional[int]:
    if not _FUSED_PREFILL_SHADOW:
        return None
    state = _fused_prefill_shadow_process_state()
    records = state["records"]
    selected_bucket = None
    for bucket_index, bucket in enumerate(_FUSED_PREFILL_SHADOW_CONTEXTS):
        upper_bound = (
            _FUSED_PREFILL_SHADOW_CONTEXTS[bucket_index + 1]
            if bucket_index + 1 < len(_FUSED_PREFILL_SHADOW_CONTEXTS)
            else None)
        used = sum(
            record["bucket_min_context_tokens"] == bucket
            for record in records)
        if (
            block_context_len >= bucket
            and (upper_bound is None or block_context_len < upper_bound)
            and used < _FUSED_PREFILL_SHADOW_MAX_CALLS_PER_CONTEXT
        ):
            selected_bucket = bucket
            break
    if selected_bucket is None:
        return None
    record = {
        "index": len(records),
        "status": "pending",
        "bucket_min_context_tokens": selected_bucket,
        "context_tokens": block_context_len,
        "query_shape": list(query.shape),
        "query_heads": num_q_heads,
        "kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "block_size": block_size,
        "candidate_finite": None,
        "reference_finite": None,
        "relative_l2": None,
        "max_abs": None,
        "error_stage": None,
        "error_type": None,
    }
    if _FUSED_PREFILL_SHADOW_NUMERIC_MODE == "calibrated":
        record.update({
            "candidate_to_fp32_relative_l2": None,
            "candidate_to_fp32_max_abs": None,
            "rounded_to_fp32_relative_l2": None,
            "rounded_to_fp32_max_abs": None,
            "relative_l2_baseline_ratio": None,
            "max_abs_baseline_ratio": None,
        })
    records.append(record)
    _atomic_write_fused_prefill_shadow_report(
        _build_fused_prefill_shadow_report(records))
    return record["index"]


def _finish_fused_prefill_shadow(
    index: int,
    *,
    status: str,
    candidate_finite: Optional[bool] = None,
    reference_finite: Optional[bool] = None,
    relative_l2: Optional[float] = None,
    max_abs: Optional[float] = None,
    candidate_to_fp32_relative_l2: Optional[float] = None,
    candidate_to_fp32_max_abs: Optional[float] = None,
    rounded_to_fp32_relative_l2: Optional[float] = None,
    rounded_to_fp32_max_abs: Optional[float] = None,
    relative_l2_baseline_ratio: Optional[float] = None,
    max_abs_baseline_ratio: Optional[float] = None,
    error_stage: Optional[str] = None,
    error_type: Optional[str] = None,
) -> None:
    state = _fused_prefill_shadow_process_state()
    records = state["records"]
    if index < 0 or index >= len(records):
        raise RuntimeError("fused-prefill shadow record index is invalid")
    if status not in {"pass", "fail", "invalid"}:
        raise RuntimeError("fused-prefill shadow status is invalid")
    record = records[index]
    updates = {
        "status": status,
        "candidate_finite": candidate_finite,
        "reference_finite": reference_finite,
        "relative_l2": relative_l2,
        "max_abs": max_abs,
        "error_stage": error_stage,
        "error_type": error_type,
    }
    if _FUSED_PREFILL_SHADOW_NUMERIC_MODE == "calibrated":
        updates.update({
            "candidate_to_fp32_relative_l2": (
                candidate_to_fp32_relative_l2),
            "candidate_to_fp32_max_abs": candidate_to_fp32_max_abs,
            "rounded_to_fp32_relative_l2": (
                rounded_to_fp32_relative_l2),
            "rounded_to_fp32_max_abs": rounded_to_fp32_max_abs,
            "relative_l2_baseline_ratio": relative_l2_baseline_ratio,
            "max_abs_baseline_ratio": max_abs_baseline_ratio,
        })
    record.update(updates)
    _atomic_write_fused_prefill_shadow_report(
        _build_fused_prefill_shadow_report(records))


def _calibrated_shadow_metrics_qualified(metrics: dict) -> bool:
    return (
        metrics["relative_l2"]
        <= _FUSED_PREFILL_SHADOW_RELATIVE_L2_LIMIT
        and metrics["candidate_to_fp32_relative_l2"]
        <= (
            _FUSED_PREFILL_SHADOW_ERROR_MULTIPLIER
            * metrics["rounded_to_fp32_relative_l2"]
            + _FUSED_PREFILL_SHADOW_RATIO_FLOOR
        )
        and metrics["candidate_to_fp32_max_abs"]
        <= (
            _FUSED_PREFILL_SHADOW_ERROR_MULTIPLIER
            * metrics["rounded_to_fp32_max_abs"]
            + _FUSED_PREFILL_SHADOW_RATIO_FLOOR
        )
    )


def _error_metrics(
    actual: torch.Tensor,
    reference: torch.Tensor,
    denominator: float,
) -> Tuple[float, float]:
    difference = actual - reference
    relative_l2 = float(torch.norm(difference).item()) / denominator
    max_abs = float(difference.abs().max().item())
    return relative_l2, max_abs


def _compare_fused_prefill_shadow_outputs(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    reference_fp32: Optional[torch.Tensor] = None,
) -> dict:
    candidate_float = candidate.float()
    reference_float = reference.float()
    candidate_finite = bool(torch.isfinite(candidate_float).all().item())
    reference_finite = bool(torch.isfinite(reference_float).all().item())
    if _FUSED_PREFILL_SHADOW_NUMERIC_MODE == "calibrated":
        if reference_fp32 is None:
            raise RuntimeError(
                "calibrated fused-prefill shadow requires FP32 reference")
        reference_fp32 = reference_fp32.float()
        reference_finite = (
            reference_finite
            and bool(torch.isfinite(reference_fp32).all().item())
        )
    if not candidate_finite or not reference_finite:
        result = {
            "status": "fail" if not candidate_finite else "invalid",
            "candidate_finite": candidate_finite,
            "reference_finite": reference_finite,
            "relative_l2": None,
            "max_abs": None,
        }
        if _FUSED_PREFILL_SHADOW_NUMERIC_MODE == "calibrated":
            result.update({
                "candidate_to_fp32_relative_l2": None,
                "candidate_to_fp32_max_abs": None,
                "rounded_to_fp32_relative_l2": None,
                "rounded_to_fp32_max_abs": None,
                "relative_l2_baseline_ratio": None,
                "max_abs_baseline_ratio": None,
            })
        return result
    denominator = max(float(torch.norm(reference_float).item()), 1.0e-12)
    relative_l2, max_abs = _error_metrics(
        candidate_float, reference_float, denominator)
    if _FUSED_PREFILL_SHADOW_NUMERIC_MODE == "calibrated":
        if reference_fp32 is None:
            raise RuntimeError(
                "calibrated fused-prefill shadow lost its FP32 reference")
        fp32_denominator = max(
            float(torch.norm(reference_fp32).item()),
            _FUSED_PREFILL_SHADOW_RATIO_FLOOR,
        )
        candidate_fp32_relative_l2, candidate_fp32_max_abs = (
            _error_metrics(
                candidate_float, reference_fp32, fp32_denominator))
        rounded_fp32_relative_l2, rounded_fp32_max_abs = (
            _error_metrics(
                reference_float, reference_fp32, fp32_denominator))
        metrics = {
            "relative_l2": relative_l2,
            "max_abs": max_abs,
            "candidate_to_fp32_relative_l2": (
                candidate_fp32_relative_l2),
            "candidate_to_fp32_max_abs": candidate_fp32_max_abs,
            "rounded_to_fp32_relative_l2": rounded_fp32_relative_l2,
            "rounded_to_fp32_max_abs": rounded_fp32_max_abs,
            "relative_l2_baseline_ratio": (
                candidate_fp32_relative_l2
                / max(
                    rounded_fp32_relative_l2,
                    _FUSED_PREFILL_SHADOW_RATIO_FLOOR,
                )
            ),
            "max_abs_baseline_ratio": (
                candidate_fp32_max_abs
                / max(
                    rounded_fp32_max_abs,
                    _FUSED_PREFILL_SHADOW_RATIO_FLOOR,
                )
            ),
        }
        return {
            "status": (
                "pass"
                if _calibrated_shadow_metrics_qualified(metrics)
                else "fail"
            ),
            "candidate_finite": True,
            "reference_finite": True,
            **metrics,
        }
    qualified = (
        relative_l2 <= _FUSED_PREFILL_SHADOW_RELATIVE_L2_LIMIT
        and max_abs <= _FUSED_PREFILL_SHADOW_MAX_ABS_LIMIT)
    return {
        "status": "pass" if qualified else "fail",
        "candidate_finite": True,
        "reference_finite": True,
        "relative_l2": relative_l2,
        "max_abs": max_abs,
    }


def _validate_decode_layout(
    num_seqs: int,
    seq_lens_count: int,
    block_table_rows: int,
    block_table_width: int,
    actual_max: int,
    block_size: int,
    physical_key_blocks: int,
    physical_value_blocks: int,
    num_heads: int,
    num_kv_heads: int,
) -> int:
    """Validate host-visible decode metadata before a native kernel launch."""
    if num_seqs <= 0:
        raise RuntimeError(f"decode requires num_seqs > 0, got {num_seqs}")
    if seq_lens_count != num_seqs:
        raise RuntimeError(
            f"seq_lens has {seq_lens_count} entries for {num_seqs} sequences")
    if block_table_rows < num_seqs:
        raise RuntimeError(
            f"block table has {block_table_rows} rows for {num_seqs} sequences")
    if actual_max <= 0:
        raise RuntimeError(f"decode sequence length must be > 0, got {actual_max}")
    if block_size <= 0:
        raise RuntimeError(f"KV block_size must be > 0, got {block_size}")
    if physical_key_blocks != physical_value_blocks:
        raise RuntimeError(
            "key/value cache block counts differ: "
            f"{physical_key_blocks} != {physical_value_blocks}")
    if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        raise RuntimeError(
            f"invalid GQA layout: num_heads={num_heads}, "
            f"num_kv_heads={num_kv_heads}")

    required_blocks = (actual_max + block_size - 1) // block_size
    if required_blocks > block_table_width:
        raise RuntimeError(
            f"decode needs {required_blocks} blocks for seq_len={actual_max}, "
            f"but block table width is {block_table_width}")
    return required_blocks


def _strict_prefix_query_segments(
    context_len: int,
    query_len: int,
    block_size: int,
) -> List[Tuple[int, int, int]]:
    """Split a query at the strict prefix-cache checkpoint, if it crosses it."""
    if query_len <= 0:
        return []
    total_len = context_len + query_len
    strict_prefix_len = ((total_len - 1) // block_size) * block_size
    split = strict_prefix_len - context_len
    if 0 < split < query_len:
        return [(0, split, context_len),
                (split, query_len, strict_prefix_len)]
    return [(0, query_len, context_len)]


def _is_supported_corex_fused_paged_prefill_request(
    kv_cache_dtype: str,
    max_query_len: int,
    total_query_len: int,
    alibi_slopes: Optional[torch.Tensor],
    sliding_window: Optional[int],
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    is_causal_decoder: bool,
) -> bool:
    """Check request-wide properties that are outside the native ABI."""
    return bool(
        is_causal_decoder
        and _PREFIX_BLOCKS_PER_TILE == 32
        and kv_cache_dtype == "auto"
        and max_query_len == total_query_len
        and alibi_slopes is None
        and sliding_window is None
        and float(k_scale.item() if isinstance(k_scale, torch.Tensor) else k_scale) == 1.0
        and float(v_scale.item() if isinstance(v_scale, torch.Tensor) else v_scale) == 1.0
    )


def _can_enable_corex_fused_paged_prefill_request(
    kv_cache_dtype: str,
    max_query_len: int,
    total_query_len: int,
    alibi_slopes: Optional[torch.Tensor],
    sliding_window: Optional[int],
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    is_causal_decoder: bool,
) -> bool:
    return bool(
        _USE_COREX_FUSED_PAGED_PREFILL
        and _is_supported_corex_fused_paged_prefill_request(
            kv_cache_dtype,
            max_query_len,
            total_query_len,
            alibi_slopes,
            sliding_window,
            k_scale,
            v_scale,
            is_causal_decoder,
        )
    )


def _is_single_sequence_fused_prefill_metadata(
    batch_size: int,
    block_table_rows: int,
    query_start_count: int,
    query_start_first: int,
    query_start_last: int,
    seq_lens_count: int,
    seq_len: int,
    context_lens_count: int,
    context_len: int,
    total_query_len: int,
) -> bool:
    """Validate the exact single-sequence metadata used by qualification."""
    return bool(
        batch_size == 1
        and block_table_rows == 1
        and query_start_count == 2
        and query_start_first == 0
        and query_start_last == total_query_len
        and seq_lens_count == 1
        and context_lens_count == 1
        and context_len >= 0
        and seq_len == context_len + total_query_len
        and seq_len <= 262144
    )


def _is_supported_corex_fused_paged_prefill_segment(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    prefix_key: torch.Tensor,
    prefix_value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_index: int,
    block_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    gqa_ratio: int,
    block_size: int,
) -> bool:
    """Accept only the fixed M1-47 production shape."""
    query_len = query.shape[0]
    if (
        query_len <= 16
        or query_len > 8192
        or block_context_len < 0
        or block_context_len % 16 != 0
        or block_context_len + query_len > 262144
    ):
        return False
    if (num_q_heads, num_kv_heads, head_dim, gqa_ratio, block_size) != (
        4,
        1,
        256,
        4,
        16,
    ):
        return False
    if prefix_key.shape[0] != 0 or prefix_value.shape[0] != 0:
        return False
    if (
        tuple(query.shape) != (query_len, 4, 256)
        or tuple(key.shape) != (query_len, 1, 256)
        or tuple(value.shape) != (query_len, 1, 256)
    ):
        return False
    if (
        len(key_cache.shape) != 5
        or tuple(key_cache.shape[1:]) != (1, 32, 16, 8)
        or len(value_cache.shape) != 4
        or tuple(value_cache.shape[1:]) != (1, 256, 16)
        or key_cache.shape[0] != value_cache.shape[0]
    ):
        return False
    if (
        len(block_tables.shape) != 2
        or block_tables.shape[0] != 1
        or seq_index < 0
        or seq_index >= block_tables.shape[0]
        or block_tables.shape[1] < block_context_len // block_size
    ):
        return False

    half_tensors = (query, key, value, key_cache, value_cache)
    if any(tensor.dtype != torch.float16 for tensor in half_tensors):
        return False
    if block_tables.dtype != torch.int32:
        return False
    tensors = half_tensors + (block_tables,)
    if any(not tensor.is_cuda for tensor in tensors):
        return False
    if any(tensor.device != query.device for tensor in tensors):
        return False
    if any(not tensor.is_contiguous() for tensor in tensors):
        return False
    return True


def _can_use_corex_fused_paged_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    prefix_key: torch.Tensor,
    prefix_value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_index: int,
    block_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    gqa_ratio: int,
    block_size: int,
) -> bool:
    return bool(
        _USE_COREX_FUSED_PAGED_PREFILL
        and _is_supported_corex_fused_paged_prefill_segment(
            query,
            key,
            value,
            prefix_key,
            prefix_value,
            key_cache,
            value_cache,
            block_tables,
            seq_index,
            block_context_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            gqa_ratio,
            block_size,
        )
    )


def _prefix_context_tile_spans(
    block_context_len: int,
    prefix_query_len: int,
    tile_size: int,
) -> List[Tuple[int, int, int, int]]:
    """Map context tiles to block-cache and preceding-query token ranges.

    Each tuple is ``(block_start, block_end, prefix_start, prefix_end)``.
    Concatenating both ranges reconstructs one tile in the logical context.
    Keeping tiles aligned to absolute token positions makes cold segmented
    prefill use the same online-softmax partitions as a warm cached request.
    """
    if block_context_len < 0 or prefix_query_len < 0 or tile_size <= 0:
        raise ValueError("context lengths must be non-negative and tile_size > 0")
    spans = []
    total_context_len = block_context_len + prefix_query_len
    for tile_start in range(0, total_context_len, tile_size):
        tile_end = min(tile_start + tile_size, total_context_len)
        block_start = min(tile_start, block_context_len)
        block_end = min(tile_end, block_context_len)
        prefix_start = max(0, tile_start - block_context_len)
        prefix_end = max(0, tile_end - block_context_len)
        spans.append((block_start, block_end, prefix_start, prefix_end))
    return spans


@dataclass
class PagedAttentionMetadata:
    """Metadata for PagedAttention."""
    # (batch_size,). The length of sequences (entire tokens seen so far) per
    # sequence.
    seq_lens_tensor: Optional[torch.Tensor]
    # Maximum sequence length in the batch. 0 if it is prefill-only batch.
    max_decode_seq_len: int
    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    block_tables: Optional[torch.Tensor]


class PagedAttention:

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [32, 64, 80, 96, 112, 120, 128, 192, 256]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size * num_kv_heads * head_size)

    @staticmethod
    def split_kv_cache(
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = 16 // kv_cache.element_size()
        num_blocks = kv_cache.shape[1]

        key_cache = kv_cache[0]
        key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x,
                                   -1, x)
        value_cache = kv_cache[1]
        value_cache = value_cache.view(num_blocks, num_kv_heads, head_size, -1)
        return key_cache, value_cache

    @staticmethod
    def write_to_paged_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> None:
        global _CACHE_WRITE_LOGGED
        flat_slots = slot_mapping.flatten()
        if key.shape[0] != value.shape[0]:
            raise RuntimeError(
                f"key/value token counts differ: {key.shape[0]} != "
                f"{value.shape[0]}")
        if flat_slots.numel() != key.shape[0]:
            raise RuntimeError(
                f"slot_mapping has {flat_slots.numel()} entries for "
                f"{key.shape[0]} KV tokens")
        if key_cache.shape[0] != value_cache.shape[0]:
            raise RuntimeError(
                "key/value cache block counts differ before cache write: "
                f"{key_cache.shape[0]} != {value_cache.shape[0]}")

        if _PAGED_ATTN_DIAGNOSTICS and flat_slots.numel() > 0:
            min_slot = int(flat_slots.min().item())
            max_slot = int(flat_slots.max().item())
            max_valid_slot = key_cache.shape[0] * value_cache.shape[3] - 1
            if min_slot < -1 or max_slot > max_valid_slot:
                raise RuntimeError(
                    f"slot_mapping range [{min_slot}, {max_slot}] outside "
                    f"[-1, {max_valid_slot}]")

        if _PAGED_ATTN_DIAGNOSTICS and not _CACHE_WRITE_LOGGED:
            print(
                "[BI100 PAGED_ATTN] cache_write "
                f"pid={os.getpid()} rank={os.environ.get('RANK', '?')} "
                f"local_rank={os.environ.get('LOCAL_RANK', '?')} "
                f"key={tuple(key.shape)} value={tuple(value.shape)} "
                f"slots={tuple(flat_slots.shape)} "
                f"key_cache={tuple(key_cache.shape)} "
                f"value_cache={tuple(value_cache.shape)}",
                file=sys.stderr,
                flush=True,
            )
            _CACHE_WRITE_LOGGED = True

        ops.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            flat_slots,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
        if _PAGED_ATTN_DIAGNOSTICS:
            try:
                torch.cuda.synchronize()
            except Exception as exc:
                print(
                    "[BI100 PAGED_ATTN] cache_write_sync_failed "
                    f"pid={os.getpid()} error={type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                raise

    @staticmethod
    def _forward_decode_pytorch(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Pure-PyTorch decode attention for long contexts (no hardware kernel).

        paged_attention_v1 hangs on BI-V100 when max_seq_len > ~32K due to
        shared memory limits. For decode, q_len=1 per sequence so no Q-tiling
        is needed — the attention weight tensor is [H, 1, seq_len] which is
        trivially small (~5 MB at 50K).

        Shapes
        ------
        query       : [num_seqs, num_heads, head_dim]
        key_cache   : [num_blocks, num_kv_heads, head_dim//x, block_size, x]
        value_cache : [num_blocks, num_kv_heads, head_dim,    block_size]
        block_tables: [num_seqs, max_blocks_per_seq]
        seq_lens    : [num_seqs]
        """
        num_seqs, num_heads, head_dim = query.shape
        num_kv_heads = key_cache.shape[1]
        block_size = value_cache.shape[3]
        gqa_ratio = num_heads // num_kv_heads
        orig_dtype = query.dtype

        output = torch.empty_like(query)

        try:
            for i in range(num_seqs):
                seq_len = int(seq_lens[i].item())
                num_blocks = (seq_len + block_size - 1) // block_size
                blk_ids = block_tables[i, :num_blocks]

                use_corex_gather = (
                    _USE_COREX_PAGED_KV_GATHER
                    and query.dtype == torch.float16
                    and key_cache.dtype == torch.float16
                    and value_cache.dtype == torch.float16
                    and block_tables.dtype == torch.int32
                    and key_cache.is_contiguous()
                    and value_cache.is_contiguous()
                    and blk_ids.is_contiguous())
                if use_corex_gather:
                    k_t, v_t = _corex_paged_kv_gather.gather(
                        key_cache, value_cache, blk_ids, seq_len)
                else:
                    # Gather K: [kv_h, head_dim, seq_len] fp32 without GQA
                    # expansion. The CoreX path above fuses these layout copies
                    # and FP16-to-FP32 conversions into one kernel.
                    k_t = (key_cache[blk_ids]
                           .permute(0, 3, 1, 2, 4)
                           .contiguous()
                           .view(-1, num_kv_heads, head_dim))[:seq_len] \
                          .permute(1, 2, 0).contiguous().float()
                    v_t = (value_cache[blk_ids]
                           .permute(0, 3, 1, 2)
                           .contiguous()
                           .view(-1, num_kv_heads, head_dim))[:seq_len] \
                          .permute(1, 0, 2).contiguous().float()

                # Reshape Q for lazy GQA: [kv_h, gqa_ratio, 1, d]
                q_grouped = (query[i].float()
                             .view(num_kv_heads, gqa_ratio, head_dim)
                             .unsqueeze(2))

                # [kv_h, gqa_ratio, 1, seq_len]
                attn_w = torch.matmul(
                    q_grouped * scale,       # [kv_h, gqa, 1, d]
                    k_t.unsqueeze(1))        # [kv_h, 1, d, seq_len]
                attn_w = torch.softmax(attn_w, dim=-1)

                # [kv_h, gqa_ratio, 1, d] → [num_heads, head_dim]
                out_i = torch.matmul(attn_w, v_t.unsqueeze(1))
                output[i] = out_i.view(num_heads, head_dim).to(orig_dtype)

        except Exception as e:
            print(f"[decode_pytorch ERROR] {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            raise

        return output

    # paged_attention_v1 on BI-V100 fails for long contexts.
    # Route on actual sequence length (seq_lens.max()), not the max_seq_len
    # parameter which is inflated to max_model_len in CUDA graph mode.
    _PYTORCH_DECODE_THRESHOLD = _PYTORCH_DECODE_THRESHOLD
    _FORCE_PAGED_ATTN_V2 = _FORCE_PAGED_ATTN_V2

    @staticmethod
    def _should_use_paged_attention_v1(
        max_seq_len: int,
        max_num_partitions: int,
        num_seqs: int,
        num_heads: int,
    ) -> bool:
        if PagedAttention._FORCE_PAGED_ATTN_V2:
            return False
        # Keep the stable BI100 default: V1 is used unless long-context decode
        # has already routed to the PyTorch fallback above.
        return True

    @staticmethod
    def _validate_prefix_block_table(
        seq_index: int,
        num_ctx_blocks: int,
        block_table_width: int,
        ctx_len: int,
    ) -> int:
        if num_ctx_blocks <= block_table_width:
            return num_ctx_blocks
        msg = (
            f"seq {seq_index}: num_ctx_blocks={num_ctx_blocks} "
            f"> block_tables.shape[1]={block_table_width}, "
            f"ctx_len={ctx_len}. Block table is undersized; "
            "refusing to truncate context because attention would be incorrect.")
        if env_bool("BI100_ALLOW_PREFIX_GUARD_CAP", False):
            print(
                "[paged_attn RISK] BI100_ALLOW_PREFIX_GUARD_CAP=1; "
                f"{msg} Debug cap is enabled and may corrupt output.",
                file=sys.stderr,
                flush=True)
            return block_table_width
        raise RuntimeError(msg)

    @staticmethod
    def forward_decode(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        kv_cache_dtype: str,
        num_kv_heads: int,
        scale: float,
        alibi_slopes: Optional[torch.Tensor],
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        tp_rank: int = 0,
        blocksparse_local_blocks: int = 0,
        blocksparse_vert_stride: int = 0,
        blocksparse_block_size: int = 64,
        blocksparse_head_sliding_step: int = 0,
    ) -> torch.Tensor:
        # Build head_mapping from num_kv_heads for BI-V100 native kernels
        num_queries_per_kv = query.shape[1] // num_kv_heads
        head_mapping = torch.repeat_interleave(
            torch.arange(num_kv_heads, dtype=torch.int32, device=query.device),
            num_queries_per_kv)
        actual_max = int(seq_lens.max().item()) if seq_lens.numel() > 0 else max_seq_len
        # Guard against uninitialized seq_lens entries (0x7FFF7FFF pattern)
        # from chunked prefill + GDN capture boundary metadata race.
        if actual_max > max_seq_len:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "[BI100 PAGED_ATTN] seq_lens contains value %d > max_seq_len %d, "
                "clamping (likely uninitialized metadata from chunked prefill)",
                actual_max, max_seq_len)
            seq_lens = seq_lens.clamp(max=max_seq_len)
            actual_max = max_seq_len
        block_size = value_cache.shape[3]
        num_seqs, num_heads, head_size = query.shape
        if key_cache.shape[1] != value_cache.shape[1]:
            raise RuntimeError(
                "key/value cache KV-head counts differ: "
                f"{key_cache.shape[1]} != {value_cache.shape[1]}")
        required_blocks = _validate_decode_layout(
            num_seqs=num_seqs,
            seq_lens_count=seq_lens.numel(),
            block_table_rows=block_tables.shape[0],
            block_table_width=block_tables.shape[1],
            actual_max=actual_max,
            block_size=block_size,
            physical_key_blocks=key_cache.shape[0],
            physical_value_blocks=value_cache.shape[0],
            num_heads=num_heads,
            num_kv_heads=key_cache.shape[1],
        )
        if actual_max > max_seq_len:
            raise RuntimeError(
                f"actual decode length {actual_max} exceeds max_seq_len "
                f"{max_seq_len}")

        if actual_max > PagedAttention._PYTORCH_DECODE_THRESHOLD:
            path = ("pytorch_corex_gather" if _USE_COREX_PAGED_KV_GATHER
                    else "pytorch")
        else:
            path = "native_v1"
        log_key = (path, None)
        if (_DECODE_LOG_INTERVAL > 0 and
                actual_max % _DECODE_LOG_INTERVAL == 0):
            log_key = (path, actual_max)
        if log_key not in _DECODE_DISPATCH_LOGGED:
            print(
                "[BI100 PAGED_ATTN] decode_dispatch "
                f"pid={os.getpid()} rank={os.environ.get('RANK', '?')} "
                f"local_rank={os.environ.get('LOCAL_RANK', '?')} "
                f"path={path} actual_max={actual_max} "
                f"max_seq_len={max_seq_len} query={tuple(query.shape)} "
                f"key_cache={tuple(key_cache.shape)} "
                f"value_cache={tuple(value_cache.shape)} "
                f"block_tables={tuple(block_tables.shape)} "
                f"required_blocks={required_blocks} "
                f"threshold={PagedAttention._PYTORCH_DECODE_THRESHOLD}",
                file=sys.stderr,
                flush=True,
            )
            _DECODE_DISPATCH_LOGGED.add(log_key)

        if _PAGED_ATTN_DIAGNOSTICS:
            for seq_index in range(num_seqs):
                seq_len = int(seq_lens[seq_index].item())
                if seq_len <= 0:
                    raise RuntimeError(
                        f"seq {seq_index}: decode length must be > 0, got {seq_len}")
                seq_blocks = (seq_len + block_size - 1) // block_size
                block_ids = block_tables[seq_index, :seq_blocks]
                min_block = int(block_ids.min().item())
                max_block = int(block_ids.max().item())
                if min_block < 0 or max_block >= key_cache.shape[0]:
                    raise RuntimeError(
                        f"seq {seq_index}: physical block range "
                        f"[{min_block}, {max_block}] outside "
                        f"[0, {key_cache.shape[0] - 1}]")

        # BI-V100: paged_attention_v1 supports max_context_len<=32768.
        # For longer contexts, use v2 with layout conversion (5D→4D).
        # v1 key: [blocks, kv_h, head_dim//x, block_size, x]
        # v2 key: [blocks, kv_h, block_size, head_dim]
        if actual_max > 32768:
            num_kv_heads = key_cache.shape[1]
            key_cache_v2 = (key_cache
                            .permute(0, 1, 3, 2, 4)
                            .contiguous()
                            .view(key_cache.shape[0], num_kv_heads,
                                  block_size, head_size))
            value_cache_v2 = (value_cache
                              .permute(0, 1, 3, 2)
                              .contiguous())
            output = torch.empty_like(query)
            _partition = 512
            max_num_partitions = ((max_seq_len + _partition - 1) //
                                  _partition)
            tmp_output = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions, head_size),
                dtype=output.dtype, device=output.device)
            exp_sums = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions),
                dtype=torch.float32, device=output.device)
            max_logits = torch.empty_like(exp_sums)
            import ixformer.functions as _ixf_F
            _ixf_F.vllm_single_query_cached_kv_attention_v2(
                output, _partition, exp_sums, max_logits, tmp_output,
                query, key_cache_v2, value_cache_v2, head_mapping, scale,
                block_tables, seq_lens, block_size, max_seq_len)
            return output

        if blocksparse_vert_stride is not None and blocksparse_vert_stride > 1:
            # use blocksparse paged attention
            block_size = value_cache.size(-1)
            assert (blocksparse_block_size > 0 and
                    blocksparse_block_size % block_size == 0), \
                (f"{blocksparse_block_size=} needs to be a multiple of"
                 f"{block_size=} used in block_tables.")

        output = torch.empty_like(query)
        max_num_partitions = ((max_seq_len + _PARTITION_SIZE - 1) //
                              _PARTITION_SIZE)
        # NOTE(woosuk): We use a simple heuristic to decide whether to use
        # PagedAttention V1 or V2. If the number of partitions is 1, we use
        # V1 to avoid the overhead of reduction. Also, if the number of
        # sequences or heads is large, we use V1 since there is enough work
        # to parallelize.
        # TODO(woosuk): Tune this heuristic.
        # For context len > 8192, use V2 kernel to avoid shared memory shortage.
        use_v1 = PagedAttention._should_use_paged_attention_v1(
            max_seq_len, max_num_partitions, num_seqs, num_heads)
        if use_v1:
            # Run PagedAttention V1.
            ops.paged_attention_v1(
                output,
                query,
                key_cache,
                value_cache,
                head_mapping,
                scale,
                block_tables,
                seq_lens,
                block_size,
                max_seq_len,
                alibi_slopes,
            )
        else:
            # Run PagedAttention V2.
            assert _PARTITION_SIZE % block_size == 0
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
            ops.paged_attention_v2(
                output,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                value_cache,
                head_mapping,
                scale,
                block_tables,
                seq_lens,
                block_size,
                max_seq_len,
                alibi_slopes,
                kv_cache_dtype,
                k_scale,
                v_scale,
                tp_rank,
                blocksparse_local_blocks,
                blocksparse_vert_stride,
                blocksparse_block_size,
                blocksparse_head_sliding_step,
            )
        return output

    @staticmethod
    def forward_prefix(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache_dtype: str,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens_tensor: torch.Tensor,
        max_query_len: int,
        alibi_slopes: Optional[torch.Tensor],
        sliding_window: Optional[int],
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> torch.Tensor:
        # NOTE: The Triton context_attention_fwd kernel hangs on Iluvatar
        # BI-V100 hardware (same class of issue as cudnnFlashAttnForward).
        # Use a pure-PyTorch fallback that reads the paged KV cache directly.

        # Derive context_lens from seq_lens and query_start_loc.
        # context_len[i] = seq_len[i] - query_len[i]
        # query_len[i] = query_start_loc[i+1] - query_start_loc[i]
        batch_size = seq_lens_tensor.shape[0]
        query_lens = query_start_loc[1:batch_size + 1] - query_start_loc[:batch_size]
        context_lens = seq_lens_tensor - query_lens

        is_causal_decoder = True  # BI-V100 always decoder-only
        supported_request = bool(
            (_USE_COREX_FUSED_PAGED_PREFILL
             or _ACTIVATION_CAPTURE_ENABLED)
            and _is_supported_corex_fused_paged_prefill_request(
                kv_cache_dtype,
                max_query_len,
                query.shape[0],
                alibi_slopes,
                sliding_window,
                k_scale,
                v_scale,
                is_causal_decoder,
            ))
        fused_request_eligible = bool(
            _USE_COREX_FUSED_PAGED_PREFILL and supported_request)
        capture_request_eligible = bool(
            _ACTIVATION_CAPTURE_ENABLED and supported_request)
        _log_corex_fused_prefill_diagnostic(
            "request",
            eligible=fused_request_eligible,
            use_native=_USE_COREX_FUSED_PAGED_PREFILL,
            causal=is_causal_decoder,
            kv_cache_dtype=kv_cache_dtype,
            max_query_len=max_query_len,
            total_query_len=query.shape[0],
            tile_blocks=_PREFIX_BLOCKS_PER_TILE,
            alibi_none=alibi_slopes is None,
            sliding_window=sliding_window,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        return PagedAttention._forward_prefix_pytorch(
            query, key, value,
            key_cache, value_cache,
            block_tables, query_start_loc,
            seq_lens_tensor, context_lens,
            fused_request_eligible=fused_request_eligible,
            capture_request_eligible=capture_request_eligible,
        )

    @staticmethod
    def _forward_prefix_pytorch(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens_tensor: torch.Tensor,
        context_lens: torch.Tensor,
        fused_request_eligible: bool = False,
        capture_request_eligible: bool = False,
    ) -> torch.Tensor:
        """Pure-PyTorch prefix-attention with K-tiling (Flash-Attention online softmax).

        Memory complexity: O(q_len), independent of kv_len.
        Query segments end at the same strict block boundary used by prefix
        caching. This keeps online-softmax reduction partitions identical when
        an otherwise equivalent request reuses that prefix.

        Algorithm: Flash Attention online softmax.
        Q is reshaped once to [kv_h, gqa, q_len, d] (24 MB) and held for all
        K-tiles.  For each tile a running (m, l, o) accumulator is updated —
        the [q_len × kv_len] attention matrix is NEVER materialised in full.

        Tile budget (kv_h=1, gqa=6, q_len=4096, tile=256 tokens):
            q_seq   [1, 6, 4096, 256] fp32  24 MB  (held all tiles)
            o_acc   same shape               24 MB  (held all tiles)
            s       same shape               24 MB  (per tile, freed before exp_s)
            exp_s   same shape               24 MB  (per tile, brief overlap with s)
            Peak ≈ 96 MB  (s and exp_s briefly coexist during update).

        Shapes
        ------
        query          : [total_q_tokens, num_q_heads,  head_dim]
        key            : [total_q_tokens, num_kv_heads, head_dim]
        value          : [total_q_tokens, num_kv_heads, head_dim]
        key_cache      : [num_blocks, num_kv_heads, head_dim//x, block_size, x]
        value_cache    : [num_blocks, num_kv_heads, head_dim,    block_size]
        block_tables   : [batch_size, max_blocks_per_seq]
        query_start_loc: [batch_size + 1]
        seq_lens_tensor: [batch_size]  total length (context + query)
        context_lens   : [batch_size]  tokens already in KV cache
        """
        try:
            profile_name = "paged_attn.prefix_pytorch"
            # Paged-block tiles for context phase.
            # tile_sz = _BLOCKS_PER_TILE × block_size  (e.g. 16×16 = 256 tokens).
            # Score tensor [kv_h, gqa, q_len, tile_sz] fp32 = 24 MB per tile.
            # Same tile size reused for the current-chunk phase.
            _BLOCKS_PER_TILE = _PREFIX_BLOCKS_PER_TILE

            batch_size   = seq_lens_tensor.shape[0]
            num_q_heads  = query.shape[1]
            num_kv_heads = key_cache.shape[1]
            head_dim     = query.shape[2]
            gqa_ratio    = num_q_heads // num_kv_heads
            block_size   = value_cache.shape[3]
            tile_sz      = _BLOCKS_PER_TILE * block_size
            scale        = head_dim ** -0.5
            orig_dtype   = query.dtype
            output       = torch.empty_like(query)

            if fused_request_eligible or capture_request_eligible:
                query_start_count = query_start_loc.numel()
                seq_lens_count = seq_lens_tensor.numel()
                context_lens_count = context_lens.numel()
                query_start_first = (
                    int(query_start_loc[0].item())
                    if query_start_count == 2 else -1)
                query_start_last = (
                    int(query_start_loc[1].item())
                    if query_start_count == 2 else -1)
                seq_len = (
                    int(seq_lens_tensor[0].item())
                    if seq_lens_count == 1 else -1)
                context_len = (
                    int(context_lens[0].item())
                    if context_lens_count == 1 else -1)
                metadata_eligible = (
                    _is_single_sequence_fused_prefill_metadata(
                        batch_size=batch_size,
                        block_table_rows=block_tables.shape[0],
                        query_start_count=query_start_count,
                        query_start_first=query_start_first,
                        query_start_last=query_start_last,
                        seq_lens_count=seq_lens_count,
                        seq_len=seq_len,
                        context_lens_count=context_lens_count,
                        context_len=context_len,
                        total_query_len=query.shape[0],
                    ))
                _log_corex_fused_prefill_diagnostic(
                    "metadata",
                    eligible=metadata_eligible,
                    batch_size=batch_size,
                    block_table_rows=block_tables.shape[0],
                    query_start_count=query_start_count,
                    query_start_first=query_start_first,
                    query_start_last=query_start_last,
                    seq_lens_count=seq_lens_count,
                    seq_len=seq_len,
                    context_lens_count=context_lens_count,
                    context_len=context_len,
                    total_query_len=query.shape[0],
                )
                fused_request_eligible = bool(
                    fused_request_eligible and metadata_eligible)
                capture_request_eligible = bool(
                    capture_request_eligible and metadata_eligible)

            # Guard against uninitialized context_lens entries (0x7FFF7FFF
            # pattern) from chunked prefill + GDN capture boundary metadata
            # race.  Per-entry clamp inside the loop below, using block_tables
            # physical capacity as absolute ceiling.
            _max_physical_ctx = block_tables.shape[1] * block_size if block_tables.numel() > 0 else 0

            for i in range(batch_size):
                ctx_len = int(context_lens[i].item())
                # Per-entry guard: if ctx_len exceeds physical KV capacity,
                # it's an uninitialized 0x7FFF7FFF sentinel — force to 0.
                if ctx_len < 0 or ctx_len > _max_physical_ctx:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "[BI100 PAGED_ATTN] context_lens[%d]=%d exceeds "
                        "physical capacity %d, forcing to 0",
                        i, ctx_len, _max_physical_ctx)
                    ctx_len = 0
                q_start = int(query_start_loc[i].item())
                q_end   = int(query_start_loc[i + 1].item())
                q_len   = q_end - q_start

                for seg_start, seg_end, seg_ctx_len in (
                        _strict_prefix_query_segments(
                            ctx_len, q_len, block_size)):
                    absolute_start = q_start + seg_start
                    absolute_end = q_start + seg_end
                    bi100_profile_count(
                        "paged_attn.prefix_dispatch",
                        path="pytorch",
                        query_len=seg_end - seg_start,
                        request_query_len=q_len,
                        context_len=seg_ctx_len,
                        block_size=block_size,
                        query_heads=num_q_heads,
                        kv_heads=num_kv_heads,
                        head_dim=head_dim,
                    )
                    with bi100_timer(profile_name):
                        output[absolute_start:absolute_end] = (
                            PagedAttention._forward_prefix_segment_pytorch(
                                query[absolute_start:absolute_end],
                                key[absolute_start:absolute_end],
                                value[absolute_start:absolute_end],
                                key[q_start:absolute_start],
                                value[q_start:absolute_start],
                                key_cache,
                                value_cache,
                                block_tables,
                                i,
                                ctx_len,
                                num_q_heads,
                                num_kv_heads,
                                head_dim,
                                gqa_ratio,
                                block_size,
                                tile_sz,
                                scale,
                                orig_dtype,
                                fused_request_eligible=(
                                    fused_request_eligible),
                                capture_request_eligible=(
                                    capture_request_eligible),
                            ))

        except Exception as e:
            print(f"[paged_attn ERROR] {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            raise
        return output

    @staticmethod
    def _forward_prefix_segment_pytorch(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        prefix_key: torch.Tensor,
        prefix_value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_index: int,
        block_context_len: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        gqa_ratio: int,
        block_size: int,
        tile_sz: int,
        scale: float,
        orig_dtype,
        fused_request_eligible: bool = False,
        return_fp32: bool = False,
        capture_request_eligible: bool = False,
    ) -> torch.Tensor:
        """Run online-softmax attention for one strict-prefix query segment."""
        q_len = query.shape[0]
        supported_segment = (
            _is_supported_corex_fused_paged_prefill_segment(
                query,
                key,
                value,
                prefix_key,
                prefix_value,
                key_cache,
                value_cache,
                block_tables,
                seq_index,
                block_context_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                gqa_ratio,
                block_size,
            ))
        segment_eligible = bool(
            fused_request_eligible
            and _USE_COREX_FUSED_PAGED_PREFILL
            and supported_segment)
        capture_segment_eligible = bool(
            capture_request_eligible
            and _ACTIVATION_CAPTURE_ENABLED
            and supported_segment)
        _log_corex_fused_prefill_diagnostic(
            "segment",
            eligible=segment_eligible,
            request_eligible=fused_request_eligible,
            query_shape=tuple(query.shape),
            key_shape=tuple(key.shape),
            value_shape=tuple(value.shape),
            prefix_key_shape=tuple(prefix_key.shape),
            prefix_value_shape=tuple(prefix_value.shape),
            key_cache_shape=tuple(key_cache.shape),
            value_cache_shape=tuple(value_cache.shape),
            block_table_shape=tuple(block_tables.shape),
            context_len=block_context_len,
            seq_index=seq_index,
            q_dtype=query.dtype,
            block_table_dtype=block_tables.dtype,
            query_cuda=query.is_cuda,
            query_contiguous=query.is_contiguous(),
            key_contiguous=key.is_contiguous(),
            value_contiguous=value.is_contiguous(),
            block_table_contiguous=block_tables.is_contiguous(),
            heads=f"{num_q_heads}/{num_kv_heads}/{head_dim}",
            gqa_ratio=gqa_ratio,
            block_size=block_size,
        )
        if segment_eligible or capture_segment_eligible:
            required_blocks = block_context_len // block_size
            active_block_table = block_tables[
                seq_index, :required_blocks].contiguous()
        if capture_segment_eligible:
            reservation = _reserve_activation_capture(block_context_len)
            if reservation is not None:
                _capture_fused_prefill_activation(
                    reservation,
                    query,
                    key,
                    value,
                    key_cache,
                    value_cache,
                    active_block_table,
                    block_context_len,
                    scale,
                )
        if segment_eligible:
            shadow_index = _reserve_fused_prefill_shadow(
                query,
                block_context_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                block_size,
            )
            try:
                fused_result = _corex_fused_paged_prefill.forward(
                    query,
                    key,
                    value,
                    key_cache,
                    value_cache,
                    active_block_table,
                    block_context_len,
                    scale,
                )
            except Exception as exc:
                if shadow_index is not None:
                    _finish_fused_prefill_shadow(
                        shadow_index,
                        status="invalid",
                        error_stage="candidate-execution",
                        error_type=type(exc).__name__,
                    )
                raise
            if (
                not isinstance(fused_result, (list, tuple))
                or len(fused_result) != 2
            ):
                if shadow_index is not None:
                    _finish_fused_prefill_shadow(
                        shadow_index,
                        status="invalid",
                        error_stage="candidate-contract",
                        error_type="InvalidResult",
                    )
                raise RuntimeError(
                    "corex fused paged-prefill returned an invalid result")
            fused_output = fused_result[0]
            if (
                tuple(fused_output.shape) != tuple(query.shape)
                or fused_output.dtype != query.dtype
                or fused_output.device != query.device
            ):
                if shadow_index is not None:
                    _finish_fused_prefill_shadow(
                        shadow_index,
                        status="invalid",
                        error_stage="candidate-contract",
                        error_type="InvalidOutput",
                    )
                raise RuntimeError(
                    "corex fused paged-prefill returned an invalid output")
            if shadow_index is not None:
                try:
                    reference_result = (
                        PagedAttention._forward_prefix_segment_pytorch(
                            query,
                            key,
                            value,
                            prefix_key,
                            prefix_value,
                            key_cache,
                            value_cache,
                            block_tables,
                            seq_index,
                            block_context_len,
                            num_q_heads,
                            num_kv_heads,
                            head_dim,
                            gqa_ratio,
                            block_size,
                            tile_sz,
                            scale,
                            orig_dtype,
                            fused_request_eligible=False,
                            capture_request_eligible=False,
                            return_fp32=(
                                _FUSED_PREFILL_SHADOW_NUMERIC_MODE
                                == "calibrated"),
                        ))
                    reference_fp32 = (
                        reference_result
                        if _FUSED_PREFILL_SHADOW_NUMERIC_MODE
                        == "calibrated"
                        else None
                    )
                    reference_output = (
                        reference_result.to(orig_dtype)
                        if reference_fp32 is not None
                        else reference_result
                    )
                    shadow_metrics = _compare_fused_prefill_shadow_outputs(
                        fused_output,
                        reference_output,
                        reference_fp32,
                    )
                except Exception as exc:
                    _finish_fused_prefill_shadow(
                        shadow_index,
                        status="invalid",
                        error_stage="reference-execution",
                        error_type=type(exc).__name__,
                    )
                    raise
                _finish_fused_prefill_shadow(
                    shadow_index,
                    **shadow_metrics,
                )
                if (
                    shadow_metrics["status"] != "pass"
                    and (
                        _FUSED_PREFILL_SHADOW_FAILURE_ACTION == "raise"
                        or shadow_metrics["status"] == "invalid"
                        or shadow_metrics.get("candidate_finite") is not True
                        or shadow_metrics.get("reference_finite") is not True
                    )
                ):
                    raise RuntimeError(
                        "corex fused paged-prefill failed the real-activation "
                        "shadow-reference numerical gate")
            log_key = "corex_split4"
            if log_key not in _PREFIX_DISPATCH_LOGGED:
                print(
                    "[BI100 PAGED_ATTN] prefix_dispatch "
                    f"pid={os.getpid()} rank={os.environ.get('RANK', '?')} "
                    f"local_rank={os.environ.get('LOCAL_RANK', '?')} "
                    f"path={log_key} context_len={block_context_len} "
                    f"query_len={q_len} required_blocks={required_blocks}",
                    file=sys.stderr,
                    flush=True,
                )
                _PREFIX_DISPATCH_LOGGED.add(log_key)
            return fused_output

        dev = query.device
        q_seq = (query.permute(1, 0, 2)
                      .float()
                      .view(num_kv_heads, gqa_ratio, q_len, head_dim)
                      .mul(scale))
        m = torch.full((num_kv_heads, gqa_ratio, q_len),
                       float('-inf'), dtype=torch.float32, device=dev)
        l = torch.zeros_like(m)
        o = torch.zeros((num_kv_heads, gqa_ratio, q_len, head_dim),
                        dtype=torch.float32, device=dev)

        if block_context_len > 0:
            num_ctx_blocks = (block_context_len + block_size - 1) // block_size
            num_ctx_blocks = PagedAttention._validate_prefix_block_table(
                seq_index, num_ctx_blocks, block_tables.shape[1],
                block_context_len)

        for block_start, block_end, prefix_start, prefix_end in (
                _prefix_context_tile_spans(
                    block_context_len, prefix_key.shape[0], tile_sz)):
            k_parts = []
            v_parts = []
            if block_end > block_start:
                first_block = block_start // block_size
                last_block = (block_end + block_size - 1) // block_size
                blk_ids = block_tables[seq_index, first_block:last_block]
                k_blocks = (key_cache[blk_ids]
                            .permute(0, 3, 1, 2, 4)
                            .contiguous()
                            .view(-1, num_kv_heads, head_dim))
                v_blocks = (value_cache[blk_ids]
                            .permute(0, 3, 1, 2)
                            .contiguous()
                            .view(-1, num_kv_heads, head_dim))
                offset = block_start - first_block * block_size
                length = block_end - block_start
                k_parts.append(k_blocks[offset:offset + length])
                v_parts.append(v_blocks[offset:offset + length])
            if prefix_end > prefix_start:
                k_parts.append(prefix_key[prefix_start:prefix_end])
                v_parts.append(prefix_value[prefix_start:prefix_end])
            k_context = (k_parts[0] if len(k_parts) == 1
                         else torch.cat(k_parts, dim=0))
            v_context = (v_parts[0] if len(v_parts) == 1
                         else torch.cat(v_parts, dim=0))
            k_t = (k_context.permute(1, 0, 2)
                   .unsqueeze(1).transpose(-1, -2).float())
            v_t = v_context.permute(1, 0, 2).unsqueeze(1).float()
            PagedAttention._update_online_softmax(q_seq, k_t, v_t, m, l, o)

        for key_start in range(0, q_len, tile_sz):
            key_end = min(key_start + tile_sz, q_len)
            k_t = (key[key_start:key_end].permute(1, 0, 2)
                   .unsqueeze(1).transpose(-1, -2).float())
            v_t = (value[key_start:key_end].permute(1, 0, 2)
                   .unsqueeze(1).float())
            scores = torch.matmul(q_seq, k_t)
            del k_t
            key_positions = torch.arange(key_start, key_end, device=dev)
            query_positions = torch.arange(q_len, device=dev)
            mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
            scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            del mask, key_positions, query_positions
            PagedAttention._update_online_softmax_from_scores(
                scores, v_t, m, l, o)

        o.div_(l.unsqueeze(-1))
        output = (
            o.view(num_q_heads, q_len, head_dim)
            .permute(1, 0, 2)
        )
        return output if return_fp32 else output.to(orig_dtype)

    @staticmethod
    def _update_online_softmax(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        running_max: torch.Tensor,
        running_sum: torch.Tensor,
        running_output: torch.Tensor,
    ) -> None:
        scores = torch.matmul(query, key)
        PagedAttention._update_online_softmax_from_scores(
            scores, value, running_max, running_sum, running_output)

    @staticmethod
    def _update_online_softmax_from_scores(
        scores: torch.Tensor,
        value: torch.Tensor,
        running_max: torch.Tensor,
        running_sum: torch.Tensor,
        running_output: torch.Tensor,
    ) -> None:
        block_max = scores.amax(dim=-1)
        new_max = torch.maximum(running_max, block_max)
        exp_scores = scores - new_max.unsqueeze(-1)
        del scores
        exp_scores.exp_()
        correction = torch.exp(running_max - new_max)
        running_max.copy_(new_max)
        running_sum.mul_(correction).add_(exp_scores.sum(dim=-1))
        running_output.mul_(correction.unsqueeze(-1)).add_(
            torch.matmul(exp_scores, value))

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache = src_kv_cache[0]
        dst_key_cache = dst_kv_cache[0]
        ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)

        src_value_cache = src_kv_cache[1]
        dst_value_cache = dst_kv_cache[1]
        ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        for kv_cache in kv_caches:
            src = src_to_dists[:, 0].long()
            dst = src_to_dists[:, 1].long()
            kv_cache[0][dst] = kv_cache[0][src]
            kv_cache[1][dst] = kv_cache[1][src]