import contextlib
import fnmatch
import functools
import json
import os
import re
import threading
import time

from vllm.logger import init_logger

logger = init_logger(__name__)
_EVENT_SCHEMA = "bi100-profile-event-v1"
_EVENT_VERSION = 1
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_FILTER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.*?-]{0,63}$")


def _strict_bool(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default).strip()
    if value not in {"0", "1"}:
        raise RuntimeError(f"{name} must be exactly 0 or 1, got {value!r}")
    return value == "1"


_ENABLED = _strict_bool("BI100_PROFILE")
_INCLUDE_STARTUP = _strict_bool("BI100_PROFILE_INCLUDE_STARTUP")
_MODE = os.getenv("BI100_PROFILE_MODE", "sync").strip().lower()
_FILTERS = tuple(
    item.strip()
    for item in os.getenv("BI100_PROFILE_FILTER", "").split(",")
    if item.strip()
)
if _ENABLED and _MODE not in {"sync", "event"}:
    raise RuntimeError(f"unsupported BI100_PROFILE_MODE={_MODE!r}")
if _ENABLED and any(_FILTER_RE.fullmatch(pattern) is None
                    for pattern in _FILTERS):
    raise RuntimeError("BI100_PROFILE_FILTER contains an invalid pattern")

_EVENT_RECORDS = []
_COUNTERS = {}
_LOCK = threading.Lock()
_FORWARD_INDEX = 0
_LAST_FLUSH_NS = None
_ACTIVE_FORWARD_TOKEN = None
_NEXT_FORWARD_TOKEN = 0


def _enabled_for(name: str) -> bool:
    return (_ENABLED
            and (not _FILTERS
                 or any(fnmatch.fnmatchcase(name, pattern)
                        for pattern in _FILTERS)))


def _skip_startup() -> bool:
    return (not _INCLUDE_STARTUP
            and os.getenv("BI100_IN_STARTUP_PROFILE") == "1")


def bi100_profile_event_enabled() -> bool:
    return _ENABLED and _MODE == "event" and not _skip_startup()


def _begin_profile_forward():
    global _ACTIVE_FORWARD_TOKEN, _NEXT_FORWARD_TOKEN
    if not bi100_profile_event_enabled():
        return None
    with _LOCK:
        _EVENT_RECORDS.clear()
        _COUNTERS.clear()
        token = _NEXT_FORWARD_TOKEN
        _NEXT_FORWARD_TOKEN += 1
        _ACTIVE_FORWARD_TOKEN = token
    return token


def _abort_profile_forward(token) -> None:
    global _ACTIVE_FORWARD_TOKEN
    if token is None:
        return
    with _LOCK:
        if _ACTIVE_FORWARD_TOKEN != token:
            return
        _EVENT_RECORDS.clear()
        _COUNTERS.clear()
        _ACTIVE_FORWARD_TOKEN = None


def bi100_profile_transaction(function):
    """Keep one top-level model forward isolated from failed forwards."""
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        token = _begin_profile_forward()
        if token is None:
            return function(*args, **kwargs)
        try:
            result = function(*args, **kwargs)
        except BaseException:
            _abort_profile_forward(token)
            raise
        with _LOCK:
            was_flushed = _ACTIVE_FORWARD_TOKEN != token
        if not was_flushed:
            _abort_profile_forward(token)
            raise RuntimeError(
                "BI100 profile transaction completed without a flush")
        return result

    return wrapped


def _normalize_metadata(metadata):
    normalized = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or _NAME_RE.fullmatch(key) is None:
            raise TypeError("profile metadata keys must be bounded names")
        if isinstance(value, bool):
            normalized[key] = value
        elif isinstance(value, int) and not isinstance(value, bool):
            normalized[key] = value
        elif isinstance(value, str) and len(value) <= 64:
            normalized[key] = value
        else:
            raise TypeError(
                "profile metadata values must be bool, int, or short strings")
    return normalized


def bi100_profile_count(name: str, **metadata) -> None:
    """Record privacy-safe path metadata for the current model forward."""
    if not bi100_profile_event_enabled() or not _enabled_for(name):
        return
    if not isinstance(name, str) or _NAME_RE.fullmatch(name) is None:
        raise TypeError("profile counter name must be a bounded name")
    normalized = _normalize_metadata(metadata)
    encoded = json.dumps(
        {"name": name, **normalized}, sort_keys=True, separators=(",", ":"))
    with _LOCK:
        _COUNTERS[encoded] = _COUNTERS.get(encoded, 0) + 1


@contextlib.contextmanager
def bi100_timer(name: str):
    if not _enabled_for(name) or _skip_startup():
        yield
        return
    import torch

    if _MODE == "event":
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        host_started_ns = time.monotonic_ns()
        started.record()
        try:
            yield
        finally:
            finished.record()
            with _LOCK:
                _EVENT_RECORDS.append(
                    (name, started, finished, host_started_ns))
        return

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        logger.info("[BI100_PROFILE] %s %.3f ms", name,
                    (time.perf_counter() - t0) * 1000)


def bi100_profile_flush(*, tp_rank, **metadata):
    """Synchronize once and emit one aggregate event record per model forward."""
    global _ACTIVE_FORWARD_TOKEN, _FORWARD_INDEX, _LAST_FLUSH_NS
    if not bi100_profile_event_enabled():
        return None
    if (not isinstance(tp_rank, int) or isinstance(tp_rank, bool)
            or not 0 <= tp_rank < 256):
        raise TypeError("profile TP rank must be an integer in [0, 255]")
    normalized_metadata = _normalize_metadata(metadata)

    with _LOCK:
        records = list(_EVENT_RECORDS)
        counters = dict(_COUNTERS)
        _EVENT_RECORDS.clear()
        _COUNTERS.clear()
        _ACTIVE_FORWARD_TOKEN = None
    if not records:
        return None

    import torch

    torch.cuda.synchronize()
    flushed_ns = time.monotonic_ns()
    regions = {}
    model_started_ns = []
    for name, started, finished, host_started_ns in records:
        stats = regions.setdefault(name, {"count": 0, "total_ms": 0.0})
        stats["count"] += 1
        stats["total_ms"] += float(started.elapsed_time(finished))
        if name == "model.forward":
            model_started_ns.append(host_started_ns)

    counter_rows = []
    for encoded, count in sorted(counters.items()):
        row = json.loads(encoded)
        row["count"] = count
        counter_rows.append(row)

    first_model_started_ns = (
        min(model_started_ns) if model_started_ns else None)
    payload = {
        "schema": _EVENT_SCHEMA,
        "version": _EVENT_VERSION,
        "tp_rank": tp_rank,
        "forward_index": _FORWARD_INDEX,
        "metadata": normalized_metadata,
        "event_count": len(records),
        "model_forward_event_count": len(model_started_ns),
        "regions": regions,
        "counters": counter_rows,
        "host_model_start_to_flush_ms": (
            (flushed_ns - first_model_started_ns) / 1_000_000
            if first_model_started_ns is not None else None),
        "host_gap_since_previous_flush_ms": (
            (first_model_started_ns - _LAST_FLUSH_NS) / 1_000_000
            if first_model_started_ns is not None
            and _LAST_FLUSH_NS is not None
            else None),
    }
    _FORWARD_INDEX += 1
    _LAST_FLUSH_NS = flushed_ns
    logger.info("[BI100_PROFILE_EVENT] %s",
                json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return payload
