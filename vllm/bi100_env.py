import os


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    if raw in ("1", "true", "True", "yes", "YES", "on", "ON"):
        return True
    if raw in ("0", "false", "False", "no", "NO", "off", "OFF"):
        return False
    raise RuntimeError(f"{name} must be boolean, got {raw!r}")


def env_int(name: str, default: int, min_value: int, max_value: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be int, got {raw!r}") from exc
    if not (min_value <= value <= max_value):
        raise RuntimeError(
            f"{name}={value} outside [{min_value}, {max_value}]")
    return value
