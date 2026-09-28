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
    digits = raw.strip()
    digits = digits[1:] if digits[:1] in ("+", "-") else digits
    if not digits.isdigit():
        raise RuntimeError(f"{name} must be int, got {raw!r}")
    value = int(raw.strip())
    if not (min_value <= value <= max_value):
        raise RuntimeError(
            f"{name}={value} outside [{min_value}, {max_value}]")
    return value


_SWITCH_STATE = {}


def _rank_tag():
    for key in ("LOCAL_RANK", "RANK"):
        raw = os.getenv(key)
        if raw is not None:
            return "r%s " % raw
    return ""


def switch_wants(label, env, default):
    want = env_bool(env, default)
    tag = _rank_tag()
    if want:
        print("========{}switch {} REQUESTED {}=1, importing...============".format(
            tag, label, env), flush=True)
    else:
        _SWITCH_STATE[label] = ("OFF", "%s=0" % env)
        print("========{}switch {} OFF {}=0============".format(tag, label, env),
              flush=True)
    return want


def switch_live(label, module, source=None):
    exports = [n for n in dir(module) if not n.startswith("_")]
    _SWITCH_STATE[label] = ("LIVE", source or getattr(module, "__file__", "?"))
    print("========{}switch {} LIVE from={} exports={}============".format(
        _rank_tag(), label, source or getattr(module, "__file__", "?"),
        exports), flush=True)
    return module


def switch_probe(label, path, hit):
    print("========{}switch {} probe {} {}============".format(
        _rank_tag(), label, "HIT " if hit else "MISS", path), flush=True)


def switch_missing(label, env, searched):
    tag = _rank_tag()
    _SWITCH_STATE[label] = ("DEAD", "not found in %d paths" % len(searched))
    print("@@@@@@@@{}switch {} DEAD {}=1 but no .so found@@@@@@@@".format(
        tag, label, env), flush=True)
    for p in searched:
        print("@@@@@@@@{}switch {} searched {}@@@@@@@@".format(tag, label, p),
              flush=True)
    raise RuntimeError(
        "%s=1 but %s.so not found; searched: %s" % (env, label, searched))


def switch_report():
    tag = _rank_tag()
    buckets = {"LIVE": [], "OFF": [], "DEAD": []}
    for k, (state, why) in _SWITCH_STATE.items():
        buckets[state].append((k, why))
    print("========{}switch_report live={} off={} dead={}============".format(
        tag, len(buckets["LIVE"]), len(buckets["OFF"]),
        len(buckets["DEAD"])), flush=True)
    for k, why in buckets["LIVE"]:
        print("========{}switch_report LIVE {} {}============".format(tag, k, why),
              flush=True)
    for k, why in buckets["OFF"]:
        print("========{}switch_report OFF {} {}============".format(tag, k, why),
              flush=True)
    for k, why in buckets["DEAD"]:
        print("@@@@@@@@{}switch_report DEAD {} {}@@@@@@@@".format(tag, k, why),
              flush=True)
    return dict(_SWITCH_STATE)

def switch_state(label):
    """LIVE / OFF / DEAD for a label that went through switch_wants.

    UNKNOWN for anything else. Modules loaded by their own path never reach
    switch_wants, and calling those DEAD would be a lie -- the caller should
    fall back to its own boolean for those.
    """
    return _SWITCH_STATE.get(label, ("UNKNOWN", "not routed through switch_wants"))[0]
