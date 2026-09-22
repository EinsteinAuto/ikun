from __future__ import annotations

import importlib.util
import pathlib
import shlex
from typing import Iterable, Optional, Sequence, Tuple


def package_root(pkg: str) -> pathlib.Path:
    spec = importlib.util.find_spec(pkg)
    if spec is None:
        raise RuntimeError(f"package not found: {pkg}")
    if not spec.submodule_search_locations:
        raise RuntimeError(f"package has no package root: {pkg}")
    return pathlib.Path(next(iter(spec.submodule_search_locations))).resolve()


def ensure_file(path: pathlib.Path) -> pathlib.Path:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def ensure_dir(path: pathlib.Path) -> pathlib.Path:
    if not path.is_dir():
        raise FileNotFoundError(str(path))
    return path


def replace_once(path: pathlib.Path,
                 old: str,
                 new: str,
                 *,
                 required: bool = True,
                 already_contains: Optional[str] = None) -> bool:
    path = ensure_file(path)
    text = path.read_text()
    marker = already_contains if already_contains is not None else new
    if marker in text:
        print(f"[skip] already patched: {path}")
        return False
    if old not in text:
        msg = f"anchor not found in {path}: {old[:120]!r}"
        if required:
            raise RuntimeError(msg)
        print(f"[warn] {msg}")
        return False
    path.write_text(text.replace(old, new, 1))
    print(f"[ok] patched: {path}")
    return True


def replace_one_of(path: pathlib.Path,
                   replacements: Sequence[Tuple[str, str]],
                   *,
                   required: bool = True,
                   already_contains: Optional[str] = None) -> bool:
    path = ensure_file(path)
    text = path.read_text()
    if already_contains is not None and already_contains in text:
        print(f"[skip] already patched: {path}")
        return False
    for _, new in replacements:
        if new in text:
            print(f"[skip] already patched: {path}")
            return False
    for old, new in replacements:
        if old in text:
            path.write_text(text.replace(old, new, 1))
            print(f"[ok] patched: {path}")
            return True
    anchors = ", ".join(repr(old[:80]) for old, _ in replacements)
    msg = f"anchor not found in {path}; tried: {anchors}"
    if required:
        raise RuntimeError(msg)
    print(f"[warn] {msg}")
    return False


def shell_env_line(name: str, value: pathlib.Path) -> str:
    return f"{name}={shlex.quote(str(value))}"
