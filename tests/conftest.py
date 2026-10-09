"""Shared helpers for the cuda-aur packaging test suite.

These helpers deliberately avoid the network and any installed CUDA toolkit:
they parse the packaging metadata as data so the suite is fast, hermetic, and
safe to run in CI.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def read_pkgbuild(repo_root: Path | None = None) -> str:
    """Return the raw text of a repo's PKGBUILD."""
    root = repo_root or REPO_ROOT
    return (root / "PKGBUILD").read_text(encoding="utf-8")


def _strip_comment(line: str) -> str:
    """Drop a trailing ``#`` comment, ignoring ``#`` inside quotes."""
    out, quote = [], None
    for ch in line:
        if quote:
            if ch == quote:
                quote = None
            out.append(ch)
        elif ch in "\"'":
            quote = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out)


def pkgbuild_assignments(repo_root: Path | None = None) -> dict[str, str]:
    """Parse top-level ``key=value`` assignments out of a PKGBUILD.

    Only the last assignment for a key wins, mirroring bash semantics for the
    simple scalar variables these tests care about. Values keep their quoting so
    callers can decide how to interpret them.
    """
    text = read_pkgbuild(repo_root)
    result: dict[str, str] = {}
    for raw in text.splitlines():
        line = _strip_comment(raw).strip()
        if not line or line.startswith(("function", "}", "{")):
            continue
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        # Skip continuations such as ``makedepends+=(...)``.
        if raw.strip().startswith(f"{key}+="):
            continue
        result[key] = value
    return result


def pkgbuild_array(text: str, name: str) -> list[str]:
    """Return the words of a bash array assignment, ``name`` -> list.

    Handles both ``name=(a b)`` and ``name=('a' 'b')`` forms, including a
    multi-line array body. Returns an empty list when the array is absent.
    """
    pattern = re.compile(
        rf"^\s*{re.escape(name)}\s*=\s*\((.*?)\)\s*$",
        re.DOTALL | re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        return []
    body = _strip_comment(match.group(1))
    body = body.replace("\n", " ")
    try:
        return shlex.split(body)
    except ValueError:
        return body.split()


def pkgbuild_scalar(text: str, name: str) -> str | None:
    """Return the unquoted value of a scalar assignment, or ``None``."""
    pattern = re.compile(rf"^\s*{re.escape(name)}=(.+)$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None
    value = _strip_comment(match.group(1)).strip()
    try:
        parts = shlex.split(value)
    except ValueError:
        return value.strip("'\"")
    return parts[0] if parts else ""


def git_tracked_files(repo_root: Path) -> list[str]:
    """Return the repo-relative paths git currently tracks."""
    out = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line for line in out.splitlines() if line]
