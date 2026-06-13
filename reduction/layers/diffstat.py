"""Diff compressor — collapse a unified diff to a per-file +/- stat summary.

A 5,000-line ``git diff`` is mostly noise to a model that just needs to know
*what changed where*. This renders the shortstat form:

    diff: 3 files, +142/-89
      src/main.py  +90/-12
      README.md    +40/-2
      tests/x.py   +12/-75

Pair with CCR so the full hunks stay retrievable.
"""

from __future__ import annotations

import re

_FILE_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)
_PLUSPLUS_RE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)


def crush_diff(text: str) -> tuple[str, bool]:
    """Summarize a unified diff. Returns (summary, was_compressed)."""
    lines = text.split("\n")
    files: list[tuple[str, int, int]] = []
    current: str | None = None
    adds = dels = 0

    def flush() -> None:
        nonlocal current, adds, dels
        if current is not None:
            files.append((current, adds, dels))
        current, adds, dels = None, 0, 0

    for line in lines:
        m = _FILE_RE.match(line)
        if not m:
            m2 = _PLUSPLUS_RE.match(line)
        else:
            m2 = None
        if m:
            flush()
            current = m.group(2)
        elif m2 and current is None:
            current = m2.group(1)
        elif line.startswith("+") and not line.startswith("+++"):
            adds += 1
        elif line.startswith("-") and not line.startswith("---"):
            dels += 1
    flush()

    if not files:
        return text, False

    total_add = sum(a for _, a, _ in files)
    total_del = sum(d for _, _, d in files)
    out = [f"diff: {len(files)} files, +{total_add}/-{total_del}"]
    width = max(len(f) for f, _, _ in files)
    for fname, a, d in files:
        out.append(f"  {fname:<{width}}  +{a}/-{d}")
    summary = "\n".join(out)
    return (summary, True) if len(summary) < len(text) else (text, False)
