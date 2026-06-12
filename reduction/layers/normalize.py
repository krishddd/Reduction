"""Always-on cheap normalization — dependency-free, lossless-ish cleanup.

Runs on every text input before the heavier layers. Strips trailing
whitespace, collapses blank-line runs, and (optionally) deduplicates
consecutive identical lines with an ``(xN)`` marker — the same idea as a
shell log filter, but applied to any string the agent feeds the model.
"""

from __future__ import annotations

import re

_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)
_BLANK_RUNS = re.compile(r"\n{3,}")


def strip_whitespace(text: str) -> str:
    text = _TRAILING_WS.sub("", text)
    text = _BLANK_RUNS.sub("\n\n", text)
    return text.strip()


def dedupe_lines(text: str) -> str:
    """Collapse runs of identical consecutive non-blank lines into ``line  (xN)``.

    Blank lines are never annotated — they are left for ``strip_whitespace`` to
    collapse, so dedupe markers only ever appear on real content.
    """
    out: list[str] = []
    prev: str | None = None
    count = 0

    def flush() -> None:
        if prev is not None:
            out.append(prev if count == 1 else f"{prev}  (x{count})")

    for line in text.split("\n"):
        if not line.strip():
            flush()
            prev, count = None, 0
            out.append(line)
            continue
        if line == prev:
            count += 1
            continue
        flush()
        prev, count = line, 1
    flush()
    return "\n".join(out)


def normalize(text: str, *, strip: bool = True, dedupe: bool = True) -> str:
    if not text:
        return text
    if dedupe:
        text = dedupe_lines(text)
    if strip:
        text = strip_whitespace(text)
    return text
