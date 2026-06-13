"""Log compressor — dedupe repeats and focus on the signal lines.

Build/CI/server logs are dominated by repeated and informational lines. This:

  * collapses consecutive duplicates (``line  (xN)``);
  * when still long, keeps error/warning lines plus a head+tail of context,
    eliding the rest with a marker.

Pair with CCR so the elided lines stay retrievable.
"""

from __future__ import annotations

import re

from reduction.layers.normalize import normalize

_SIGNAL_RE = re.compile(
    r"\b(ERROR|WARN|WARNING|FATAL|FAIL|FAILED|Exception|Traceback)\b", re.IGNORECASE
)


def crush_log(text: str, *, max_lines: int = 120, context: int = 10) -> tuple[str, bool]:
    """Compress a log. Returns (compressed, was_compressed)."""
    deduped = normalize(text, strip=True, dedupe=True)
    lines = deduped.split("\n")
    if len(lines) <= max_lines:
        changed = deduped != text.strip()
        return deduped, changed

    keep: set[int] = set(range(context)) | set(range(len(lines) - context, len(lines)))
    for i, line in enumerate(lines):
        if _SIGNAL_RE.search(line):
            keep.add(i)

    out: list[str] = []
    prev_kept = -1
    for i in sorted(keep):
        if i < 0 or i >= len(lines):
            continue
        if prev_kept != -1 and i > prev_kept + 1:
            out.append(f"  ... ({i - prev_kept - 1} lines elided) ...")
        out.append(lines[i])
        prev_kept = i
    return "\n".join(out), True
