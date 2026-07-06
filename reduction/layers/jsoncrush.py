"""SmartCrusher-lite — statistical compression for large JSON.

Tool outputs are often huge JSON arrays of near-identical objects. Sending all
N rows is wasteful when the model only needs the shape plus a representative
sample. This compressor:

  * for a large uniform array, keeps a head+tail sample and replaces the middle
    with a count, preserving the schema;
  * renders uniform arrays as TOON (tabular) which is far denser than JSON;
  * leaves small or non-uniform JSON essentially untouched.

It returns ``(compressed_text, was_compressed)``. Pair it with the CCR store so
the dropped rows remain retrievable.

Inspired by Headroom's SmartCrusher (https://github.com/chopratejas/headroom).
"""

from __future__ import annotations

import json
from typing import Any

from reduction.layers import toon

# Arrays at/above this length are sampled rather than sent whole.
SAMPLE_THRESHOLD = 20
SAMPLE_HEAD = 5
SAMPLE_TAIL = 5


def _largest_array(data: Any) -> list[Any] | None:
    """Find the top-level (or one-level-nested) largest list to sample."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        best: list[Any] | None = None
        for value in data.values():
            if isinstance(value, list) and (best is None or len(value) > len(best)):
                best = value
        return best
    return None


def crush_json(text: str) -> tuple[str, bool]:
    """Compress a JSON string. Returns (compressed, was_lossy).

    Truncated JSON (a captured tool output cut off mid-stream) is recovered by
    parsing the longest complete prefix — the result is flagged lossy so CCR
    keeps the raw original retrievable.
    """
    data, repaired = _parse_lenient(text)
    if data is None:
        return text, False

    arr = _largest_array(data)
    if arr is not None and len(arr) >= SAMPLE_THRESHOLD:
        compressed = _crush_array(data, arr, truncated=repaired)
        return compressed, True

    compact = json.dumps(data, separators=(",", ":"))
    if repaired:
        # The tail was cut off mid-value; the complete prefix is lossy.
        return f"# truncated JSON; complete prefix follows\n{compact}", True

    # No big array — fall back to compact JSON (drops pretty-print whitespace).
    # Compaction is lossless, so it is never flagged as lossy (no CCR needed).
    return (compact, False) if len(compact) < len(text) else (text, False)


def _parse_lenient(text: str) -> tuple[Any, bool]:
    """Parse JSON, recovering the longest complete prefix of truncated input.

    Returns ``(data, was_repaired)``; ``(None, False)`` when unrecoverable.
    """
    try:
        return json.loads(text), False
    except ValueError:
        pass
    repaired = _repair_truncated(text)
    return repaired, repaired is not None


def _repair_truncated(text: str, max_attempts: int = 40) -> Any:
    """Recover truncated JSON by closing brackets at the last complete value.

    One pass tracks bracket/string state, remembering it at the trailing
    structural characters (``"`` close, ``}``, ``]``, ``,``); each candidate cut
    point is then tried newest-first with the matching closers appended.
    """
    from collections import deque

    s = text.strip()
    if not s or s[0] not in "{[":
        return None

    states: deque[tuple[int, tuple[str, ...]]] = deque(maxlen=max_attempts)
    stack: list[str] = []
    in_str = esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
                states.append((i, tuple(stack)))
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if not stack or stack[-1] != ch:
                return None  # mismatched brackets — not truncation, not JSON
            stack.pop()
            states.append((i, tuple(stack)))
        elif ch == ",":
            states.append((i, tuple(stack)))

    for pos, st in reversed(states):
        # Cutting at a trailing comma leaves the bracket state unchanged.
        prefix = s[: pos + 1].rstrip().rstrip(",")
        try:
            return json.loads(prefix + "".join(reversed(st)))
        except ValueError:
            continue
    return None


def _crush_array(root: Any, arr: list[Any], *, truncated: bool = False) -> str:
    head = arr[:SAMPLE_HEAD]
    tail = arr[-SAMPLE_TAIL:]
    omitted = len(arr) - len(head) - len(tail)
    sample = head + tail

    # Sibling keys of a dict root (counts, ids, status fields) must survive
    # sampling — only the big array itself is reduced.
    key = "items"
    rest: dict[str, Any] | None = None
    if isinstance(root, dict):
        key = next((k for k, v in root.items() if v is arr), "items")
        rest = {k: v for k, v in root.items() if v is not arr} or None

    # If the sample is a uniform array of flat dicts, TOON is densest.
    if toon.is_uniform_array(sample):
        body = toon.encode(sample)
        note = (
            f"# {len(arr)} items total; {omitted} sampled out "
            f"(head {SAMPLE_HEAD} + tail {SAMPLE_TAIL})"
        )
        if truncated:
            note += "; input was truncated JSON"
        lines = []
        if rest is not None:
            lines.append(json.dumps(rest, separators=(",", ":")))
        lines.append(note)
        if isinstance(root, dict):
            # Note which key held the big array.
            lines.append(f"{key}:")
        lines.append(body)
        return "\n".join(lines)

    # Otherwise emit compact JSON sample with an omission marker.
    payload: dict[str, Any] = {
        "_sample_head": head,
        "_omitted": omitted,
        "_sample_tail": tail,
    }
    if rest is not None:
        return json.dumps({**rest, key: payload}, separators=(",", ":"))
    return json.dumps(payload, separators=(",", ":"))
