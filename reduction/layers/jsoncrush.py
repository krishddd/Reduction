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
    """Compress a JSON string. Returns (compressed, was_compressed)."""
    try:
        data = json.loads(text)
    except ValueError:
        return text, False

    arr = _largest_array(data)
    if arr is not None and len(arr) >= SAMPLE_THRESHOLD:
        compressed = _crush_array(data, arr)
        return compressed, True

    # No big array — fall back to compact JSON (drops pretty-print whitespace).
    compact = json.dumps(data, separators=(",", ":"))
    return (compact, True) if len(compact) < len(text) else (text, False)


def _crush_array(root: Any, arr: list[Any]) -> str:
    head = arr[:SAMPLE_HEAD]
    tail = arr[-SAMPLE_TAIL:]
    omitted = len(arr) - len(head) - len(tail)
    sample = head + tail

    # If the sample is a uniform array of flat dicts, TOON is densest.
    if toon.is_uniform_array(sample):
        body = toon.encode(sample)
        note = (
            f"# {len(arr)} items total; {omitted} sampled out "
            f"(head {SAMPLE_HEAD} + tail {SAMPLE_TAIL})"
        )
        if isinstance(root, dict):
            # Note which key held the big array.
            key = next((k for k, v in root.items() if v is arr), "items")
            return f"{note}\n{key}:\n{body}"
        return f"{note}\n{body}"

    # Otherwise emit compact JSON sample with an omission marker.
    payload = {
        "_sample_head": head,
        "_omitted": omitted,
        "_sample_tail": tail,
    }
    return json.dumps(payload, separators=(",", ":"))
