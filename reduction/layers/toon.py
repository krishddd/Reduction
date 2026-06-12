"""Layer 5 — TOON (Token-Oriented Object Notation) encoder.

TOON combines YAML-style indentation with a CSV-like tabular layout for
uniform arrays of objects, cutting 30-60% of tokens vs JSON. Its sweet spot
is uniform arrays; for non-uniform or deeply nested data JSON wins, so
``encode`` falls back to compact JSON automatically.

Spec: https://github.com/toon-format/toon
"""

from __future__ import annotations

import json
from typing import Any

INDENT = "  "


def is_uniform_array(value: Any) -> bool:
    """True when value is a non-empty list of flat dicts sharing one key set."""
    if not isinstance(value, list) or not value:
        return False
    if not all(isinstance(item, dict) for item in value):
        return False
    keys = list(value[0].keys())
    if not keys:
        return False
    for item in value:
        if list(item.keys()) != keys:
            return False
        if any(isinstance(v, dict | list) for v in item.values()):
            return False
    return True


def is_scalar_array(value: Any) -> bool:
    """True when value is a non-empty list of plain scalars (no dict/list)."""
    if not isinstance(value, list) or not value:
        return False
    return all(not isinstance(v, dict | list) for v in value)


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value) if isinstance(value, float) else str(value)
    text = str(value)
    # Quote only when the value would be ambiguous in a comma-separated row.
    if any(c in text for c in ",\n:") or text != text.strip():
        return json.dumps(text)
    return text


def _encode_uniform_array(name: str, rows: list[dict[str, Any]], depth: int) -> list[str]:
    pad = INDENT * depth
    keys = list(rows[0].keys())
    lines = [f"{pad}{name}[{len(rows)}]{{{','.join(keys)}}}:"]
    for row in rows:
        lines.append(f"{pad}{INDENT}{','.join(_scalar(row[k]) for k in keys)}")
    return lines


def _encode_value(name: str, value: Any, depth: int) -> list[str]:
    pad = INDENT * depth
    if is_uniform_array(value):
        return _encode_uniform_array(name, value, depth)
    if isinstance(value, dict):
        lines = [f"{pad}{name}:"]
        for key, val in value.items():
            lines.extend(_encode_value(key, val, depth + 1))
        return lines
    if is_scalar_array(value):
        # Inline CSV row: ``tags[3]: a,b,c`` — no per-item key repetition.
        return [f"{pad}{name}[{len(value)}]: {','.join(_scalar(v) for v in value)}"]
    if isinstance(value, list):
        # Non-uniform / nested array — compact JSON is more token-efficient.
        return [f"{pad}{name}: {json.dumps(value, separators=(',', ':'))}"]
    return [f"{pad}{name}: {_scalar(value)}"]


def encode(data: Any) -> str:
    """Encode a JSON-compatible value as TOON text."""
    if is_uniform_array(data):
        return "\n".join(_encode_uniform_array("items", data, 0))
    if isinstance(data, dict):
        lines: list[str] = []
        for key, value in data.items():
            lines.extend(_encode_value(key, value, 0))
        return "\n".join(lines)
    # Scalars / non-uniform top-level lists: JSON fallback.
    return json.dumps(data, separators=(",", ":"))


# In-prompt example used to teach the model the format on first use.
TOON_INSTRUCTION = (
    "Return structured data as TOON (Token-Oriented Object Notation). "
    "Uniform arrays use a tabular header then comma rows. Example:\n"
    "users[2]{id,name,role}:\n  1,alice,admin\n  2,bob,user"
)
