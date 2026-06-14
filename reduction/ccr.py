"""CCR — Compress-Cache-Retrieve: reversible compression.

The problem with any lossy compression (sampling a huge JSON array, truncating
a log, summarizing a diff) is that the model occasionally needs the part you
dropped. CCR makes compression *safe*: every time content is compressed, the
original is stored under a short content hash and the compressed text carries a
marker — ``[reduction: 4,000 lines compressed, ref=ab12cd34]``. The agent can
call the ``reduction_retrieve`` tool with that ref to get the original back.

This module is provider-agnostic. It gives you:

  * ``CompressionStore`` — hash -> original, in-memory with optional JSON persist
  * ``retrieve_tool_definition`` — the tool schema (Anthropic / OpenAI shapes)
  * ``inject_retrieve_tool`` — add the tool to a request's ``tools`` array once
  * ``handle_retrieve_call`` — resolve a tool call back to the original text

Inspired by Headroom's CCR architecture (https://github.com/chopratejas/headroom).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RETRIEVE_TOOL_NAME = "reduction_retrieve"
REF_PREFIX = "ref="

# Bound the in-memory store so a long-running agent session can't grow it
# without limit. LRU by access; entries also expire after a TTL.
DEFAULT_MAX_ENTRIES = 2048
DEFAULT_TTL_SECONDS = 3600.0


def content_ref(text: str) -> str:
    """Stable 8-hex-char content hash used as a retrieval ref."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


@dataclass
class CompressionStore:
    """Stores originals so compressed content stays reversible.

    In-memory by default with LRU + TTL eviction so it can't leak. Pass ``path``
    to persist as JSON so refs survive across processes (e.g. a proxy writing,
    an MCP server reading). ``max_entries``/``ttl_seconds`` bound the cache;
    set ``ttl_seconds=0`` to disable expiry.
    """

    path: str | Path | None = None
    max_entries: int = DEFAULT_MAX_ENTRIES
    ttl_seconds: float = DEFAULT_TTL_SECONDS
    # ref -> (original, stored_at)
    _data: OrderedDict[str, tuple[str, float]] = field(default_factory=OrderedDict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.path and Path(self.path).exists():
            try:
                raw = json.loads(Path(self.path).read_text(encoding="utf-8"))
                now = time.time()
                # Accept both the current [text, ts] shape and a legacy text-only map.
                for ref, val in raw.items():
                    if isinstance(val, list) and len(val) == 2:
                        self._data[ref] = (val[0], float(val[1]))
                    else:
                        self._data[ref] = (val, now)
            except (OSError, ValueError):
                self._data = OrderedDict()

    def _expired(self, stored_at: float, now: float) -> bool:
        return self.ttl_seconds > 0 and (now - stored_at) > self.ttl_seconds

    def put(self, original: str) -> str:
        """Store ``original`` and return its ref (idempotent by content)."""
        ref = content_ref(original)
        now = time.time()
        with self._lock:
            self._data[ref] = (original, now)
            self._data.move_to_end(ref)
            # Evict expired, then oldest beyond the cap.
            for k in [k for k, (_, ts) in self._data.items() if self._expired(ts, now)]:
                del self._data[k]
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
            self._flush_locked()
        return ref

    def get(self, ref: str) -> str | None:
        ref = ref.strip()
        if ref.startswith(REF_PREFIX):
            ref = ref[len(REF_PREFIX) :]
        now = time.time()
        with self._lock:
            entry = self._data.get(ref)
            if entry is None:
                return None
            original, stored_at = entry
            if self._expired(stored_at, now):
                del self._data[ref]
                self._flush_locked()
                return None
            self._data.move_to_end(ref)  # LRU touch
            return original

    def __len__(self) -> int:
        return len(self._data)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = sum(len(v) for v, _ in self._data.values())
            return {"entries": len(self._data), "stored_chars": total}

    def _flush_locked(self) -> None:
        if self.path:
            serializable = {ref: [text, ts] for ref, (text, ts) in self._data.items()}
            Path(self.path).write_text(json.dumps(serializable), encoding="utf-8")


# --- module-level default store ---------------------------------------

_default_store: CompressionStore | None = None
_default_lock = threading.Lock()


def get_default_store(path: str | Path | None = None) -> CompressionStore:
    global _default_store
    with _default_lock:
        if _default_store is None:
            _default_store = CompressionStore(path=path)
    return _default_store


def reset_default_store() -> None:
    global _default_store
    with _default_lock:
        _default_store = None


# --- markers ----------------------------------------------------------


def make_marker(summary: str, ref: str) -> str:
    """Render the inline CCR marker appended to compressed content."""
    return f"[reduction: {summary}, {REF_PREFIX}{ref}]"


def store_and_mark(
    original: str, compressed: str, summary: str, store: CompressionStore | None = None
) -> str:
    """Store ``original`` and append a retrieval marker to ``compressed``."""
    if store is None:
        store = get_default_store()
    ref = store.put(original)
    return f"{compressed}\n{make_marker(summary, ref)}"


# --- retrieval tool ---------------------------------------------------


def retrieve_tool_definition(provider: str = "anthropic") -> dict[str, Any]:
    """Tool schema the agent calls to expand a ref. Anthropic or OpenAI shape."""
    description = (
        "Retrieve the original, uncompressed content that was compressed to save "
        "tokens. Use when you need more than the compressed view shows. The ref is "
        "in markers like [reduction: ... ref=ab12cd34]."
    )
    schema = {
        "type": "object",
        "properties": {
            "ref": {
                "type": "string",
                "description": "The 8-character ref from a compression marker.",
            }
        },
        "required": ["ref"],
    }
    if provider == "openai":
        return {
            "type": "function",
            "function": {
                "name": RETRIEVE_TOOL_NAME,
                "description": description,
                "parameters": schema,
            },
        }
    # Anthropic shape
    return {"name": RETRIEVE_TOOL_NAME, "description": description, "input_schema": schema}


def inject_retrieve_tool(
    tools: list[dict[str, Any]] | None, provider: str = "anthropic"
) -> list[dict[str, Any]]:
    """Return ``tools`` with the retrieve tool added once (no duplicates)."""
    tools = list(tools or [])
    if any(_tool_name(t) == RETRIEVE_TOOL_NAME for t in tools):
        return tools
    tools.append(retrieve_tool_definition(provider))
    return tools


def handle_retrieve_call(
    name: str, arguments: dict[str, Any], store: CompressionStore | None = None
) -> str | None:
    """If ``name`` is the retrieve tool, return the original text for the ref."""
    if name != RETRIEVE_TOOL_NAME:
        return None
    if store is None:
        store = get_default_store()
    ref = str(arguments.get("ref", ""))
    return store.get(ref) or f"[reduction: ref {ref!r} not found or expired]"


def _tool_name(tool: dict[str, Any]) -> str:
    if "function" in tool and isinstance(tool["function"], dict):
        return tool["function"].get("name", "")
    return tool.get("name", "")
