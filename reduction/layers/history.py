"""Conversation-history compression — shrink old turns in a multi-turn message list.

For long-horizon agents the largest token sink is rarely any single tool output;
it is the *accumulation* of past turns. Every tool result, file dump, and log
stays in context on every subsequent call, so an agent that runs for 40 steps
re-sends the first step's 9,000-token scan 39 more times. This layer compresses
the older messages while keeping the most recent turns verbatim — the model
keeps full-fidelity recent context and a compressed-but-retrievable history of
everything before it.

Old-message text is routed through the same content-aware compressors as
Layer 1 (JSON / diff / log / code) and stays CCR-reversible: the agent can call
``reduction_retrieve`` to expand any elided block. System messages (instructions)
are never touched, and the most recent ``keep_last`` messages pass through
untouched so nothing the model is actively reasoning over is degraded.

Both message shapes are handled:

  * OpenAI style — ``{"role": "user", "content": "..."}`` (string content).
  * Anthropic style — ``{"role": "user", "content": [{"type": "text", ...},
    {"type": "tool_result", "content": [...]}]}`` (list-of-blocks content).

The walk is recursive, so a ``tool_result`` whose ``content`` is itself a list
of blocks is compressed at the leaves.

Inspired by ACON (Optimizing Context Compression for Long-horizon LLM Agents,
arXiv:2510.00615) and Headroom's conversation-history routing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from reduction.content import MIN_CHARS_TO_COMPRESS, compress_content

# Compressible string fields inside a content block (Anthropic block shapes).
_TEXT_KEYS = ("text", "content")


@dataclass
class HistoryResult:
    messages: list[Any]
    messages_compressed: int
    tokens_before: int
    tokens_after: int
    refs: list[str] = field(default_factory=list)

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after

    @property
    def compression_ratio(self) -> float:
        return self.tokens_saved / self.tokens_before if self.tokens_before else 0.0


def _compress_value(value: Any, ccr: bool, store: Any, acc: dict) -> tuple[Any, bool]:
    """Recursively compress every large string leaf in a message-content value.

    Returns ``(new_value, changed)``. Token accounting is accumulated into
    ``acc`` only for fields actually run through the compressor, so the reported
    savings reflect what was touched (small fields are skipped and not counted).
    """
    if isinstance(value, str):
        if len(value) < MIN_CHARS_TO_COMPRESS:
            return value, False
        result = compress_content(value, ccr=ccr, store=store)
        acc["before"] += result.tokens_before
        acc["after"] += result.tokens_after
        if result.ref:
            acc["refs"].append(result.ref)
        return result.text, result.text != value

    if isinstance(value, list):
        changed = False
        out = []
        for item in value:
            new_item, item_changed = _compress_value(item, ccr, store, acc)
            out.append(new_item)
            changed = changed or item_changed
        return out, changed

    if isinstance(value, dict):
        changed = False
        new = dict(value)
        for key in _TEXT_KEYS:
            if key in new:
                new[key], key_changed = _compress_value(new[key], ccr, store, acc)
                changed = changed or key_changed
        return new, changed

    return value, False


def compress_history(
    messages: list[Any],
    *,
    keep_last: int = 4,
    ccr: bool = True,
    store: Any = None,
) -> HistoryResult:
    """Compress old turns in ``messages``, keeping the last ``keep_last`` verbatim.

    System messages and the ``keep_last`` most recent messages pass through
    unchanged. Everything older has its large content compressed (content-aware
    + CCR). The original list is never mutated; new message dicts are returned.
    """
    n = len(messages)
    cutoff = max(0, n - keep_last) if keep_last >= 0 else n
    acc: dict = {"before": 0, "after": 0, "refs": []}
    out: list[Any] = []
    compressed_count = 0

    for i, msg in enumerate(messages):
        keep = i >= cutoff
        if keep or not isinstance(msg, dict) or msg.get("role") == "system":
            out.append(msg)
            continue

        new_content, changed = _compress_value(msg.get("content"), ccr, store, acc)
        if changed:
            compressed_count += 1
            new_msg = dict(msg)
            new_msg["content"] = new_content
            out.append(new_msg)
        else:
            out.append(msg)

    return HistoryResult(
        messages=out,
        messages_compressed=compressed_count,
        tokens_before=acc["before"],
        tokens_after=acc["after"],
        refs=acc["refs"],
    )
