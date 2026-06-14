"""Batch-API CCR processing.

The Batch API is asynchronous: you submit many requests and collect results
later, out of band. If a compressed tool result made the model call
``reduction_retrieve``, there is no live loop to satisfy it — the retrieval
must be resolved when results come back. This module does that:

  1. on submit, ``BatchContextStore`` remembers each request's messages/tools
     keyed by its ``custom_id``;
  2. on results, ``process_batch_results`` scans each result for
     ``reduction_retrieve`` tool calls, resolves them against the CCR store, and
     produces continuation messages (the original content as a tool result) so
     a follow-up batch can be submitted.

Works with Anthropic and OpenAI batch result shapes.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from reduction.ccr import RETRIEVE_TOOL_NAME, CompressionStore, get_default_store


@dataclass
class BatchRequestContext:
    custom_id: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BatchContextStore:
    _data: dict[str, BatchRequestContext] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def put(self, ctx: BatchRequestContext) -> None:
        with self._lock:
            self._data[ctx.custom_id] = ctx

    def get(self, custom_id: str) -> BatchRequestContext | None:
        with self._lock:
            return self._data.get(custom_id)


@dataclass
class ResolvedRetrieval:
    custom_id: str
    tool_use_id: str
    ref: str
    original: str
    continuation_message: dict[str, Any]


def _message_body(result: dict[str, Any]) -> dict[str, Any]:
    """Locate the completion body across the known batch-result nestings.

    Anthropic batch:  {"custom_id", "result": {"type": "succeeded",
                        "message": {"content": [...]}}}
    Anthropic inline: {"message": {"content": [...]}}
    OpenAI batch:     {"custom_id", "response": {"body": {"choices": [...]}}}
    OpenAI inline:    {"body"/"choices": ...}
    """
    # Anthropic nested under result.message
    res = result.get("result")
    if isinstance(res, dict) and isinstance(res.get("message"), dict):
        return res["message"]
    if isinstance(result.get("message"), dict):
        return result["message"]
    # OpenAI nested under response.body
    resp = result.get("response")
    if isinstance(resp, dict) and isinstance(resp.get("body"), dict):
        return resp["body"]
    if isinstance(result.get("body"), dict):
        return result["body"]
    if "choices" in result or "content" in result:
        return result
    return {}


def _iter_tool_calls(result: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """Yield (tool_use_id, name, arguments) across Anthropic + OpenAI shapes."""
    import json

    calls: list[tuple[str, str, dict[str, Any]]] = []
    message = _message_body(result)

    # Anthropic: content[] with type == "tool_use".
    for block in message.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            calls.append((block.get("id", ""), block.get("name", ""), block.get("input", {}) or {}))

    # OpenAI: choices[].message.tool_calls[].function.{name,arguments}.
    for choice in message.get("choices", []):
        for tc in (choice.get("message", {}) or {}).get("tool_calls", []) or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except ValueError:
                args = {}
            calls.append((tc.get("id", ""), fn.get("name", ""), args))
    return calls


def process_batch_results(
    results: list[dict[str, Any]],
    *,
    store: CompressionStore | None = None,
    provider: str = "anthropic",
) -> list[ResolvedRetrieval]:
    """Resolve reduction_retrieve tool calls found in batch results."""
    if store is None:
        store = get_default_store()
    resolved: list[ResolvedRetrieval] = []

    for result in results:
        custom_id = result.get("custom_id", "")
        for tool_use_id, name, args in _iter_tool_calls(result):
            if name != RETRIEVE_TOOL_NAME:
                continue
            ref = str(args.get("ref", ""))
            original = store.get(ref) or f"[reduction: ref {ref!r} not found]"
            resolved.append(
                ResolvedRetrieval(
                    custom_id=custom_id,
                    tool_use_id=tool_use_id,
                    ref=ref,
                    original=original,
                    continuation_message=_tool_result_message(tool_use_id, original, provider),
                )
            )
    return resolved


def _tool_result_message(tool_use_id: str, content: str, provider: str) -> dict[str, Any]:
    if provider == "openai":
        return {"role": "tool", "tool_call_id": tool_use_id, "content": content}
    # Anthropic
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": content}],
    }
