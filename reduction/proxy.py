"""Drop-in compression proxy — OpenAI- and Anthropic-compatible.

Point any client at this proxy instead of the provider. It:

  1. compresses large message content (tool outputs, pasted blobs) with the
     content-aware + CCR pipeline;
  2. injects the ``reduction_retrieve`` tool so the model can ask for originals;
  3. forwards to the real upstream;
  4. transparently satisfies any ``reduction_retrieve`` tool call from the CCR
     store and continues the turn — so the client never even sees the retrieval
     round-trip.

The request/response transforms and SSE tool-call collectors are pure functions
(unit-tested without a network); the FastAPI handlers wire them to an httpx
upstream. Both buffered and streaming (SSE) responses are supported, with
mid-stream CCR retrieval resolved transparently.

Run:  ``reduction proxy --port 8788``   (set OPENAI_BASE_URL / ANTHROPIC_BASE_URL)
"""

from __future__ import annotations

import os
from typing import Any

from reduction import ccr
from reduction.content import compress_content

# Compress message strings at/above this length (smaller isn't worth a ref).
MIN_CONTENT_CHARS = 400
MAX_RETRIEVE_HOPS = 4


def _compress_text_block(text: str) -> str:
    if len(text) < MIN_CONTENT_CHARS:
        return text
    return compress_content(text, ccr=True).text


def _compress_content_field(content: Any) -> Any:
    """Compress a message ``content`` that may be a string or a block list."""
    if isinstance(content, str):
        return _compress_text_block(content)
    if isinstance(content, list):
        out = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                out.append({**block, "text": _compress_text_block(block["text"])})
            elif isinstance(block, dict) and isinstance(block.get("content"), str):
                # Anthropic tool_result content.
                out.append({**block, "content": _compress_text_block(block["content"])})
            else:
                out.append(block)
        return out
    return content


def compress_openai_request(body: dict[str, Any]) -> dict[str, Any]:
    """Compress messages + inject retrieve tool for an OpenAI chat request."""
    body = dict(body)
    messages = []
    for msg in body.get("messages", []):
        msg = dict(msg)
        if "content" in msg:
            msg["content"] = _compress_content_field(msg["content"])
        messages.append(msg)
    body["messages"] = messages
    body["tools"] = ccr.inject_retrieve_tool(body.get("tools"), provider="openai")
    return body


def compress_anthropic_request(body: dict[str, Any]) -> dict[str, Any]:
    """Compress messages + inject retrieve tool for an Anthropic messages request."""
    body = dict(body)
    messages = []
    for msg in body.get("messages", []):
        msg = dict(msg)
        if "content" in msg:
            msg["content"] = _compress_content_field(msg["content"])
        messages.append(msg)
    body["messages"] = messages
    body["tools"] = ccr.inject_retrieve_tool(body.get("tools"), provider="anthropic")
    return body


# --- CCR retrieve loop (provider-agnostic helpers) ---------------------


def extract_openai_retrievals(response: dict[str, Any]) -> list[tuple[str, str]]:
    """Return [(tool_call_id, ref)] for reduction_retrieve calls in an OAI response."""
    import json

    out: list[tuple[str, str]] = []
    for choice in response.get("choices", []):
        for tc in (choice.get("message", {}) or {}).get("tool_calls", []) or []:
            fn = tc.get("function", {})
            if fn.get("name") == ccr.RETRIEVE_TOOL_NAME:
                try:
                    ref = json.loads(fn.get("arguments", "{}")).get("ref", "")
                except ValueError:
                    ref = ""
                out.append((tc.get("id", ""), ref))
    return out


def extract_anthropic_retrievals(response: dict[str, Any]) -> list[tuple[str, str]]:
    """Return [(tool_use_id, ref)] for reduction_retrieve calls in an Anthropic response."""
    out: list[tuple[str, str]] = []
    for block in response.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            if block.get("name") == ccr.RETRIEVE_TOOL_NAME:
                out.append((block.get("id", ""), (block.get("input") or {}).get("ref", "")))
    return out


# --- streaming: pure SSE tool-call collectors (unit-tested) ------------


def parse_sse_data(line: str) -> dict[str, Any] | None:
    """Parse one ``data: {...}`` SSE line to a dict; None for [DONE]/blank/non-data."""
    import json

    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        return json.loads(payload)
    except ValueError:
        return None


class OpenAIToolCallCollector:
    """Reassembles OpenAI streaming ``delta.tool_calls`` fragments into calls."""

    def __init__(self) -> None:
        self._calls: dict[int, dict[str, str]] = {}

    def feed(self, chunk: dict[str, Any]) -> None:
        for choice in chunk.get("choices", []):
            for tc in (choice.get("delta", {}) or {}).get("tool_calls", []) or []:
                idx = tc.get("index", 0)
                slot = self._calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function", {}) or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]

    @staticmethod
    def chunk_has_tool_calls(chunk: dict[str, Any]) -> bool:
        return any(
            (choice.get("delta", {}) or {}).get("tool_calls") for choice in chunk.get("choices", [])
        )

    def has_tool_calls(self) -> bool:
        return bool(self._calls)

    def finalize(self) -> list[dict[str, Any]]:
        import json

        out = []
        for slot in self._calls.values():
            try:
                args = json.loads(slot["arguments"] or "{}")
            except ValueError:
                args = {}
            out.append({"id": slot["id"], "name": slot["name"], "arguments": args})
        return out

    def retrieve_refs(self) -> list[tuple[str, str]]:
        return [
            (c["id"], c["arguments"].get("ref", ""))
            for c in self.finalize()
            if c["name"] == ccr.RETRIEVE_TOOL_NAME
        ]


class AnthropicToolUseCollector:
    """Reassembles Anthropic streaming ``tool_use`` blocks from SSE events."""

    def __init__(self) -> None:
        self._blocks: dict[int, dict[str, str]] = {}

    def feed(self, event: dict[str, Any]) -> None:
        etype = event.get("type")
        if etype == "content_block_start":
            cb = event.get("content_block", {}) or {}
            if cb.get("type") == "tool_use":
                self._blocks[event.get("index", 0)] = {
                    "id": cb.get("id", ""),
                    "name": cb.get("name", ""),
                    "partial": "",
                }
        elif etype == "content_block_delta":
            idx = event.get("index", 0)
            if idx in self._blocks:
                delta = event.get("delta", {}) or {}
                if delta.get("type") == "input_json_delta":
                    self._blocks[idx]["partial"] += delta.get("partial_json", "")

    @staticmethod
    def event_is_tool_use_block(event: dict[str, Any], known_indices: set[int]) -> bool:
        etype = event.get("type")
        if etype == "content_block_start":
            return (event.get("content_block", {}) or {}).get("type") == "tool_use"
        if etype in ("content_block_delta", "content_block_stop"):
            return event.get("index", 0) in known_indices
        return False

    def tool_use_indices(self) -> set[int]:
        return set(self._blocks)

    def has_tool_calls(self) -> bool:
        return bool(self._blocks)

    def finalize(self) -> list[dict[str, Any]]:
        import json

        out = []
        for block in self._blocks.values():
            try:
                inp = json.loads(block["partial"] or "{}")
            except ValueError:
                inp = {}
            out.append({"id": block["id"], "name": block["name"], "input": inp})
        return out

    def retrieve_refs(self) -> list[tuple[str, str]]:
        return [
            (b["id"], b["input"].get("ref", ""))
            for b in self.finalize()
            if b["name"] == ccr.RETRIEVE_TOOL_NAME
        ]


def build_app():  # pragma: no cover - exercised via integration, not unit tests
    """Construct the FastAPI proxy app (requires the gateway extra)."""
    import httpx
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import StreamingResponse

    app = FastAPI(title="Reduction Proxy")
    openai_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
    anthropic_base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    async def _forward(client, url, headers, body):
        return await client.post(url, json=body, headers=headers, timeout=120.0)

    @app.post("/v1/chat/completions")
    async def openai_chat(request: Request) -> Response:
        body = compress_openai_request(await request.json())
        headers = _passthrough_headers(request)
        url = f"{openai_base}/v1/chat/completions"
        if body.get("stream"):
            return StreamingResponse(
                _stream_openai(url, headers, body), media_type="text/event-stream"
            )
        async with httpx.AsyncClient() as client:
            for _ in range(MAX_RETRIEVE_HOPS):
                r = await _forward(client, url, headers, body)
                retrievals = extract_openai_retrievals(r.json())
                if not retrievals:
                    return Response(r.content, r.status_code, media_type="application/json")
                body = _continue_openai(body, r.json(), retrievals)
        return Response(r.content, r.status_code, media_type="application/json")

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request) -> Response:
        body = compress_anthropic_request(await request.json())
        headers = _passthrough_headers(request)
        url = f"{anthropic_base}/v1/messages"
        if body.get("stream"):
            return StreamingResponse(
                _stream_anthropic(url, headers, body), media_type="text/event-stream"
            )
        async with httpx.AsyncClient() as client:
            for _ in range(MAX_RETRIEVE_HOPS):
                r = await _forward(client, url, headers, body)
                retrievals = extract_anthropic_retrievals(r.json())
                if not retrievals:
                    return Response(r.content, r.status_code, media_type="application/json")
                body = _continue_anthropic(body, r.json(), retrievals)
        return Response(r.content, r.status_code, media_type="application/json")

    return app


def _passthrough_headers(request) -> dict[str, str]:
    skip = {"host", "content-length", "accept-encoding"}
    return {k: v for k, v in request.headers.items() if k.lower() not in skip}


def _continue_openai(body, response, retrievals):  # pragma: no cover
    body = dict(body)
    assistant = response["choices"][0]["message"]
    messages = list(body["messages"]) + [assistant]
    for tool_call_id, ref in retrievals:
        original = ccr.handle_retrieve_call(ccr.RETRIEVE_TOOL_NAME, {"ref": ref}) or ""
        messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": original})
    body["messages"] = messages
    return body


def _continue_anthropic(body, response, retrievals):  # pragma: no cover
    body = dict(body)
    messages = list(body["messages"]) + [{"role": "assistant", "content": response["content"]}]
    tool_results = []
    for tool_use_id, ref in retrievals:
        original = ccr.handle_retrieve_call(ccr.RETRIEVE_TOOL_NAME, {"ref": ref}) or ""
        tool_results.append(
            {"type": "tool_result", "tool_use_id": tool_use_id, "content": original}
        )
    messages.append({"role": "user", "content": tool_results})
    body["messages"] = messages
    return body


# --- streaming generators (integration; pure collectors above are tested) ---


def _reconstruct_openai_assistant(collector) -> dict[str, Any]:  # pragma: no cover
    import json

    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": c["id"],
                "type": "function",
                "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])},
            }
            for c in collector.finalize()
        ],
    }


async def _stream_openai(url, headers, body, hops=MAX_RETRIEVE_HOPS):  # pragma: no cover
    """Stream OpenAI SSE, satisfying reduction_retrieve calls transparently.

    Content chunks forward immediately (low latency). Tool-call chunks are
    buffered; at stream end, if every buffered call is reduction_retrieve we
    resolve them and continue with a fresh upstream stream; otherwise we replay
    the buffered chunks so the client's own tools still work.
    """
    import httpx

    async with httpx.AsyncClient() as client:
        for _ in range(hops):
            collector = OpenAIToolCallCollector()
            buffered: list[str] = []
            saw_done = False
            async with client.stream("POST", url, json=body, headers=headers, timeout=120.0) as r:
                async for raw in r.aiter_lines():
                    if not raw:
                        continue
                    chunk = parse_sse_data(raw)
                    if chunk is None:
                        if raw.strip() == "data: [DONE]":
                            saw_done = True
                            continue
                        yield raw + "\n"
                        continue
                    if OpenAIToolCallCollector.chunk_has_tool_calls(chunk):
                        collector.feed(chunk)
                        buffered.append(raw)
                    else:
                        yield raw + "\n"
            refs = collector.retrieve_refs()
            only_retrieve = collector.has_tool_calls() and len(refs) == len(collector.finalize())
            if refs and only_retrieve:
                assistant = _reconstruct_openai_assistant(collector)
                body = _continue_openai(body, {"choices": [{"message": assistant}]}, refs)
                continue
            # Not a pure-retrieval turn: replay any buffered tool-call chunks.
            for raw in buffered:
                yield raw + "\n"
            if saw_done:
                yield "data: [DONE]\n\n"
            return
        yield "data: [DONE]\n\n"


async def _stream_anthropic(url, headers, body, hops=MAX_RETRIEVE_HOPS):  # pragma: no cover
    """Stream Anthropic SSE, satisfying reduction_retrieve calls transparently."""
    import httpx

    async with httpx.AsyncClient() as client:
        for _ in range(hops):
            collector = AnthropicToolUseCollector()
            buffered: list[str] = []
            async with client.stream("POST", url, json=body, headers=headers, timeout=120.0) as r:
                async for raw in r.aiter_lines():
                    if not raw:
                        continue
                    event = parse_sse_data(raw)
                    if event is None:
                        yield raw + "\n"
                        continue
                    if AnthropicToolUseCollector.event_is_tool_use_block(
                        event, collector.tool_use_indices()
                    ):
                        collector.feed(event)
                        buffered.append(raw)
                    elif (
                        event.get("type") == "content_block_start"
                        and (event.get("content_block", {}) or {}).get("type") == "tool_use"
                    ):
                        collector.feed(event)
                        buffered.append(raw)
                    else:
                        yield raw + "\n"
            refs = collector.retrieve_refs()
            only_retrieve = collector.has_tool_calls() and len(refs) == len(collector.finalize())
            if refs and only_retrieve:
                content = [
                    {"type": "tool_use", "id": b["id"], "name": b["name"], "input": b["input"]}
                    for b in collector.finalize()
                ]
                body = _continue_anthropic(body, {"content": content}, refs)
                continue
            for raw in buffered:
                yield raw + "\n"
            return
