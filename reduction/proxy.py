"""Drop-in compression proxy — OpenAI- and Anthropic-compatible.

Point any client at this proxy instead of the provider. It:

  1. compresses large message content (tool outputs, pasted blobs) with the
     content-aware + CCR pipeline;
  2. injects the ``reduction_retrieve`` tool so the model can ask for originals;
  3. forwards to the real upstream;
  4. transparently satisfies any ``reduction_retrieve`` tool call from the CCR
     store and continues the turn — so the client never even sees the retrieval
     round-trip.

The request/response transforms are pure functions (unit-tested without a
network); the FastAPI handlers wire them to an httpx upstream. Non-streaming.

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


def build_app():  # pragma: no cover - exercised via integration, not unit tests
    """Construct the FastAPI proxy app (requires the gateway extra)."""
    import httpx
    from fastapi import FastAPI, Request, Response

    app = FastAPI(title="Reduction Proxy")
    openai_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
    anthropic_base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    async def _forward(client, url, headers, body):
        r = await client.post(url, json=body, headers=headers, timeout=120.0)
        return r

    @app.post("/v1/chat/completions")
    async def openai_chat(request: Request) -> Response:
        body = compress_openai_request(await request.json())
        headers = _passthrough_headers(request)
        async with httpx.AsyncClient() as client:
            for _ in range(MAX_RETRIEVE_HOPS):
                r = await _forward(client, f"{openai_base}/v1/chat/completions", headers, body)
                data = r.json()
                retrievals = extract_openai_retrievals(data)
                if not retrievals:
                    return Response(
                        r.content, status_code=r.status_code, media_type="application/json"
                    )
                body = _continue_openai(body, data, retrievals)
        return Response(r.content, status_code=r.status_code, media_type="application/json")

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request) -> Response:
        body = compress_anthropic_request(await request.json())
        headers = _passthrough_headers(request)
        async with httpx.AsyncClient() as client:
            for _ in range(MAX_RETRIEVE_HOPS):
                r = await _forward(client, f"{anthropic_base}/v1/messages", headers, body)
                data = r.json()
                retrievals = extract_anthropic_retrievals(data)
                if not retrievals:
                    return Response(
                        r.content, status_code=r.status_code, media_type="application/json"
                    )
                body = _continue_anthropic(body, data, retrievals)
        return Response(r.content, status_code=r.status_code, media_type="application/json")

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
