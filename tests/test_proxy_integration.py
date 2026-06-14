"""End-to-end proxy tests: canned upstream via httpx.MockTransport through build_app.

Covers the streaming generators (and non-streaming handlers) that unit tests
can't reach, including transparent mid-stream CCR retrieval resolution.
"""

import json

import httpx
from fastapi.testclient import TestClient

from reduction.ccr import get_default_store, reset_default_store
from reduction.proxy import build_app


def _sse(*events: str) -> bytes:
    """Join SSE event blocks with blank-line separators."""
    return ("".join(e + "\n\n" for e in events)).encode("utf-8")


def _mock_app(responses):
    """build_app whose upstream returns each response in `responses` in order."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        idx = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        status, content, ctype = responses[idx]
        return httpx.Response(status, content=content, headers={"content-type": ctype})

    transport = httpx.MockTransport(handler)
    app = build_app(client_factory=lambda: httpx.AsyncClient(transport=transport))
    return TestClient(app), calls


# --- non-streaming --------------------------------------------------------


def test_nonstream_passthrough_no_retrieval():
    body = json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode()
    client, calls = _mock_app([(200, body, "application/json")])
    r = client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": []})
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "hi"
    assert calls["n"] == 1  # no retrieval loop


def test_nonstream_resolves_retrieval_then_continues():
    reset_default_store()
    ref = get_default_store().put("THE ORIGINAL PAYLOAD")
    first = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": "reduction_retrieve",
                                    "arguments": json.dumps({"ref": ref}),
                                },
                            }
                        ]
                    }
                }
            ]
        }
    ).encode()
    second = json.dumps({"choices": [{"message": {"content": "done"}}]}).encode()
    client, calls = _mock_app([(200, first, "application/json"), (200, second, "application/json")])
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "done"
    assert calls["n"] == 2  # retrieval triggered exactly one continuation


# --- streaming ------------------------------------------------------------


def test_stream_passthrough_content():
    stream = _sse(
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        'data: {"choices":[{"delta":{"content":" world"}}]}',
        "data: [DONE]",
    )
    client, _ = _mock_app([(200, stream, "text/event-stream")])
    r = client.post("/v1/chat/completions", json={"messages": [], "stream": True})
    assert r.status_code == 200
    text = r.text
    assert "Hello" in text and "world" in text
    # framing preserved: events separated by blank lines, terminated by [DONE]
    assert "\n\n" in text
    assert "data: [DONE]" in text


def test_stream_resolves_retrieval_midstream():
    reset_default_store()
    ref = get_default_store().put("FULL ORIGINAL ROWS")
    # First upstream stream: a reduction_retrieve tool call split across deltas.
    first = _sse(
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1",'
        '"function":{"name":"reduction_retrieve","arguments":""}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
        '"function":{"arguments":"{\\"ref\\": \\"REF\\"}"}}]}}]}'.replace("REF", ref),
        "data: [DONE]",
    )
    # Continuation stream: the final answer.
    second = _sse('data: {"choices":[{"delta":{"content":"answer"}}]}', "data: [DONE]")
    client, calls = _mock_app(
        [(200, first, "text/event-stream"), (200, second, "text/event-stream")]
    )
    r = client.post("/v1/chat/completions", json={"messages": [], "stream": True})
    assert r.status_code == 200
    # Client sees the final answer, never the suppressed retrieve tool call.
    assert "answer" in r.text
    assert "reduction_retrieve" not in r.text
    assert calls["n"] == 2


def test_stream_passes_through_other_tool_calls():
    # A non-retrieval tool call must reach the client (its harness handles it).
    stream = _sse(
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c9",'
        '"function":{"name":"get_weather","arguments":"{}"}}]}}]}',
        "data: [DONE]",
    )
    client, calls = _mock_app([(200, stream, "text/event-stream")])
    r = client.post("/v1/chat/completions", json={"messages": [], "stream": True})
    assert "get_weather" in r.text
    assert calls["n"] == 1  # no continuation for non-retrieval tools


def test_stream_anthropic_resolves_retrieval():
    reset_default_store()
    ref = get_default_store().put("ANTHROPIC ORIGINAL")
    first = _sse(
        "event: content_block_start\n"
        'data: {"type":"content_block_start","index":0,'
        '"content_block":{"type":"tool_use","id":"u1","name":"reduction_retrieve"}}',
        "event: content_block_delta\n"
        'data: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"input_json_delta","partial_json":"{\\"ref\\": \\"REF\\"}"}}'.replace(
            "REF", ref
        ),
        'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}',
    )
    second = _sse(
        "event: content_block_delta\n"
        'data: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"text_delta","text":"final"}}'
    )
    client, calls = _mock_app(
        [(200, first, "text/event-stream"), (200, second, "text/event-stream")]
    )
    r = client.post("/v1/messages", json={"messages": [], "stream": True})
    assert "final" in r.text
    assert "reduction_retrieve" not in r.text
    assert calls["n"] == 2


def test_stream_surfaces_upstream_error():
    # A non-200 upstream must be passed through, not fed to the SSE parser.
    err = json.dumps({"error": {"message": "rate limited"}}).encode()
    client, _ = _mock_app([(429, err, "application/json")])
    r = client.post("/v1/chat/completions", json={"messages": [], "stream": True})
    assert "rate limited" in r.text


def test_anthropic_stream_preserves_multiline_event_framing():
    # Anthropic events are multi-line (event: + data:); framing must survive.
    stream = _sse(
        "event: content_block_delta\n"
        'data: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"text_delta","text":"hi"}}'
    )
    client, _ = _mock_app([(200, stream, "text/event-stream")])
    r = client.post("/v1/messages", json={"messages": [], "stream": True})
    assert "event: content_block_delta" in r.text
    assert "hi" in r.text
