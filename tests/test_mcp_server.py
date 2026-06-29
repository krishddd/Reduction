"""Unit tests for the MCP tool implementations (mcp package not required)."""

import json

from reduction import mcp_server

BIG_JSON = json.dumps([{"id": i, "host": f"h-{i}", "open": True} for i in range(300)])


def test_impl_compress_and_retrieve_roundtrip():
    result = mcp_server._impl_compress(BIG_JSON)
    assert result["tokens_after"] < result["tokens_before"]
    assert result["ref"]
    assert mcp_server._impl_retrieve(result["ref"]) == BIG_JSON


def test_impl_retrieve_unknown_ref():
    assert "not found" in mcp_server._impl_retrieve("deadbeef")


def test_impl_compress_history():
    messages = [
        {"role": "system", "content": "be an agent"},
        {"role": "tool", "content": BIG_JSON},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "next"},
    ]
    out = mcp_server._impl_compress_history(messages, keep_last=2)
    assert out["messages_compressed"] == 1
    assert out["tokens_saved"] > 0
    assert out["refs"]
    # original retrievable through the same store
    assert mcp_server._impl_retrieve(out["refs"][0]) == BIG_JSON


def test_impl_fit_context():
    out = mcp_server._impl_fit_context(["small note", BIG_JSON], token_budget=120)
    assert out["included"] >= 1
    assert out["tokens_used"] <= 120


def test_impl_route_effort():
    out = mcp_server._impl_route_effort("read the file and show it")
    assert out["level"] == "minimal"
    assert out["thinking_budget"] == 0
    out = mcp_server._impl_route_effort("analyze why latency regressed")
    assert out["level"] == "high"


def test_impl_stats_shape():
    stats = mcp_server._impl_stats()
    assert "input_savings_pct" in stats
