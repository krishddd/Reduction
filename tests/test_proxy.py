import json

from reduction.ccr import CompressionStore, get_default_store, reset_default_store
from reduction.proxy import (
    compress_anthropic_request,
    compress_openai_request,
    extract_anthropic_retrievals,
    extract_openai_retrievals,
)


def _big_json_blob() -> str:
    return json.dumps({"items": [{"id": i, "ok": True} for i in range(300)]})


def test_compress_openai_request_shrinks_and_injects_tool():
    reset_default_store()
    body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": _big_json_blob()},
        ],
    }
    out = compress_openai_request(body)
    user = out["messages"][1]["content"]
    assert len(user) < len(_big_json_blob())
    assert "ref=" in user  # CCR marker embedded
    names = [t["function"]["name"] for t in out["tools"]]
    assert "reduction_retrieve" in names


def test_compress_anthropic_request_blocks_and_tool_result():
    reset_default_store()
    body = {
        "model": "claude-sonnet-4-6",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "short"},
                    {"type": "tool_result", "tool_use_id": "t1", "content": _big_json_blob()},
                ],
            }
        ],
    }
    out = compress_anthropic_request(body)
    tr = out["messages"][0]["content"][1]
    assert "ref=" in tr["content"]
    assert any(t["name"] == "reduction_retrieve" for t in out["tools"])


def test_small_content_untouched():
    body = {"messages": [{"role": "user", "content": "hello"}]}
    out = compress_openai_request(body)
    assert out["messages"][0]["content"] == "hello"


def test_extract_openai_retrievals():
    resp = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {
                                "name": "reduction_retrieve",
                                "arguments": '{"ref": "abc12345"}',
                            },
                        },
                        {"id": "c2", "function": {"name": "other", "arguments": "{}"}},
                    ]
                }
            }
        ]
    }
    assert extract_openai_retrievals(resp) == [("c1", "abc12345")]


def test_extract_anthropic_retrievals():
    resp = {
        "content": [
            {"type": "text", "text": "hi"},
            {
                "type": "tool_use",
                "id": "u1",
                "name": "reduction_retrieve",
                "input": {"ref": "deadbeef"},
            },
        ]
    }
    assert extract_anthropic_retrievals(resp) == [("u1", "deadbeef")]


def test_roundtrip_compressed_ref_resolvable():
    reset_default_store()
    store: CompressionStore = get_default_store()
    body = {"messages": [{"role": "user", "content": _big_json_blob()}]}
    out = compress_openai_request(body)
    marker = out["messages"][0]["content"]
    ref = marker.split("ref=")[1].rstrip("]").strip()
    assert store.get(ref) == _big_json_blob()
