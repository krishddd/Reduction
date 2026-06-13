import json

from reduction.proxy import (
    AnthropicToolUseCollector,
    OpenAIToolCallCollector,
    parse_sse_data,
)


def test_parse_sse_data():
    assert parse_sse_data('data: {"a": 1}') == {"a": 1}
    assert parse_sse_data("data: [DONE]") is None
    assert parse_sse_data("") is None
    assert parse_sse_data("event: ping") is None
    assert parse_sse_data("data: not json") is None


def test_openai_collector_reassembles_split_tool_call():
    # arguments arrive fragmented across chunks (as OpenAI streams them).
    chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "reduction_retrieve", "arguments": ""},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"ref": "ab1'}}]}}
            ]
        },
        {
            "choices": [
                {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '2cd34"}'}}]}}
            ]
        },
    ]
    c = OpenAIToolCallCollector()
    for ch in chunks:
        assert OpenAIToolCallCollector.chunk_has_tool_calls(ch)
        c.feed(ch)
    assert c.has_tool_calls()
    assert c.retrieve_refs() == [("c1", "ab12cd34")]


def test_openai_collector_ignores_content_chunks():
    c = OpenAIToolCallCollector()
    chunk = {"choices": [{"delta": {"content": "hello"}}]}
    assert not OpenAIToolCallCollector.chunk_has_tool_calls(chunk)
    c.feed(chunk)
    assert not c.has_tool_calls()


def test_openai_collector_non_retrieve_tool():
    c = OpenAIToolCallCollector()
    c.feed(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c9",
                                "function": {"name": "get_weather", "arguments": "{}"},
                            }
                        ]
                    }
                }
            ]
        }
    )
    assert c.has_tool_calls()
    assert c.retrieve_refs() == []  # not a retrieval call


def test_anthropic_collector_reassembles_tool_use():
    events = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "u1", "name": "reduction_retrieve"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"ref":'},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": ' "dead0001"}'},
        },
        {"type": "content_block_stop", "index": 0},
    ]
    c = AnthropicToolUseCollector()
    for e in events:
        c.feed(e)
    assert c.retrieve_refs() == [("u1", "dead0001")]


def test_anthropic_collector_text_block_not_tool():
    c = AnthropicToolUseCollector()
    start = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    assert not AnthropicToolUseCollector.event_is_tool_use_block(start, c.tool_use_indices())
    c.feed(
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}
    )
    assert not c.has_tool_calls()


def test_anthropic_event_is_tool_use_block_tracks_indices():
    c = AnthropicToolUseCollector()
    start = {
        "type": "content_block_start",
        "index": 2,
        "content_block": {"type": "tool_use", "id": "u", "name": "x"},
    }
    assert AnthropicToolUseCollector.event_is_tool_use_block(start, c.tool_use_indices())
    c.feed(start)
    delta = {
        "type": "content_block_delta",
        "index": 2,
        "delta": {"type": "input_json_delta", "partial_json": "{}"},
    }
    assert AnthropicToolUseCollector.event_is_tool_use_block(delta, c.tool_use_indices())


def test_collector_finalize_roundtrips_json():
    c = OpenAIToolCallCollector()
    c.feed(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "f", "arguments": json.dumps({"x": 1})},
                            }
                        ]
                    }
                }
            ]
        }
    )
    assert c.finalize()[0]["arguments"] == {"x": 1}
