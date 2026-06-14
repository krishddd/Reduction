import json

from reduction.ccr import CompressionStore
from reduction.ccr_batch import (
    BatchContextStore,
    BatchRequestContext,
    process_batch_results,
)


def test_batch_context_store():
    store = BatchContextStore()
    ctx = BatchRequestContext(custom_id="req-1", messages=[{"role": "user", "content": "hi"}])
    store.put(ctx)
    assert store.get("req-1").messages[0]["content"] == "hi"
    assert store.get("missing") is None


def test_process_anthropic_batch_retrievals():
    ccr_store = CompressionStore()
    ref = ccr_store.put("the original 5000-row payload")
    results = [
        {
            "custom_id": "req-1",
            "message": {
                "content": [
                    {"type": "text", "text": "let me look"},
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "reduction_retrieve",
                        "input": {"ref": ref},
                    },
                ]
            },
        }
    ]
    resolved = process_batch_results(results, store=ccr_store, provider="anthropic")
    assert len(resolved) == 1
    assert resolved[0].original == "the original 5000-row payload"
    assert resolved[0].continuation_message["role"] == "user"
    block = resolved[0].continuation_message["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "tu_1"


def test_process_openai_batch_retrievals():
    ccr_store = CompressionStore()
    ref = ccr_store.put("original openai payload")
    results = [
        {
            "custom_id": "req-2",
            "response": {
                "body": {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "call_1",
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
            },
        }
    ]
    resolved = process_batch_results(results, store=ccr_store, provider="openai")
    assert len(resolved) == 1
    assert resolved[0].original == "original openai payload"
    assert resolved[0].continuation_message["role"] == "tool"
    assert resolved[0].continuation_message["tool_call_id"] == "call_1"


def test_anthropic_nested_result_message_shape():
    # Real Anthropic batch: {"custom_id", "result": {"message": {"content": [...]}}}
    ccr_store = CompressionStore()
    ref = ccr_store.put("nested anthropic original")
    results = [
        {
            "custom_id": "rq",
            "result": {
                "type": "succeeded",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu",
                            "name": "reduction_retrieve",
                            "input": {"ref": ref},
                        }
                    ]
                },
            },
        }
    ]
    resolved = process_batch_results(results, store=ccr_store, provider="anthropic")
    assert len(resolved) == 1
    assert resolved[0].original == "nested anthropic original"


def test_unknown_ref_returns_marker():
    results = [
        {
            "custom_id": "r",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t",
                        "name": "reduction_retrieve",
                        "input": {"ref": "deadbeef"},
                    }
                ]
            },
        }
    ]
    resolved = process_batch_results(results, store=CompressionStore())
    assert "not found" in resolved[0].original
