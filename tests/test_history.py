import json

from reduction import TokenOptimizer
from reduction.ccr import CompressionStore
from reduction.layers import history

# A large, uniform JSON array — compresses ~97% via content routing.
BIG_JSON = json.dumps([{"id": i, "name": f"host-{i}", "open": True} for i in range(300)])


def _convo():
    return [
        {"role": "system", "content": "You are an agent. " * 30},
        {"role": "user", "content": "scan the network"},
        {"role": "assistant", "content": "running scan"},
        {"role": "tool", "content": BIG_JSON},  # old, large -> should compress
        {"role": "assistant", "content": "found 300 hosts"},
        {"role": "user", "content": "summarize the open ports"},
    ]


def test_compresses_old_large_message():
    store = CompressionStore()
    msgs = _convo()
    result = history.compress_history(msgs, keep_last=2, store=store)
    assert result.messages_compressed == 1
    assert result.tokens_after < result.tokens_before
    # the tool message (index 3) was compressed and carries a CCR ref marker
    tool_msg = result.messages[3]["content"]
    assert "reduction:" in tool_msg
    assert result.refs
    # original recoverable
    assert store.get(result.refs[0]) == BIG_JSON


def test_keeps_recent_and_system_verbatim():
    store = CompressionStore()
    msgs = _convo()
    result = history.compress_history(msgs, keep_last=2, store=store)
    # system message untouched even though it is "old"
    assert result.messages[0]["content"] == msgs[0]["content"]
    # last two messages untouched
    assert result.messages[-1]["content"] == msgs[-1]["content"]
    assert result.messages[-2]["content"] == msgs[-2]["content"]


def test_does_not_mutate_input():
    msgs = _convo()
    before = json.dumps(msgs)
    history.compress_history(msgs, keep_last=2)
    assert json.dumps(msgs) == before  # input list/dicts unchanged


def test_anthropic_block_content():
    store = CompressionStore()
    msgs = [
        {"role": "user", "content": "go"},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "content": BIG_JSON},
                {"type": "text", "text": "small note"},
            ],
        },
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "next"},
    ]
    result = history.compress_history(msgs, keep_last=2, store=store)
    block = result.messages[1]["content"][0]
    assert "reduction:" in block["content"]  # tool_result leaf compressed
    assert result.messages[1]["content"][1]["text"] == "small note"  # small text untouched


def test_keep_last_zero_compresses_all_old():
    result = history.compress_history(_convo(), keep_last=0)
    assert result.messages_compressed == 1  # only the big one clears the threshold


def test_small_history_no_change():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    result = history.compress_history(msgs, keep_last=4)
    assert result.messages_compressed == 0
    assert result.tokens_before == 0


def test_sdk_records_metrics():
    opt = TokenOptimizer()
    msgs = _convo()
    new_msgs = opt.compress_messages(msgs, keep_last=2)
    assert len(new_msgs) == len(msgs)
    summary = opt.report()
    assert summary["input_tokens_saved"] > 0
    assert summary["input_savings_pct"] > 0


def test_sdk_uses_config_default():
    opt = TokenOptimizer()
    opt.config.history_keep_last = 2
    new_msgs = opt.compress_messages(_convo())
    # index 3 (old, large) compressed; index 5/4 (last 2) preserved
    assert "reduction:" in new_msgs[3]["content"]
    assert new_msgs[5]["content"] == "summarize the open ports"
