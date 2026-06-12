from reduction.layers.ordering import assemble_messages, flatten_for_openai


def test_static_context_lands_in_system_prefix():
    blocks, messages = assemble_messages(
        "system prompt",
        "user question",
        static_context=["doc A", "doc B"],
        volatile_context=["timestamp: now"],
    )
    assert [b["text"] for b in blocks] == ["system prompt", "doc A", "doc B"]
    assert messages == [{"role": "user", "content": "timestamp: now\n\nuser question"}]


def test_cache_breakpoint_on_last_stable_block():
    blocks, _ = assemble_messages("sys", "q", static_context=["doc"])
    assert "cache_control" not in blocks[0]
    assert blocks[-1]["cache_control"] == {"type": "ephemeral"}


def test_no_cache_control_when_disabled():
    blocks, _ = assemble_messages("sys", "q", anthropic_cache=False)
    assert all("cache_control" not in b for b in blocks)


def test_volatile_context_never_in_prefix():
    blocks, messages = assemble_messages("sys", "q", volatile_context=["volatile"])
    assert all("volatile" not in b["text"] for b in blocks)
    assert "volatile" in messages[0]["content"]


def test_flatten_for_openai():
    blocks, _ = assemble_messages("sys", "q", static_context=["doc"])
    assert flatten_for_openai(blocks) == "sys\n\ndoc"
