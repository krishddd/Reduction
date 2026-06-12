from reduction import OptimizerConfig, TokenOptimizer


def test_prepare_injects_caveman_and_format():
    opt = TokenOptimizer(OptimizerConfig(caveman=True, output_format="toon"))
    req = opt.prepare(system="You are a planner.", user="do the thing")
    assert "Caveman" in req.system_text
    assert "TOON" in req.system_text
    assert req.output_format == "toon"
    # cache breakpoint present on the last system block
    assert req.system_blocks[-1]["cache_control"] == {"type": "ephemeral"}


def test_prepare_normalizes_user_turn():
    opt = TokenOptimizer(OptimizerConfig(caveman=False, output_format="text"))
    req = opt.prepare(system="s", user="dup\ndup\ndup\n\n\n\n")
    assert "(x3)" in req.messages[0]["content"]


def test_static_context_in_prefix_user_volatile_last():
    opt = TokenOptimizer(OptimizerConfig(caveman=False))
    req = opt.prepare(
        system="s",
        user="question",
        static_context=["reused schema"],
        volatile_context=["ts: 123"],
    )
    joined = "\n".join(b["text"] for b in req.system_blocks)
    assert "reused schema" in joined
    assert "ts: 123" in req.messages[0]["content"]
    assert "ts: 123" not in joined


def test_decode_toon_roundtrip():
    opt = TokenOptimizer(OptimizerConfig(output_format="toon"))
    text = "items[2]{id,name}:\n  1,alice\n  2,bob"
    assert opt.decode_output(text) == [
        {"id": "1", "name": "alice"},
        {"id": "2", "name": "bob"},
    ]


def test_record_usage_counts_cache_reads_both_shapes():
    opt = TokenOptimizer()
    opt.record_usage({"output_tokens": 100, "cache_read_input_tokens": 5000})
    opt.record_usage({"completion_tokens": 50, "prompt_tokens_details": {"cached_tokens": 200}})
    s = opt.report()
    assert s["calls"] == 2
    assert s["cache_read_tokens"] == 5200


def test_disabled_layers_are_noops():
    cfg = OptimizerConfig(
        caveman=False, output_format="text", strip_whitespace=False, dedupe_lines=False
    )
    opt = TokenOptimizer(cfg)
    req = opt.prepare(system="s", user="u")
    assert req.system_text == "s"
    assert req.messages[0]["content"] == "u"
