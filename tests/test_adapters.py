from reduction import TokenOptimizer
from reduction.adapters import wrap_message_fn


class _Usage:
    output_tokens = 120
    cache_read_input_tokens = 4000


class _Resp:
    text = "ok"
    usage = _Usage()


def test_wrap_message_fn_injects_and_records():
    captured = {}

    def fake_message(*, model, system, user, **kwargs):
        captured["system"] = system
        captured["user"] = user
        return _Resp()

    opt = TokenOptimizer()
    wrapped = wrap_message_fn(fake_message, opt, output_format="toon")
    resp = wrapped(model="claude-sonnet-4-6", system="You plan.", user="a\na\n\n\n\n")

    assert resp is not None
    assert "Caveman" in captured["system"]
    assert "TOON" in captured["system"]
    assert "(x2)" in captured["user"]  # normalized
    assert opt.report()["calls"] == 1
    assert opt.report()["cache_read_tokens"] == 4000


def test_wrap_passes_tools_through():
    seen = {}

    def fake_message(*, model, system, user, tools=None, **kwargs):
        seen["tools"] = tools
        return _Resp()

    opt = TokenOptimizer()
    wrapped = wrap_message_fn(fake_message, opt)
    wrapped(model="m", system="s", user="u", tools=[{"name": "t"}])
    assert seen["tools"] == [{"name": "t"}]
