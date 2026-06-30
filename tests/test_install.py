"""Tests for reduction.install() — the zero-touch global SDK patch.

The patched method calls the *original* SDK ``create``; we stub that original
(via monkeypatch, before install captures it) so no network call is made.
"""

import pytest

import reduction


@pytest.fixture(autouse=True)
def _clean():
    yield
    reduction.uninstall()  # never leak a patch between tests


class _Usage:
    input_tokens = 120
    output_tokens = 30
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0


class _Resp:
    usage = _Usage()


def test_report_before_install():
    assert "not installed" in reduction.report()


def test_install_anthropic_optimizes_system_and_records(monkeypatch):
    pytest.importorskip("anthropic")
    from anthropic.resources.messages import Messages

    captured = {}

    def fake_create(self, *, model, messages, system=None, **kwargs):
        captured["system"] = system
        captured["messages"] = messages
        return _Resp()

    monkeypatch.setattr(Messages, "create", fake_create)
    opt = reduction.install(openai=False)

    inst = Messages.__new__(Messages)  # no real client/auth needed
    Messages.create(
        inst,
        model="claude-sonnet-4-6",
        max_tokens=64,
        system="You are a planner.",
        messages=[{"role": "user", "content": "scan   the    network"}],
    )

    # system was rewritten into optimized cache-control blocks (a list, not a str)
    assert isinstance(captured["system"], list)
    assert captured["system"]  # non-empty
    # usage recorded into the shared optimizer
    assert opt.report()["calls"] == 1


def test_install_openai_optimizes_messages(monkeypatch):
    pytest.importorskip("openai")
    from openai.resources.chat.completions import Completions

    captured = {}

    def fake_create(self, *, messages, model, **kwargs):
        captured["messages"] = messages
        return _Resp()

    monkeypatch.setattr(Completions, "create", fake_create)
    opt = reduction.install(anthropic=False)

    inst = Completions.__new__(Completions)
    Completions.create(
        inst,
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You plan."},
            # trailing whitespace + duplicate line -> normalize trims and dedupes
            {"role": "user", "content": "find the bug   \nfind the bug   \ndone"},
        ],
    )

    system_out, user_out = (m["content"] for m in captured["messages"])
    assert system_out != "You plan."  # caveman/format contract injected
    assert "   \n" not in user_out  # trailing whitespace stripped
    assert "(x2)" in user_out  # duplicate line deduped
    assert opt.report()["calls"] == 1


def test_idempotent_install(monkeypatch):
    pytest.importorskip("anthropic")
    from anthropic.resources.messages import Messages

    monkeypatch.setattr(Messages, "create", lambda self, **kw: _Resp())
    reduction.install(openai=False)
    patched_once = Messages.create
    reduction.install(openai=False)  # second call is a no-op
    assert Messages.create is patched_once


def test_uninstall_restores(monkeypatch):
    pytest.importorskip("anthropic")
    from anthropic.resources.messages import Messages

    sentinel = lambda self, **kw: _Resp()  # noqa: E731
    monkeypatch.setattr(Messages, "create", sentinel)
    reduction.install(openai=False)
    assert Messages.create is not sentinel
    reduction.uninstall()
    assert Messages.create is sentinel
    assert reduction.get_optimizer() is None


def test_install_returns_shared_optimizer(monkeypatch):
    opt = reduction.install()
    assert opt is reduction.get_optimizer()
    assert isinstance(reduction.report(), str)


def test_install_anthropic_async(monkeypatch):
    pytest.importorskip("anthropic")
    import asyncio

    from anthropic.resources.messages import AsyncMessages

    captured = {}

    async def fake_create(self, *, model, messages, system=None, **kwargs):
        captured["system"] = system
        return _Resp()

    monkeypatch.setattr(AsyncMessages, "create", fake_create)
    opt = reduction.install(openai=False)

    inst = AsyncMessages.__new__(AsyncMessages)
    asyncio.run(
        AsyncMessages.create(
            inst,
            model="claude-sonnet-4-6",
            max_tokens=32,
            system="You plan.",
            messages=[{"role": "user", "content": "hi"}],
        )
    )
    assert isinstance(captured["system"], list)  # optimized blocks
    assert opt.report()["calls"] == 1


def test_install_openai_async(monkeypatch):
    pytest.importorskip("openai")
    import asyncio

    from openai.resources.chat.completions import AsyncCompletions

    captured = {}

    async def fake_create(self, *, messages, model, **kwargs):
        captured["messages"] = messages
        return _Resp()

    monkeypatch.setattr(AsyncCompletions, "create", fake_create)
    opt = reduction.install(anthropic=False)

    inst = AsyncCompletions.__new__(AsyncCompletions)
    asyncio.run(
        AsyncCompletions.create(
            inst,
            model="gpt-4o",
            messages=[{"role": "system", "content": "You plan."}],
        )
    )
    assert captured["messages"][0]["content"] != "You plan."  # caveman injected
    assert opt.report()["calls"] == 1


def test_uninstall_restores_async(monkeypatch):
    pytest.importorskip("anthropic")
    from anthropic.resources.messages import AsyncMessages

    async def sentinel(self, **kw):
        return _Resp()

    monkeypatch.setattr(AsyncMessages, "create", sentinel)
    reduction.install(openai=False)
    assert AsyncMessages.create is not sentinel
    reduction.uninstall()
    assert AsyncMessages.create is sentinel
