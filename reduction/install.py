"""``reduction.install()`` — the zero-touch, one-line way to start saving tokens.

The adapters in ``reduction.adapters`` still ask you to swap your constructor
(``OptimizedAnthropic(...)`` instead of ``anthropic.Anthropic(...)``). This is
the path with *no* code change at all: call ``install()`` once at startup and
every ``anthropic`` / ``openai`` client in the process — sync *or* async, already
created or created later — routes through the optimizer.

    import reduction
    reduction.install()                 # that's the whole integration

    # ... your existing code, unchanged ...
    client = anthropic.Anthropic()
    client.messages.create(model="claude-sonnet-4-6", max_tokens=512,
                           system="You plan.", messages=[...])

    print(reduction.report())           # token-savings summary

It works by patching the SDKs' ``create`` methods in place (the same technique
Instructor uses to add structured output, and observability tools use to trace
calls). Both the sync (``Messages`` / ``Completions``) and async
(``AsyncMessages`` / ``AsyncCompletions``) clients are covered. ``uninstall()``
restores the originals. Idempotent; providers whose SDK is not installed are
silently skipped.

Reduction-specific kwargs (``output_format``, ``caveman_on``) are honored if
passed but never required; everything else falls back to config / env vars.
"""

from __future__ import annotations

import inspect
from typing import Any

from reduction.adapters.anthropic import _split_last_user, _system_to_text
from reduction.config import OptimizerConfig
from reduction.layers import normalize
from reduction.sdk import TokenOptimizer

# The optimizer all patched clients share, so metrics aggregate across the app.
_active_optimizer: TokenOptimizer | None = None
_patched: list[tuple[type, str, Any]] = []  # (class, attr, original) for uninstall


def get_optimizer() -> TokenOptimizer | None:
    """The optimizer installed clients report into (None if not installed)."""
    return _active_optimizer


def report() -> str:
    """Render the token-savings summary for the installed optimizer."""
    if _active_optimizer is None:
        return "reduction not installed — call reduction.install() first"
    return _active_optimizer.render()


def install(
    config: OptimizerConfig | None = None,
    *,
    optimizer: TokenOptimizer | None = None,
    anthropic: bool = True,
    openai: bool = True,
) -> TokenOptimizer:
    """Patch installed LLM SDKs in place so every call is optimized.

    Returns the shared :class:`TokenOptimizer` (call ``.render()`` for metrics, or
    use :func:`report`). Safe to call more than once; re-patching is skipped.
    """
    global _active_optimizer
    if _active_optimizer is None:
        _active_optimizer = optimizer or TokenOptimizer(config)
    opt = _active_optimizer

    if anthropic:
        _patch_anthropic(opt)
    if openai:
        _patch_openai(opt)
    return opt


def uninstall() -> None:
    """Restore the original SDK methods and forget the active optimizer."""
    global _active_optimizer
    for cls, attr, original in reversed(_patched):
        setattr(cls, attr, original)
    _patched.clear()
    _active_optimizer = None


# ---- shared transforms ------------------------------------------------


def _record(opt: TokenOptimizer, resp: Any) -> None:
    usage = getattr(resp, "usage", None)
    if usage is not None:
        opt.record_usage(usage)


def _anthropic_request(
    opt: TokenOptimizer,
    messages: list[dict[str, Any]],
    system: Any,
    output_format: str | None,
    caveman_on: bool | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build (optimized system_blocks, final messages) for an Anthropic call."""
    user_text, prior = _split_last_user(messages)
    req = opt.prepare(
        system=_system_to_text(system),
        user=user_text,
        output_format=output_format,
        caveman_on=caveman_on,
    )
    return req.system_blocks, prior + req.messages


def _openai_messages(
    opt: TokenOptimizer,
    messages: list[dict[str, Any]],
    output_format: str | None,
    caveman_on: bool | None,
) -> list[dict[str, Any]]:
    """Rewrite system/user messages through the optimizer for an OpenAI call."""
    cfg = opt.config
    out: list[dict[str, Any]] = []
    for msg in messages:
        role, content = msg.get("role"), msg.get("content")
        if role == "system" and isinstance(content, str):
            out.append(
                {
                    "role": "system",
                    "content": opt.build_system(
                        content, output_format=output_format, caveman_on=caveman_on
                    ),
                }
            )
        elif role == "user" and isinstance(content, str):
            out.append(
                {
                    "role": "user",
                    "content": normalize.normalize(
                        content, strip=cfg.strip_whitespace, dedupe=cfg.dedupe_lines
                    ),
                }
            )
        else:
            out.append(msg)
    return out


# ---- patchers ---------------------------------------------------------


def _patch(cls: type, original: Any, wrapper: Any) -> None:
    wrapper._reduction_patched = True  # type: ignore[attr-defined]
    cls.create = wrapper  # type: ignore[attr-defined]
    _patched.append((cls, "create", original))


def _is_patched(cls: type) -> bool:
    return getattr(getattr(cls, "create", None), "_reduction_patched", False)


def _patch_anthropic(opt: TokenOptimizer) -> None:
    try:
        from anthropic.resources.messages import AsyncMessages, Messages
    except ImportError:
        return

    if not _is_patched(Messages):
        sync_original = Messages.create

        def create(self: Any, *, model: str, messages: list, system: Any = None,
                   output_format: str | None = None, caveman_on: bool | None = None,
                   **kwargs: Any) -> Any:  # fmt: skip
            system_blocks, final = _anthropic_request(
                opt, messages, system, output_format, caveman_on
            )
            resp = sync_original(self, model=model, system=system_blocks, messages=final, **kwargs)
            _record(opt, resp)
            return resp

        _patch(Messages, sync_original, create)

    if not _is_patched(AsyncMessages):
        async_original = AsyncMessages.create

        async def acreate(self: Any, *, model: str, messages: list, system: Any = None,
                          output_format: str | None = None, caveman_on: bool | None = None,
                          **kwargs: Any) -> Any:  # fmt: skip
            system_blocks, final = _anthropic_request(
                opt, messages, system, output_format, caveman_on
            )
            resp = async_original(self, model=model, system=system_blocks, messages=final, **kwargs)
            if inspect.isawaitable(resp):
                resp = await resp
            _record(opt, resp)
            return resp

        _patch(AsyncMessages, async_original, acreate)


def _patch_openai(opt: TokenOptimizer) -> None:
    try:
        from openai.resources.chat.completions import AsyncCompletions, Completions
    except ImportError:
        return

    if not _is_patched(Completions):
        sync_original = Completions.create

        def create(self: Any, *, messages: list, model: str,
                   output_format: str | None = None, caveman_on: bool | None = None,
                   **kwargs: Any) -> Any:  # fmt: skip
            new_messages = _openai_messages(opt, messages, output_format, caveman_on)
            resp = sync_original(self, messages=new_messages, model=model, **kwargs)
            _record(opt, resp)
            return resp

        _patch(Completions, sync_original, create)

    if not _is_patched(AsyncCompletions):
        async_original = AsyncCompletions.create

        async def acreate(self: Any, *, messages: list, model: str,
                          output_format: str | None = None, caveman_on: bool | None = None,
                          **kwargs: Any) -> Any:  # fmt: skip
            new_messages = _openai_messages(opt, messages, output_format, caveman_on)
            resp = async_original(self, messages=new_messages, model=model, **kwargs)
            if inspect.isawaitable(resp):
                resp = await resp
            _record(opt, resp)
            return resp

        _patch(AsyncCompletions, async_original, acreate)
