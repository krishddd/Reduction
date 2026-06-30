"""``reduction.install()`` — the zero-touch, one-line way to start saving tokens.

The adapters in ``reduction.adapters`` still ask you to swap your constructor
(``OptimizedAnthropic(...)`` instead of ``anthropic.Anthropic(...)``). This is
the path with *no* code change at all: call ``install()`` once at startup and
every ``anthropic.Anthropic`` / ``openai.OpenAI`` client in the process — already
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
calls). ``uninstall()`` restores the originals. Idempotent: calling ``install()``
twice is a no-op. Providers whose SDK is not installed are silently skipped.

Reduction-specific kwargs (``output_format``, ``caveman_on``) are honored if
passed but never required; everything else falls back to config / env vars.
"""

from __future__ import annotations

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


def _already_patched(method: Any) -> bool:
    return getattr(method, "_reduction_patched", False)


def _patch_anthropic(opt: TokenOptimizer) -> None:
    try:
        from anthropic.resources.messages import Messages
    except ImportError:
        return
    if _already_patched(Messages.create):
        return
    original = Messages.create

    def create(
        self: Any,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: Any = None,
        output_format: str | None = None,
        caveman_on: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        system_text = _system_to_text(system)
        user_text, prior = _split_last_user(messages)
        req = opt.prepare(
            system=system_text,
            user=user_text,
            output_format=output_format,
            caveman_on=caveman_on,
        )
        resp = original(
            self,
            model=model,
            system=req.system_blocks,
            messages=prior + req.messages,
            **kwargs,
        )
        usage = getattr(resp, "usage", None)
        if usage is not None:
            opt.record_usage(usage)
        return resp

    create._reduction_patched = True  # type: ignore[attr-defined]
    Messages.create = create  # type: ignore[method-assign]
    _patched.append((Messages, "create", original))


def _patch_openai(opt: TokenOptimizer) -> None:
    try:
        from openai.resources.chat.completions import Completions
    except ImportError:
        return
    if _already_patched(Completions.create):
        return
    original = Completions.create
    cfg = opt.config

    def create(
        self: Any,
        *,
        messages: list[dict[str, Any]],
        model: str,
        output_format: str | None = None,
        caveman_on: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        new_messages: list[dict[str, Any]] = []
        for msg in messages:
            role, content = msg.get("role"), msg.get("content")
            if role == "system" and isinstance(content, str):
                new_messages.append(
                    {
                        "role": "system",
                        "content": opt.build_system(
                            content, output_format=output_format, caveman_on=caveman_on
                        ),
                    }
                )
            elif role == "user" and isinstance(content, str):
                new_messages.append(
                    {
                        "role": "user",
                        "content": normalize.normalize(
                            content, strip=cfg.strip_whitespace, dedupe=cfg.dedupe_lines
                        ),
                    }
                )
            else:
                new_messages.append(msg)

        resp = original(self, messages=new_messages, model=model, **kwargs)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            opt.record_usage(usage)
        return resp

    create._reduction_patched = True  # type: ignore[attr-defined]
    Completions.create = create  # type: ignore[method-assign]
    _patched.append((Completions, "create", original))
