"""Drop-in wrapper around ``anthropic.Anthropic``.

Exposes a ``.messages.create(...)`` that mirrors the real SDK but routes the
request through the optimizer first: caveman/format contract on the system
prompt, stable-prefix ordering with ``cache_control``, normalized inputs, and
usage metrics. Anything the optimizer does not understand is passed straight
through to the underlying client.

    from reduction.adapters import OptimizedAnthropic
    client = OptimizedAnthropic(api_key=...)        # same ctor as anthropic
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        system="You are a planner.",                # plain string is fine
        messages=[{"role": "user", "content": "..."}],
        max_tokens=1024,
        output_format="toon",                        # optimizer extra (optional)
    )
    print(client.optimizer.render())
"""

from __future__ import annotations

from typing import Any

from reduction.config import OptimizerConfig
from reduction.sdk import TokenOptimizer


class _Messages:
    def __init__(self, parent: OptimizedAnthropic) -> None:
        self._parent = parent

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        output_format: str | None = None,
        caveman_on: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        opt = self._parent.optimizer
        system_text = _system_to_text(system)

        # Pull the latest user string out so we can normalize + re-order it.
        user_text, prior = _split_last_user(messages)
        req = opt.prepare(
            system=system_text,
            user=user_text,
            output_format=output_format,
            caveman_on=caveman_on,
        )

        final_messages = prior + req.messages
        resp = self._parent._client.messages.create(
            model=model,
            system=req.system_blocks,
            messages=final_messages,
            **kwargs,
        )
        usage = getattr(resp, "usage", None)
        if usage is not None:
            opt.record_usage(usage)
        return resp


class OptimizedAnthropic:
    def __init__(
        self,
        *,
        config: OptimizerConfig | None = None,
        optimizer: TokenOptimizer | None = None,
        **anthropic_kwargs: Any,
    ) -> None:
        import anthropic

        self._client = anthropic.Anthropic(**anthropic_kwargs)
        self.optimizer = optimizer or TokenOptimizer(config)
        self.messages = _Messages(self)

    def __getattr__(self, name: str) -> Any:
        # Delegate everything else (models, beta, etc.) to the real client.
        return getattr(self._client, name)


def _system_to_text(system: str | list[dict[str, Any]] | None) -> str:
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    return "\n\n".join(b.get("text", "") for b in system if isinstance(b, dict))


def _split_last_user(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Return (last user text, all messages before it)."""
    if messages and messages[-1].get("role") == "user":
        content = messages[-1]["content"]
        text = content if isinstance(content, str) else _blocks_to_text(content)
        return text, messages[:-1]
    return "", messages


def _blocks_to_text(content: Any) -> str:
    if isinstance(content, list):
        return "\n\n".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)
