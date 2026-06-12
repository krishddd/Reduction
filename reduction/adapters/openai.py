"""Drop-in wrapper around ``openai.OpenAI``.

Exposes ``.chat.completions.create(...)`` mirroring the real SDK. OpenAI
caches stable prefixes automatically, so the optimizer's job here is the
input/output shaping: caveman + format contract on the system message,
normalized user content, and usage metrics. Stable content stays first,
volatile last, to maximize automatic prefix-cache hits.
"""

from __future__ import annotations

from typing import Any

from reduction.config import OptimizerConfig
from reduction.layers import normalize
from reduction.sdk import TokenOptimizer


class _Completions:
    def __init__(self, parent: OptimizedOpenAI) -> None:
        self._parent = parent

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        output_format: str | None = None,
        caveman_on: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        opt = self._parent.optimizer
        cfg = opt.config
        new_messages: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system" and isinstance(msg.get("content"), str):
                new_messages.append(
                    {
                        "role": "system",
                        "content": opt.build_system(
                            msg["content"], output_format=output_format, caveman_on=caveman_on
                        ),
                    }
                )
            elif msg.get("role") == "user" and isinstance(msg.get("content"), str):
                new_messages.append(
                    {
                        "role": "user",
                        "content": normalize.normalize(
                            msg["content"], strip=cfg.strip_whitespace, dedupe=cfg.dedupe_lines
                        ),
                    }
                )
            else:
                new_messages.append(msg)

        resp = self._parent._client.chat.completions.create(
            model=model, messages=new_messages, **kwargs
        )
        usage = getattr(resp, "usage", None)
        if usage is not None:
            opt.record_usage(usage)
        return resp


class _Chat:
    def __init__(self, parent: OptimizedOpenAI) -> None:
        self.completions = _Completions(parent)


class OptimizedOpenAI:
    def __init__(
        self,
        *,
        config: OptimizerConfig | None = None,
        optimizer: TokenOptimizer | None = None,
        **openai_kwargs: Any,
    ) -> None:
        import openai

        self._client = openai.OpenAI(**openai_kwargs)
        self.optimizer = optimizer or TokenOptimizer(config)
        self.chat = _Chat(self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
