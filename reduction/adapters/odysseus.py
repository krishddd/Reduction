"""Adapter for the odysseus security agent (Agent_security_testing).

That agent routes every LLM call through a thin client:

    ClaudeClient.message(*, model, system, user, tools=None,
                         max_tokens=2048, temperature=0.2) -> LLMResponse

The client already does Layer 4 (it cache-marks the system prompt). This
adapter layers the *rest* of the pipeline on top without touching call sites:

  - Layer 5: caveman persona + TOON/YAML output contract appended to system
  - always-on: normalize the user turn (dedupe + whitespace)
  - metrics: record per-call savings and cache reads

The agent enforces a prompt-cache contract (no profile data in the system
prompt). Caveman/format text is generic and profile-free, so it is safe to
append; this adapter never moves per-target data into the system prompt.

Usage (in Security_module, no other change needed):

    from reduction import TokenOptimizer
    from reduction.adapters import wrap_message_fn

    client = ClaudeClient()
    opt = TokenOptimizer()
    client.message = wrap_message_fn(client.message, opt, output_format="toon")
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from reduction.layers import normalize
from reduction.sdk import TokenOptimizer


def wrap_message_fn(
    message_fn: Callable[..., Any],
    optimizer: TokenOptimizer,
    *,
    output_format: str | None = None,
    caveman_on: bool | None = None,
) -> Callable[..., Any]:
    """Return a wrapped ``message`` callable that optimizes inputs/outputs.

    Keeps the original signature and return type (``LLMResponse``). The
    response's ``usage`` is fed to the optimizer's metrics.
    """
    cfg = optimizer.config

    def wrapped(
        *,
        model: str,
        system: str,
        user: str | list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> Any:
        # Layer 5 (instructions side) — append caveman + format contract.
        new_system = optimizer.build_system(
            system, output_format=output_format, caveman_on=caveman_on
        )

        # Always-on — normalize the user turn (string form only; block lists
        # are passed through untouched to stay compatible with tool results).
        new_user = user
        if isinstance(user, str) and (cfg.strip_whitespace or cfg.dedupe_lines):
            new_user = normalize.normalize(
                user, strip=cfg.strip_whitespace, dedupe=cfg.dedupe_lines
            )
            optimizer.metrics.record_input(user, new_user, layer="normalize")

        kwargs: dict[str, Any] = {
            "model": model,
            "system": new_system,
            "user": new_user,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools is not None:
            kwargs["tools"] = tools

        resp = message_fn(**kwargs)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            optimizer.record_usage(usage)
        return resp

    return wrapped
