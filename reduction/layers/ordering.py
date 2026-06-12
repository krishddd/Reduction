"""Layer 4 — stable-prefix message assembly for native provider caching.

Providers cache the *static prefix* of a prompt:
- Anthropic: explicit ``cache_control`` breakpoints; reads cost 0.10x input.
- OpenAI: automatic for prefixes > 1,024 tokens; 50% discount.

The discipline both reward is the same: byte-stable content first (system
prompt, tool definitions, schemas, static context), volatile content last
(user message, timestamps). This module is the single place messages get
assembled so that ordering can never regress.
"""

from __future__ import annotations

from typing import Any

CACHE_CONTROL = {"type": "ephemeral"}


def assemble_messages(
    system_prompt: str,
    user_message: str,
    *,
    static_context: list[str] | None = None,
    volatile_context: list[str] | None = None,
    anthropic_cache: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(system_blocks, messages)`` ordered for maximum cache reuse.

    ``static_context`` is content reused across requests (schemas, docs,
    filtered logs) — it lands in the cacheable prefix. ``volatile_context``
    (timestamps, request-specific data) goes after the cache breakpoint.

    With ``anthropic_cache`` the last stable block carries a ``cache_control``
    breakpoint; pass the blocks as the Anthropic ``system`` parameter. For
    OpenAI-style providers, flatten the blocks into one system message and
    rely on automatic prefix caching.
    """
    system_blocks: list[dict[str, Any]] = [{"type": "text", "text": system_prompt}]
    for doc in static_context or []:
        system_blocks.append({"type": "text", "text": doc})

    if anthropic_cache:
        # Breakpoint on the LAST stable block: everything up to and including
        # it is written to / read from the provider cache.
        system_blocks[-1] = {**system_blocks[-1], "cache_control": CACHE_CONTROL}

    user_parts = list(volatile_context or []) + [user_message]
    messages = [{"role": "user", "content": "\n\n".join(user_parts)}]
    return system_blocks, messages


def flatten_for_openai(system_blocks: list[dict[str, Any]]) -> str:
    """Join system blocks into one string for providers without block syntax."""
    return "\n\n".join(b["text"] for b in system_blocks)
