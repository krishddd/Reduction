"""Layer 5 (instructions side) — Caveman persona injection.

Appends a terse-output skill to the system prompt so the model drops
articles, hedging and conversational wrappers (~45% fewer output tokens with
neutral-to-positive accuracy impact). Restrict to machine/tool legs; keep
prose mode for user-facing text.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "caveman.md"


@lru_cache(maxsize=1)
def caveman_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8").strip()


def apply(system_prompt: str) -> str:
    """Append the caveman skill to a system prompt."""
    return f"{system_prompt}\n\n{caveman_prompt()}"
