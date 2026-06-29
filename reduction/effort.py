"""Effort routing — match reasoning budget to the task so routine steps stay cheap.

Extended thinking is the largest *output*-side cost in agent loops: a model that
"thinks" for 8k tokens before reading a file burns budget on a step that needs
none. Conversely, disabling thinking on a hard debugging step costs accuracy.
This classifies a task description and recommends a reasoning effort level — and
the concrete knobs both providers expose:

  * Anthropic extended thinking: ``{"type": "enabled", "budget_tokens": N}``
    (or omit thinking entirely when N == 0).
  * OpenAI reasoning models: ``reasoning_effort`` ("minimal" | "low" | "medium"
    | "high").

The classifier is heuristic and dependency-free: routine verbs (read, list,
grep, show, status) route to minimal effort; analytical verbs (debug, why, root
cause, design, refactor, prove) route to high. Everything else is medium. It is
advisory — a caller can always override — but on a long agent run, steering the
many trivial steps to minimal effort is where the output savings come from.

Inspired by Headroom's effort routing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Suggested Anthropic thinking budgets per level (budget_tokens; 0 disables).
_LEVEL_BUDGET = {
    "minimal": 0,
    "low": 1024,
    "medium": 4096,
    "high": 16384,
}
_LEVELS = list(_LEVEL_BUDGET)

# Routine, mechanical tasks — little or no reasoning needed.
_ROUTINE = re.compile(
    r"\b(?:read|cat|show|display|print|list|ls|get|fetch|open|view|head|tail|"
    r"grep|find|search for|count|status|format|lint|rename|copy|move|echo|"
    r"look\s?up|retrieve)\b",
    re.IGNORECASE,
)
# Analytical tasks — worth a large thinking budget.
_COMPLEX = re.compile(
    r"\b(?:why|debug|root\s?cause|diagnos\w*|analy[sz]e|investigat\w*|design|"
    r"architect\w*|plan|strateg\w*|refactor|optimi[sz]e|prove|derive|"
    r"trade[-\s]?off|compare|reason\w*|explain|vulnerab\w*|exploit|"
    r"fix the bug|figure out|how should|what causes)\b",
    re.IGNORECASE,
)


@dataclass
class EffortDecision:
    level: str  # "minimal" | "low" | "medium" | "high"
    thinking_budget: int  # Anthropic budget_tokens (0 = thinking disabled)
    reasoning_effort: str  # OpenAI reasoning_effort value
    rationale: str

    def anthropic_thinking(self) -> dict | None:
        """Anthropic ``thinking`` param, or None to omit it (minimal effort)."""
        if self.thinking_budget <= 0:
            return None
        return {"type": "enabled", "budget_tokens": self.thinking_budget}


def route_effort(task: str, *, default: str = "medium") -> EffortDecision:
    """Classify ``task`` and recommend a reasoning effort level.

    Analytical signals win ties over routine ones (under-thinking a hard task is
    costlier than over-thinking an easy one). With no signal, ``default`` applies.
    """
    text = task or ""
    complex_hit = bool(_COMPLEX.search(text))
    routine_hit = bool(_ROUTINE.search(text))

    if complex_hit:
        level, why = "high", "analytical task — allocate a large thinking budget"
    elif routine_hit:
        level, why = "minimal", "routine/mechanical task — reasoning adds cost, not value"
    else:
        level = default if default in _LEVEL_BUDGET else "medium"
        why = f"no strong signal — default ({level})"

    return EffortDecision(
        level=level,
        thinking_budget=_LEVEL_BUDGET[level],
        reasoning_effort=level,
        rationale=why,
    )
