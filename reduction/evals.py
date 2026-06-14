"""Accuracy evaluation — does compression preserve the model's answers?

Saving tokens is only useful if the model still answers correctly. This harness
runs each case twice — once with the raw context, once with the compressed
context — through an injectable ``model_fn`` and checks whether the answer still
passes. It reports the **answer-preservation rate** (of the cases the model got
right on raw input, how many it still gets right compressed) alongside the token
savings, so you can see the trade-off instead of guessing.

``model_fn`` is injected so this is testable offline and provider-agnostic:

    def model_fn(context: str, question: str) -> str: ...

Wire a real model by closing over your client:

    def model_fn(context, question):
        resp = client.messages.create(model="claude-sonnet-4-6", max_tokens=256,
            messages=[{"role": "user", "content": f"{context}\n\nQ: {question}"}])
        return resp.content[0].text
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from reduction.metrics import estimate_tokens
from reduction.sdk import TokenOptimizer

ModelFn = Callable[[str, str], str]
CheckFn = Callable[[str], bool]


@dataclass
class EvalCase:
    name: str
    context: str  # the large blob that gets compressed (tool output, document)
    question: str
    check: CheckFn  # returns True if an answer is acceptable


@dataclass
class CaseResult:
    name: str
    passed_raw: bool
    passed_compressed: bool
    tokens_raw: int
    tokens_compressed: int

    @property
    def preserved(self) -> bool:
        # Only meaningful where raw passed: did compression keep it passing?
        return (not self.passed_raw) or self.passed_compressed


@dataclass
class EvalReport:
    results: list[CaseResult] = field(default_factory=list)

    @property
    def raw_passes(self) -> int:
        return sum(r.passed_raw for r in self.results)

    @property
    def compressed_passes(self) -> int:
        return sum(r.passed_compressed for r in self.results)

    @property
    def preservation_rate(self) -> float:
        """Of cases correct on raw input, fraction still correct compressed."""
        eligible = [r for r in self.results if r.passed_raw]
        if not eligible:
            return 1.0
        return sum(r.passed_compressed for r in eligible) / len(eligible)

    @property
    def token_savings_pct(self) -> float:
        raw = sum(r.tokens_raw for r in self.results)
        comp = sum(r.tokens_compressed for r in self.results)
        return (1 - comp / raw) * 100 if raw else 0.0

    def summary(self) -> dict:
        return {
            "cases": len(self.results),
            "raw_passes": self.raw_passes,
            "compressed_passes": self.compressed_passes,
            "preservation_rate_pct": round(self.preservation_rate * 100, 1),
            "token_savings_pct": round(self.token_savings_pct, 1),
            "regressions": [
                r.name for r in self.results if r.passed_raw and not r.passed_compressed
            ],
        }

    def render(self) -> str:
        s = self.summary()
        lines = [
            "Reduction - Accuracy Eval",
            "============================================",
            f"Cases:               {s['cases']}",
            f"Passed (raw):        {s['raw_passes']}",
            f"Passed (compressed): {s['compressed_passes']}",
            f"Answer preservation: {s['preservation_rate_pct']}%",
            f"Token savings:       {s['token_savings_pct']}%",
        ]
        if s["regressions"]:
            lines.append(f"REGRESSIONS:         {', '.join(s['regressions'])}")
        return "\n".join(lines)


def run_evals(
    cases: list[EvalCase],
    model_fn: ModelFn,
    *,
    optimizer: TokenOptimizer | None = None,
) -> EvalReport:
    """Run each case raw and compressed; measure answer preservation + savings."""
    opt = optimizer or TokenOptimizer()
    report = EvalReport()
    for case in cases:
        compressed = opt.filter_tool_output(case.context)
        ans_raw = model_fn(case.context, case.question)
        ans_comp = model_fn(compressed, case.question)
        report.results.append(
            CaseResult(
                name=case.name,
                passed_raw=bool(case.check(ans_raw)),
                passed_compressed=bool(case.check(ans_comp)),
                tokens_raw=estimate_tokens(case.context),
                tokens_compressed=estimate_tokens(compressed),
            )
        )
    return report
