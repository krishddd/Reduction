"""Token accounting for the pipeline.

Tracks *input* tokens before vs after optimization (the part we actually
reduce pre-call), the provider-reported billing breakdown (cache reads/writes,
billed input), and observed output tokens.

Honesty note: output-shaping (Caveman/TOON) savings are NOT reported as
"saved", because we have no counterfactual — we never see what the model
*would* have produced uncompressed. We report observed output tokens only.
``savings_pct`` therefore reflects input savings; use ``reduction.evals`` to
measure the output/accuracy trade-off empirically.

Tokenizer note: counts use ``tiktoken`` when installed. ``cl100k_base`` /
``o200k_base`` are exact for OpenAI models; for Claude they are an
*approximation* (Anthropic's tokenizer is not bundled), good enough for
relative savings but not exact billing.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

CHARS_PER_TOKEN = 4
_encoders: dict[str, object] = {}
_tiktoken_ok: bool | None = None


def _encoding_name(model: str | None) -> str:
    m = (model or "").lower()
    if m.startswith(("gpt-4o", "gpt-5", "o1", "o3", "o4")):
        return "o200k_base"
    return "cl100k_base"  # gpt-4/3.5 exact; Claude approximation


def _get_encoder(model: str | None = None):
    """Lazily load a tiktoken encoder for the model family; None if unavailable."""
    global _tiktoken_ok
    if _tiktoken_ok is False:
        return None
    name = _encoding_name(model)
    if name not in _encoders:
        try:
            import tiktoken

            _encoders[name] = tiktoken.get_encoding(name)
            _tiktoken_ok = True
        except Exception:
            _tiktoken_ok = False
            return None
    return _encoders[name]


def estimate_tokens(text: str, model: str | None = None) -> int:
    """Token count via tiktoken when installed, else a ~4-chars/token estimate.

    Pass ``model`` to pick the right encoding (exact for OpenAI; Claude is
    approximated with cl100k_base).
    """
    if not text:
        return 0
    enc = _get_encoder(model)
    if enc is not None:
        return len(enc.encode(text))
    return max(1, round(len(text) / CHARS_PER_TOKEN))


@dataclass
class Metrics:
    calls: int = 0
    input_tokens_raw: int = 0
    input_tokens_optimized: int = 0
    output_tokens: int = 0  # observed only — NOT reported as a saving
    # Provider-reported billing breakdown (when available).
    billed_input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    events: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Not a dataclass field — keeps it out of asdict/serialization and
        # avoids "cannot pickle lock" when persisting.
        self._lock = threading.Lock()

    def record_input(self, raw: str, optimized: str, *, layer: str) -> None:
        r, o = estimate_tokens(raw), estimate_tokens(optimized)
        with self._lock:
            self.input_tokens_raw += r
            self.input_tokens_optimized += o
            if r != o:
                self.events.append({"kind": "input", "layer": layer, "raw": r, "opt": o})

    def record_output(self, tokens: int) -> None:
        """Record observed output tokens (not treated as a saving)."""
        with self._lock:
            self.output_tokens += tokens

    def record_call(
        self,
        *,
        billed_input_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> None:
        with self._lock:
            self.calls += 1
            self.billed_input_tokens += billed_input_tokens
            self.cache_read_tokens += cache_read_tokens
            self.cache_write_tokens += cache_write_tokens

    @property
    def input_saved(self) -> int:
        return self.input_tokens_raw - self.input_tokens_optimized

    @property
    def input_savings_pct(self) -> float:
        return (self.input_saved / self.input_tokens_raw * 100) if self.input_tokens_raw else 0.0

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens_raw": self.input_tokens_raw,
            "input_tokens_optimized": self.input_tokens_optimized,
            "input_tokens_saved": self.input_saved,
            "input_savings_pct": round(self.input_savings_pct, 1),
            "output_tokens_observed": self.output_tokens,
            "billed_input_tokens": self.billed_input_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
        }

    def render(self) -> str:
        # ASCII-only so it prints on any console (incl. Windows cp1252).
        s = self.summary()
        bar_len = 24
        pct = s["input_savings_pct"]
        filled = max(0, min(bar_len, round(bar_len * pct / 100)))
        bar = "#" * filled + "-" * (bar_len - filled)
        return (
            "Reduction - Token Savings (input)\n"
            "============================================\n"
            f"Calls:            {s['calls']:>12,}\n"
            f"Input  raw->opt:  {s['input_tokens_raw']:>12,} -> {s['input_tokens_optimized']:,}\n"
            f"Input saved:      {s['input_tokens_saved']:>12,} ({pct}%)\n"
            f"Output (observed):{s['output_tokens_observed']:>12,}\n"
            f"Native cache rd:  {s['cache_read_tokens']:>12,}\n"
            f"Efficiency: {bar} {pct}%  (output/accuracy: see reduction.evals)"
        )

    def persist(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
