"""Token-savings accounting — the `zap gain` of the in-process pipeline.

Tracks tokens before vs after optimization across every input and output, plus
the input the provider actually billed (native cache reads, cache writes,
uncached input). Token counts use ``tiktoken`` when available for accuracy and
fall back to a ~4-chars-per-token heuristic otherwise.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

CHARS_PER_TOKEN = 4
_encoder = None
_encoder_tried = False


def _get_encoder():
    """Lazily load a tiktoken encoder; None if tiktoken is unavailable."""
    global _encoder, _encoder_tried
    if not _encoder_tried:
        _encoder_tried = True
        try:
            import tiktoken

            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _encoder = None
    return _encoder


def estimate_tokens(text: str) -> int:
    """Token count via tiktoken when installed, else a ~4-chars/token estimate."""
    if not text:
        return 0
    enc = _get_encoder()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, round(len(text) / CHARS_PER_TOKEN))


@dataclass
class Metrics:
    calls: int = 0
    input_tokens_raw: int = 0
    input_tokens_optimized: int = 0
    output_tokens_raw: int = 0
    output_tokens_optimized: int = 0
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

    def record_output(self, raw_tokens: int, optimized_tokens: int) -> None:
        with self._lock:
            self.output_tokens_raw += raw_tokens
            self.output_tokens_optimized += optimized_tokens

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
    def total_saved(self) -> int:
        return self.input_saved + (self.output_tokens_raw - self.output_tokens_optimized)

    @property
    def savings_pct(self) -> float:
        total_raw = self.input_tokens_raw + self.output_tokens_raw
        return (self.total_saved / total_raw * 100) if total_raw else 0.0

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens_raw": self.input_tokens_raw,
            "input_tokens_optimized": self.input_tokens_optimized,
            "output_tokens_raw": self.output_tokens_raw,
            "output_tokens_optimized": self.output_tokens_optimized,
            "billed_input_tokens": self.billed_input_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "tokens_saved": self.total_saved,
            "savings_pct": round(self.savings_pct, 1),
        }

    def render(self) -> str:
        # ASCII-only so it prints on any console (incl. Windows cp1252).
        s = self.summary()
        bar_len = 24
        filled = max(0, min(bar_len, round(bar_len * s["savings_pct"] / 100)))
        bar = "#" * filled + "-" * (bar_len - filled)
        return (
            "Reduction - Token Savings\n"
            "============================================\n"
            f"Calls:            {s['calls']:>12,}\n"
            f"Input  raw->opt:  {s['input_tokens_raw']:>12,} -> {s['input_tokens_optimized']:,}\n"
            f"Output raw->opt:  {s['output_tokens_raw']:>12,} -> {s['output_tokens_optimized']:,}\n"
            f"Native cache rd:  {s['cache_read_tokens']:>12,}\n"
            f"Tokens saved:     {s['tokens_saved']:>12,} ({s['savings_pct']}%)\n"
            f"Efficiency: {bar} {s['savings_pct']}%"
        )

    def persist(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
