"""Configuration for the Reduction token-optimization pipeline.

Every knob has an env-var fallback so an agent can opt in to layers without
touching code. Heavy layers (LLMLingua compression, redis semantic cache)
default OFF — they pull large dependencies and only pay off on large, reused
context. The cheap, dependency-free layers (caveman, TOON, stable-prefix
ordering, shell filtering, whitespace dedup) default ON.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _flag(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class OptimizerConfig:
    # --- Layer 1: shell / tool-output filtering ---
    shell_filter: bool = _flag("REDUCTION_SHELL_FILTER", True)
    zap_binary: str = os.environ.get("REDUCTION_ZAP_BINARY", "zap")
    max_tool_output_lines: int = int(os.environ.get("REDUCTION_MAX_TOOL_LINES", "200"))
    # Content-aware compression (JSON/diff/log routing) for tool output.
    content_routing: bool = _flag("REDUCTION_CONTENT_ROUTING", True)
    # CCR: keep compression reversible (store originals, embed retrieval refs).
    ccr: bool = _flag("REDUCTION_CCR", True)
    ccr_store_path: str | None = os.environ.get("REDUCTION_CCR_STORE")
    # Conversation-history compression: shrink old turns, keep the last N verbatim.
    compress_history: bool = _flag("REDUCTION_HISTORY", False)
    history_keep_last: int = int(os.environ.get("REDUCTION_HISTORY_KEEP_LAST", "4"))

    # --- Layer 2: context compression (LLMLingua-2) ---
    compress_context: bool = _flag("REDUCTION_COMPRESS", False)
    compression_rate: float = float(os.environ.get("REDUCTION_COMPRESSION_RATE", "0.5"))

    # --- Layer 3: semantic cache ---
    semantic_cache: bool = _flag("REDUCTION_SEMANTIC_CACHE", False)
    semantic_threshold: float = float(os.environ.get("REDUCTION_SEMANTIC_THRESHOLD", "0.92"))

    # --- Layer 4: native provider prompt caching ---
    native_cache: bool = _flag("REDUCTION_NATIVE_CACHE", True)

    # --- Layer 5: output shaping ---
    caveman: bool = _flag("REDUCTION_CAVEMAN", True)
    # one of: "text" | "toon" | "yaml"
    output_format: str = os.environ.get("REDUCTION_OUTPUT_FORMAT", "text")

    # --- always-on cheap normalization ---
    strip_whitespace: bool = _flag("REDUCTION_STRIP_WS", True)
    dedupe_lines: bool = _flag("REDUCTION_DEDUPE", True)

    # --- metrics ---
    track_metrics: bool = _flag("REDUCTION_METRICS", True)
    metrics_path: str | None = os.environ.get("REDUCTION_METRICS_PATH")

    VALID_FORMATS = ("text", "toon", "yaml")

    def __post_init__(self) -> None:
        if self.output_format not in self.VALID_FORMATS:
            raise ValueError(
                f"output_format must be one of {self.VALID_FORMATS}, got {self.output_format!r}"
            )
        if not 0.0 < self.compression_rate <= 1.0:
            raise ValueError(f"compression_rate must be in (0, 1], got {self.compression_rate}")
        if not 0.0 < self.semantic_threshold <= 1.0:
            raise ValueError(f"semantic_threshold must be in (0, 1], got {self.semantic_threshold}")
        if self.history_keep_last < 0:
            raise ValueError(f"history_keep_last must be >= 0, got {self.history_keep_last}")

    def with_overrides(self, **kwargs: object) -> OptimizerConfig:
        """Return a copy with per-call overrides applied."""
        data = {**self.__dict__, **{k: v for k, v in kwargs.items() if v is not None}}
        return OptimizerConfig(**data)  # type: ignore[arg-type]
