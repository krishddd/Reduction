"""Content router — detect type, route to the right compressor, keep it reversible.

This is the unified entry point that ties detection + per-type compressors + CCR
together. Given any blob of text (a tool output, a retrieved document), it:

  1. detects the content type (JSON / diff / log / code / markdown / text);
  2. runs the specialized compressor for that type;
  3. if compression was lossy enough to matter, stores the original in the CCR
     store and appends a retrieval marker so the agent can expand it on demand.

Returns a ``CompressionResult`` with the compressed text, the ref (if stored),
and token accounting.
"""

from __future__ import annotations

from dataclasses import dataclass

from reduction import ccr as ccr_mod
from reduction.ccr import CompressionStore
from reduction.layers import codecrush, diffstat, jsoncrush, logcrush
from reduction.layers.detect import ContentType, detect
from reduction.layers.normalize import normalize
from reduction.metrics import estimate_tokens

# Below this, compression overhead (markers) outweighs the savings.
MIN_CHARS_TO_COMPRESS = 200
# Only attach a CCR marker when at least this fraction of tokens was removed.
CCR_MIN_SAVINGS = 0.25


@dataclass
class CompressionResult:
    text: str
    content_type: ContentType
    ref: str | None
    tokens_before: int
    tokens_after: int

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after

    @property
    def compression_ratio(self) -> float:
        return self.tokens_saved / self.tokens_before if self.tokens_before else 0.0


def _compress_by_type(text: str, ctype: ContentType) -> tuple[str, bool]:
    if ctype is ContentType.JSON:
        return jsoncrush.crush_json(text)
    if ctype is ContentType.DIFF:
        return diffstat.crush_diff(text)
    if ctype is ContentType.LOG:
        return logcrush.crush_log(text)
    if ctype is ContentType.CODE:
        return codecrush.crush_code(text)
    # markdown / text: lossless normalization only (safe, no CCR needed).
    return normalize(text, strip=True, dedupe=True), False


def compress_content(
    text: str,
    *,
    ccr: bool = True,
    store: CompressionStore | None = None,
) -> CompressionResult:
    """Detect, compress, and (optionally) make the result reversible via CCR."""
    tokens_before = estimate_tokens(text)
    if len(text) < MIN_CHARS_TO_COMPRESS:
        return CompressionResult(text, ContentType.TEXT, None, tokens_before, tokens_before)

    ctype = detect(text)
    compressed, lossy = _compress_by_type(text, ctype)
    tokens_after = estimate_tokens(compressed)

    ref = None
    if ccr and lossy and tokens_before:
        savings = (tokens_before - tokens_after) / tokens_before
        if savings >= CCR_MIN_SAVINGS:
            summary = (
                f"{ctype.value} compressed {savings:.0%} ({tokens_before}->{tokens_after} tok)"
            )
            if store is None:
                store = ccr_mod.get_default_store()
            ref = store.put(text)
            compressed = f"{compressed}\n{ccr_mod.make_marker(summary, ref)}"
            tokens_after = estimate_tokens(compressed)

    return CompressionResult(compressed, ctype, ref, tokens_before, tokens_after)
