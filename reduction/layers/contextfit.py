"""Budget-aware context fitting — pack the most useful context into a token budget.

RAG and agent loops routinely assemble more candidate context than is worth
sending: a dozen retrieved chunks, several files, prior notes. Sending all of it
wastes tokens; truncating blindly drops the relevant part. This fits a list of
chunks into a fixed token budget by:

  1. **scoring** each chunk for relevance to an optional query (lexical overlap,
     length-normalized so a huge chunk does not win on size alone);
  2. **greedily including** chunks in priority order while they fit verbatim;
  3. **compressing** a chunk that does not fit (content-aware + CCR) and including
     it if the compressed form fits;
  4. **truncating** a still-too-large chunk to the remaining room with a CCR
     marker (reversible) when there is meaningful space left, else **dropping** it.

Included chunks are returned in their original input order (document coherence);
selection is by score. Everything dropped or truncated is reported, so a caller
never mistakes a budget-trimmed context for the full set.

This is the dependency-free analogue of Headroom's IntelligentContext / score-based
context fitting.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from reduction import ccr as ccr_mod
from reduction.content import compress_content
from reduction.metrics import CHARS_PER_TOKEN, estimate_tokens

# Split on non-alphanumerics so identifiers like AWS_PROFILE or camelCase
# boundaries surface their parts ("aws", "profile") for query matching.
_WORD = re.compile(r"[a-z0-9]+")
# Only truncate-to-fit when at least this much budget remains; below it the
# fragment is too small to be useful, so the chunk is dropped instead.
MIN_FIT_TOKENS = 32


@dataclass
class FitResult:
    chunks: list[str]
    included: int
    compressed: int
    dropped: int
    tokens_used: int
    token_budget: int
    refs: list[str] = field(default_factory=list)


def _terms(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _score(chunk: str, query_terms: set[str]) -> float:
    """Relevance of a chunk to the query: query-term hits / log(length).

    Length normalization keeps a long, mostly-irrelevant chunk from outscoring a
    short, on-topic one purely by containing more words. With no query every
    chunk scores 0 and the stable sort preserves input order (recency/priority).
    """
    if not query_terms:
        return 0.0
    counts = Counter(_terms(chunk))
    if not counts:
        return 0.0
    hits = sum(counts[t] for t in query_terms)
    if not hits:
        return 0.0
    length = sum(counts.values())
    return hits / math.log2(length + 2)


def _truncate_to_budget(
    original: str,
    compressed: str,
    budget_tokens: int,
    ccr: bool,
    store: Any,
    existing_ref: str | None,
) -> tuple[str, str | None]:
    """Cut ``compressed`` to ~``budget_tokens`` and attach a reversible CCR marker."""
    ref = existing_ref
    if ccr and ref is None:
        if store is None:
            store = ccr_mod.get_default_store()
        ref = store.put(original)

    note = "context truncated to fit budget"
    marker = ccr_mod.make_marker(note, ref) if ref else "[... truncated]"
    body_budget = max(0, budget_tokens - estimate_tokens(marker) - 1)  # -1 for the newline

    # Coarse char cut, then *measure* and shrink until within budget. The char
    # heuristic under-cuts on number-heavy text (tiktoken counts denser than
    # 4 chars/token), so a measured loop is required to actually fit.
    body = compressed
    approx_chars = body_budget * CHARS_PER_TOKEN
    if len(body) > approx_chars:
        body = body[:approx_chars]
    while body and estimate_tokens(body) > body_budget:
        body = body[: int(len(body) * 0.8)]
    body = body.rstrip()
    return f"{body}\n{marker}", ref


def fit_context(
    chunks: list[str],
    *,
    token_budget: int,
    query: str | None = None,
    ccr: bool = True,
    store: Any = None,
) -> FitResult:
    """Select/compress ``chunks`` to fit ``token_budget``, prioritizing relevance."""
    if token_budget <= 0:
        return FitResult([], 0, 0, len(chunks), 0, token_budget, [])

    query_terms = set(_terms(query)) if query else set()
    # Priority order: highest score first, ties broken by original index (stable).
    order = sorted(range(len(chunks)), key=lambda i: (-_score(chunks[i], query_terms), i))

    selected: dict[int, str] = {}
    refs: list[str] = []
    used = 0
    compressed = dropped = 0

    for idx in order:
        chunk = chunks[idx]
        raw_cost = estimate_tokens(chunk)
        if used + raw_cost <= token_budget:
            selected[idx] = chunk
            used += raw_cost
            continue

        result = compress_content(chunk, ccr=ccr, store=store)
        if result.text != chunk and used + result.tokens_after <= token_budget:
            selected[idx] = result.text
            used += result.tokens_after
            compressed += 1
            if result.ref:
                refs.append(result.ref)
            continue

        remaining = token_budget - used
        if remaining >= MIN_FIT_TOKENS:
            truncated, ref = _truncate_to_budget(
                chunk, result.text, remaining, ccr, store, result.ref
            )
            selected[idx] = truncated
            used += estimate_tokens(truncated)
            compressed += 1
            if ref:
                refs.append(ref)
            continue

        dropped += 1

    out = [selected[i] for i in sorted(selected)]
    return FitResult(
        chunks=out,
        included=len(out),
        compressed=compressed,
        dropped=dropped,
        tokens_used=used,
        token_budget=token_budget,
        refs=refs,
    )
