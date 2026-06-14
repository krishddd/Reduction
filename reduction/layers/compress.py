"""Layer 2 — LLMLingua-2 context compression.

Compresses retrieved/context documents before prompt assembly. System
instructions and schemas are NEVER compressed — only the context payload.

llmlingua is an optional heavy dependency (pulls torch); when it is not
installed the compressor degrades to a no-op so the rest of the pipeline
still works.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Compressing tiny prompts costs more (model inference) than it saves.
MIN_CHARS_TO_COMPRESS = 2000

_compressor: object | None = None


def _get_compressor() -> object | None:
    global _compressor
    if _compressor is None:
        try:
            from llmlingua import PromptCompressor

            _compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                use_llmlingua2=True,
            )
        except ImportError:
            logger.warning("llmlingua not installed — Layer 2 compression disabled")
            _compressor = False
    return _compressor or None


def compress_documents(
    documents: list[str], rate: float = 0.5, *, compressor: object | None = None
) -> list[str]:
    """Compress each context document toward ``rate`` of its original tokens.

    Documents below ``MIN_CHARS_TO_COMPRESS`` pass through untouched. ``compressor``
    is injectable (anything with ``compress_prompt(text, rate=...) -> {"compressed_prompt": ...}``)
    so the gate/rate logic is testable without loading LLMLingua.
    """
    compressor = compressor or _get_compressor()
    if not compressor:
        return documents

    out: list[str] = []
    for doc in documents:
        if len(doc) < MIN_CHARS_TO_COMPRESS:
            out.append(doc)
            continue
        result = compressor.compress_prompt(doc, rate=rate)  # type: ignore[attr-defined]
        out.append(result["compressed_prompt"])
    return out
