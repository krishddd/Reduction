"""Layer 3 — semantic cache via LiteLLM.

Wraps completions so a new query that is *semantically* close to a previous
one (cosine > threshold) returns the cached response in single-digit
milliseconds instead of regenerating. Backed by redis-semantic (RediSearch
VSS) or qdrant-semantic.

litellm is an optional dependency; when absent or unconfigured the cache is a
no-op and calls go straight through.
"""

from __future__ import annotations

import os
from typing import Any


def configure_cache(
    *,
    host: str | None = None,
    port: int | None = None,
    threshold: float = 0.92,
    embedding_model: str = "text-embedding-3-small",
) -> bool:
    """Enable litellm's redis-semantic cache. Returns True if enabled."""
    host = host or os.environ.get("REDIS_HOST")
    if not host:
        return False
    try:
        import litellm
    except ImportError:
        return False
    litellm.cache = litellm.Cache(
        type="redis-semantic",
        host=host,
        port=int(port or os.environ.get("REDIS_PORT", "6379")),
        similarity_threshold=threshold,
        redis_semantic_cache_embedding_model=embedding_model,
    )
    return True


async def acomplete(model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
    """Async completion routed through litellm (and its cache layers)."""
    import litellm

    return await litellm.acompletion(model=model, messages=messages, **kwargs)
