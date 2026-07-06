"""Layer 3 — semantic cache.

A new query that is *semantically* close to a previous one (cosine >
threshold) returns the cached response instead of regenerating. Two backends:

  * ``LocalSemanticCache`` — in-process, dependency-free, used by the SDK path
    (``TokenOptimizer.cached_call``). Embeddings come from
    sentence-transformers when installed (``[memory]`` extra), else a hashing
    embedding that matches near-duplicate prompts by lexical overlap.
  * LiteLLM redis-semantic / qdrant-semantic — shared cross-process cache for
    the HTTP gateway (``configure_cache``). litellm is an optional dependency;
    when absent or unconfigured this backend is a no-op.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from typing import Any


class LocalSemanticCache:
    """In-process semantic response cache — Layer 3 without infrastructure.

    Stores ``(prompt embedding, response)``; a lookup whose cosine similarity
    to a stored prompt clears ``threshold`` returns that response. LRU-bounded
    by ``max_entries``. With the hashing-embedding fallback, similarity is
    lexical (bag-of-tokens): reworded-but-same-words prompts hit, true
    paraphrases may not — keep the threshold high either way, since a false
    hit returns a wrong answer.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.92,
        max_entries: int = 512,
        embed_fn: Any = None,
    ) -> None:
        from reduction.memory import _default_embed

        self.threshold = threshold
        self.max_entries = max_entries
        self._embed = embed_fn or _default_embed()
        self._lock = threading.Lock()
        # key -> (embedding, response)
        self._entries: OrderedDict[str, tuple[list[float], str]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, prompt: str) -> str | None:
        """Return the cached response for the closest prompt, or None."""
        from reduction.memory import cosine

        vec = self._embed(prompt)
        with self._lock:
            best_key: str | None = None
            best_score = 0.0
            for key, (stored_vec, _) in self._entries.items():
                score = cosine(vec, stored_vec)
                if score > best_score:
                    best_key, best_score = key, score
            if best_key is not None and best_score >= self.threshold:
                self._entries.move_to_end(best_key)  # LRU touch
                self.hits += 1
                return self._entries[best_key][1]
            self.misses += 1
            return None

    def put(self, prompt: str, response: str) -> None:
        vec = self._embed(prompt)
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        with self._lock:
            self._entries[key] = (vec, response)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)


def build_cache_params(
    *,
    host: str | None = None,
    port: int | None = None,
    threshold: float = 0.92,
    embedding_model: str = "text-embedding-3-small",
) -> dict | None:
    """Build the litellm redis-semantic cache kwargs, or None if no host.

    Pure (no litellm import) so the configuration logic is unit-testable.
    """
    host = host or os.environ.get("REDIS_HOST")
    if not host:
        return None
    return {
        "type": "redis-semantic",
        "host": host,
        "port": int(port or os.environ.get("REDIS_PORT", "6379")),
        "similarity_threshold": threshold,
        "redis_semantic_cache_embedding_model": embedding_model,
    }


def configure_cache(
    *,
    host: str | None = None,
    port: int | None = None,
    threshold: float = 0.92,
    embedding_model: str = "text-embedding-3-small",
) -> bool:
    """Enable litellm's redis-semantic cache. Returns True if enabled."""
    params = build_cache_params(
        host=host, port=port, threshold=threshold, embedding_model=embedding_model
    )
    if params is None:
        return False
    try:
        import litellm
    except ImportError:
        return False
    litellm.cache = litellm.Cache(**params)
    return True


async def acomplete(model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
    """Async completion routed through litellm (and its cache layers)."""
    import litellm

    return await litellm.acompletion(model=model, messages=messages, **kwargs)
