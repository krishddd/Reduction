"""Persistent vector memory for cross-turn / cross-agent recall.

A per-project store of text + embedding + metadata backed by SQLite, with
semantic search. Embeddings come from (in priority order):

  1. a user-supplied ``embed_fn``;
  2. ``sentence-transformers`` if installed;
  3. a dependency-free hashing embedding (deterministic bag-of-tokens) so the
     store works out of the box and in tests.

Search is exact cosine over stored vectors (fine to tens of thousands of rows);
``hnswlib`` is used automatically when installed for larger corpora.

Namespacing: each ``Memory`` is scoped to a ``namespace`` (default "default").
Rows from other namespaces are never returned, so projects don't bleed into
each other even when they share one SQLite file.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

EmbedFn = Callable[[str], list[float]]

_HASH_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def hashing_embedding(text: str, dim: int = _HASH_DIM) -> list[float]:
    """Deterministic, dependency-free embedding: hashed token counts, L2-normed."""
    vec = [0.0] * dim
    for tok in _TOKEN_RE.findall(text.lower()):
        # crc32 is process-stable (unlike str hash) so vectors persist correctly.
        vec[zlib.crc32(tok.encode("utf-8")) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm:
        vec = [v / norm for v in vec]
    return vec


def _default_embed() -> EmbedFn:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")

        def embed(text: str) -> list[float]:
            return model.encode(text, normalize_embeddings=True).tolist()

        return embed
    except Exception:
        return hashing_embedding


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class MemoryHit:
    id: int
    text: str
    score: float
    metadata: dict


class _AnnIndex:
    """Optional hnswlib cosine index. Falls back to None when hnswlib absent.

    Labels are dense ints assigned in insertion order; ``_meta[label]`` holds
    the (row id, text, metadata) so a search maps back to the stored row.
    """

    def __init__(self) -> None:
        self._index = None
        self._meta: list[tuple[int, str, dict]] = []
        self._count = 0

    def _ensure(self, dim: int) -> None:
        if self._index is None:
            import hnswlib

            self._index = hnswlib.Index(space="cosine", dim=dim)
            self._index.init_index(max_elements=1024, ef_construction=200, M=16)
            self._index.set_ef(64)

    def add(self, vec: list[float], meta: tuple[int, str, dict]) -> None:
        self._ensure(len(vec))
        if self._count >= self._index.get_max_elements():
            self._index.resize_index(self._index.get_max_elements() * 2)
        self._index.add_items([vec], [self._count])
        self._meta.append(meta)
        self._count += 1

    def search(self, vec: list[float], k: int) -> list[MemoryHit]:
        if self._index is None or self._count == 0:
            return []
        k = min(k, self._count)
        labels, dists = self._index.knn_query([vec], k=k)
        hits = []
        for label, dist in zip(labels[0], dists[0], strict=False):
            rid, text, meta = self._meta[int(label)]
            hits.append(MemoryHit(id=rid, text=text, score=1.0 - float(dist), metadata=meta))
        return hits


def _hnswlib_available() -> bool:
    try:
        import hnswlib  # noqa: F401

        return True
    except ImportError:
        return False


class Memory:
    def __init__(
        self,
        path: str | Path = "reduction-memory.db",
        *,
        namespace: str = "default",
        embed_fn: EmbedFn | None = None,
    ) -> None:
        self.namespace = namespace
        self.embed = embed_fn or _default_embed()
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS memory ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, namespace TEXT, text TEXT, "
            "embedding TEXT, metadata TEXT)"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_ns ON memory(namespace)")
        self._conn.commit()

        # Build an ANN index from existing rows when hnswlib is available.
        self._ann: _AnnIndex | None = _AnnIndex() if _hnswlib_available() else None
        if self._ann is not None:
            for rid, text, emb, meta in self._conn.execute(
                "SELECT id, text, embedding, metadata FROM memory WHERE namespace=?",
                (self.namespace,),
            ).fetchall():
                self._ann.add(json.loads(emb), (rid, text, json.loads(meta)))

    def add(self, text: str, metadata: dict | None = None) -> int:
        vec = self.embed(text)
        meta = metadata or {}
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO memory (namespace, text, embedding, metadata) VALUES (?,?,?,?)",
                (self.namespace, text, json.dumps(vec), json.dumps(meta)),
            )
            self._conn.commit()
            rid = int(cur.lastrowid)
            if self._ann is not None:
                self._ann.add(vec, (rid, text, meta))
            return rid

    def search(self, query: str, k: int = 5) -> list[MemoryHit]:
        qvec = self.embed(query)
        # Fast path: hnswlib ANN index (cosine) when available.
        if self._ann is not None:
            with self._lock:
                return self._ann.search(qvec, k)
        # Fallback: exact cosine scan over the namespace's rows.
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, text, embedding, metadata FROM memory WHERE namespace=?",
                (self.namespace,),
            ).fetchall()
        hits = [
            MemoryHit(
                id=rid,
                text=text,
                score=cosine(qvec, json.loads(emb)),
                metadata=json.loads(meta),
            )
            for rid, text, emb, meta in rows
        ]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def count(self) -> int:
        with self._lock:
            (n,) = self._conn.execute(
                "SELECT COUNT(*) FROM memory WHERE namespace=?", (self.namespace,)
            ).fetchone()
        return int(n)

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM memory WHERE namespace=?", (self.namespace,))
            self._conn.commit()
            # Rebuild an empty ANN index (hnswlib has no stable per-item delete).
            self._ann = _AnnIndex() if _hnswlib_available() else None

    def close(self) -> None:
        self._conn.close()
