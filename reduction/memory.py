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

import array
import json
import math
import os
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


def _vec_to_blob(vec: list[float]) -> bytes:
    return array.array("f", vec).tobytes()


def _blob_to_vec(blob: bytes) -> list[float]:
    a = array.array("f")
    a.frombytes(blob)
    return list(a)


def _load_vec(raw) -> list[float]:
    # Current rows store float32 blobs; tolerate legacy JSON-text rows.
    if isinstance(raw, bytes | bytearray):
        return _blob_to_vec(bytes(raw))
    return json.loads(raw)


class _AnnIndex:
    """Optional hnswlib cosine index. Falls back to None when hnswlib absent.

    Labels are dense ints assigned in insertion order; ``_meta[label]`` holds
    the (row id, text, metadata) so a search maps back to the stored row. The
    index can be persisted with ``save`` / ``load`` so it isn't rebuilt from
    scratch on every open.
    """

    def __init__(self) -> None:
        self._index = None
        self._meta: list[tuple[int, str, dict]] = []
        self._count = 0
        self._dim: int | None = None

    def _new_index(self, dim: int, capacity: int):
        import hnswlib

        idx = hnswlib.Index(space="cosine", dim=dim)
        idx.init_index(max_elements=max(capacity, 1024), ef_construction=200, M=16)
        idx.set_ef(64)
        return idx

    def build(self, vectors: list[list[float]], metas: list[tuple[int, str, dict]]) -> None:
        if not vectors:
            return
        self._dim = len(vectors[0])
        self._index = self._new_index(self._dim, len(vectors))
        self._index.add_items(vectors, list(range(len(vectors))))
        self._meta = list(metas)
        self._count = len(vectors)

    def load(self, path: str, dim: int, metas: list[tuple[int, str, dict]]) -> None:
        import hnswlib

        idx = hnswlib.Index(space="cosine", dim=dim)
        idx.load_index(path, max_elements=max(len(metas), 1024))
        if idx.get_current_count() != len(metas):
            raise ValueError("stale ANN index: element count != row count")
        idx.set_ef(64)
        self._index, self._dim, self._meta, self._count = idx, dim, list(metas), len(metas)

    def add(self, vec: list[float], meta: tuple[int, str, dict]) -> None:
        if self._index is None:
            self._dim = len(vec)
            self._index = self._new_index(self._dim, 1024)
        if self._count >= self._index.get_max_elements():
            self._index.resize_index(self._index.get_max_elements() * 2)
        self._index.add_items([vec], [self._count])
        self._meta.append(meta)
        self._count += 1

    def save(self, path: str) -> None:
        if self._index is not None:
            self._index.save_index(path)

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
            "embedding BLOB, metadata TEXT)"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_ns ON memory(namespace)")
        self._conn.commit()
        self._ann_path = f"{path}.{namespace}.hnsw"

        # Build/load an ANN index when hnswlib is available.
        self._ann: _AnnIndex | None = _AnnIndex() if _hnswlib_available() else None
        if self._ann is not None:
            rows = self._conn.execute(
                "SELECT id, text, embedding, metadata FROM memory WHERE namespace=? ORDER BY id",
                (self.namespace,),
            ).fetchall()
            vectors = [_load_vec(emb) for _, _, emb, _ in rows]
            metas = [(rid, text, json.loads(meta)) for rid, text, _, meta in rows]
            loaded = False
            if vectors and os.path.exists(self._ann_path):
                try:
                    self._ann.load(self._ann_path, len(vectors[0]), metas)
                    loaded = True
                except Exception:
                    loaded = False
            if not loaded and vectors:
                self._ann.build(vectors, metas)

    def add(self, text: str, metadata: dict | None = None) -> int:
        vec = self.embed(text)
        meta = metadata or {}
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO memory (namespace, text, embedding, metadata) VALUES (?,?,?,?)",
                (self.namespace, text, _vec_to_blob(vec), json.dumps(meta)),
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
                id=rid, text=text, score=cosine(qvec, _load_vec(emb)), metadata=json.loads(meta)
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
            # hnswlib has no stable per-item delete — drop the index entirely.
            self._ann = _AnnIndex() if _hnswlib_available() else None
            if os.path.exists(self._ann_path):
                try:
                    os.remove(self._ann_path)
                except OSError:
                    pass

    def close(self) -> None:
        # Persist the ANN index so the next open loads instead of rebuilding.
        if self._ann is not None:
            try:
                self._ann.save(self._ann_path)
            except Exception:
                pass
        self._conn.close()
