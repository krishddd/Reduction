"""Tests for the L2/L3 testability fixes and the docstring-safe code heuristic."""

from reduction.layers import codecrush, compress, semantic_cache

# --- L2: injectable compressor + min-chars gate ---


class _FakeCompressor:
    def __init__(self):
        self.calls = []

    def compress_prompt(self, text, rate):
        self.calls.append((len(text), rate))
        return {"compressed_prompt": text[: len(text) // 2]}


def test_compress_documents_respects_min_chars_gate():
    fake = _FakeCompressor()
    small = "x" * 100
    big = "y" * 5000
    out = compress.compress_documents([small, big], rate=0.4, compressor=fake)
    assert out[0] == small  # below gate, untouched
    assert len(out[1]) < len(big)  # compressed
    assert fake.calls == [(5000, 0.4)]  # only the big doc, with the given rate


def test_compress_documents_noop_without_compressor(monkeypatch):
    monkeypatch.setattr(compress, "_get_compressor", lambda: None)
    docs = ["a" * 5000]
    assert compress.compress_documents(docs) == docs


# --- L3: pure cache-params builder ---


def test_build_cache_params_none_without_host(monkeypatch):
    monkeypatch.delenv("REDIS_HOST", raising=False)
    assert semantic_cache.build_cache_params() is None


def test_build_cache_params_shape():
    params = semantic_cache.build_cache_params(host="localhost", port=6380, threshold=0.95)
    assert params["type"] == "redis-semantic"
    assert params["host"] == "localhost"
    assert params["port"] == 6380
    assert params["similarity_threshold"] == 0.95
    assert params["redis_semantic_cache_embedding_model"] == "text-embedding-3-small"


# --- codecrush: docstring safety ---


def test_heuristic_does_not_mangle_docstrings():
    src = '''def real():
    """A docstring that mentions def fake(): and class Fake: inside it.

    More lines here describing behavior in prose.
    """
    x = compute()
    return x
'''
    out, lossy = codecrush.crush_code_heuristic(src)
    assert "def real():" in out
    assert lossy
    # The fake signatures inside the docstring must NOT appear as kept code lines
    # (they were part of an elided/quoted body, not treated as real defs).
    assert "def fake()" not in out
    assert "class Fake" not in out


def test_guess_language():
    assert codecrush.guess_language("def f():\n    pass\nimport os") == "python"
    assert codecrush.guess_language("fn main() {\n    let x = 1;\n}\nuse std;") == "rust"
