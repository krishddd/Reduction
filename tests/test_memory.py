from reduction.memory import Memory, cosine, hashing_embedding


def test_hashing_embedding_is_stable_and_normalized():
    v1 = hashing_embedding("the quick brown fox")
    v2 = hashing_embedding("the quick brown fox")
    assert v1 == v2  # deterministic across calls (crc32, not salted hash)
    assert abs(sum(x * x for x in v1) ** 0.5 - 1.0) < 1e-6


def test_add_and_search_relevance(tmp_path):
    mem = Memory(tmp_path / "m.db")
    mem.add("how to configure the semantic cache threshold")
    mem.add("the cat sat on the mat in the sun")
    mem.add("redis vector similarity search setup guide")
    hits = mem.search("configure cache threshold", k=2)
    assert len(hits) == 2
    assert "cache" in hits[0].text  # most relevant first
    assert hits[0].score >= hits[1].score


def test_namespaces_isolated(tmp_path):
    db = tmp_path / "m.db"
    a = Memory(db, namespace="proj-a")
    b = Memory(db, namespace="proj-b")
    a.add("secret from project a")
    assert a.count() == 1
    assert b.count() == 0  # no cross-project bleed
    assert b.search("secret", k=5) == []


def test_persists_across_instances(tmp_path):
    db = tmp_path / "m.db"
    m1 = Memory(db)
    m1.add("durable memory entry about tokenization", metadata={"tag": "x"})
    m1.close()
    m2 = Memory(db)  # reopen
    hits = m2.search("tokenization", k=1)
    assert hits and hits[0].metadata["tag"] == "x"


def test_cosine_basics():
    assert cosine([1, 0], [1, 0]) == 1.0
    assert abs(cosine([1, 0], [0, 1])) < 1e-9
