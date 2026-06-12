import json

from reduction.metrics import Metrics, estimate_tokens


def test_persist_does_not_crash_and_roundtrips(tmp_path):
    m = Metrics()
    m.record_input("a" * 100, "a" * 40, layer="shell")
    m.record_call(billed_input_tokens=10, cache_read_tokens=4000, cache_write_tokens=50)
    out = tmp_path / "metrics.json"
    m.persist(out)  # previously raised: cannot pickle threading.Lock
    data = json.loads(out.read_text())
    assert data["calls"] == 1
    assert data["cache_read_tokens"] == 4000
    assert data["cache_write_tokens"] == 50
    assert data["billed_input_tokens"] == 10


def test_input_savings_tracked():
    m = Metrics()
    m.record_input("x" * 400, "x" * 100, layer="normalize")
    s = m.summary()
    assert s["input_tokens_raw"] > s["input_tokens_optimized"]
    assert s["tokens_saved"] > 0
    assert s["savings_pct"] > 0


def test_estimate_tokens_nonzero_for_text():
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world this is a sentence") > 0


def test_render_is_ascii_only():
    m = Metrics()
    m.record_input("y" * 80, "y" * 20, layer="shell")
    text = m.render()
    assert text.isascii()
    assert "Token Savings" in text
