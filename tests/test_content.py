import json

from reduction.ccr import CompressionStore
from reduction.content import compress_content
from reduction.layers import diffstat, jsoncrush, logcrush
from reduction.layers.detect import ContentType, detect


def test_detect_json():
    assert detect('{"a": 1, "b": [1,2,3]}') is ContentType.JSON


def test_detect_diff():
    diff = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+new"
    assert detect(diff) is ContentType.DIFF


def test_detect_log():
    log = "\n".join(f"2026-01-01 12:00:0{i} ERROR boom" for i in range(10))
    assert detect(log) is ContentType.LOG


def test_detect_code():
    assert detect("def foo():\n    return 1\n\nclass Bar:\n    pass") is ContentType.CODE


def test_jsoncrush_samples_large_uniform_array():
    data = {"items": [{"id": i, "ok": True} for i in range(100)]}
    out, changed = jsoncrush.crush_json(json.dumps(data))
    assert changed
    assert len(out) < len(json.dumps(data))
    assert "100 items total" in out


def test_jsoncrush_small_array_untouched_or_compact():
    data = {"items": [{"id": 1}, {"id": 2}]}
    out, changed = jsoncrush.crush_json(json.dumps(data, indent=2))
    # compacting still counts as compressed, but no sampling marker
    assert "items total" not in out


def test_diffstat_summary():
    diff = (
        "diff --git a/a.py b/a.py\n+++ b/a.py\n+line\n+line2\n-old\n"
        "diff --git a/b.py b/b.py\n+++ b/b.py\n+x\n"
    )
    out, changed = diffstat.crush_diff(diff)
    assert changed
    assert "2 files" in out
    assert "+3/-1" in out


def test_logcrush_keeps_errors():
    lines = ["2026-01-01 INFO ok"] * 200 + ["2026-01-01 ERROR boom"]
    out, changed = logcrush.crush_log("\n".join(lines))
    assert changed
    assert "ERROR boom" in out
    assert len(out) < len("\n".join(lines))


def test_compress_content_attaches_ccr_ref():
    store = CompressionStore()
    data = json.dumps({"items": [{"id": i, "ok": True} for i in range(200)]})
    result = compress_content(data, ccr=True, store=store)
    assert result.content_type is ContentType.JSON
    assert result.ref is not None
    assert result.tokens_after < result.tokens_before
    assert store.get(result.ref) == data  # original is recoverable


def test_compress_content_skips_tiny_input():
    result = compress_content("short", ccr=True, store=CompressionStore())
    assert result.ref is None
    assert result.text == "short"
