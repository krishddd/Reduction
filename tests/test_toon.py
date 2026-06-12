import json

from reduction.layers import toon


def test_uniform_array_tabular():
    data = [
        {"id": 1, "name": "alice", "role": "admin"},
        {"id": 2, "name": "bob", "role": "user"},
    ]
    assert toon.encode(data) == "items[2]{id,name,role}:\n  1,alice,admin\n  2,bob,user"


def test_nested_object_with_uniform_array():
    data = {
        "summary": "scan ok",
        "vulns": [
            {"cve": "CVE-2026-1", "sev": "high"},
            {"cve": "CVE-2026-2", "sev": "low"},
        ],
    }
    out = toon.encode(data)
    assert "summary: scan ok" in out
    assert "vulns[2]{cve,sev}:" in out
    assert "  CVE-2026-1,high" in out


def test_non_uniform_array_falls_back_to_json():
    out = toon.encode({"mixed": [{"a": 1}, {"b": 2}]})
    assert 'mixed: [{"a":1},{"b":2}]' in out


def test_values_with_commas_are_quoted():
    out = toon.encode([{"id": 1, "msg": "hello, world"}])
    assert '"hello, world"' in out


def test_scalar_types():
    out = toon.encode({"ok": True, "count": 3, "ratio": 0.5, "missing": None})
    assert "ok: true" in out
    assert "count: 3" in out
    assert "ratio: 0.5" in out
    assert "missing: null" in out


def test_toon_is_smaller_than_json():
    data = [{"id": i, "name": f"user{i}", "active": True} for i in range(50)]
    assert len(toon.encode(data)) < len(json.dumps(data))


def test_is_uniform_array_rejects_nested_values():
    assert not toon.is_uniform_array([{"a": {"b": 1}}])
    assert not toon.is_uniform_array([])
    assert not toon.is_uniform_array("not a list")
