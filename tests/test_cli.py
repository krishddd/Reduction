import json

from reduction.cli import main


def test_compress_command(tmp_path, capsys):
    src = tmp_path / "big.json"
    src.write_text(json.dumps({"items": [{"id": i, "ok": True} for i in range(100)]}))
    store = tmp_path / "ccr.json"
    rc = main(["compress", str(src), "--store", str(store)])
    out = capsys.readouterr()
    assert rc == 0
    assert "items total" in out.out
    assert "saved" in out.err
    assert store.exists()


def test_compress_then_retrieve_roundtrip(tmp_path, capsys):
    src = tmp_path / "big.json"
    payload = json.dumps({"items": [{"id": i, "ok": True} for i in range(200)]})
    src.write_text(payload)
    store = tmp_path / "ccr.json"
    main(["compress", str(src), "--store", str(store)])
    err = capsys.readouterr().err
    ref = err.split("ref=")[1].strip()

    rc = main(["retrieve", ref, "--store", str(store)])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.strip() == payload


def test_wrap_snippets(capsys):
    for agent in ("anthropic", "openai", "odysseus", "mcp"):
        assert main(["wrap", agent]) == 0
    out = capsys.readouterr().out
    assert "OptimizedAnthropic" in out


def test_stats_missing_file(tmp_path, capsys):
    rc = main(["stats", str(tmp_path / "nope.json")])
    assert rc == 1


def test_demo_runs(capsys):
    rc = main(["demo"])
    assert rc == 0
    assert "Token Savings" in capsys.readouterr().out
