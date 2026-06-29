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


def test_fit_command(tmp_path, capsys):
    big = tmp_path / "big.json"
    big.write_text(json.dumps([{"id": i, "host": f"h-{i}"} for i in range(300)]))
    small = tmp_path / "note.txt"
    small.write_text("deployment uses prod profile")
    store = tmp_path / "ccr.json"
    rc = main(
        ["fit", str(small), str(big), "--budget", "120", "--store", str(store), "--query", "deploy"]
    )
    out = capsys.readouterr()
    assert rc == 0
    assert "[fit]" in out.err
    assert "chunks kept" in out.err


def test_history_command(tmp_path, capsys):
    big = json.dumps([{"id": i, "ok": True} for i in range(300)])
    msgs = tmp_path / "convo.json"
    msgs.write_text(
        json.dumps(
            [
                {"role": "user", "content": "scan"},
                {"role": "tool", "content": big},
                {"role": "assistant", "content": "done"},
                {"role": "user", "content": "next"},
            ]
        )
    )
    store = tmp_path / "ccr.json"
    rc = main(["history", str(msgs), "--keep-last", "2", "--store", str(store)])
    out = capsys.readouterr()
    assert rc == 0
    assert "[history]" in out.err
    assert "compressed" in out.err
    result = json.loads(out.out)
    assert isinstance(result, list) and len(result) == 4


def test_history_rejects_non_list(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"role": "user"}))
    rc = main(["history", str(bad)])
    assert rc == 1


def test_effort_command(capsys):
    rc = main(["effort", "debug", "why", "the", "deploy", "fails"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "level: high" in out
    rc = main(["effort", "read the config file"])
    assert "level: minimal" in capsys.readouterr().out
