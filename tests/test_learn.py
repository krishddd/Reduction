from reduction.learn import (
    BEGIN_MARKER,
    END_MARKER,
    FailureLog,
    write_corrections,
)


def test_record_and_derive(tmp_path):
    log = FailureLog(tmp_path / "f.jsonl")
    for _ in range(3):
        log.record(
            context="run tests", action="pytest -k foo", outcome="fail", error="no tests ran"
        )
    log.record(context="build", action="make", outcome="success")
    log.record(context="once", action="rare-cmd", outcome="fail", error="boom")

    corrections = log.derive_corrections(min_occurrences=2)
    assert len(corrections) == 1
    assert corrections[0].occurrences == 3
    assert "pytest -k foo" in corrections[0].guidance


def test_write_corrections_managed_block(tmp_path):
    log = FailureLog(tmp_path / "f.jsonl")
    for _ in range(2):
        log.record(context="c", action="bad-cmd", outcome="fail", error="nope")
    corrections = log.derive_corrections(min_occurrences=2)

    claude = tmp_path / "CLAUDE.md"
    claude.write_text("# Project\n\nExisting notes.\n", encoding="utf-8")
    assert write_corrections(claude, corrections)
    text = claude.read_text(encoding="utf-8")
    assert "Existing notes." in text
    assert BEGIN_MARKER in text and END_MARKER in text
    assert "bad-cmd" in text


def test_write_corrections_replaces_block_idempotent(tmp_path):
    log = FailureLog(tmp_path / "f.jsonl")
    for _ in range(2):
        log.record(context="c", action="x", outcome="fail", error="e")
    corrections = log.derive_corrections(min_occurrences=2)
    target = tmp_path / "AGENTS.md"
    write_corrections(target, corrections)
    write_corrections(target, corrections)  # second write must not duplicate
    assert target.read_text(encoding="utf-8").count(BEGIN_MARKER) == 1


def test_no_corrections_returns_false(tmp_path):
    log = FailureLog(tmp_path / "f.jsonl")
    log.record(context="c", action="x", outcome="fail", error="e")
    assert write_corrections(tmp_path / "X.md", log.derive_corrections(min_occurrences=2)) is False


def test_normalize_error_clusters_variants():
    from reduction.learn import normalize_error

    a = normalize_error("Timeout after 30s connecting to 10.0.0.5:8080")
    b = normalize_error("Timeout after 5s connecting to 192.168.1.1:443")
    assert a == b  # numbers/paths/addresses normalized away


def test_variant_wordings_cluster_into_one_correction(tmp_path):
    log = FailureLog(tmp_path / "f.jsonl")
    log.record(context="c", action="curl api", outcome="fail", error="Timeout after 30000ms")
    log.record(context="c", action="curl api", outcome="fail", error="Timeout after 500ms")
    corrections = log.derive_corrections(min_occurrences=2)
    assert len(corrections) == 1
    assert corrections[0].occurrences == 2
