from reduction.layers import normalize, shell


def test_strip_whitespace_collapses_blank_runs():
    assert normalize.strip_whitespace("a   \n\n\n\nb  ") == "a\n\nb"


def test_dedupe_consecutive_lines():
    out = normalize.dedupe_lines("err\nerr\nerr\nok")
    assert out == "err  (x3)\nok"


def test_normalize_combines():
    assert normalize.normalize("x\nx\n\n\n\n") == "x  (x2)"


def test_builtin_filter_truncates_long_output():
    big = "\n".join(f"line{i}" for i in range(1000))
    out = shell.builtin_filter(big, max_lines=100)
    assert "lines elided by reduction" in out
    assert len(out.split("\n")) <= 102


def test_builtin_filter_keeps_short_output():
    out = shell.builtin_filter("a\nb\nc", max_lines=100)
    assert out == "a\nb\nc"


def test_is_safe_to_rerun_allows_read_only_commands():
    assert shell.is_safe_to_rerun(["git", "status"])
    assert shell.is_safe_to_rerun(["git", "-C", "repo", "log", "--oneline"])
    assert shell.is_safe_to_rerun(["git", "stash", "list"])
    assert shell.is_safe_to_rerun(["ls", "-la"])
    assert shell.is_safe_to_rerun(["/usr/bin/grep", "-r", "foo"])
    assert shell.is_safe_to_rerun(["C:\\Program Files\\Git\\git.exe", "diff"])


def test_is_safe_to_rerun_blocks_side_effects():
    assert not shell.is_safe_to_rerun([])
    assert not shell.is_safe_to_rerun(["rm", "-rf", "/tmp/x"])
    assert not shell.is_safe_to_rerun(["git", "push"])
    assert not shell.is_safe_to_rerun(["git", "stash"])  # bare stash mutates
    assert not shell.is_safe_to_rerun(["pytest", "-q"])
    assert not shell.is_safe_to_rerun(["npm", "install"])


def test_filter_tool_output_never_reruns_unsafe_commands(monkeypatch):
    # zap "installed" and would blow up if invoked — unsafe commands must not
    # reach it, only be filtered from the captured output.
    monkeypatch.setattr(shell, "zap_available", lambda binary="zap": True)

    def _boom(command, binary="zap", timeout=15.0):
        raise AssertionError(f"re-executed unsafe command: {command}")

    monkeypatch.setattr(shell, "filter_with_zap", _boom)
    out = shell.filter_tool_output("output line\n" * 5, command=["pytest", "-q"])
    assert "output line" in out
