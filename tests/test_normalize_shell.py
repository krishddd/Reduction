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
