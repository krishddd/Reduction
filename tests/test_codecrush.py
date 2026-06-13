from reduction.layers import codecrush
from reduction.layers.detect import ContentType, detect

PY = """import os
from typing import Any


@cache
def parse(data: bytes) -> dict:
    result = {}
    for line in data.splitlines():
        result[line] = len(line)
    return result


class Widget:
    def render(self) -> str:
        body = self.compute()
        return f"<{body}>"
"""


def test_keeps_signatures_elides_bodies():
    out, lossy = codecrush.crush_code(PY)
    assert lossy
    assert "import os" in out
    assert "def parse(data: bytes) -> dict:" in out
    assert "@cache" in out
    assert "class Widget:" in out
    assert "def render(self) -> str:" in out
    # bodies elided
    assert "result[line]" not in out
    assert "... (" in out
    assert len(out) < len(PY)


def test_detect_routes_code():
    assert detect(PY) is ContentType.CODE


def test_small_code_no_bodies_unchanged():
    src = "import sys\nx = 1\n"
    out, lossy = codecrush.crush_code(src)
    assert "import sys" in out
    assert not lossy  # nothing to elide


def test_js_function():
    js = "export function add(a, b) {\n  const sum = a + b;\n  return sum;\n}\n"
    out, lossy = codecrush.crush_code(js)
    assert "export function add(a, b) {" in out
    assert lossy
