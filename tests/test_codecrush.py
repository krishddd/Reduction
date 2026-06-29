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


GO = """package main

func add(a int, b int) int {
	sum := a + b
	return sum
}
"""

JAVA = """public class Calc {
    public void run(int a) {
        int sum = a + 1;
        log(sum);
    }
}
"""

CPP = """#include <vector>

int main() {
    int x = compute();
    return x;
}
"""

TS = """export class Service {
  run(id: number): void {
    const data = fetch(id);
    return data;
  }
}
"""


def test_go_func_elided():
    out, lossy = codecrush.crush_code(GO)
    assert lossy
    assert "package main" in out
    assert "func add(a int, b int) int {" in out
    assert "sum := a + b" not in out  # body elided


def test_java_keeps_class_and_method_signature():
    out, lossy = codecrush.crush_code(JAVA)
    assert lossy
    assert "public class Calc {" in out
    assert "public void run(int a) {" in out  # method signature kept
    assert "int sum = a + 1;" not in out  # method body elided


def test_cpp_keeps_includes():
    out, lossy = codecrush.crush_code(CPP)
    assert "#include <vector>" in out
    assert lossy
    assert "int x = compute();" not in out


def test_ts_class_compressed():
    out, lossy = codecrush.crush_code(TS)
    assert lossy
    assert "export class Service {" in out
    assert len(out) < len(TS)


def test_guess_language_new_langs():
    assert codecrush.guess_language(GO) == "go"
    assert codecrush.guess_language(JAVA) == "java"
