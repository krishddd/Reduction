"""AST-aware code compressor.

Source files are mostly function bodies; an agent navigating a codebase usually
needs the *shape* — imports, class/function signatures, decorators, type
annotations — far more often than every line of every body. This compressor
keeps the structure and elides bodies:

    def parse(data: bytes) -> Doc:
        ... (12 lines)

Two backends:

  * **tree-sitter** (when ``tree_sitter_language_pack`` is installed): a real
    parser. Body nodes of function/class definitions are located by AST node
    type and their line spans elided — accurate across languages.
  * **heuristic** (always available, dependency-free): an indentation/keyword
    scanner that tracks triple-quoted strings so docstrings and string literals
    containing ``def``/``class`` are not mistaken for code.

Either way the original is recoverable via CCR, so eliding bodies is safe.
"""

from __future__ import annotations

import re

# Lines that are structural and must be kept verbatim (heuristic backend).
_KEEP_RE = re.compile(
    r"^\s*(?:"
    r"import\s|from\s+\S+\s+import|"  # py imports
    r"@|"  # decorators
    r"(?:async\s+)?def\s|class\s|"  # py defs/classes
    r"(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s|"  # js functions
    r"(?:export\s+)?(?:abstract\s+)?class\s|"  # js/ts classes
    r"(?:public|private|protected|static|export|const|let|var)\s|"
    r"interface\s|type\s+\w+\s*=|enum\s|"  # ts types
    r"#include|#define|using\s+namespace|namespace\s|template\s*<|"  # c/c++
    r"package\s|func\s|"  # go (also matches go's func)
    r"(?:pub\s+)?(?:async\s+)?fn\s|impl\s|struct\s|trait\s|mod\s|use\s"  # rust
    r")"
)
# A signature line that opens a body (ends with : or {).
_OPENS_BODY = re.compile(r"[:{]\s*(?://.*)?$")
_TRIPLE = ('"""', "'''")

# Map a guessed language to the tree-sitter-language-pack name. Order matters
# only for ties; the language with the most signature-line hits wins.
_LANG_HINTS = {
    "python": re.compile(r"^\s*(?:def |class |import |from \S+ import)", re.MULTILINE),
    "rust": re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn |^\s*impl |^\s*use ", re.MULTILINE),
    "go": re.compile(r"^\s*(?:package |func |import \(|type \w+ struct)", re.MULTILINE),
    "java": re.compile(
        r"^\s*(?:public |private |protected )?(?:static |final |abstract )*"
        r"(?:class |interface |void |[A-Z]\w*(?:<[^>]*>)? \w+\s*\()|^\s*package \w",
        re.MULTILINE,
    ),
    "cpp": re.compile(
        r"^\s*(?:#include|#define|using namespace|namespace |template\s*<|"
        r"std::|class \w+\s*[:{])",
        re.MULTILINE,
    ),
    "typescript": re.compile(
        r"^\s*(?:interface \w|type \w+\s*=|enum \w|"
        r"(?:export )?(?:abstract )?class \w|\w+\s*:\s*\w+(?:\[\])?\s*[=;,)])",
        re.MULTILINE,
    ),
    # JavaScript last: its signatures (function/const/=>) also appear in TS, so
    # TS-specific hits above should win when present.
    "javascript": re.compile(r"(?:^|\s)(?:function |const |let |=> )", re.MULTILINE),
}


def guess_language(text: str) -> str | None:
    if text.lstrip().startswith("#!") and "python" in text.splitlines()[0]:
        return "python"
    best, best_hits = None, 0
    for lang, rx in _LANG_HINTS.items():
        hits = len(rx.findall(text))
        if hits > best_hits:
            best, best_hits = lang, hits
    return best


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def crush_code_heuristic(text: str) -> tuple[str, bool]:
    """Keep imports/signatures/decorators; elide indented bodies. (compressed, lossy).

    Tracks triple-quoted strings so a docstring line like ``def f():`` inside a
    string is never treated as a real signature.
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    elided_any = False
    in_triple: str | None = None

    def toggles_triple(line: str, state: str | None) -> str | None:
        # Return the triple-quote state after this line (very small scanner).
        for q in _TRIPLE:
            count = line.count(q)
            if count:
                # odd number of a given triple flips state if it matches/open
                for _ in range(count):
                    if state is None:
                        state = q
                    elif state == q:
                        state = None
        return state

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if in_triple is not None:
            out.append(line)
            in_triple = toggles_triple(line, in_triple)
            i += 1
            continue

        if not stripped:
            out.append(line)
            i += 1
            continue

        new_state = toggles_triple(line, None)

        if _KEEP_RE.match(line) and new_state is None:
            out.append(line)
            if _OPENS_BODY.search(line) and not stripped.startswith(("import", "from", "#")):
                sig_indent = _indent(line)
                j = i + 1
                body_lines = 0
                body_triple: str | None = None
                while j < n:
                    nxt = lines[j]
                    if body_triple is not None:
                        body_triple = toggles_triple(nxt, body_triple)
                        body_lines += 1
                        j += 1
                        continue
                    if not nxt.strip():
                        j += 1
                        continue
                    if _indent(nxt) <= sig_indent:
                        break
                    if _KEEP_RE.match(nxt) and toggles_triple(nxt, None) is None:
                        break  # nested def/class — keep structure
                    body_triple = toggles_triple(nxt, None)
                    body_lines += 1
                    j += 1
                if body_lines > 0:
                    pad = " " * (sig_indent + 4)
                    out.append(f"{pad}... ({body_lines} lines)")
                    elided_any = True
                i = j
                continue
            i += 1
            continue

        # Non-structural line. Keep top-level; drop stray indented lines.
        if _indent(line) == 0:
            out.append(line)
        else:
            elided_any = True
        in_triple = new_state
        i += 1

    compressed = "\n".join(out)
    return compressed, (elided_any and len(compressed) < len(text))


def _tree_sitter_parser(language: str):
    from tree_sitter_language_pack import get_parser

    return get_parser(language)


def crush_code_treesitter(text: str, language: str) -> tuple[str, bool]:
    """Elide function/class body node spans using a real parser. (compressed, lossy)."""
    parser = _tree_sitter_parser(language)
    data = text.encode("utf-8")
    tree = parser.parse(data)
    lines = text.split("\n")

    body_types = {"block", "statement_block", "function_body", "compound_statement"}
    def_types = {
        "function_definition",
        "function_declaration",
        "method_definition",
        "method_declaration",  # go / java
        "constructor_declaration",  # java
        "function_item",
        "class_definition",
        "class_declaration",
        "impl_item",
    }
    # Collect (start_row, end_row) spans of body nodes whose parent is a def.
    spans: list[tuple[int, int]] = []

    def walk(node) -> None:
        if node.type in body_types and node.parent and node.parent.type in def_types:
            sr, er = node.start_point[0], node.end_point[0]
            if er - sr >= 2:  # only elide multi-line bodies
                spans.append((sr, er))
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    if not spans:
        return text, False

    # Replace inner body lines (keep the signature line + closing brace line).
    drop: dict[int, int] = {}  # first inner line -> count elided
    blocked: set[int] = set()
    for sr, er in spans:
        inner = list(range(sr + 1, er))  # keep sr (sig/brace) and er (close)
        if inner:
            drop[inner[0]] = len(inner)
            blocked.update(inner[1:])

    out: list[str] = []
    for idx, line in enumerate(lines):
        if idx in drop:
            pad = " " * (_indent(line) if line.strip() else 4)
            out.append(f"{pad}... ({drop[idx]} lines)")
        elif idx in blocked:
            continue
        else:
            out.append(line)
    compressed = "\n".join(out)
    return compressed, len(compressed) < len(text)


def _tree_sitter_available() -> bool:
    try:
        import tree_sitter_language_pack  # noqa: F401

        return True
    except ImportError:
        return False


def crush_code(text: str, language: str | None = None) -> tuple[str, bool]:
    """Compress source code. Returns (compressed, was_compressed).

    Uses tree-sitter when available and a language can be determined; otherwise
    the docstring-safe heuristic.
    """
    lang = language or guess_language(text)
    if lang and _tree_sitter_available():
        try:
            return crush_code_treesitter(text, lang)
        except Exception:
            pass  # parser/language missing -> heuristic
    return crush_code_heuristic(text)
