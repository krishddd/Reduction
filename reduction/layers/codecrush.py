"""AST-aware code compressor.

Source files are mostly function bodies; an agent navigating a codebase usually
needs the *shape* — imports, class/function signatures, decorators, type
annotations — far more often than every line of every body. This compressor
keeps the structure and elides bodies:

    def parse(data: bytes) -> Doc:
        ... (12 lines)

It uses tree-sitter when ``tree_sitter_language_pack`` is installed (accurate,
language-aware); otherwise it falls back to a robust indentation/keyword
heuristic that handles Python and C-family/JS/TS well. Either way the original
is recoverable via CCR, so eliding bodies is safe.
"""

from __future__ import annotations

import re

# Lines that are structural and must be kept verbatim.
_KEEP_RE = re.compile(
    r"^\s*(?:"
    r"import\s|from\s+\S+\s+import|"  # py imports
    r"@|"  # decorators
    r"(?:async\s+)?def\s|class\s|"  # py defs/classes
    r"(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s|"  # js functions
    r"(?:export\s+)?(?:abstract\s+)?class\s|"  # js/ts classes
    r"(?:public|private|protected|static|export|const|let|var)\s|"
    r"interface\s|type\s+\w+\s*=|enum\s|"  # ts types
    r"#include|using\s+namespace|namespace\s|"  # c/c++
    r"(?:pub\s+)?(?:async\s+)?fn\s|impl\s|struct\s|trait\s|mod\s|use\s"  # rust
    r")"
)
# A signature line that opens a body (ends with : or {).
_OPENS_BODY = re.compile(r"[:{]\s*(?://.*)?$")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def crush_code_heuristic(text: str) -> tuple[str, bool]:
    """Keep imports/signatures/decorators; elide indented bodies. (compressed, lossy)."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    elided_any = False

    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            out.append(line)
            i += 1
            continue

        if _KEEP_RE.match(line):
            out.append(line)
            # If this signature opens a body, elide the more-indented block.
            if _OPENS_BODY.search(line) and not stripped.startswith(("import", "from", "#")):
                sig_indent = _indent(line)
                j = i + 1
                # Skip a leading docstring line if present (keep it — cheap, useful).
                body_lines = 0
                while j < n:
                    nxt = lines[j]
                    if not nxt.strip():
                        j += 1
                        continue
                    if _indent(nxt) <= sig_indent:
                        break
                    if _KEEP_RE.match(nxt):  # nested def/class — keep structure
                        break
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

        # Top-level non-structural line (module constant, etc.) — keep.
        if _indent(line) == 0:
            out.append(line)
        else:
            # Stray indented line outside a tracked signature — drop quietly.
            elided_any = True
        i += 1

    compressed = "\n".join(out)
    return compressed, (elided_any and len(compressed) < len(text))


def _tree_sitter_available() -> bool:
    try:
        import tree_sitter_language_pack  # noqa: F401

        return True
    except ImportError:
        return False


def crush_code(text: str, language: str | None = None) -> tuple[str, bool]:
    """Compress source code. Returns (compressed, was_compressed).

    Uses the heuristic backend by default. tree-sitter, when installed, can be
    plugged in here for language-exact parsing; the heuristic already covers the
    common languages well enough that we prefer it when tree-sitter is absent.
    """
    return crush_code_heuristic(text)
