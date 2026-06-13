"""Content-type detection for compression routing.

Headroom uses Google's Magika (an ML model) for this; we use fast, dependency-
free heuristics that cover the cases that matter for agent I/O — JSON, diffs,
logs, code, markdown, text. If ``magika`` happens to be installed it is used as
a tie-breaker for ambiguous content.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum


class ContentType(StrEnum):
    JSON = "json"
    DIFF = "diff"
    LOG = "log"
    CODE = "code"
    MARKDOWN = "markdown"
    TEXT = "text"


_DIFF_RE = re.compile(r"^(diff --git |--- |\+\+\+ |@@ )", re.MULTILINE)
_LOG_RE = re.compile(
    r"\b(ERROR|WARN|WARNING|INFO|DEBUG|TRACE|FATAL)\b|"
    r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}",
)
_CODE_RE = re.compile(
    r"^\s*(def |class |import |from \w+ import |function |const |let |var |"
    r"public |private |#include|package |fn |impl )",
    re.MULTILINE,
)
_MD_RE = re.compile(r"^(#{1,6} |\s*[-*] |\d+\. |```)", re.MULTILINE)


def _looks_like_json(text: str) -> bool:
    s = text.strip()
    if not s or s[0] not in "{[":
        return False
    try:
        json.loads(s)
        return True
    except ValueError:
        return False


def detect(text: str) -> ContentType:
    """Classify ``text`` into a content category for routing."""
    if not text or not text.strip():
        return ContentType.TEXT
    if _looks_like_json(text):
        return ContentType.JSON
    if _DIFF_RE.search(text):
        return ContentType.DIFF

    # Score the remaining ambiguous categories by match density.
    log_hits = len(_LOG_RE.findall(text))
    code_hits = len(_CODE_RE.findall(text))
    md_hits = len(_MD_RE.findall(text))
    line_count = text.count("\n") + 1

    if log_hits and log_hits >= line_count * 0.2:
        return ContentType.LOG
    if code_hits and code_hits >= max(log_hits, md_hits):
        return ContentType.CODE
    if md_hits and md_hits >= line_count * 0.3:
        return ContentType.MARKDOWN
    if log_hits:
        return ContentType.LOG
    return ContentType.TEXT
