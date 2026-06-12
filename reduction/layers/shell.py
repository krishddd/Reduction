"""Layer 1 — shell / tool-output filtering.

When an agent runs a command (``git status``, ``pytest``, ``npm install``)
and feeds the output back to the model, that raw output is pure token waste.
This layer shells out to the ``zap`` / RTK binary when present (60-90%
reduction across 12 strategies). If zap is not installed it falls back to a
built-in heuristic filter so the layer always does *something*.

zap is a separately-installed Rust binary (https://github.com/rtk-ai/rtk);
the pipeline does not vendor it.
"""

from __future__ import annotations

import shutil
import subprocess

from reduction.layers.normalize import normalize


def zap_available(binary: str = "zap") -> bool:
    return shutil.which(binary) is not None


def filter_with_zap(command: list[str], binary: str = "zap", timeout: float = 15.0) -> str | None:
    """Run ``zap <command>`` and return its filtered stdout, or None on failure."""
    if not zap_available(binary):
        return None
    try:
        proc = subprocess.run(
            [binary, *command],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.stdout
    except (subprocess.SubprocessError, OSError):
        return None


def builtin_filter(output: str, *, max_lines: int = 200) -> str:
    """Heuristic fallback when zap is absent.

    Dedupe consecutive lines, drop noise (progress bars / blank runs), and
    head+tail truncate very long output with an elision marker.
    """
    text = normalize(output, strip=True, dedupe=True)
    lines = [ln for ln in text.split("\n") if "\r" not in ln]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    head = lines[: max_lines // 2]
    tail = lines[-max_lines // 2 :]
    elided = len(lines) - len(head) - len(tail)
    return "\n".join([*head, f"... ({elided} lines elided by reduction) ...", *tail])


def filter_tool_output(
    output: str,
    *,
    command: list[str] | None = None,
    binary: str = "zap",
    max_lines: int = 200,
) -> str:
    """Best-effort filter for tool/command output.

    If ``command`` is given and zap is installed, re-runs it through zap for
    structure-aware filtering. Otherwise applies the built-in heuristic to the
    already-captured ``output``.
    """
    if command:
        filtered = filter_with_zap(command, binary=binary)
        if filtered is not None:
            return filtered
    return builtin_filter(output, max_lines=max_lines)
