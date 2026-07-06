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

# Programs that are read-only and side-effect free, so re-executing them
# through zap is safe. Anything else must be filtered from the *captured*
# output — re-running a command like ``rm``, ``pytest`` or ``npm install``
# to filter its output would repeat its side effects.
_SAFE_RERUN_PROGRAMS = frozenset(
    {
        "ls", "cat", "head", "tail", "grep", "rg", "find", "wc", "du", "df",
        "ps", "env", "printenv", "pwd", "which", "tree", "file", "stat", "uname",
    }
)  # fmt: skip
_SAFE_GIT_SUBCOMMANDS = frozenset(
    {
        "status", "log", "diff", "show", "branch", "blame", "shortlog",
        "ls-files", "rev-parse", "remote", "tag", "describe", "stash",
    }
)  # fmt: skip


def is_safe_to_rerun(command: list[str]) -> bool:
    """True when re-executing ``command`` cannot cause side effects.

    Conservative allowlist: read-only inspection commands and read-only git
    subcommands. ``git stash`` is only safe as ``git stash list``/``show``.
    """
    if not command:
        return False
    # Basename across both separators so Windows paths work on any host.
    prog = command[0].replace("\\", "/").rsplit("/", 1)[-1].lower().removesuffix(".exe")
    if prog == "git":
        args = _skip_git_global_flags(command[1:])
        sub = args[0] if args else ""
        if sub == "stash":
            rest = [a for a in args[1:] if not a.startswith("-")]
            return bool(rest) and rest[0] in ("list", "show")
        return sub in _SAFE_GIT_SUBCOMMANDS
    return prog in _SAFE_RERUN_PROGRAMS


def _skip_git_global_flags(args: list[str]) -> list[str]:
    """Drop git's pre-subcommand global flags (``-C dir``, ``-c k=v``, ...)."""
    value_flags = ("-C", "-c", "--git-dir", "--work-tree", "--exec-path", "--namespace")
    i = 0
    while i < len(args):
        a = args[i]
        if a in value_flags:
            i += 2  # flag consumes the next arg as its value
        elif a.startswith("-"):
            i += 1  # standalone or --flag=value form
        else:
            return args[i:]
    return []


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

    If ``command`` is given, is safe to re-execute (read-only allowlist), and
    zap is installed, re-runs it through zap for structure-aware filtering.
    Otherwise applies the built-in heuristic to the already-captured ``output``
    — commands with side effects are never re-executed.
    """
    if command and is_safe_to_rerun(command):
        filtered = filter_with_zap(command, binary=binary)
        if filtered is not None:
            return filtered
    return builtin_filter(output, max_lines=max_lines)
