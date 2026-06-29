"""MCP server exposing Reduction to any MCP host (Claude Code, Cursor, ...).

Tools:
  * ``reduction_compress`` — compress a blob of text; returns compressed text,
    content type, token savings, and a CCR ref.
  * ``reduction_retrieve`` — expand a CCR ref back to the original content.
  * ``reduction_compress_history`` — compress old turns in a message list,
    keeping the last N verbatim (CCR-reversible).
  * ``reduction_fit_context`` — pack context chunks into a token budget,
    relevance-scored against an optional query.
  * ``reduction_route_effort`` — recommend a reasoning-effort level / thinking
    budget for a task.
  * ``reduction_stats``    — token-savings summary for the session.

This is the headroom-style "connect to any agent" path: instead of wrapping a
client SDK, the agent calls these tools directly. Requires the ``mcp`` package
(``pip install reduction[mcp]``); importing without it raises a clear error.

The tool implementations live as module-level ``_impl_*`` functions (so they are
unit-testable without the ``mcp`` package); ``_build_server`` registers thin
wrappers under their public ``reduction_*`` names.

Run it with:  ``reduction mcp``  (or ``python -m reduction.mcp_server``)
"""

from __future__ import annotations

from reduction import TokenOptimizer, ccr
from reduction.content import compress_content

_optimizer = TokenOptimizer()


# ---- tool implementations (mcp-free, unit-testable) -------------------


def _impl_compress(text: str) -> dict:
    result = compress_content(text, ccr=True, store=_optimizer._ccr_store())
    _optimizer.metrics.record_input(text, result.text, layer="mcp")
    return {
        "compressed": result.text,
        "content_type": result.content_type.value,
        "tokens_before": result.tokens_before,
        "tokens_after": result.tokens_after,
        "tokens_saved": result.tokens_saved,
        "ref": result.ref,
    }


def _impl_retrieve(ref: str) -> str:
    return (
        ccr.handle_retrieve_call(
            ccr.RETRIEVE_TOOL_NAME, {"ref": ref}, store=_optimizer._ccr_store()
        )
        or "[reduction: not found]"
    )


def _impl_compress_history(messages: list, keep_last: int = 4) -> dict:
    from reduction.layers.history import compress_history

    result = compress_history(
        messages, keep_last=keep_last, ccr=True, store=_optimizer._ccr_store()
    )
    if result.tokens_before != result.tokens_after:
        _optimizer.metrics.record_input_tokens(
            result.tokens_before, result.tokens_after, layer="mcp_history"
        )
    return {
        "messages": result.messages,
        "messages_compressed": result.messages_compressed,
        "tokens_before": result.tokens_before,
        "tokens_after": result.tokens_after,
        "tokens_saved": result.tokens_saved,
        "refs": result.refs,
    }


def _impl_fit_context(chunks: list, token_budget: int, query: str = "") -> dict:
    from reduction.layers.contextfit import fit_context

    result = fit_context(
        chunks,
        token_budget=token_budget,
        query=query or None,
        ccr=True,
        store=_optimizer._ccr_store(),
    )
    return {
        "chunks": result.chunks,
        "included": result.included,
        "compressed": result.compressed,
        "dropped": result.dropped,
        "tokens_used": result.tokens_used,
        "token_budget": result.token_budget,
        "refs": result.refs,
    }


def _impl_route_effort(task: str) -> dict:
    from reduction.effort import route_effort

    d = route_effort(task)
    return {
        "level": d.level,
        "thinking_budget": d.thinking_budget,
        "reasoning_effort": d.reasoning_effort,
        "rationale": d.rationale,
    }


def _impl_stats() -> dict:
    return _optimizer.report()


# ---- server wiring ----------------------------------------------------


def _build_server():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - exercised only without mcp
        raise SystemExit(
            "The MCP server needs the 'mcp' package. Install it with:\n"
            "    pip install 'reduction[mcp]'"
        ) from exc

    server = FastMCP("reduction")

    @server.tool()
    def reduction_compress(text: str) -> dict:
        """Compress text (auto-detects JSON/diff/log/code); reversible via CCR."""
        return _impl_compress(text)

    @server.tool()
    def reduction_retrieve(ref: str) -> str:
        """Retrieve the original uncompressed content for a CCR ref."""
        return _impl_retrieve(ref)

    @server.tool()
    def reduction_compress_history(messages: list, keep_last: int = 4) -> dict:
        """Compress old turns in a conversation; keep the last ``keep_last`` verbatim.

        ``messages`` is a list of {role, content} objects (string or block
        content). Returns the new messages plus token savings; compressed blocks
        carry CCR refs that ``reduction_retrieve`` can expand.
        """
        return _impl_compress_history(messages, keep_last)

    @server.tool()
    def reduction_fit_context(chunks: list, token_budget: int, query: str = "") -> dict:
        """Pack ``chunks`` into ``token_budget``, prioritizing relevance to ``query``.

        Includes what fits, compresses (CCR) what doesn't, truncates/drops the
        rest. Returns the selected chunks and what was trimmed.
        """
        return _impl_fit_context(chunks, token_budget, query)

    @server.tool()
    def reduction_route_effort(task: str) -> dict:
        """Recommend a reasoning-effort level + thinking budget for a task."""
        return _impl_route_effort(task)

    @server.tool()
    def reduction_stats() -> dict:
        """Token-savings summary for this session."""
        return _impl_stats()

    return server


def main() -> None:
    _build_server().run()


if __name__ == "__main__":
    main()
