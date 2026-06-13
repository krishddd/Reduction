"""MCP server exposing Reduction to any MCP host (Claude Code, Cursor, ...).

Tools:
  * ``reduction_compress`` — compress a blob of text; returns compressed text,
    content type, token savings, and a CCR ref.
  * ``reduction_retrieve`` — expand a CCR ref back to the original content.
  * ``reduction_stats``    — token-savings summary for the session.

This is the headroom-style "connect to any agent" path: instead of wrapping a
client SDK, the agent calls these tools directly. Requires the ``mcp`` package
(``pip install reduction[mcp]``); importing without it raises a clear error.

Run it with:  ``reduction mcp``  (or ``python -m reduction.mcp_server``)
"""

from __future__ import annotations

from reduction import TokenOptimizer, ccr
from reduction.content import compress_content

_optimizer = TokenOptimizer()


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

    @server.tool()
    def reduction_retrieve(ref: str) -> str:
        """Retrieve the original uncompressed content for a CCR ref."""
        return (
            ccr.handle_retrieve_call(
                ccr.RETRIEVE_TOOL_NAME, {"ref": ref}, store=_optimizer._ccr_store()
            )
            or "[reduction: not found]"
        )

    @server.tool()
    def reduction_stats() -> dict:
        """Token-savings summary for this session."""
        return _optimizer.report()

    return server


def main() -> None:
    _build_server().run()


if __name__ == "__main__":
    main()
