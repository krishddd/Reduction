"""``reduction`` command-line interface.

Mirrors headroom's top-level UX so the pipeline is usable without writing code:

    reduction compress <file>     # compress a file, show savings + CCR ref
    reduction retrieve <ref>      # expand a CCR ref (needs a persisted store)
    reduction stats               # token-savings summary (from a metrics file)
    reduction simulate ...        # compounded cost model
    reduction mcp                 # run the MCP server (stdio)
    reduction serve               # run the FastAPI gateway
    reduction wrap <agent>        # print copy-paste integration for an agent
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reduction import TokenOptimizer, __version__
from reduction.config import OptimizerConfig
from reduction.content import compress_content


def _cmd_compress(args: argparse.Namespace) -> int:
    text = Path(args.file).read_text(encoding="utf-8", errors="replace")
    store_path = args.store or "reduction-ccr.json"
    from reduction.ccr import CompressionStore

    result = compress_content(text, ccr=not args.no_ccr, store=CompressionStore(path=store_path))
    sys.stdout.write(result.text + "\n")
    sys.stderr.write(
        f"[{result.content_type.value}] {result.tokens_before} -> {result.tokens_after} tok "
        f"({result.compression_ratio:.0%} saved)"
        + (f"  ref={result.ref}" if result.ref else "")
        + "\n"
    )
    return 0


def _cmd_retrieve(args: argparse.Namespace) -> int:
    from reduction.ccr import CompressionStore

    store = CompressionStore(path=args.store or "reduction-ccr.json")
    original = store.get(args.ref)
    if original is None:
        sys.stderr.write(f"ref {args.ref!r} not found in {args.store or 'reduction-ccr.json'}\n")
        return 1
    sys.stdout.write(original + "\n")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    import json

    path = Path(args.file)
    if not path.exists():
        sys.stderr.write(f"no metrics file at {path}\n")
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    for key, value in data.items():
        sys.stdout.write(f"{key:>22}: {value}\n")
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    from simulator.simulate import simulate

    simulate(args.daily_input_tokens, args.daily_output_tokens, args.cached_prefix_fraction)
    return 0


def _cmd_mcp(_: argparse.Namespace) -> int:
    from reduction.mcp_server import main as mcp_main

    mcp_main()
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        sys.stderr.write("The gateway needs extras. Install: pip install 'reduction[gateway]'\n")
        return 1
    uvicorn.run("reduction.gateway.main:app", host=args.host, port=args.port)
    return 0


def _cmd_proxy(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from reduction.proxy import build_app
    except ImportError:
        sys.stderr.write("The proxy needs extras. Install: pip install 'reduction[gateway]'\n")
        return 1
    uvicorn.run(build_app(), host=args.host, port=args.port)
    return 0


def _cmd_memory(args: argparse.Namespace) -> int:
    from reduction.memory import Memory

    mem = Memory(args.db, namespace=args.namespace)
    if args.action == "add":
        mid = mem.add(args.text)
        sys.stdout.write(f"stored id={mid} (namespace={args.namespace})\n")
    elif args.action == "search":
        for hit in mem.search(args.text, k=args.k):
            sys.stdout.write(f"{hit.score:.3f}  {hit.text[:100]}\n")
    return 0


def _cmd_learn(args: argparse.Namespace) -> int:
    from reduction.learn import FailureLog, write_corrections

    log = FailureLog(args.log)
    corrections = log.derive_corrections(min_occurrences=args.min_occurrences)
    if not corrections:
        sys.stderr.write("no recurring failures found\n")
        return 0
    if args.write:
        write_corrections(args.write, corrections)
        sys.stdout.write(f"wrote {len(corrections)} corrections to {args.write}\n")
    else:
        from reduction.learn import render_corrections

        sys.stdout.write(render_corrections(corrections) + "\n")
    return 0


_WRAP_SNIPPETS = {
    "anthropic": (
        "from reduction.adapters import OptimizedAnthropic\n"
        "client = OptimizedAnthropic(api_key=...)\n"
        "# same as anthropic.Anthropic; pass output_format='toon' to messages.create\n"
    ),
    "openai": (
        "from reduction.adapters import OptimizedOpenAI\n"
        "client = OptimizedOpenAI(api_key=...)\n"
        "# same as openai.OpenAI; pass output_format='toon' to chat.completions.create\n"
    ),
    "odysseus": (
        "from reduction import TokenOptimizer\n"
        "from reduction.adapters import wrap_message_fn\n"
        "opt = TokenOptimizer()\n"
        "client.message = wrap_message_fn(client.message, opt, output_format='toon')\n"
    ),
    "mcp": (
        "Add to your MCP host config (Claude Code / Cursor):\n"
        '  { "mcpServers": { "reduction": { "command": "reduction", "args": ["mcp"] } } }\n'
    ),
}


def _cmd_wrap(args: argparse.Namespace) -> int:
    snippet = _WRAP_SNIPPETS.get(args.agent)
    if snippet is None:
        sys.stderr.write(f"unknown agent {args.agent!r}; choose from {list(_WRAP_SNIPPETS)}\n")
        return 1
    sys.stdout.write(snippet)
    return 0


def _cmd_demo(_: argparse.Namespace) -> int:
    opt = TokenOptimizer(OptimizerConfig(output_format="toon"))
    big = '{"items":[' + ",".join(f'{{"id":{i},"ok":true}}' for i in range(100)) + "]}"
    filtered = opt.filter_tool_output(big)
    sys.stdout.write(filtered + "\n\n")
    sys.stdout.write(opt.render() + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reduction", description=__doc__)
    p.add_argument("--version", action="version", version=f"reduction {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("compress", help="compress a file (content-aware + CCR)")
    c.add_argument("file")
    c.add_argument("--store", help="CCR store path (default reduction-ccr.json)")
    c.add_argument("--no-ccr", action="store_true", help="disable reversible CCR markers")
    c.set_defaults(func=_cmd_compress)

    r = sub.add_parser("retrieve", help="expand a CCR ref to the original")
    r.add_argument("ref")
    r.add_argument("--store", help="CCR store path (default reduction-ccr.json)")
    r.set_defaults(func=_cmd_retrieve)

    s = sub.add_parser("stats", help="print a persisted metrics summary")
    s.add_argument("file", nargs="?", default="reduction-metrics.json")
    s.set_defaults(func=_cmd_stats)

    sim = sub.add_parser("simulate", help="compounded cost model")
    sim.add_argument("--daily-input-tokens", type=float, default=5_000_000)
    sim.add_argument("--daily-output-tokens", type=float, default=800_000)
    sim.add_argument("--cached-prefix-fraction", type=float, default=0.6)
    sim.set_defaults(func=_cmd_simulate)

    m = sub.add_parser("mcp", help="run the MCP server (stdio)")
    m.set_defaults(func=_cmd_mcp)

    srv = sub.add_parser("serve", help="run the FastAPI gateway")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    srv.set_defaults(func=_cmd_serve)

    px = sub.add_parser("proxy", help="run the OpenAI/Anthropic-compatible compression proxy")
    px.add_argument("--host", default="127.0.0.1")
    px.add_argument("--port", type=int, default=8788)
    px.set_defaults(func=_cmd_proxy)

    mem = sub.add_parser("memory", help="add/search persistent vector memory")
    mem.add_argument("action", choices=["add", "search"])
    mem.add_argument("text")
    mem.add_argument("--db", default="reduction-memory.db")
    mem.add_argument("--namespace", default="default")
    mem.add_argument("-k", type=int, default=5)
    mem.set_defaults(func=_cmd_memory)

    lrn = sub.add_parser("learn", help="derive corrections from a failure log")
    lrn.add_argument("--log", default="reduction-failures.jsonl")
    lrn.add_argument("--min-occurrences", type=int, default=2)
    lrn.add_argument("--write", help="instructions file to update (e.g. CLAUDE.md)")
    lrn.set_defaults(func=_cmd_learn)

    w = sub.add_parser("wrap", help="print integration snippet for an agent")
    w.add_argument("agent", choices=list(_WRAP_SNIPPETS))
    w.set_defaults(func=_cmd_wrap)

    d = sub.add_parser("demo", help="compress a sample and show savings")
    d.set_defaults(func=_cmd_demo)

    return p


def _force_utf8_stdio() -> None:
    # Compressed text / memory hits can contain non-ASCII; on a Windows cp1252
    # console a bare write would raise UnicodeEncodeError. Reconfigure to UTF-8
    # (replace on failure) so the CLI never crashes on output encoding.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdio()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
