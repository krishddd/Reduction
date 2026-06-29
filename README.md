# Reduction

> Five-layer token-optimization pipeline for AI agents.
> Attacks token waste at **every layer** — from the shell to the serialized
> response — on **every input and output**. Compression is **reversible**: the
> agent can retrieve any dropped detail on demand (CCR). Connect via the
> in-process SDK, zero-touch client adapters, an **MCP server**, a CLI, or a
> shared HTTP gateway.

```
            ┌──────────────────────────── your agent ────────────────────────────┐
 tool /     │                                                                     │
 command ──▶│  L1 shell filter ─┐                                                 │
 output     │                   │   prepare()                                     │
            │  context docs ────┼─▶ L2 compress ─▶ L4 stable-prefix + cache_ctrl  │──▶ provider
 user turn ─┼─▶ normalize ──────┘   L5 caveman + TOON contract                    │
            │                                                                     │
            │  response ──▶ L5 decode (TOON/YAML→objects) ──▶ metrics             │◀── provider
            └─────────────────────────────────────────────────────────────────────┘
                              L3 semantic cache wraps the call (optional)
```

## The five layers

| # | Layer | Technique | Default | Savings |
|---|-------|-----------|---------|---------|
| 1 | Shell | content-aware tool-output compression (JSON/diff/log routing) + reversible CCR; [zap](https://github.com/rtk-ai/rtk)/RTK or heuristic fallback | on | 60–97% on tool output |
| 2 | Context | [LLMLingua-2](https://arxiv.org/abs/2403.12968) compresses retrieved docs (never instructions) | off* | 2–5× on context |
| 3 | Cache | [LiteLLM semantic cache](https://docs.litellm.ai/docs/proxy/caching) (Redis VSS / Qdrant) | off* | skips generation on hit |
| 4 | Provider | [Native prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) via stable-prefix ordering + `cache_control` | on | 90% input discount (Anthropic) |
| 5 | Output | Caveman persona + [TOON](https://github.com/toon-format/toon)/YAML serialization | on | ~45% output; 30–60% structured |

\* Layers 2 and 3 pull heavy optional dependencies (`torch`, `litellm`/Redis) and
only pay off on large or reused context, so they default off. Everything else
is dependency-free and on by default.

## Content-aware compression + CCR (reversible)

Layer 1 doesn't just truncate lines. Tool/command output is **classified**
(JSON / diff / log / code / text) and routed to a specialized compressor:

| Content | Compressor | Example result |
|---------|-----------|----------------|
| JSON | SmartCrusher-lite — sample large uniform arrays, render as TOON | 300-row scan: **9,006 → 231 tokens (97%)** |
| diff | per-file `+/-` shortstat | 5,000-line diff → `3 files, +142/-89` |
| log | dedupe + keep error/warn lines + head/tail | noisy build log → signal only |
| code / text | lossless dedupe + whitespace | safe, no data dropped |

Because lossy compression occasionally hides the one row the model needs,
every compression is **reversible** via **CCR (Compress-Cache-Retrieve)**: the
original is stored under a short content hash and the compressed text carries a
marker — `[reduction: json compressed 97% ..., ref=14a9cd0d]`. The agent calls
the `reduction_retrieve` tool (or `opt.retrieve(ref)`) to get the original back.

```python
opt = TokenOptimizer()
small = opt.filter_tool_output(huge_json_scan_output)   # 97% smaller, carries a ref
# ...later, if the model needs everything:
original = opt.retrieve("14a9cd0d")
```

## Install

```bash
pip install -e .                # core SDK (zero heavy deps)
pip install -e ".[gateway]"     # + FastAPI/LiteLLM HTTP gateway
pip install -e ".[compress]"    # + LLMLingua-2 (Layer 2)
pip install -e ".[mcp]"         # + MCP server (reduction_compress/retrieve/stats)
pip install -e ".[proxy]"       # + OpenAI/Anthropic compression proxy
pip install -e ".[code]"        # + tree-sitter AST code compression
pip install -e ".[memory]"      # + sentence-transformers + hnswlib vector memory
pip install -e ".[tokenizer]"   # + tiktoken (accurate token counts)
pip install -e ".[dev]"         # + test/lint tooling
```

## Use it — in-process SDK

```python
from reduction import TokenOptimizer

opt = TokenOptimizer()
req = opt.prepare(
    system="You are a security planner.",
    user=target_profile_json,          # per-target / volatile → user turn
    static_context=[taxonomy, schema], # reused → cacheable prefix
    output_format="toon",
)
resp = client.messages.create(
    model="claude-sonnet-4-6",
    system=req.system_blocks,          # cache_control already attached
    messages=req.messages,
    max_tokens=2048,
)
opt.record_usage(resp.usage)
data = opt.decode_output(resp.content[0].text, req.output_format)
print(opt.render())                    # token-savings report
```

## Use it — zero-touch adapters

Wrap an existing client and change nothing else:

```python
from reduction.adapters import OptimizedAnthropic
client = OptimizedAnthropic(api_key=...)      # same ctor as anthropic.Anthropic
resp = client.messages.create(model="claude-sonnet-4-6",
                              system="You plan.", messages=[...],
                              max_tokens=1024, output_format="toon")
print(client.optimizer.render())
```

`OptimizedOpenAI` does the same for `openai.OpenAI`.

## Connect it to an agent — the odysseus example

The odysseus security agent (`Agent_security_testing/Security_module`) routes
every call through `ClaudeClient.message(...)`. One line wraps it — the whole
scan gets caveman output, TOON serialization, normalized inputs, and savings
metrics, with no change to the planner/synthesizer/triager call sites:

```python
from reduction import TokenOptimizer
from reduction.adapters import wrap_message_fn

client = ClaudeClient()
opt = TokenOptimizer()
client.message = wrap_message_fn(client.message, opt, output_format="toon")
```

See [examples/odysseus_integration.py](examples/odysseus_integration.py)
(`python examples/odysseus_integration.py` runs an offline demo).

## Use it — HTTP gateway

For non-Python agents or one shared service:

```bash
docker compose up --build         # gateway + redis-stack (semantic cache)
curl localhost:8000/v1/pipeline/chat -H 'content-type: application/json' \
  -d '{"user_message":"summarize failures","output_format":"toon"}'
```

Endpoints: `/v1/pipeline/chat`, `/v1/optimize`, `/v1/encode/toon`,
`/v1/metrics`, `/healthz`.

## Use it — MCP server (any MCP host)

The most universal "connect to an agent" path: instead of wrapping a client,
the agent calls compression tools directly. Works with Claude Code, Cursor, or
any MCP host.

```bash
pip install -e ".[mcp]"
reduction mcp                       # runs the stdio MCP server
```

```jsonc
// Claude Code / Cursor MCP config
{ "mcpServers": { "reduction": { "command": "reduction", "args": ["mcp"] } } }
```

Exposes `reduction_compress` (content-aware + CCR), `reduction_retrieve`
(expand a ref), and `reduction_stats` (savings summary).

## Use it — CLI

```bash
reduction compress scan.json            # content-aware compress, prints CCR ref
reduction retrieve 14a9cd0d             # expand a ref back to the original
reduction simulate --daily-input-tokens 5000000
reduction wrap anthropic                # print a copy-paste integration snippet
reduction demo                          # compress a sample and show savings
reduction serve / reduction mcp         # gateway / MCP server
reduction proxy --port 8788             # OpenAI/Anthropic-compatible compression proxy
reduction memory add "..." / search "..."   # persistent vector memory
reduction learn --log f.jsonl --write CLAUDE.md   # failure-learning corrections
```

## Advanced subsystems

These close the gap with full context-optimization platforms. All have
dependency-free fallbacks, so they work before you install any extras.

### Compression proxy ([reduction/proxy.py](reduction/proxy.py))
A drop-in OpenAI- and Anthropic-compatible HTTP proxy. Point any client at it;
it compresses large message content, injects the `reduction_retrieve` tool, and
**transparently satisfies retrieval tool calls** from the CCR store so the
client never sees the round-trip. **Streaming (SSE) is supported**: content
tokens forward as they arrive, while `reduction_retrieve` tool-call events are
buffered, resolved mid-stream, and the turn continues — all transparent to the
client. Non-retrieval tool calls pass straight through.

```bash
pip install -e ".[proxy]"
OPENAI_BASE_URL=https://api.openai.com reduction proxy --port 8788
# point your client's base_url at http://127.0.0.1:8788
```

### AST-aware code compression ([reduction/layers/codecrush.py](reduction/layers/codecrush.py))
`CODE` content keeps imports, decorators, and class/function signatures while
eliding bodies (`... (12 lines)`) — the agent sees the shape, retrieves a body
via CCR when it needs one. tree-sitter (`[code]` extra) for language-exact
parsing; robust Python/JS/TS/Go/Java/C++/Rust heuristic otherwise.

### Persistent vector memory ([reduction/memory.py](reduction/memory.py))
Per-project SQLite store with semantic search for cross-turn / cross-agent
recall. Namespaced so projects never bleed into each other.

```python
from reduction.memory import Memory
mem = Memory("proj.db", namespace="my-project")
mem.add("the deploy step needs AWS_PROFILE=prod", metadata={"src": "runbook"})
hits = mem.search("how do I deploy", k=3)
```

Real embeddings with `[memory]` (sentence-transformers); a deterministic
hashing embedding otherwise. When `hnswlib` is installed, search uses an ANN
index (built from SQLite on open, updated on add) for sub-linear lookups;
otherwise it falls back to an exact cosine scan.

### Failure-learning ([reduction/learn.py](reduction/learn.py))
Record agent outcomes; recurring failures become corrections written into a
managed block in `CLAUDE.md` / `AGENTS.md`, so the next run starts smarter.

```python
from reduction.learn import FailureLog, write_corrections
log = FailureLog()
log.record(context="run tests", action="pytest -k foo", outcome="fail", error="no tests ran")
write_corrections("CLAUDE.md", log.derive_corrections(min_occurrences=2))
```

### Batch-API CCR ([reduction/ccr_batch.py](reduction/ccr_batch.py))
Resolves `reduction_retrieve` tool calls that arrive in asynchronous Batch API
results, producing continuation messages — CCR stays reversible even off the
live request path.

## Configuration

Every knob has an env-var fallback (see [reduction/config.py](reduction/config.py)):

| Env var | Purpose | Default |
|---------|---------|---------|
| `REDUCTION_CAVEMAN` | inject terse-output persona | `true` |
| `REDUCTION_OUTPUT_FORMAT` | `text` / `toon` / `yaml` | `text` |
| `REDUCTION_SHELL_FILTER` | filter tool output (Layer 1) | `true` |
| `REDUCTION_CONTENT_ROUTING` | content-aware tool-output compression | `true` |
| `REDUCTION_CCR` | reversible compression (store + retrieve refs) | `true` |
| `REDUCTION_CCR_STORE` | path to persist the CCR store as JSON | _(memory)_ |
| `REDUCTION_COMPRESS` | LLMLingua-2 (Layer 2) | `false` |
| `REDUCTION_SEMANTIC_CACHE` | LiteLLM semantic cache (Layer 3) | `false` |
| `REDUCTION_SEMANTIC_THRESHOLD` | cosine hit threshold | `0.92` |
| `REDUCTION_NATIVE_CACHE` | stable-prefix + `cache_control` (Layer 4) | `true` |

## Cost simulator

```bash
python simulator/simulate.py --daily-input-tokens 5000000 --daily-output-tokens 800000
```

Models the compounded savings against provider pricing as a before/after
waterfall.

## CI/CD

- **ci.yml** — ruff lint + format, pytest on Python 3.11/3.12, simulator smoke test.
- **docker.yml** — builds the gateway image on `main`, publishes to GHCR on `v*` tags.

## Accuracy evaluation (does compression keep answers correct?)

Saving tokens is only safe if the model still answers correctly. The eval
harness runs each case raw and compressed through an injectable `model_fn` and
reports **answer preservation** alongside token savings, so you see the
trade-off instead of guessing:

```bash
reduction eval        # offline self-check (synthetic log case)
# -> Answer preservation: 100.0%   Token savings: 98.8%
```

```python
from reduction.evals import EvalCase, run_evals
report = run_evals(cases, model_fn)   # model_fn(context, question) -> answer
print(report.render())                # flags any REGRESSIONS
```

Wire `model_fn` to a real client to validate on your own traffic. This is the
number that actually matters — a high savings % with a low preservation % means
the compression is too aggressive for that content.

## Honesty notes (what the metrics do and don't claim)

- **Input savings are measured; output savings are not.** Caveman/TOON shrink
  output, but we have no counterfactual (we never see the uncompressed
  generation), so the metrics report observed output tokens, never "output
  saved." Use the eval harness to quantify the output/accuracy effect.
- **Token counts for Claude are approximate.** tiktoken (`cl100k`/`o200k`) is
  exact for OpenAI; Anthropic's tokenizer isn't bundled, so Claude counts are a
  close proxy, not exact billing.
- **codecrush** uses a real tree-sitter parser when `[code]` is installed and a
  docstring-safe heuristic otherwise — both are honest about which ran.

## Caveats (read before production)

- Compounded ≠ additive — measure end-to-end, don't multiply marketing numbers.
- Semantic cache can return a wrong answer for a subtly different query; keep
  the threshold high (≥0.92) on high-stakes paths.
- LLMLingua is itself a model — only worth running on large, reused context.
- TOON loses to JSON on deeply nested / non-uniform data; the encoder falls
  back automatically.
- Caveman output reads as terse — restrict it to machine/tool legs.
- CCR refs in an in-memory store don't survive a restart — set
  `REDUCTION_CCR_STORE` to a file path if a later process must retrieve them.

## Credits

The content-aware compression, CCR (Compress-Cache-Retrieve), and MCP-tool
design are inspired by [Headroom](https://github.com/chopratejas/headroom)
(Apache-2.0). Reduction is an independent Python implementation of those ideas
layered onto its own caveman/TOON/native-cache pipeline.

## License

Apache-2.0
