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
```

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
