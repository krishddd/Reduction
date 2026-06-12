# Token-Reduction Pipeline — Build Plan

> Goal: a low-cost, production-grade AI-agent infrastructure that attacks token
> waste at every layer — from the shell to the serialized response. `zap`
> (already in this repo) is Layer 1. This plan wires the remaining four layers
> around it into one deployable system.

Date: 2026-06-12 · Status: planning · Owner: Harish

---

## 1. What already exists

`./zap` is a complete, high-quality Rust CLI proxy (the "RTK / Rust Token
Killer" pattern). It intercepts shell output (`git status`, `cargo test`,
`pytest`, …), applies 12 filtering strategies, and cuts terminal tokens
**60–90%**. It ships hooks for 10+ agents, a SQLite analytics store (`zap
gain`), and TOML filter recipes. **We reuse it as-is for Layer 1** — no rewrite.

The rest of the pipeline (compression, caching, output shaping, the gateway
that ties them together) does not exist yet. That is what this plan builds.

---

## 2. The five layers (verified against real tools/papers)

| # | Layer | Tool / technique | Where it runs | Verified savings | Source |
|---|-------|------------------|---------------|------------------|--------|
| 1 | Shell | **zap / RTK** | local, pre-agent hook | 60–90% on command output | [rtk-ai/rtk](https://github.com/rtk-ai/rtk) |
| 2 | Orchestration | **LLMLingua / LLMLingua-2** | gateway pre-processing | 2–5× (up to 20×) prompt compression | [arXiv 2310.05736](https://arxiv.org/abs/2310.05736), [arXiv 2403.12968](https://arxiv.org/abs/2403.12968) |
| 3 | Gateway | **LiteLLM semantic cache** (Redis/Qdrant) | Dockerized proxy | hit returns in 3–8 ms vs ~2 s gen | [LiteLLM caching docs](https://docs.litellm.ai/docs/proxy/caching) |
| 4 | Provider | **Native prompt caching** | Anthropic/OpenAI API | 90% input discount (Anthropic), 50% (OpenAI) | [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) |
| 5 | Generation | **Caveman prompt + TOON/YAML** | system prompt + output schema | ~45% fewer output tokens; 30–60% on structured arrays | [JuliusBrussee/caveman](https://github.com/juliusbrussee/caveman), [toon-format/toon](https://github.com/toon-format/toon) |

Savings **compound multiplicatively** — each layer operates on the output of the
previous one.

---

## 3. Request flow

```
                ┌─────────────────────────────────────────────────────────┐
   agent run    │                  FastAPI Gateway (Docker)               │
   shell cmd    │                                                         │
      │         │   ┌──────────┐   ┌───────────┐   ┌──────────────────┐   │
      ▼         │   │ Layer 2  │   │  Layer 3  │   │    Layer 4/5     │   │
  ┌────────┐    │   │LLMLingua │   │ LiteLLM   │   │  Provider call   │   │
  │  zap   │───▶│──▶│ compress │──▶│ semantic  │──▶│  native cache +  │──▶ response
  │(Layer1)│    │   │ context  │   │ cache?    │   │  Caveman + TOON  │   │
  └────────┘    │   └──────────┘   └─────┬─────┘   └──────────────────┘   │
                │                        │ hit                            │
                │                        ▼ 3–8 ms                         │
                │                  cached response ──────────────────────▶│
                └─────────────────────────────────────────────────────────┘
```

Ordering rule that makes Layer 4 work: **stable content first** (system prompt,
tool defs, schemas, static logs), **volatile content last** (user msg,
timestamps). Anthropic caches the static prefix; OpenAI auto-caches prefixes
>1,024 tokens.

---

## 4. Proposed repo layout

Build a sibling to `zap/` so the Rust binary stays independent:

```
Reduction/
├── zap/                      # Layer 1 — already here, unchanged
├── gateway/                  # NEW — the Python orchestration service
│   ├── pyproject.toml
│   ├── Dockerfile
│   ├── docker-compose.yml    # gateway + redis-stack (vector) + litellm
│   ├── app/
│   │   ├── main.py           # FastAPI entrypoint
│   │   ├── compress.py       # Layer 2: LLMLingua-2 wrapper
│   │   ├── router.py         # Layer 3: LiteLLM client + cache config
│   │   ├── prompts/
│   │   │   └── caveman.md     # Layer 5: output system prompt
│   │   ├── formats/
│   │   │   └── toon.py        # Layer 5: JSON→TOON encoder (or python-toon dep)
│   │   └── ordering.py       # Layer 4: prefix-stable message assembly
│   ├── config/
│   │   └── litellm.config.yaml
│   └── tests/
├── simulator/                # NEW — cost simulator (matches the spec's tool)
│   └── simulate.py           # CLI/notebook: model compounded savings
└── PIPELINE_PLAN.md          # this file
```

---

## 5. Layer-by-layer build notes

### Layer 1 — zap (done)
Action: install the hook (`zap init -g`), point the gateway at zap-filtered
output. No code change. Confirm `zap gain --format json` is readable by the
simulator for real baseline numbers.

### Layer 2 — LLMLingua-2 (context compression)
- Dependency: `llmlingua` (pip). Use **LLMLingua-2** (`xlm-roberta` token
  classifier) — 3–6× faster than v1, task-agnostic, bidirectional.
- Apply **only to retrieved/context documents**, never to system instructions
  or the schema (budget controller keeps instructions at ratio 1.0).
- Tunable `compression_ratio` per request; default 0.5 (2× shrink), escalate to
  0.2 for large RAG dumps.
- Caveat: needs a small model loaded in-process (~CPU or small GPU). Run as a
  warm singleton in the FastAPI worker.

### Layer 3 — LiteLLM semantic cache (gateway)
- Run LiteLLM proxy in Docker. `redis-stack` provides RediSearch + vector (VSS).
- `config/litellm.config.yaml`:
  ```yaml
  litellm_settings:
    cache: true
    cache_params:
      type: redis-semantic
      similarity_threshold: 0.90          # cosine; spec's threshold
      redis_semantic_cache_embedding_model: text-embedding-3-small
      redis_semantic_cache_use_async: true
  ```
- Qdrant alternative: `type: qdrant-semantic`, `vector_size: 1536`,
  `quantization: binary`.
- Requires `redis-py >= 4.2.0` with the redisearch module loaded (known gotcha,
  litellm#12401).

### Layer 4 — native provider caching (ordering discipline)
- `ordering.py` assembles messages so the cacheable prefix is byte-stable.
- Anthropic: add `cache_control: {type: "ephemeral"}` on the last stable block
  (system + tools + schema). Write costs 1.25× input (5-min TTL) / 2.0× (1-hr);
  reads cost **0.10×** — a 90% discount ($0.30/M on Sonnet 4.6 vs $3/M).
- OpenAI: nothing to set — automatic for prefixes >1,024 tokens, 50% off. Just
  keep volatile tokens at the bottom.
- Stacks with batch API (50%) → up to ~95% combined on eligible workloads.

### Layer 5 — Caveman + TOON (output shaping)
- **Caveman**: ship `prompts/caveman.md` as a system-prompt skill that strips
  articles, pleasantries, hedging. ~45% fewer output tokens; brevity also
  reduces scale-dependent verbosity (accuracy-neutral or better per the
  efficient-reasoning literature). Make it toggleable — keep prose mode for
  user-facing text, Caveman for machine/tool legs.
- **TOON**: serialize uniform arrays (vuln lists, state arrays, test results) as
  TOON instead of JSON — 30–60% fewer tokens. Use `python-toon` or the official
  spec. Fall back to JSON for deeply nested / non-uniform data (TOON loses there).
  Teach the model the format by one in-prompt example since it's newer than JSON.

---

## 6. The cost simulator

Mirrors the "Token Optimization & Cost Simulator" in the spec. Pure function so
it's testable and embeddable in a notebook or a small web UI later.

Inputs: baseline daily tokens (in/out split), per-layer reduction %, provider
rates, cache-hit rate. Output: daily/monthly $ before vs after, per-layer
waterfall. Seed defaults from real `zap gain --format json` data so Layer 1's
number isn't guessed.

---

## 7. Milestones

1. **M1 — Gateway skeleton.** FastAPI passthrough to one provider via LiteLLM,
   Dockerized, no optimization. Establishes the baseline + the seam.
2. **M2 — Layer 4 ordering + native cache.** Stable-prefix assembly; verify
   Anthropic cache-read billing in logs. Cheapest win, zero new infra.
3. **M3 — Layer 5 output shaping.** Caveman system prompt + TOON encoder +
   format toggle. Measure output-token drop on a fixed task set.
4. **M4 — Layer 3 semantic cache.** Add redis-stack, wire LiteLLM semantic
   cache, tune threshold against a replayed query log.
5. **M5 — Layer 2 LLMLingua-2.** Add context compression with per-request
   budget; guard instructions/schema from compression.
6. **M6 — Simulator + dashboard.** Compounded-savings model fed by live
   analytics; one-screen before/after.

Order is deliberate: M2 first (free, immediate), heavy infra (M4/M5) last.

---

## 8. Risks & caveats

- **Compounded ≠ additive.** Report measured end-to-end savings, not the product
  of marketing numbers. Each layer's % is measured on the *already-reduced*
  stream.
- **Semantic cache false hits.** 0.90 cosine can return a wrong answer for a
  subtly different query. Start conservative (0.92–0.95) on high-stakes paths.
- **LLMLingua compute cost.** The compressor is itself a model — only worth it
  when context is large and reused. Skip for short prompts.
- **TOON is young.** Models know JSON far better; validate output parses and
  keep JSON fallback. Don't use TOON for non-uniform/nested data.
- **Caveman on user-facing text** can read as terse/rude — restrict it to
  tool/machine legs unless the user opts in.
- **Native cache TTL.** Anthropic 5-min default; bursty traffic may miss. Watch
  write-cost amplification (1.25×) on low-reuse prompts.

---

## 9. Sources

- RTK / Rust Token Killer — https://github.com/rtk-ai/rtk · https://www.rtk-ai.app/
- LLMLingua — https://arxiv.org/abs/2310.05736
- LLMLingua-2 — https://arxiv.org/abs/2403.12968
- LiteLLM caching — https://docs.litellm.ai/docs/proxy/caching · https://docs.litellm.ai/docs/caching/all_caches
- Anthropic prompt caching — https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- Anthropic 2026 pricing — https://platform.claude.com/docs/en/about-claude/pricing
- TOON spec + benchmarks — https://github.com/toon-format/toon · https://blog.logrocket.com/reduce-tokens-with-toon/
- python-toon — https://github.com/xaviviro/python-toon
- Caveman skill — https://github.com/juliusbrussee/caveman
- Token-efficient reasoning survey context — https://arxiv.org/html/2505.07961v3
