"""FastAPI gateway exposing the Reduction pipeline over HTTP.

Use this when agents are written in another language or you want one shared
optimization service. In-process Python agents should prefer the
``reduction.TokenOptimizer`` SDK or an adapter — no network hop.

Endpoints:
    GET  /healthz             liveness
    POST /v1/pipeline/chat    optimize + (optionally) call a provider
    POST /v1/optimize         optimize only, return the assembled request
    POST /v1/encode/toon      JSON -> TOON preview
    GET  /v1/metrics          token-savings summary
"""

from __future__ import annotations

from typing import Any

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field

from reduction import TokenOptimizer, __version__
from reduction.layers import semantic_cache, toon

app = FastAPI(title="Reduction Gateway", version=__version__)
optimizer = TokenOptimizer()
_JSON_BODY = Body(...)


class ChatRequest(BaseModel):
    model: str = "claude-sonnet-4-6"
    system: str = "You are a precise engineering assistant."
    user_message: str
    static_context: list[str] = Field(default_factory=list)
    volatile_context: list[str] = Field(default_factory=list)
    caveman: bool = True
    output_format: str = Field(default="text", pattern="^(text|toon|yaml)$")
    call_provider: bool = False


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/v1/optimize")
async def optimize(req: ChatRequest) -> dict:
    r = optimizer.prepare(
        system=req.system,
        user=req.user_message,
        static_context=req.static_context,
        volatile_context=req.volatile_context,
        output_format=req.output_format,
        caveman_on=req.caveman,
    )
    return {
        "system_blocks": r.system_blocks,
        "messages": r.messages,
        "output_format": r.output_format,
        "layers_applied": r.layers_applied,
        "raw_input_chars": r.raw_input_chars,
        "optimized_input_chars": r.optimized_input_chars,
    }


@app.post("/v1/pipeline/chat")
async def pipeline_chat(req: ChatRequest) -> dict:
    r = optimizer.prepare(
        system=req.system,
        user=req.user_message,
        static_context=req.static_context,
        volatile_context=req.volatile_context,
        output_format=req.output_format,
        caveman_on=req.caveman,
    )
    if not req.call_provider:
        return {"optimized": True, "messages": r.messages, "note": "call_provider=false"}

    try:
        messages = [{"role": "system", "content": r.system_blocks}, *r.messages]
        resp = await semantic_cache.acomplete(req.model, messages)
    except Exception as exc:  # provider/network/litellm errors -> 502
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    content = resp.choices[0].message.content or ""
    if resp.usage:
        optimizer.record_usage(resp.usage.model_dump())
    return {
        "content": content,
        "model": getattr(resp, "model", req.model),
        "decoded": optimizer.decode_output(content, req.output_format),
    }


@app.post("/v1/encode/toon")
async def encode_toon(data: Any = _JSON_BODY) -> dict:
    return {"toon": toon.encode(data)}


@app.get("/v1/metrics")
async def metrics() -> dict:
    return optimizer.report()
