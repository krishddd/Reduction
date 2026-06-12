"""TokenOptimizer — the in-process entry point agents wrap their LLM calls with.

It applies token optimization to *every input and output*:

  inputs   system prompt + retrieved context + user turn + tool/command output
  outputs  caveman persona, TOON/YAML serialization, decode back to objects

Typical use (provider-agnostic):

    from reduction import TokenOptimizer
    opt = TokenOptimizer()

    req = opt.prepare(
        system="You are a security planner.",
        user=profile_json,                 # per-target / volatile -> user turn
        static_context=[taxonomy, schema], # reused -> cacheable prefix
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

For a zero-touch wrap, see ``reduction.adapters``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from reduction.config import OptimizerConfig
from reduction.layers import caveman, compress, normalize, ordering, shell, toon
from reduction.metrics import Metrics


@dataclass
class OptimizedRequest:
    system_blocks: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    system_text: str
    output_format: str
    raw_input_chars: int = 0
    optimized_input_chars: int = 0
    layers_applied: list[str] = field(default_factory=list)

    @property
    def system_flat(self) -> str:
        """System prompt as one string (for providers without block syntax)."""
        return ordering.flatten_for_openai(self.system_blocks)


class TokenOptimizer:
    def __init__(self, config: OptimizerConfig | None = None) -> None:
        self.config = config or OptimizerConfig()
        self.metrics = Metrics()

    # ---- inputs -------------------------------------------------------

    def build_system(
        self, system: str, *, output_format: str | None = None, caveman_on: bool | None = None
    ) -> str:
        """Layer 5 (instructions): inject caveman + output-format contract."""
        cfg = self.config
        fmt = output_format if output_format is not None else cfg.output_format
        use_caveman = cfg.caveman if caveman_on is None else caveman_on

        out = system
        if use_caveman:
            out = caveman.apply(out)
        if fmt == "toon":
            out = f"{out}\n\n{toon.TOON_INSTRUCTION}"
        elif fmt == "yaml":
            out = f"{out}\n\nReturn structured data as YAML, never JSON."
        return out

    def prepare(
        self,
        *,
        system: str,
        user: str,
        static_context: list[str] | None = None,
        volatile_context: list[str] | None = None,
        output_format: str | None = None,
        caveman_on: bool | None = None,
    ) -> OptimizedRequest:
        """Run all input layers and assemble a cache-optimal request."""
        cfg = self.config
        fmt = output_format if output_format is not None else cfg.output_format
        applied: list[str] = []

        raw_chars = len(system) + len(user)
        for c in (static_context or []) + (volatile_context or []):
            raw_chars += len(c)

        # Layer 2: compress reused context documents (never instructions).
        context = list(static_context or [])
        if cfg.compress_context and context:
            before = "\n".join(context)
            context = compress.compress_documents(context, rate=cfg.compression_rate)
            self.metrics.record_input(before, "\n".join(context), layer="compress")
            applied.append("compress")

        # Always-on: normalize every text input.
        if cfg.strip_whitespace or cfg.dedupe_lines:
            user = normalize.normalize(user, strip=cfg.strip_whitespace, dedupe=cfg.dedupe_lines)
            context = [
                normalize.normalize(c, strip=cfg.strip_whitespace, dedupe=cfg.dedupe_lines)
                for c in context
            ]
            applied.append("normalize")

        # Layer 5 (instructions): caveman + format contract.
        system_text = self.build_system(system, output_format=fmt, caveman_on=caveman_on)
        if system_text != system:
            applied.append("caveman/format")

        # Layer 4: stable-prefix assembly + cache_control breakpoint.
        system_blocks, messages = ordering.assemble_messages(
            system_text,
            user,
            static_context=context,
            volatile_context=volatile_context,
            anthropic_cache=cfg.native_cache,
        )
        applied.append("ordering")

        opt_chars = len(system_text) + len(user) + sum(len(c) for c in context)
        for c in volatile_context or []:
            opt_chars += len(c)

        return OptimizedRequest(
            system_blocks=system_blocks,
            messages=messages,
            system_text=system_text,
            output_format=fmt,
            raw_input_chars=raw_chars,
            optimized_input_chars=opt_chars,
            layers_applied=applied,
        )

    def filter_tool_output(self, output: str, *, command: list[str] | None = None) -> str:
        """Layer 1: shrink raw tool/command output before it re-enters context."""
        if not self.config.shell_filter:
            return output
        filtered = shell.filter_tool_output(
            output,
            command=command,
            binary=self.config.zap_binary,
            max_lines=self.config.max_tool_output_lines,
        )
        self.metrics.record_input(output, filtered, layer="shell")
        return filtered

    # ---- outputs ------------------------------------------------------

    def encode_output(self, data: Any) -> str:
        """Serialize a structured object the way the model was told to."""
        if self.config.output_format == "toon":
            return toon.encode(data)
        return _json_compact(data)

    def decode_output(self, text: str, output_format: str | None = None) -> Any:
        """Parse a model response back into a Python object when structured.

        TOON/YAML/JSON are attempted in turn; on failure the raw text is
        returned so callers never lose data.
        """
        fmt = output_format or self.config.output_format
        text = text.strip()
        if fmt == "yaml":
            try:
                import yaml

                return yaml.safe_load(text)
            except Exception:
                return text
        if fmt == "toon":
            return _decode_toon(text)
        try:
            import json

            return json.loads(text)
        except Exception:
            return text

    # ---- metrics ------------------------------------------------------

    def record_usage(self, usage: Any) -> None:
        """Record provider usage (Anthropic/OpenAI shapes both supported)."""
        cache_read = (
            _get(usage, "cache_read_input_tokens")
            or _get(usage, "cache_read_tokens")
            or _get_nested(usage, "prompt_tokens_details", "cached_tokens")
            or 0
        )
        out = _get(usage, "output_tokens") or _get(usage, "completion_tokens") or 0
        self.metrics.record_call(cache_read_tokens=int(cache_read))
        # Caveman/TOON savings are realized as a smaller actual output; we
        # record it as both raw and optimized so the call count stays honest
        # while the cache-read tokens still surface in the summary.
        self.metrics.record_output(int(out), int(out))

    def report(self) -> dict:
        return self.metrics.summary()

    def render(self) -> str:
        return self.metrics.render()

    def persist(self) -> None:
        if self.config.metrics_path:
            self.metrics.persist(self.config.metrics_path)


# ---- helpers ----------------------------------------------------------


def _json_compact(data: Any) -> str:
    import json

    return json.dumps(data, separators=(",", ":"))


def _get(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _get_nested(obj: Any, outer: str, inner: str) -> Any:
    o = _get(obj, outer)
    return _get(o, inner) if o is not None else None


def _decode_toon(text: str) -> Any:
    """Decode the tabular TOON subset back to JSON-compatible objects.

    Handles the uniform-array header form ``name[N]{k1,k2}:`` followed by
    comma rows. Falls back to returning the raw text for anything else.
    """
    import csv
    import io

    lines = text.split("\n")
    if not lines:
        return text
    header = lines[0].strip()
    if "{" not in header or "}" not in header or not header.endswith(":"):
        return text
    keys = header[header.index("{") + 1 : header.index("}")].split(",")
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        values = next(csv.reader(io.StringIO(line.strip())))
        rows.append(dict(zip(keys, values, strict=False)))
    return rows
