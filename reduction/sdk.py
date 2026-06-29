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

        # Char accounting tracks the *payload we actually shrink* (user turn +
        # context) — never the system instructions, which caveman intentionally
        # grows. This keeps optimized <= raw and the savings number honest.
        raw_chars = len(user) + sum(len(c) for c in (static_context or []))
        raw_chars += sum(len(c) for c in (volatile_context or []))

        # Layer 2: compress reused context documents (never instructions).
        context = list(static_context or [])
        if cfg.compress_context and context:
            before = "\n".join(context)
            context = compress.compress_documents(context, rate=cfg.compression_rate)
            self.metrics.record_input(before, "\n".join(context), layer="compress")
            applied.append("compress")

        # Always-on: normalize every text input, recording the savings.
        if cfg.strip_whitespace or cfg.dedupe_lines:
            norm_user = normalize.normalize(
                user, strip=cfg.strip_whitespace, dedupe=cfg.dedupe_lines
            )
            self.metrics.record_input(user, norm_user, layer="normalize")
            user = norm_user
            norm_context = []
            for c in context:
                nc = normalize.normalize(c, strip=cfg.strip_whitespace, dedupe=cfg.dedupe_lines)
                self.metrics.record_input(c, nc, layer="normalize")
                norm_context.append(nc)
            context = norm_context
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

        opt_chars = len(user) + sum(len(c) for c in context)
        opt_chars += sum(len(c) for c in (volatile_context or []))

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
        """Layer 1: shrink raw tool/command output before it re-enters context.

        With ``content_routing`` on, the output is classified (JSON/diff/log/...)
        and sent to the specialized compressor, and — when ``ccr`` is on — the
        original is stored so the agent can retrieve it via ``reduction_retrieve``.
        Otherwise it falls back to the zap binary / heuristic line filter.
        """
        if not self.config.shell_filter:
            return output

        # Prefer zap when a command is known and the binary is installed — it is
        # structure-aware per command. zap is the *command-wrapping* path.
        if command and shell.zap_available(self.config.zap_binary):
            filtered = shell.filter_tool_output(
                output,
                command=command,
                binary=self.config.zap_binary,
                max_lines=self.config.max_tool_output_lines,
            )
            self.metrics.record_input(output, filtered, layer="shell")
            return filtered

        # Otherwise classify the raw text and route to a content compressor.
        if self.config.content_routing:
            from reduction.content import compress_content

            result = compress_content(output, ccr=self.config.ccr, store=self._ccr_store())
            self.metrics.record_input(output, result.text, layer="content")
            return result.text

        # Last resort: built-in heuristic line filter.
        filtered = shell.builtin_filter(output, max_lines=self.config.max_tool_output_lines)
        self.metrics.record_input(output, filtered, layer="shell")
        return filtered

    def compress_messages(
        self,
        messages: list[Any],
        *,
        keep_last: int | None = None,
        ccr: bool | None = None,
    ) -> list[Any]:
        """Compress old turns in a multi-turn message list (keeps recent verbatim).

        The biggest token sink for long-horizon agents is accumulated history:
        every past tool result re-sent on every call. This compresses messages
        older than the last ``keep_last`` (content-aware + CCR-reversible),
        leaving system messages and recent turns untouched. Returns a new list;
        the input is not mutated. Savings are recorded under the ``history`` layer.

        ``keep_last``/``ccr`` default to the config (``history_keep_last``/``ccr``).
        """
        from reduction.layers import history

        kl = self.config.history_keep_last if keep_last is None else keep_last
        use_ccr = self.config.ccr if ccr is None else ccr
        result = history.compress_history(
            messages, keep_last=kl, ccr=use_ccr, store=self._ccr_store()
        )
        if result.tokens_before != result.tokens_after:
            self.metrics.record_input_tokens(
                result.tokens_before, result.tokens_after, layer="history"
            )
        return result.messages

    def _ccr_store(self):
        from reduction.ccr import CompressionStore, get_default_store

        # With an explicit path, use a dedicated persistent store (cached on the
        # instance). Without one, share the process-wide default store so the
        # MCP server / proxy can retrieve what this optimizer compressed.
        if self.config.ccr_store_path:
            cached = getattr(self, "_ccr_store_obj", None)
            if cached is None:
                cached = CompressionStore(path=self.config.ccr_store_path)
                self._ccr_store_obj = cached
            return cached
        return get_default_store()

    def retrieve(self, ref: str) -> str | None:
        """Expand a CCR ref back to its original content (None if unknown)."""
        return self._ccr_store().get(ref)

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
        """Record provider usage (Anthropic and OpenAI shapes both supported).

        Anthropic exposes ``input_tokens`` (uncached), ``cache_read_input_tokens``
        and ``cache_creation_input_tokens``; OpenAI exposes ``prompt_tokens`` with
        a nested ``prompt_tokens_details.cached_tokens``. Extraction is
        None-safe — a legitimate 0 is never confused with "field absent".
        """
        cache_read = _first_present(
            _get(usage, "cache_read_input_tokens"),
            _get(usage, "cache_read_tokens"),
            _get_nested(usage, "prompt_tokens_details", "cached_tokens"),
        )
        cache_write = _first_present(_get(usage, "cache_creation_input_tokens"))
        billed_input = _first_present(_get(usage, "input_tokens"), _get(usage, "prompt_tokens"))
        out = _first_present(_get(usage, "output_tokens"), _get(usage, "completion_tokens"))

        self.metrics.record_call(
            billed_input_tokens=billed_input,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        )
        # Output is recorded as observed only — Caveman/TOON output savings have
        # no counterfactual, so we never claim them as "saved" (see reduction.evals).
        self.metrics.record_output(out)

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


def _first_present(*values: Any) -> int:
    """Return the first non-None value as int (so a real 0 wins over absence)."""
    for v in values:
        if v is not None:
            return int(v)
    return 0


def _coerce_scalar(raw: str) -> Any:
    """Reverse the TOON scalar encoding: numbers, bools, null, quoted strings."""
    s = raw.strip()
    if s.startswith('"'):
        try:
            import json

            return json.loads(s)
        except Exception:
            return s
    if s == "null":
        return None
    if s == "true":
        return True
    if s == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _decode_toon(text: str) -> Any:
    """Decode the tabular TOON subset back to JSON-compatible objects.

    Handles the uniform-array header form ``name[N]{k1,k2}:`` followed by
    comma rows, restoring scalar types and unquoting quoted cells. Falls back
    to returning the raw text for anything it does not recognize.
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
        cells = next(csv.reader(io.StringIO(line.strip())))
        values = [_coerce_scalar(c) for c in cells]
        rows.append(dict(zip(keys, values, strict=False)))
    return rows
