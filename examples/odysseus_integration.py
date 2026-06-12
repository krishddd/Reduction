"""Wire the Reduction pipeline into the odysseus security agent.

The agent (Agent_security_testing/Security_module) routes every LLM call
through ``llm.client.ClaudeClient.message(...)``. Wrapping that one method
gives the whole scan caveman output + TOON serialization + normalized inputs
+ savings metrics — with no change to the planner / synthesizer / triager
call sites.

Run from inside Security_module (so its imports resolve), or copy the wiring
into ``cli.py`` where the client is constructed.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python examples/odysseus_integration.py` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def install(client, *, output_format: str = "toon", caveman: bool = True):
    """Monkeypatch a ClaudeClient instance in place; return the optimizer.

    >>> from llm.client import ClaudeClient
    >>> client = ClaudeClient()
    >>> opt = install(client, output_format="toon")
    >>> # ... run scan ...
    >>> print(opt.render())
    """
    from reduction import TokenOptimizer
    from reduction.adapters import wrap_message_fn

    optimizer = TokenOptimizer()
    client.message = wrap_message_fn(
        client.message, optimizer, output_format=output_format, caveman_on=caveman
    )
    return optimizer


def demo() -> None:
    """Offline demo with a fake client that mimics the message() signature."""
    from reduction import TokenOptimizer
    from reduction.adapters import wrap_message_fn

    class FakeUsage:
        output_tokens = 120
        cache_read_input_tokens = 4000

    class FakeResponse:
        text = "vulns[1]{id,sev}:\n  CVE-2026-1,high"
        usage = FakeUsage()

    class FakeClient:
        def message(self, *, model, system, user, **kwargs):
            assert "Caveman" in system, "caveman skill should be injected"
            assert "TOON" in system, "format contract should be injected"
            return FakeResponse()

    client = FakeClient()
    opt = TokenOptimizer()
    client.message = wrap_message_fn(client.message, opt, output_format="toon")

    resp = client.message(
        model="claude-sonnet-4-6",
        system="You are an ASI security planner.",
        user="target:\n\n\n  https://example.test   \n  https://example.test",
    )
    print("response:", resp.text)
    print()
    print(opt.render())


if __name__ == "__main__":
    demo()
