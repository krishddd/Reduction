"""Drop-in adapters that wrap an existing LLM client with the optimizer.

- ``OptimizedAnthropic``  — wraps ``anthropic.Anthropic``.
- ``OptimizedOpenAI``     — wraps ``openai.OpenAI``.
- ``wrap_message_fn``     — wraps any ``message(model, system, user, ...)``
                            callable (e.g. the odysseus security agent's
                            ``ClaudeClient.message``).
"""

from reduction.adapters.anthropic import OptimizedAnthropic
from reduction.adapters.odysseus import wrap_message_fn
from reduction.adapters.openai import OptimizedOpenAI

__all__ = ["OptimizedAnthropic", "OptimizedOpenAI", "wrap_message_fn"]
