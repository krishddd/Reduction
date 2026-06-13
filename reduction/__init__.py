"""Reduction — a five-layer token-optimization pipeline for AI agents.

Attacks token waste at every layer, from the shell to the serialized
response, on every input and output:

    Layer 1  shell / tool-output filtering (zap / RTK)
    Layer 2  context compression (LLMLingua-2)
    Layer 3  semantic cache (LiteLLM + Redis/Qdrant)
    Layer 4  native provider prompt caching (stable-prefix ordering)
    Layer 5  output shaping (Caveman persona + TOON/YAML)

Quick start:

    from reduction import TokenOptimizer
    opt = TokenOptimizer()
    req = opt.prepare(system="...", user="...", output_format="toon")
    # ...send to provider, then:
    opt.record_usage(resp.usage)
    print(opt.render())
"""

from reduction.config import OptimizerConfig
from reduction.metrics import Metrics
from reduction.sdk import OptimizedRequest, TokenOptimizer

__version__ = "0.2.0"

__all__ = [
    "TokenOptimizer",
    "OptimizedRequest",
    "OptimizerConfig",
    "Metrics",
]
