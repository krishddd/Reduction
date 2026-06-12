"""Compounded-savings cost simulator.

Models how the five layers multiply against provider pricing. Reductions are
applied to the already-reduced stream — compounded, not additive.

Usage:
    python simulator/simulate.py --daily-input-tokens 5000000 --daily-output-tokens 800000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

# $ per 1M tokens (Claude Sonnet 4.6, 2026)
INPUT_RATE = 3.00
OUTPUT_RATE = 15.00
CACHE_READ_RATE = 0.30  # 90% discount on cached prefix reads


@dataclass
class Layer:
    name: str
    input_reduction: float = 0.0  # fraction of remaining input tokens removed
    output_reduction: float = 0.0  # fraction of remaining output tokens removed


DEFAULT_LAYERS = [
    Layer("L1 shell filtering (zap)", input_reduction=0.70),
    Layer("L2 context compression (LLMLingua-2)", input_reduction=0.40),
    Layer("L3 semantic cache (25% hit rate)", input_reduction=0.25, output_reduction=0.25),
    Layer("L5 caveman + TOON output", output_reduction=0.45),
]


def simulate(
    daily_input: float,
    daily_output: float,
    cached_prefix_fraction: float = 0.6,
    layers: list[Layer] = DEFAULT_LAYERS,
) -> None:
    base_cost = (daily_input / 1e6) * INPUT_RATE + (daily_output / 1e6) * OUTPUT_RATE
    print(
        f"{'baseline':45s} in={daily_input:>12,.0f} out={daily_output:>11,.0f} "
        f"${base_cost:,.2f}/day"
    )

    inp, out = daily_input, daily_output
    for layer in layers:
        inp *= 1 - layer.input_reduction
        out *= 1 - layer.output_reduction
        cost = (inp / 1e6) * INPUT_RATE + (out / 1e6) * OUTPUT_RATE
        print(f"{'after ' + layer.name:45s} in={inp:>12,.0f} out={out:>11,.0f} ${cost:,.2f}/day")

    # L4: of the surviving input, the stable prefix bills at the cache-read rate.
    cached = inp * cached_prefix_fraction
    uncached = inp - cached
    final_cost = (
        (uncached / 1e6) * INPUT_RATE + (cached / 1e6) * CACHE_READ_RATE + (out / 1e6) * OUTPUT_RATE
    )
    print(
        f"{'after L4 native prompt cache':45s} "
        f"({cached_prefix_fraction:.0%} of input at cache-read rate) ${final_cost:,.2f}/day"
    )

    print("-" * 80)
    savings = 1 - final_cost / base_cost
    print(f"daily:   ${base_cost:,.2f} -> ${final_cost:,.2f}  ({savings:.1%} saved)")
    print(f"monthly: ${base_cost * 30:,.2f} -> ${final_cost * 30:,.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-input-tokens", type=float, default=5_000_000)
    parser.add_argument("--daily-output-tokens", type=float, default=800_000)
    parser.add_argument("--cached-prefix-fraction", type=float, default=0.6)
    args = parser.parse_args()
    simulate(args.daily_input_tokens, args.daily_output_tokens, args.cached_prefix_fraction)


if __name__ == "__main__":
    main()
