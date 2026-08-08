"""Deterministic seed streams.

SplitMix64 finalizer — the same mixer family the engine uses for hand seeds.
Everything derives from (run_seed, generation), so a resumed run regenerates
exactly the seeds it had without storing RNG state.
"""

from __future__ import annotations

MASK = (1 << 64) - 1


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
    return (x ^ (x >> 31)) & MASK


def generation_seeds(run_seed: int, generation: int, count: int) -> list[int]:
    """`count` distinct u64 seeds shared by every pairing of one generation."""
    base = splitmix64((run_seed & MASK) ^ ((generation * 0xD1B54A32D192ED03) & MASK))
    return [splitmix64(base + i) for i in range(count)]
