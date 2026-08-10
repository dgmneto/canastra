"""Plies-per-second benchmark: a random legal policy over a 64-game pool.

- `--shape pool` (default) measures the raw Rust loop — the number to watch
  when touching the pool or the encoder.
- `--shape league` measures the real generation loop through `drive_pool`:
  the same batched encode/forward/apply with the whole roster scored in one
  stacked `einsum` forward, and reports the per-ply phase split. This is the
  number to watch when touching the policy/glue. Use `--device` to compare
  cpu and cuda.

Per-ply cost scales with the batch size, so tune `--population`/`--opponents`/
`--seeds`/`--cap` to the shape you actually train with — the pop-8 defaults are
only for fast smoke; a real generation is pop 96 × 4 opponents × 8 seeds.

Run: `.venv/bin/python -m canastra_train.bench [--shape pool|league] [--device cpu]`
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from canastra_py import Pool

from canastra_train import ga, league
from canastra_train.train import TRAINING_ARCH


def _pool_shape() -> None:
    pool = Pool(list(range(64)))
    rng = np.random.default_rng(7)
    plies = 0
    start = time.perf_counter()
    while pool.has_live():
        _, _, mask, _rows = pool.encode()
        pool.apply([int(rng.integers(0, int(menu.sum()))) for menu in mask])
        plies += 1
    elapsed = time.perf_counter() - start
    print(f"{plies} plies in {elapsed:.2f}s = {plies / elapsed:.0f} plies/s")


def _league_shape(
    device: str,
    population: int,
    opponents: int,
    seed_count: int,
    cap: int,
    shards: int,
) -> None:
    pop = ga.initial_population(
        TRAINING_ARCH, ga.GAConfig(population=population), run_seed=7
    )
    hof = ga.HallOfFame()
    rng = np.random.default_rng(7)
    pairings = league.schedule_pairings(len(pop), opponents=opponents, hof=hof, rng=rng)
    seeds = list(range(11, 11 + seed_count))

    plies = 0
    phases = {"encode": 0.0, "forward": 0.0, "apply": 0.0, "glue": 0.0}

    def on_ply(encode_s: float, forward_s: float, apply_s: float) -> None:
        nonlocal plies
        plies += 1
        phases["encode"] += encode_s
        phases["forward"] += forward_s
        phases["apply"] += apply_s

    def progress_count(p: int, _finished: int) -> None:
        nonlocal plies
        plies = max(plies, p)

    began = time.perf_counter()
    if shards > 1:
        league.evaluate_generation(
            pop, hof, pairings, TRAINING_ARCH, seeds, cap, device,
            progress=progress_count, shards=shards,
        )
    else:
        stacked, game_seeds, meta = league.build_batch(
            pop, hof, pairings, TRAINING_ARCH, seeds=seeds, device=device
        )
        league.drive_pool(
            stacked, game_seeds, meta, cap=cap, device=device, on_ply=on_ply
        )
    elapsed = time.perf_counter() - began
    phases["glue"] = elapsed - sum(phases.values())
    print(
        f"{plies} plies in {elapsed:.2f}s = {plies / elapsed:.0f} plies/s "
        f"(device {device}, pop {population}, shards {shards})"
    )
    total = sum(phases.values()) or 1.0
    for name, seconds in phases.items():
        print(
            f"  {name:8s} {seconds:8.3f}s  {100 * seconds / total:5.1f}%  "
            f"{1000 * seconds / max(plies, 1):7.1f} ms/ply"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=["pool", "league"], default="pool")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--opponents", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--cap", type=int, default=30_000)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.shape == "pool":
        _pool_shape()
    else:
        _league_shape(
            args.device, args.population, args.opponents, args.seeds,
            args.cap, args.shards,
        )


if __name__ == "__main__":
    main()