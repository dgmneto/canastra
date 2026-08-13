"""Plies-per-second benchmark: a random legal policy over a 64-game pool.

- `--shape pool` (default) measures the raw Rust loop — the number to watch
  when touching the pool or the encoder.
- `--shape league` measures the real generation loop through `drive_pool`:
  the same batched encode/forward/apply with the whole roster scored in one
  stacked policy-kernel forward, and reports the per-ply phase split. This is
  the number to watch when touching the policy/glue. Use `--device` to compare
  cpu and cuda, and `--kernel` to select the policy implementation.

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

from canastra_train import elo as elo_mod
from canastra_train import ga, league, policy
from canastra_train.train import TRAINING_ARCH


def _pool_shape(
    max_rounds: int | None,
    game_count: int,
    pool_cap: int | None,
) -> None:
    pool = Pool(list(range(game_count)), max_actions_per_game=pool_cap)
    rng = np.random.default_rng(7)
    plies = 0
    actions = 0
    action_counts = np.zeros(game_count, dtype=np.int64)
    start = time.perf_counter()
    while pool.has_live():
        if max_rounds is not None and plies >= max_rounds:
            break
        _, _, mask, rows = pool.encode()
        pool.apply([int(rng.integers(0, int(menu.sum()))) for menu in mask])
        actions += len(rows)
        for game in rows[:, 0]:
            action_counts[int(game)] += 1
        plies += 1
    elapsed = time.perf_counter() - start
    unfinished = sum(result[4] for result in pool.results())
    print(
        f"{plies} batch rounds in {elapsed:.2f}s = {plies / elapsed:.0f} rounds/s; "
        f"{actions} individual actions = {actions / elapsed:.0f} actions/s; "
        f"unfinished {unfinished}/{game_count}"
    )
    print(
        f"  actions/game mean {action_counts.mean():.1f} p95 {np.percentile(action_counts, 95):.1f} "
        f"p99 {np.percentile(action_counts, 99):.1f} max {action_counts.max()}"
    )


def _league_shape(
    device: str,
    population: int,
    opponents: int,
    seed_count: int,
    cap: int,
    shards: int,
    max_rounds: int | None,
    kernel: policy.PolicyKernel,
) -> None:
    pop = ga.initial_population(
        TRAINING_ARCH, ga.GAConfig(population=population), run_seed=7
    )
    hof = ga.HallOfFame()
    rng = np.random.default_rng(7)
    pairings = league.schedule_pairings(len(pop), opponents=opponents, hof=hof, rng=rng)
    seeds = list(range(11, 11 + seed_count))

    drive_metrics: list[league.DriveMetrics] = []

    def progress_count(_p: int, _finished: int) -> None:
        return

    began = time.perf_counter()
    limited = False
    elo = elo_mod.EloTracker(population)
    if shards > 1:
        league.evaluate_generation(
            pop, hof, pairings, TRAINING_ARCH, seeds, cap, device, elo,
            progress=progress_count, shards=shards, metrics_out=drive_metrics,
            max_rounds=max_rounds, kernel=kernel,
        )
    else:
        stacked, game_seeds, meta = league.build_batch(
            pop, hof, pairings, TRAINING_ARCH, seeds=seeds, device=device
        )
        metrics = league.DriveMetrics()
        try:
            league.drive_pool(
                stacked,
                game_seeds,
                meta,
                cap=cap,
                device=device,
                metrics=metrics,
                max_rounds=max_rounds,
                kernel=kernel,
            )
        except league.BatchRoundLimitReached:
            limited = True
        drive_metrics.append(metrics)
    elapsed = time.perf_counter() - began
    batch_rounds = sum(item.batch_rounds for item in drive_metrics)
    individual_actions = sum(item.individual_actions for item in drive_metrics)
    action_counts = np.concatenate(
        [np.asarray(item.action_counts, dtype=np.int64) for item in drive_metrics]
    ) if drive_metrics else np.zeros(0, dtype=np.int64)
    print(
        f"{batch_rounds} aggregate batch rounds in {elapsed:.2f}s = "
        f"{batch_rounds / elapsed:.0f} rounds/s; "
        f"{individual_actions / elapsed:.0f} individual actions/s "
        f"(device {device}, kernel {kernel}, pop {population}, shards {shards})"
    )
    if limited:
        print(f"  bounded sample stopped at --max-rounds {max_rounds}")
    print(
        f"  critical-path rounds/s {max((item.batch_rounds for item in drive_metrics), default=0) / elapsed:.0f}; "
        f"unfinished {sum(item.unfinished_games for item in drive_metrics)}; "
        f"actions/game mean {action_counts.mean():.1f} "
        f"p95 {np.percentile(action_counts, 95):.1f} "
        f"p99 {np.percentile(action_counts, 99):.1f} max {action_counts.max()}"
    )
    phases = {
        "encode": sum(item.encode_seconds for item in drive_metrics),
        "forward": sum(item.forward_seconds for item in drive_metrics),
        "apply": sum(item.apply_seconds for item in drive_metrics),
    }
    phases["glue"] = max(elapsed - max(phases.values(), default=0.0), 0.0)
    total = sum(phases.values()) or 1.0
    for name, seconds in phases.items():
        print(
            f"  {name:8s} {seconds:8.3f}s  {100 * seconds / total:5.1f}%  "
            f"{1000 * seconds / max(batch_rounds, 1):7.1f} ms/round"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=["pool", "league"], default="pool")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--kernel", choices=["einsum", "bmm"], default="einsum")
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--opponents", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--cap", type=int, default=30_000)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument(
        "--games",
        type=int,
        default=64,
        help="number of games for the raw pool benchmark",
    )
    parser.add_argument(
        "--pool-cap",
        type=int,
        default=None,
        help="per-game action cap for the raw pool benchmark",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="stop after this many batch rounds; useful for bounded calibration runs",
    )
    args = parser.parse_args()
    if args.max_rounds is not None and args.shards > 1:
        parser.error("--max-rounds currently requires --shards 1")
    if args.shape == "pool":
        _pool_shape(args.max_rounds, args.games, args.pool_cap)
    else:
        _league_shape(
            args.device, args.population, args.opponents, args.seeds,
            args.cap, args.shards, args.max_rounds,
            args.kernel,
        )


if __name__ == "__main__":
    main()
