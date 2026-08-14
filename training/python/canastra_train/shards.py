"""Process-parallel generation evaluation: the games split into shards.

Each shard is a spawned worker process holding its own engine `Pool` and its
own stacked roster weights (built from the shared genome vectors), looping its
slice of the generation's games with the exact same `drive_pool` driver the
single-process path uses. Workers pull progress counts through a queue so the
dashboard keeps ticking; results come back in global game order, so the
aggregation — and therefore the fitness — is bit-identical to `shards == 1`.

Workers pin `torch.set_num_threads(1)`: each worker already owns the whole
population as one batched forward, and 8 processes × 16 intra-op threads would
just serialize on the memory subsystem. The win comes from parallelizing the
per-ply work (encode/apply/torch dispatch) across cores, which the GIL
serializes inside a single process.
"""

from __future__ import annotations

import multiprocessing
import queue as queue_mod
import threading
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from canastra_train import genome as genome_mod
from canastra_train import policy

if TYPE_CHECKING:
    from canastra_train.league import MatchRow

_Progress = Callable[[int, int], None]


def run_shards(
    roster: list[np.ndarray],
    game_seeds: list[int],
    meta: np.ndarray,
    arch: genome_mod.Arch,
    max_hands: int | None,
    device: str,
    shards: int,
    progress: _Progress | None,
    metrics_out: list[Any] | None = None,
    max_rounds: int | None = None,
    kernel: policy.PolicyKernel = "einsum",
) -> list[MatchRow]:
    """Evaluate `game_seeds` across `shards` worker processes.

    Games are interleaved across workers to spread paired seatings, seeds, and
    opponents across the critical path. Results carry their original indices
    and are reassembled into the same order as the single-pool path.
    """
    worker_count = min(shards, len(game_seeds))
    slices = []
    for shard_id in range(worker_count):
        indices = list(range(shard_id, len(game_seeds), worker_count))
        slices.append(
            (
                indices,
                [game_seeds[index] for index in indices],
                meta[indices],
            )
        )
    context = multiprocessing.get_context("spawn")
    results_queue: Any = context.Queue()
    progress_queue: Any = context.Queue()

    processes = []
    for shard_id, (shard_indices, shard_seeds, shard_meta) in enumerate(slices):
        if not shard_seeds:
            break
        processes.append(
            context.Process(
                target=_shard_worker,
                args=(
                    roster,
                    arch,
                    shard_indices,
                    shard_seeds,
                    shard_meta,
                    max_hands,
                    device,
                    results_queue,
                    progress_queue,
                    shard_id,
                    max_rounds,
                    kernel,
                ),
            )
        )

    stop = threading.Event()
    latest: dict[int, tuple[int, int]] = {}
    items: list[tuple[Any, ...]] = []

    def _pump() -> None:
        while not stop.is_set():
            try:
                shard_id, plies, finished = progress_queue.get(timeout=0.05)
            except queue_mod.Empty:
                continue
            latest[shard_id] = (int(plies), int(finished))
            if progress is not None:
                progress(
                    sum(entry[0] for entry in latest.values()),
                    sum(entry[1] for entry in latest.values()),
                )

    def _results_pump() -> None:
        while len(items) < len(processes) and not stop.is_set():
            try:
                items.append(results_queue.get(timeout=0.1))
            except queue_mod.Empty:
                continue

    pumps = [threading.Thread(target=_pump, daemon=True),
             threading.Thread(target=_results_pump, daemon=True)]
    for pump in pumps:
        pump.start()
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    stop.set()
    for pump in pumps:
        pump.join(timeout=5.0)

    metrics_by_shard: dict[int, Any] = {}
    results_by_index: list[Any | None] = [None] * len(game_seeds)
    for shard_id, shard_indices, shard_results, *rest in items:
        if shard_results is None:  # worker failure carries (id, indices, None, tb)
            trace = rest[0] if rest else "unknown"
            raise RuntimeError(f"shard worker {shard_id} failed:\n{trace}")
        if rest:
            metrics_by_shard[shard_id] = rest[0]
        for index, result in zip(shard_indices, shard_results):
            results_by_index[index] = result
    if len(items) < len(processes):
        raise RuntimeError(
            f"{len(processes) - len(items)} shard worker(s) produced no results"
        )

    results = [result for result in results_by_index if result is not None]
    if len(results) != len(game_seeds):
        raise RuntimeError("shards lost games while reassembling global order")
    if metrics_out is not None:
        metrics_out.extend(
            metrics_by_shard[shard_id] for shard_id in sorted(metrics_by_shard)
        )
    assert len(results) == len(game_seeds), "shards lost games"
    return results


def _shard_worker(
    roster: list[np.ndarray],
    arch: genome_mod.Arch,
    game_indices: list[int],
    game_seeds: list[int],
    meta: np.ndarray,
    max_hands: int | None,
    device: str,
    results_queue: Any,
    progress_queue: Any,
    shard_id: int,
    max_rounds: int | None,
    kernel: policy.PolicyKernel,
) -> None:
    from canastra_train import league

    try:
        import torch

        torch.set_num_threads(1)
        stacked = league.build_stacked(roster, arch, device)
        metrics = league.DriveMetrics()

        def _progress(plies: int, finished: int) -> None:
            try:
                progress_queue.put((shard_id, plies, finished), block=False)
            except queue_mod.Full:
                pass

        results = league.drive_pool(
            stacked,
            game_seeds,
            meta,
            max_hands,
            device,
            progress=_progress,
            metrics=metrics,
            max_rounds=max_rounds,
            kernel=kernel,
        )
        results_queue.put((shard_id, game_indices, results, metrics))
    except Exception:  # noqa: BLE001 - a worker must never exit silently
        results_queue.put((shard_id, game_indices, None, traceback.format_exc()))
