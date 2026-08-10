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

import math
import multiprocessing
import queue as queue_mod
import threading
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from canastra_train import genome as genome_mod

if TYPE_CHECKING:
    from canastra_train.league import MatchRow

_Progress = Callable[[int, int], None]


def run_shards(
    roster: list[np.ndarray],
    game_seeds: list[int],
    meta: np.ndarray,
    arch: genome_mod.Arch,
    cap: int,
    device: str,
    shards: int,
    progress: _Progress | None,
) -> list[MatchRow]:
    """Evaluate `game_seeds` across `shards` worker processes.

    The seed list is split into contiguous global-order slices (one per
    worker, mirror slicing), so concatenating the returned results in shard
    order reproduces the single-pool result order exactly.
    """
    chunk = math.ceil(len(game_seeds) / shards)
    slices = [
        (game_seeds[start : start + chunk], meta[start : start + chunk])
        for start in range(0, len(game_seeds), chunk)
    ]
    context = multiprocessing.get_context("spawn")
    results_queue: Any = context.Queue()
    progress_queue: Any = context.Queue()

    processes = []
    for shard_id, (shard_seeds, shard_meta) in enumerate(slices):
        if not shard_seeds:
            break
        processes.append(
            context.Process(
                target=_shard_worker,
                args=(
                    roster,
                    arch,
                    shard_seeds,
                    shard_meta,
                    cap,
                    device,
                    results_queue,
                    progress_queue,
                    shard_id,
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

    for shard_id, shard_results, *rest in items:
        if shard_results is None:  # worker failure carries (id, None, tb)
            trace = rest[0] if rest else "unknown"
            raise RuntimeError(f"shard worker {shard_id} failed:\n{trace}")
    if len(items) < len(processes):
        raise RuntimeError(
            f"{len(processes) - len(items)} shard worker(s) produced no results"
        )

    results: list[MatchRow] = []
    for shard_id, shard_results in sorted(
        (item[0], item[1]) for item in items if item[1] is not None
    ):
        results.extend(shard_results)
    assert len(results) == len(game_seeds), "shards lost games"
    return results


def _shard_worker(
    roster: list[np.ndarray],
    arch: genome_mod.Arch,
    game_seeds: list[int],
    meta: np.ndarray,
    cap: int,
    device: str,
    results_queue: Any,
    progress_queue: Any,
    shard_id: int,
) -> None:
    from canastra_train import league

    try:
        import torch

        torch.set_num_threads(1)
        stacked = league.build_stacked(roster, arch, device)

        def _progress(plies: int, finished: int) -> None:
            try:
                progress_queue.put((shard_id, plies, finished), block=False)
            except queue_mod.Full:
                pass

        results = league.drive_pool(
            stacked, game_seeds, meta, cap, device, progress=_progress
        )
        results_queue.put((shard_id, results))
    except Exception:  # noqa: BLE001 - a worker must never exit silently
        results_queue.put((shard_id, None, traceback.format_exc()))