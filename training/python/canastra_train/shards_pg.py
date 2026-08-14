"""Sharded REINFORCE: process-parallel rollouts with gradient reduction.

The single-process `Trainer.step` runs one rollout at a time, so the GPU idles
during the Rust `encode`/`apply` crossings. This module spawns N worker
processes, each with its own `Pool` and its own `CanastraNet` copy on the GPU,
each driving a slice of the update's games. Workers compute losses, call
`backward` locally (accumulating grads into their own net's `.grad`), then ship
the flat grad to the parent. The parent sums, divides by the total number of
mini-batches (the mean-of-means estimate), clips, and steps the optimizer.

Seeds are partitioned by unique deal (keeping both duplicate-deal seatings in
the same shard), so each shard computes its own per-seed diffs as rewards — the
same variance-reduction the single-process path gets from the duplicate-deal
layout. Grad accumulation order in the parent is fixed (shard order), so same-
config (`--shards N`) resume reproduces parameters exactly. Cross-config
(different `--shards`) is not bit-identical — different reduction order + bf16 —
and that's documented (the PG path intentionally relaxes the GA's bit-identity
contract for bf16 throughput).

Workers pin `torch.set_num_threads(1)`: the win is parallelizing the per-ply
CPU-bound glue across cores, not intra-op parallelism (which would just
serialize on the memory subsystem). Each worker owns its own CUDA context; 4
shards ≈ 7GB on a 16GB card, 8 shards ≈ 14GB.
"""

from __future__ import annotations

import multiprocessing
import queue as queue_mod
import threading
import traceback
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from canastra_train.pg import Trainer


def run_sharded_step(
    trainer: Trainer,
    all_seeds: list[int],
    progress: Any = None,
) -> dict[str, Any]:
    """Run one update's rollouts across `trainer.shards` worker processes.

    Returns the aggregated metrics (mean_reward, plies, wins, …) and loads the
    summed flat gradient into the trainer's optimizer params via
    `trainer._load_flat_grad`. `progress(mini_batch, total_mb, plies, games_done)`
    is called during the rollout so a dashboard can show intra-update progress.
    """
    shards = trainer.shards
    n_deals = len(all_seeds)
    worker_count = min(shards, n_deals)
    # Partition unique deals across shards (keeping dup-deal pairs together).
    slices: list[tuple[int, list[int]]] = []
    for shard_id in range(worker_count):
        shard_seeds = [all_seeds[i] for i in range(shard_id, n_deals, worker_count)]
        if shard_seeds:
            slices.append((shard_id, shard_seeds))

    context = multiprocessing.get_context("spawn")
    results_queue: Any = context.Queue()
    progress_queue: Any = context.Queue()

    from canastra_train.pg import unwrap
    param_vec = unwrap(trainer.net).to_genome_vec()
    opp_vec = trainer._opp_vec()
    gen_seed = trainer._generator_seed()

    processes = []
    for shard_id, shard_seeds in slices:
        processes.append(
            context.Process(
                target=_shard_worker,
                args=(
                    trainer.arch,
                    trainer.cfg,
                    param_vec,
                    opp_vec,
                    shard_seeds,
                    trainer.baseline,
                    gen_seed,
                    shard_id,
                    results_queue,
                    progress_queue,
                ),
            )
        )

    stop = threading.Event()
    items: list[tuple[Any, ...]] = []
    latest_progress: dict[int, tuple[int, int, int]] = {}  # shard_id -> (mb, plies, games_done)

    def _progress_pump() -> None:
        while not stop.is_set():
            try:
                shard_id, mb, plies, games_done = progress_queue.get(timeout=0.05)
            except queue_mod.Empty:
                continue
            latest_progress[shard_id] = (int(mb), int(plies), int(games_done))
            if progress is not None:
                total_plies = sum(p[1] for p in latest_progress.values())
                total_games = sum(p[2] for p in latest_progress.values())
                total_mb = max(p[0] for p in latest_progress.values()) if latest_progress else 0
                progress(total_mb, trainer.cfg.num_mini_batches, total_plies, total_games)

    def _results_pump() -> None:
        while len(items) < len(processes) and not stop.is_set():
            try:
                items.append(results_queue.get(timeout=0.5))
            except queue_mod.Empty:
                continue

    prog_thread = threading.Thread(target=_progress_pump, daemon=True)
    pump = threading.Thread(target=_results_pump, daemon=True)
    prog_thread.start()
    pump.start()
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    stop.set()
    pump.join(timeout=5.0)
    prog_thread.join(timeout=2.0)

    if len(items) < len(processes):
        raise RuntimeError(f"{len(processes) - len(items)} shard worker(s) produced no results")

    # Sum flat grads across shards and divide by total mini-batches.
    total_mini_batches = 0
    flat_grad_sum: np.ndarray | None = None
    batch_rewards: list[float] = []
    total_plies = 0
    total_unfinished = 0
    total_wins = 0
    total_losses = 0
    total_games = 0
    total_actions = 0

    for shard_id, flat_grad, shard_metrics, *rest in items:
        if flat_grad is None:  # worker failure
            trace = rest[0] if rest else "unknown"
            raise RuntimeError(f"shard worker {shard_id} failed:\n{trace}")
        fg = np.asarray(flat_grad, dtype=np.float32)
        flat_grad_sum = fg if flat_grad_sum is None else flat_grad_sum + fg
        total_mini_batches += shard_metrics["mini_batches"]
        batch_rewards.extend(shard_metrics["rewards"])
        total_plies += shard_metrics["plies"]
        total_unfinished += shard_metrics["unfinished"]
        total_wins += shard_metrics["wins"]
        total_losses += shard_metrics["losses"]
        total_games += shard_metrics["games"]
        total_actions += shard_metrics["total_actions"]

    assert flat_grad_sum is not None and total_mini_batches > 0
    flat_grad_sum /= total_mini_batches
    trainer._load_flat_grad(flat_grad_sum)

    return {
        "mean_reward": float(np.mean(batch_rewards)) if batch_rewards else 0.0,
        "plies": total_plies,
        "unfinished": total_unfinished,
        "wins": total_wins,
        "losses": total_losses,
        "games": total_games,
        "mean_actions": total_actions / max(total_games, 1),
    }


def _shard_worker(
    arch: Any,
    cfg: Any,
    param_vec: np.ndarray,
    opp_vec: np.ndarray,
    shard_seeds: list[int],
    baseline: float,
    gen_seed: int,
    shard_id: int,
    results_queue: Any,
    progress_queue: Any,
) -> None:
    """One worker: build a net from `param_vec`, run its mini-batches, ship flat grad."""
    try:
        import torch

        torch.set_num_threads(1)
        from canastra_train import model as model_mod
        from canastra_train.pg import compute_loss, rollout

        device = cfg.device
        net = model_mod.CanastraNet.from_genome_vec(param_vec, arch).to(device)

        # Per-shard generator: same base seed as the single-process path (so
        # shards=1 reproduces the single-process grad exactly), offset by
        # shard_id so multiple shards sample independently.
        gen = torch.Generator(device=device)
        gen.manual_seed((gen_seed + shard_id) & 0x7FFFFFFF)

        # Run this shard's mini-batches, accumulating grads locally.
        mb_deals = cfg.mini_batch // 2
        batch_rewards: list[float] = []
        total_plies = 0
        total_unfinished = 0
        total_wins = 0
        total_losses = 0
        total_games = 0
        total_actions = 0
        mini_batches = 0

        for mb_start in range(0, len(shard_seeds), mb_deals):
            mb_seeds = shard_seeds[mb_start : mb_start + mb_deals]
            if not mb_seeds:
                break

            _games_so_far = total_games

            def _ply_progress(plies: int, games_done: int, _mb: int = mini_batches, _base: int = _games_so_far) -> None:
                try:
                    progress_queue.put((shard_id, _mb, plies, _base + games_done), block=False)
                except queue_mod.Full:
                    pass

            result = rollout(net, opp_vec, arch, mb_seeds, cfg, generator=gen, progress=_ply_progress)
            loss = compute_loss(result, baseline, cfg.entropy_coef, device)
            loss.backward()  # type: ignore[no-untyped-call]  # no scaling — parent divides

            batch_rewards.extend(result.rewards.tolist())
            total_plies += result.plies
            total_unfinished += result.unfinished
            total_wins += result.wins
            total_losses += result.losses
            total_games += len(result.rewards)
            total_actions += int(result.mean_actions * len(result.rewards))
            mini_batches += 1

        # Extract flat grad in the same order as to_genome_vec / from_genome_vec.
        parts = []
        for param in net.parameters():
            grad = param.grad
            if grad is None:
                parts.append(np.zeros(param.numel(), dtype=np.float32))
            else:
                parts.append(grad.detach().cpu().to(torch.float32).numpy().ravel())
        flat_grad = np.concatenate(parts).astype(np.float32)

        metrics = {
            "rewards": batch_rewards,
            "plies": total_plies,
            "unfinished": total_unfinished,
            "wins": total_wins,
            "losses": total_losses,
            "games": total_games,
            "total_actions": total_actions,
            "mini_batches": mini_batches,
        }
        results_queue.put((shard_id, flat_grad, metrics))
    except Exception:  # noqa: BLE001 - a worker must never exit silently
        results_queue.put((shard_id, None, {}, traceback.format_exc()))
