"""One generation of self-play, batched into a single pool.

Every directed pairing (genome, opponent) contributes 2 x len(seeds) games —
duplicate deals, swapped seatings — and the pool's row metadata routes each
decision back to the genome that owns that seat. Fitness is the mean
duplicate-deal score differential (spec §G): score diff only, no win bonus,
unfinished matches at the cap count as-is.

The ply loop lives in `drive_pool` — shared with the bench, so the benchmark
measures exactly the code training runs.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
import torch
from canastra_py import Pool

from canastra_train import ga, policy
from canastra_train import genome as genome_mod

MatchRow = tuple[int, tuple[int, int], int | None, int, bool]


class BatchRoundLimitReached(RuntimeError):
    """Raised only by explicitly bounded calibration runs."""

    def __init__(self, max_rounds: int) -> None:
        super().__init__(f"batch round limit reached: {max_rounds}")
        self.max_rounds = max_rounds


@dataclass
class DriveMetrics:
    """Counters for one pool driver, measured in both useful units."""

    batch_rounds: int = 0
    individual_actions: int = 0
    elapsed_seconds: float = 0.0
    encode_seconds: float = 0.0
    forward_seconds: float = 0.0
    apply_seconds: float = 0.0
    action_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    unfinished_games: int = 0

    def reset(self, game_count: int) -> None:
        self.batch_rounds = 0
        self.individual_actions = 0
        self.elapsed_seconds = 0.0
        self.encode_seconds = 0.0
        self.forward_seconds = 0.0
        self.apply_seconds = 0.0
        self.action_counts = np.zeros(game_count, dtype=np.int64)
        self.unfinished_games = 0

    @property
    def batch_rounds_per_second(self) -> float:
        return self.batch_rounds / self.elapsed_seconds if self.elapsed_seconds else 0.0

    @property
    def individual_actions_per_second(self) -> float:
        return self.individual_actions / self.elapsed_seconds if self.elapsed_seconds else 0.0

    @property
    def max_actions_per_game(self) -> int:
        return int(self.action_counts.max()) if self.action_counts.size else 0

    @property
    def mean_actions_per_game(self) -> float:
        return (
            self.individual_actions / len(self.action_counts)
            if self.action_counts
            else 0.0
        )


def schedule_pairings(
    pop_size: int,
    opponents: int,
    hof: ga.HallOfFame,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Directed pairings: each genome vs `opponents` distinct others.

    Opponent indices address the combined roster used by evaluate_generation —
    population first (0..pop_size-1), then hall-of-fame entries. Spec §G: one
    of the opponents comes from the hall of fame once it exists.
    """
    pairings: list[tuple[int, int]] = []
    for me in range(pop_size):
        others = [i for i in range(pop_size) if i != me]
        chosen = list(rng.choice(others, size=min(opponents, len(others)), replace=False))
        if len(hof) and opponents >= 1:
            if not chosen:
                chosen.append(pop_size + int(rng.integers(0, len(hof))))
            else:
                chosen[-1] = pop_size + int(rng.integers(0, len(hof)))
        pairings.extend((me, int(opp)) for opp in chosen)
    return pairings


def batch_layout(
    pop: np.ndarray,
    hof: ga.HallOfFame,
    pairings: list[tuple[int, int]],
    seeds: list[int],
) -> tuple[list[np.ndarray], list[int], np.ndarray]:
    """Roster vectors, game seed list, and per-game pairing metadata.

    Returns (roster, game_seeds, meta) where roster is the population plus
    hall-of-fame vectors (indexed by roster position, which is what the
    pairing indices in meta address), and meta[i] = (a, b, seating) of game i.
    """
    roster = [pop[i] for i in range(len(pop))] + list(hof.genomes)
    game_seeds: list[int] = []
    game_meta: list[tuple[int, int, int]] = []
    for a, b in pairings:
        for seed in seeds:
            game_seeds.extend([seed, seed])
            game_meta.append((a, b, 0))
            game_meta.append((a, b, 1))
    return roster, game_seeds, np.asarray(game_meta, dtype=np.int64)


def build_stacked(roster: list[np.ndarray], arch: genome_mod.Arch, device: str) -> policy.WeightStack:
    """Stack the roster's flat genomes into one set of `[G, ...]` weight tensors.

    Built once per generation (or per shard worker), then reused every ply:
    the ply loop forwards the whole population in a handful of batched
    `einsum` calls instead of one tiny per-genome forward.
    """
    return policy.stack_weights(roster, arch, device)


def build_batch(
    pop: np.ndarray,
    hof: ga.HallOfFame,
    pairings: list[tuple[int, int]],
    arch: genome_mod.Arch,
    seeds: list[int],
    device: str,
) -> tuple[policy.WeightStack, list[int], np.ndarray]:
    """Stacked weights, game seed list, and per-game pairing metadata.

    Convenience wrapper over `batch_layout` + `build_stacked` for the
    single-process path; the sharded path uses the layout in the parent and
    builds weights inside each worker.
    """
    roster, game_seeds, meta = batch_layout(pop, hof, pairings, seeds)
    return build_stacked(roster, arch, device), game_seeds, meta


def drive_pool(
    stacked: policy.WeightStack,
    game_seeds: list[int],
    meta: np.ndarray,
    cap: int,
    device: str,
    progress: Callable[[int, int], None] | None = None,
    on_ply: Callable[[float, float, float], None] | None = None,
    metrics: DriveMetrics | None = None,
    max_rounds: int | None = None,
) -> list[MatchRow]:
    """Run every game to completion, one batched ply at a time.

    `progress`, when given, is called every few plies with
    (plies, games_finished) so a caller can render live progress and ETA.
    `on_ply(encode_s, forward_s, apply_s)`, when given, receives the per-ply
    wall-clock split — the bench uses it to attribute time to phases.
    """
    pool = Pool(game_seeds, max_actions_per_game=cap)
    if metrics is not None:
        metrics.reset(len(game_seeds))
    plies = 0
    started = time.perf_counter()
    try:
        while pool.has_live():
            if max_rounds is not None and plies >= max_rounds:
                raise BatchRoundLimitReached(max_rounds)
            began = time.perf_counter()
            obs, acts, mask, rows = pool.encode()
            encoded = time.perf_counter()
            games = rows[:, 0]
            seats = rows[:, 1]
            pairing = meta[games]
            # Genome owning this row: in seating 0, genome A holds the even seats
            # (team 0); in seating 1 the sides are swapped.
            genome_idx = np.where(
                (seats % 2 == 0) == (pairing[:, 2] == 0), pairing[:, 0], pairing[:, 1]
            )
            picks = _pick_batch(stacked, obs, acts, mask, genome_idx, device)
            scored = time.perf_counter()
            pool.apply(picks.tolist())
            applied = time.perf_counter()
            if metrics is not None:
                metrics.batch_rounds += 1
                metrics.individual_actions += len(rows)
                metrics.encode_seconds += encoded - began
                metrics.forward_seconds += scored - encoded
                metrics.apply_seconds += applied - scored
                np.add.at(metrics.action_counts, games, 1)
            if on_ply is not None:
                on_ply(encoded - began, scored - encoded, applied - scored)
            plies += 1
            if progress is not None and plies % 8 == 0:
                progress(plies, len(pool.results()))
        results = pool.results()
        if progress is not None:
            progress(plies, len(results))
        if metrics is not None:
            metrics.unfinished_games = sum(result[4] for result in results)
        return results
    finally:
        if metrics is not None:
            metrics.elapsed_seconds = time.perf_counter() - started


def _pick_batch(
    stacked: policy.WeightStack,
    obs: np.ndarray,
    acts: np.ndarray,
    mask: np.ndarray,
    genome_idx: np.ndarray,
    device: str,
) -> np.ndarray:
    """One argmax pick per pending row, routed to the row's owner genome.

    The dominant cost of a ply is moving the acts tensor to the device: the
    encoder pads every row to the ply's global max menu, so one rogue 600-action
    meld makes every row pay that width. On GPU (`device != "cpu"`) rows are
    sorted by their real menu size (p50 ≈ 2, p99 ≈ 26) and split into width
    buckets, so each bucket's acts tensor is trimmed to the bucket's max width —
    a ~2× cut in the cuda forward. CPU pays no transfer cost, so bucketing only
    adds small-op overhead there; it bypasses the bucket logic entirely.

    Both paths are bit-identical to each other (and so shard and single-process
    agree), because the einsum kernels are stable across the batch shape (N, G,
    and width), and the invariants below pin the last ulps:

    - Every group is padded to at least 4 rows. torch's batched kernels switch
      GEMM implementations when a batch element has N<=3 rows, which changes
      the last few ulps of the scores and flips near-tie picks. Padding to a
      floor keeps every ply on the stable kernel; the extra rows are masked
      and never contribute a pick.
    - The stacked weights are sliced to the genomes actually present (a
      batched-kernel-stable operation), never re-created, so the per-ply
      weights match the roster the generation was built from.
    """
    if device == "cpu":
        return _pick_batch_flat(stacked, obs, acts, mask, genome_idx)
    rw = mask.sum(axis=1)  # real menu size per row
    n_rows = len(rw)
    order = np.argsort(rw, kind="stable")
    rw_sorted = rw[order]
    picks = np.empty(n_rows, dtype=np.int64)

    # Width buckets: contiguous runs of rows by real menu size. Real menus are
    # tiny (p50 ~2, p99 ~26); the ply max can be 100s, so trimming to the
    # bucket max cuts the acts transfer to the GPU dramatically.
    caps = (8, 16, 32, 64)
    bounds = [0]
    for cap in caps:
        stop = int(np.searchsorted(rw_sorted, cap, side="right"))
        if stop > bounds[-1]:
            bounds.append(stop)
    if bounds[-1] < n_rows:
        bounds.append(n_rows)
    n_groups_full = stacked.trunk_w[0].shape[0]

    for start, end in pairwise(bounds):
        if start >= end:
            continue
        sel = order[start:end]
        bucket_width = int(rw_sorted[end - 1])
        # Sort the bucket's rows by genome for block einsum efficiency.
        o2 = np.argsort(genome_idx[sel], kind="stable")
        inner = sel[o2]
        g_sel = genome_idx[inner]
        counts = np.bincount(g_sel, minlength=n_groups_full)
        present = np.flatnonzero(counts > 0)
        n_max = max(int(counts[present].max()) if present.size else 0, 4)
        csum = np.zeros(n_groups_full, dtype=np.int64)
        csum[present] = np.cumsum(counts[present])
        grp_start = csum - counts
        padded = np.full((len(present), n_max), -1, dtype=np.int64)
        for gi, gid in enumerate(present):
            padded[gi, : counts[gid]] = np.arange(
                grp_start[gid], grp_start[gid] + counts[gid]
            )
        valid = padded >= 0
        padded = np.where(valid, padded, 0)
        row_order = inner[padded]

        # Compose the row index before indexing the source arrays. Chained
        # advanced indexing materializes an avoidable intermediate copy.
        obs_t = torch.from_numpy(obs[row_order]).to(device)
        acts_t = torch.from_numpy(acts[:, :bucket_width][row_order]).to(device)
        mask_t = (
            torch.from_numpy(mask[:, :bucket_width][row_order]).to(device)
            & torch.from_numpy(valid)[:, :, None].to(device)
        )
        pres_t = torch.from_numpy(present).to(device)
        sub = policy.WeightStack(
            [w[pres_t] for w in stacked.trunk_w],
            [b[pres_t] for b in stacked.trunk_b],
            [w[pres_t] for w in stacked.head_w],
            [b[pres_t] for b in stacked.head_b],
        )
        scores = policy.logits_stacked(sub, obs_t, acts_t, mask_t)
        picks[inner] = scores.argmax(dim=2).cpu().numpy()[valid]

    return picks


def _pick_batch_flat(
    stacked: policy.WeightStack,
    obs: np.ndarray,
    acts: np.ndarray,
    mask: np.ndarray,
    genome_idx: np.ndarray,
) -> np.ndarray:
    """The un-bucketed full-roster forward, used on CPU.

    One grouped pass over all rows at the ply's global acts width: batches over
    the full roster with every group padded to the largest, scored by
    `logits_stacked`. Bit-identical to the GPU bucket path (see `_pick_batch`).
    """
    order = np.argsort(genome_idx, kind="stable")
    sorted_idx = genome_idx[order]
    n_groups = stacked.trunk_w[0].shape[0]

    counts = np.bincount(sorted_idx, minlength=n_groups)
    ends = np.cumsum(counts)
    starts = ends - counts
    n_rows = len(sorted_idx)
    n_max = max(int(counts.max()) if n_groups else 0, 4)

    flat = np.arange(n_rows)
    padded = np.full((n_groups, n_max), -1, dtype=np.int64)
    for g in range(n_groups):
        padded[g, : counts[g]] = flat[starts[g] : ends[g]]
    valid = padded >= 0
    padded = np.where(valid, padded, 0)
    row_order = order[padded]

    obs_t = torch.from_numpy(obs[row_order])
    acts_t = torch.from_numpy(acts[row_order])
    mask_t = torch.from_numpy(mask[row_order]) & torch.from_numpy(valid)[:, :, None]

    scores = policy.logits_stacked(stacked, obs_t, acts_t, mask_t)
    picks_sorted = scores.argmax(dim=2).numpy()[valid]
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return picks_sorted[inverse]


def evaluate_generation(
    pop: np.ndarray,
    hof: ga.HallOfFame,
    pairings: list[tuple[int, int]],
    arch: genome_mod.Arch,
    seeds: list[int],
    cap: int,
    device: str,
    progress: Callable[[int, int], None] | None = None,
    shards: int = 1,
    metrics_out: list[DriveMetrics] | None = None,
    max_rounds: int | None = None,
) -> np.ndarray:
    """Mean duplicate-deal differential for every population genome.

    With `shards > 1` the generation's games are split across that many worker
    processes, each running its own batched ply loop (`shards.run_shards`).
    Results are bit-identical to the `shards == 1` path — per-game play is
    deterministic, and the workers return their results in global game order.
    `progress`, when given, receives the aggregated (plies, games_finished)
    counts across all shards.
    """
    roster, game_seeds, meta = batch_layout(pop, hof, pairings, seeds)
    if max_rounds is not None and shards > 1:
        raise ValueError("max_rounds calibration is only supported with shards=1")
    per_game = 2 * len(seeds)
    if shards > 1:
        from canastra_train import shards as shards_mod

        results = shards_mod.run_shards(
            roster, game_seeds, meta, arch, cap, device, shards, progress,
            metrics_out=metrics_out,
            max_rounds=max_rounds,
        )
    else:
        stacked = build_stacked(roster, arch, device)
        metrics = DriveMetrics()
        results = drive_pool(
            stacked,
            game_seeds,
            meta,
            cap,
            device,
            progress=progress,
            metrics=metrics,
            max_rounds=max_rounds,
        )
        if metrics_out is not None:
            metrics_out.append(metrics)

    # Aggregate duplicate-deal differentials per pairing, then per genome.
    assert len(results) == len(game_seeds)
    pairing_by_seed: dict[int, dict[int, list[float]]] = {
        p: {} for p in range(len(pairings))
    }
    for gidx, (_seed, result_scores, _winner, _hands, _unfinished) in enumerate(
        results
    ):
        pairing = gidx // per_game
        seating = gidx % 2
        seed = game_seeds[gidx]
        a_is_team_zero = seating == 0
        a_score = result_scores[0] if a_is_team_zero else result_scores[1]
        b_score = result_scores[1] if a_is_team_zero else result_scores[0]
        pairing_by_seed[pairing].setdefault(seed, []).append(a_score - b_score)

    fitness = np.zeros(len(pop), dtype=np.float64)
    counts = np.zeros(len(pop), dtype=np.float64)
    for pidx, (ga_idx, _gb_idx) in enumerate(pairings):
        diffs = []
        for seed in seeds:
            pair = pairing_by_seed[pidx][seed]
            assert len(pair) == 2, f"pairing {pidx} seed {seed} lost a seating"
            diffs.append((pair[0] + pair[1]) / 2)
        fitness[ga_idx] += float(np.mean(diffs))
        counts[ga_idx] += 1
    assert (counts > 0).all()
    return (fitness / counts).astype(np.float32)
