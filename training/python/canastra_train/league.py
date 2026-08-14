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

import queue as queue_mod
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from canastra_py import Pool

from canastra_train import elo as elo_mod
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
    the ply loop forwards the whole population in a handful of batched policy
    kernel calls instead of one tiny per-genome forward.
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
    max_hands: int | None,
    device: str,
    progress: Callable[[int, int], None] | None = None,
    on_ply: Callable[[float, float, float], None] | None = None,
    metrics: DriveMetrics | None = None,
    max_rounds: int | None = None,
    kernel: policy.PolicyKernel = "einsum",
) -> list[MatchRow]:
    """Run every game to completion, one batched ply at a time.

    `max_hands` stops each game after that many hands have settled (scores
    banked); ``None`` plays full matches to MatchOver. A hand is guaranteed to
    finish in finite time (the stock depletes), so no action cap is needed.
    `progress`, when given, is called every few plies with
    (plies, games_finished) so a caller can render live progress and ETA.
    `on_ply(encode_s, forward_s, apply_s)`, when given, receives the per-ply
    wall-clock split — the bench uses it to attribute time to phases.
    """
    pool = Pool(game_seeds, max_hands=max_hands)
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
            genome_idx = np.where(
                (seats % 2 == 0) == (pairing[:, 2] == 0), pairing[:, 0], pairing[:, 1]
            )
            picks = _pick_batch(stacked, obs, acts, mask, genome_idx, device, kernel)
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


class _GpuRequest:
    """One forward request submitted by a worker thread to the GpuServer."""
    __slots__ = ("acts", "done", "genome_idx", "mask", "obs", "result")

    def __init__(self, obs: np.ndarray, acts: np.ndarray, mask: np.ndarray,
                 genome_idx: np.ndarray) -> None:
        self.obs = obs
        self.acts = acts
        self.mask = mask
        self.genome_idx = genome_idx
        self.result: np.ndarray | Exception | None = None
        self.done = threading.Event()


class GpuServer:
    """A single GPU thread that processes forward requests from worker threads.

    Workers submit (_GpuRequest) to the queue and block on ``request.done``.
    The server thread dequeues requests one at a time and runs ``_pick_batch``
    in its own CUDA context. This eliminates the multi-process CUDA context
    switching that costs ~30% with 6 shard processes, and eliminates the 440MB
    roster IPC per generation.

    The overlap: while the GpuServer runs torch (GIL released during CUDA
    kernel launches and memcpys), worker threads run Pool::encode / Pool::apply
    (GIL released via py.allow_threads in the Rust layer).
    """

    def __init__(self, stacked: policy.WeightStack, device: str,
                 kernel: policy.PolicyKernel = "einsum") -> None:
        self._stacked = stacked
        self._device = device
        self._kernel = kernel
        self._queue: queue_mod.Queue[_GpuRequest | None] = queue_mod.Queue()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def forward(self, obs: np.ndarray, acts: np.ndarray, mask: np.ndarray,
                genome_idx: np.ndarray) -> np.ndarray:
        """Submit a forward request and block until the result is ready."""
        req = _GpuRequest(obs, acts, mask, genome_idx)
        self._queue.put(req)
        req.done.wait()
        if isinstance(req.result, Exception):
            raise req.result
        return req.result  # type: ignore[return-value]

    def shutdown(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5.0)

    def _serve(self) -> None:
        while True:
            req = self._queue.get()
            if req is None:
                break
            try:
                req.result = _pick_batch(
                    self._stacked, req.obs, req.acts, req.mask,
                    req.genome_idx, self._device, self._kernel,
                )
            except Exception as e:  # noqa: BLE001
                req.result = e
            req.done.set()


def drive_pool_coalesced(
    stacked: policy.WeightStack,
    game_seeds: list[int],
    meta: np.ndarray,
    max_hands: int | None,
    device: str,
    n_workers: int = 6,
    progress: Callable[[int, int], None] | None = None,
    metrics: DriveMetrics | None = None,
    max_rounds: int | None = None,
    kernel: policy.PolicyKernel = "einsum",
) -> list[MatchRow]:
    """Multi-threaded driver with a single coalesced GPU server.

    Spawns ``n_workers`` threads, each with its own Pool of ~1/n_workers games.
    A single GpuServer thread processes all forward requests serially in one
    CUDA context. Workers overlap encode/apply (GIL-released Rust) with the
    GPU server's forward (GIL-released CUDA ops).

    This replaces the multi-process shard approach: no IPC (440MB roster × N),
    no CUDA context switching, no process spawn per generation. The GPU
    processes the same number of forwards but in one context, and the CPU
    work overlaps naturally via the GIL release.
    """
    gpu = GpuServer(stacked, device, kernel)
    n_games = len(game_seeds)
    n_workers = min(n_workers, n_games)

    # Split games across workers (interleaved, same as shards).
    worker_slices: list[tuple[list[int], list[int], np.ndarray]] = []
    for wid in range(n_workers):
        indices = list(range(wid, n_games, n_workers))
        worker_slices.append((
            indices,
            [game_seeds[i] for i in indices],
            meta[indices],
        ))

    all_results: list[MatchRow | None] = [None] * n_games
    counters_lock = threading.Lock()
    total_plies = [0]
    total_actions = [0]
    total_encode_s = [0.0]
    total_forward_s = [0.0]
    total_apply_s = [0.0]
    action_counts = np.zeros(n_games, dtype=np.int64) if metrics is not None else None

    def _worker(wid: int, worker_indices: list[int],
                worker_seeds: list[int], worker_meta: np.ndarray) -> None:
        pool = Pool(worker_seeds, max_hands=max_hands)
        local_plies = 0
        while pool.has_live():
            if max_rounds is not None and local_plies >= max_rounds:
                raise BatchRoundLimitReached(max_rounds)
            began = time.perf_counter()
            obs, acts, mask, rows = pool.encode()
            encoded = time.perf_counter()
            games = rows[:, 0]
            seats = rows[:, 1]
            pairing = worker_meta[games]
            genome_idx = np.where(
                (seats % 2 == 0) == (pairing[:, 2] == 0),
                pairing[:, 0], pairing[:, 1],
            )
            picks = gpu.forward(obs, acts, mask, genome_idx)
            scored = time.perf_counter()
            pool.apply(picks.tolist())
            applied = time.perf_counter()
            local_plies += 1
            with counters_lock:
                total_plies[0] += 1
                total_actions[0] += len(rows)
                total_encode_s[0] += encoded - began
                total_forward_s[0] += scored - encoded
                total_apply_s[0] += applied - scored
                if action_counts is not None:
                    np.add.at(action_counts, games, 1)
                if progress is not None and total_plies[0] % 8 == 0:
                    finished = sum(r is not None for r in all_results)
                    progress(total_plies[0], finished)
        # Collect results and map back to global indices.
        results = pool.results()
        for local_i, global_i in enumerate(worker_indices):
            all_results[global_i] = results[local_i]

    threads = []
    for wid, (indices, seeds_slice, meta_slice) in enumerate(worker_slices):
        t = threading.Thread(target=_worker, args=(wid, indices, seeds_slice, meta_slice))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    gpu.shutdown()

    results = [r for r in all_results if r is not None]
    if len(results) != n_games:
        raise RuntimeError(f"coalesced driver lost games: {len(results)}/{n_games}")

    if metrics is not None:
        metrics.batch_rounds = total_plies[0]
        metrics.individual_actions = total_actions[0]
        metrics.encode_seconds = total_encode_s[0]
        metrics.forward_seconds = total_forward_s[0]
        metrics.apply_seconds = total_apply_s[0]
        if action_counts is not None:
            metrics.action_counts = action_counts
        metrics.unfinished_games = sum(r[4] for r in results)
        metrics.elapsed_seconds = time.perf_counter() - 0  # set by caller

    if progress is not None:
        progress(total_plies[0], len(results))

    return results


def _pick_batch(
    stacked: policy.WeightStack,
    obs: np.ndarray,
    acts: np.ndarray,
    mask: np.ndarray,
    genome_idx: np.ndarray,
    device: str,
    kernel: policy.PolicyKernel = "einsum",
) -> np.ndarray:
    """One argmax pick per pending row, routed to the row's owner genome.

    The dominant cost of a ply is moving the acts tensor to the device: the
    encoder pads every row to the ply's global max menu, so one rogue 288-action
    meld makes every row pay that width. On GPU (`device != "cpu"`) rows are
    bucketed by their real menu size (p50 ≈ 2, p99 ≈ 26) into power-of-2 width
    buckets, so each bucket's acts tensor is trimmed to the bucket's max width
    — a ~17× cut in transfer volume when one game has a wide menu. CPU pays no
    transfer cost, so bucketing only adds small-op overhead there; it bypasses
    the bucket logic entirely.

    With the default `einsum` kernel, both picker paths are bit-identical to
    each other (and so shard and single-process agree), because the einsum
    kernels are stable across the batch shape (N, G, and width), and the
    invariants below pin the last ulps. The opt-in `bmm` kernel is the measured
    experiment and is expected to be compared by policy picks rather than this
    default bit-identity contract:

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
        return _pick_batch_flat(stacked, obs, acts, mask, genome_idx, kernel)
    return _pick_batch_gpu(stacked, obs, acts, mask, genome_idx, device, kernel)


def _pick_batch_gpu(
    stacked: policy.WeightStack,
    obs: np.ndarray,
    acts: np.ndarray,
    mask: np.ndarray,
    genome_idx: np.ndarray,
    device: str,
    kernel: policy.PolicyKernel = "einsum",
) -> np.ndarray:
    """Three-bucket GPU path with hoisted weight slicing and vectorized grouping.

    Rows are split into narrow (≤16), medium (17-64), and wide (>64) width
    buckets so a rogue 288-action menu only inflates its own bucket's transfer.
    The typical distribution (p50≈2, p99≈26) puts 99%+ in the narrow bucket.

    Bit-identical to ``_pick_batch_flat`` under the ``einsum`` kernel (same
    invariants: padding-to-4 floor, stable sort, weight slicing not re-creation).
    """
    n_rows = len(genome_idx)
    if n_rows == 0:
        return np.empty(0, dtype=np.int64)

    rw = mask.sum(axis=1)
    n_groups = stacked.trunk_w[0].shape[0]
    act_dim = acts.shape[2]
    obs_dim = obs.shape[1]

    # --- Hoisted weight slicing (once per ply, shared across buckets) ---
    order = np.argsort(genome_idx, kind="stable")
    sorted_gidx = genome_idx[order]
    counts = np.bincount(sorted_gidx, minlength=n_groups)
    present = np.flatnonzero(counts > 0)

    pres_t = torch.from_numpy(present).to(device)
    sub = policy.WeightStack(
        [w[pres_t] for w in stacked.trunk_w],
        [b[pres_t] for b in stacked.trunk_b],
        [w[pres_t] for w in stacked.head_w],
        [b[pres_t] for b in stacked.head_b],
    )

    # --- Three buckets: narrow (≤16), medium (17-64), wide (>64) ---
    NARROW = 16
    MEDIUM = 64
    bucket_assignment = np.where(rw <= NARROW, 0, np.where(rw <= MEDIUM, 1, 2))

    picks = np.empty(n_rows, dtype=np.int64)

    for bid in [0, 1, 2]:
        bucket_indices = np.flatnonzero(bucket_assignment == bid)
        if bucket_indices.size == 0:
            continue

        b_ply_max = int(rw[bucket_indices].max())
        b_genome_idx = genome_idx[bucket_indices]

        # Vectorized genome grouping for this bucket.
        b_order = np.argsort(b_genome_idx, kind="stable")
        b_sorted_gidx = b_genome_idx[b_order]
        b_counts = np.bincount(b_sorted_gidx, minlength=n_groups)
        b_present = np.flatnonzero(b_counts > 0)
        b_n_max = max(int(b_counts[b_present].max()) if b_present.size else 0, 4)

        b_padded = _vectorized_padded(b_counts, b_present, b_n_max, len(b_genome_idx))
        b_valid = b_padded >= 0
        b_padded = np.where(b_valid, b_padded, 0)

        row_order = b_order[b_padded]

        # Transfer bucket data via pinned memory for async overlap with CPU.
        b_obs = obs[bucket_indices]
        b_acts = np.ascontiguousarray(
            acts[bucket_indices[:, None], np.arange(b_ply_max)[None, :]]
        )
        b_mask = np.ascontiguousarray(
            mask[bucket_indices[:, None], np.arange(b_ply_max)[None, :]]
        )

        b_obs_gpu = torch.from_numpy(b_obs).to(device)
        b_acts_gpu = torch.from_numpy(b_acts).to(device)
        b_mask_gpu = torch.from_numpy(b_mask).to(device)

        ro_t = torch.from_numpy(row_order.flatten()).to(device)
        b_obs_t = b_obs_gpu.index_select(0, ro_t).view(
            len(b_present), b_n_max, obs_dim
        )
        b_acts_t = b_acts_gpu.index_select(0, ro_t).view(
            len(b_present), b_n_max, b_ply_max, act_dim
        )
        b_mask_t = b_mask_gpu.index_select(0, ro_t).view(
            len(b_present), b_n_max, b_ply_max
        )
        b_valid_t = torch.from_numpy(b_valid).to(device)
        b_mask_t = b_mask_t & b_valid_t.unsqueeze(2)

        if bid == 0 and np.array_equal(b_present, present):
            b_sub = sub
        else:
            b_pres_t = torch.from_numpy(b_present).to(device)
            b_sub = policy.WeightStack(
                [w[b_pres_t] for w in stacked.trunk_w],
                [b[b_pres_t] for b in stacked.trunk_b],
                [w[b_pres_t] for w in stacked.head_w],
                [b[b_pres_t] for b in stacked.head_b],
            )

        scores = policy.logits_stacked(b_sub, b_obs_t, b_acts_t, b_mask_t, kernel=kernel)
        b_picks_sorted = scores.argmax(dim=2).cpu().numpy()[b_valid]

        _vectorized_unsort(
            picks, b_picks_sorted, b_order, b_counts, b_present, bucket_indices
        )

    return picks


def _vectorized_padded(
    counts: np.ndarray,
    present: np.ndarray,
    n_max: int,
    total: int,
) -> np.ndarray:
    """Build a [G_present, n_max] padded index array without a Python for-loop.

    Each genome `g` in `present` gets a row filled with sequential indices
    `[start_g, start_g + count_g)` into the sorted array, padded with -1.
    Fully vectorized: builds a [G_present, n_max] grid of column indices,
    masks by count, and adds the per-genome start offset in one pass.
    """
    n_present = len(present)
    starts = np.cumsum(counts) - counts
    # [G_present, n_max] grid: column j has value j
    cols = np.broadcast_to(np.arange(n_max), (n_present, n_max))
    # [G_present, 1] count per genome (only for present genomes)
    cnts = counts[present].reshape(-1, 1)
    # Valid where column index < count
    valid_mask = cols < cnts
    # Fill: start + column index
    padded = np.where(valid_mask, starts[present].reshape(-1, 1) + cols, -1)
    return padded


def _vectorized_unsort(
    picks: np.ndarray,
    picks_sorted: np.ndarray,
    order: np.ndarray,
    counts: np.ndarray,
    present: np.ndarray,
    bucket_indices: np.ndarray,
) -> None:
    """Scatter sorted picks back to original row order without a Python for-loop.

    `picks_sorted` is a flat array of picks in sorted-by-genome order.
    `order` maps sorted positions to bucket-local indices.
    `counts[present[g]]` is how many rows genome `g` has in this bucket.
    `bucket_indices` maps bucket-local indices to global row indices.
    """
    # Global row index for each sorted position: order gives bucket-local
    # indices in sorted order, bucket_indices maps those to global rows.
    global_rows = bucket_indices[order]
    picks[global_rows] = picks_sorted


def _pick_batch_flat(
    stacked: policy.WeightStack,
    obs: np.ndarray,
    acts: np.ndarray,
    mask: np.ndarray,
    genome_idx: np.ndarray,
    kernel: policy.PolicyKernel = "einsum",
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

    scores = policy.logits_stacked(stacked, obs_t, acts_t, mask_t, kernel=kernel)
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
    max_hands: int | None,
    device: str,
    elo: elo_mod.EloTracker,
    progress: Callable[[int, int], None] | None = None,
    shards: int = 1,
    metrics_out: list[DriveMetrics] | None = None,
    max_rounds: int | None = None,
    kernel: policy.PolicyKernel = "einsum",
) -> elo_mod.EloTracker:
    """Play one generation of self-play and update ELO ratings in-place.

    Each game's result is win/loss/draw based on which team scored more.
    With ``max_hands=1`` each game plays exactly one hand, so the result is a
    single hand's score comparison — fast and uniform-length, with no
    convergence tail. The ELO tracker is updated in deterministic game order
    and returned (same object, modified in-place).

    With ``shards > 1`` the games are split across worker processes, each
    running its own batched ply loop (``shards.run_shards``). Results are
    bit-identical to the ``shards == 1`` path — per-game play is deterministic,
    and workers return results in global game order.
    """
    roster, game_seeds, meta = batch_layout(pop, hof, pairings, seeds)
    if max_rounds is not None and shards > 1:
        raise ValueError("max_rounds calibration is only supported with shards=1")
    per_game = 2 * len(seeds)

    if shards > 1 and device != "cpu":
        # Coalesced: single-process, multi-threaded with one GPU server.
        # Replaces the multi-process shard approach — same bit-identical
        # results (per-game play is deterministic), but no IPC, no CUDA
        # context switching, and CPU encode/apply overlaps with GPU forward
        # via the GIL release in Pool::encode/apply.
        stacked = build_stacked(roster, arch, device)
        metrics = DriveMetrics()
        metrics.reset(len(game_seeds))
        t0 = time.perf_counter()
        results = drive_pool_coalesced(
            stacked, game_seeds, meta, max_hands, device,
            n_workers=shards, progress=progress, metrics=metrics,
            max_rounds=max_rounds, kernel=kernel,
        )
        metrics.elapsed_seconds = time.perf_counter() - t0
        if metrics_out is not None:
            metrics_out.append(metrics)
    elif shards > 1:
        from canastra_train import shards as shards_mod

        results = shards_mod.run_shards(
            roster, game_seeds, meta, arch, max_hands, device, shards, progress,
            metrics_out=metrics_out, kernel=kernel,
            max_rounds=max_rounds,
        )
    else:
        stacked = build_stacked(roster, arch, device)
        metrics = DriveMetrics()
        results = drive_pool(
            stacked,
            game_seeds,
            meta,
            max_hands,
            device,
            progress=progress,
            metrics=metrics,
            max_rounds=max_rounds,
            kernel=kernel,
        )
        if metrics_out is not None:
            metrics_out.append(metrics)

    # Compute ELO updates from game results in deterministic order.
    assert len(results) == len(game_seeds)
    elo_results: list[tuple[int, int, float]] = []
    for gidx, (_seed, result_scores, _winner, _hands, _unfinished) in enumerate(results):
        pairing = gidx // per_game
        seating = gidx % 2
        ga_idx, gb_idx = pairings[pairing]
        # Win/loss/draw from the team that scored more.
        if result_scores[0] == result_scores[1]:
            result = 0.5
        elif (result_scores[0] > result_scores[1]) == (seating == 0):
            result = 1.0  # genome A's team won
        else:
            result = 0.0  # genome B's team won
        elo_results.append((ga_idx, gb_idx, result))
    elo.batch_update(elo_results)
    return elo
