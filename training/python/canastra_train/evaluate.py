"""Duplicate-deal evaluation: same seeds, swapped seatings, score differentials.

Canastra's variance is dominated by the deal, so two genomes are compared by
playing each seed TWICE — once with genome A in seats 0/2, once with the
seats swapped — and averaging A's score differential over the pair. The deal
cancels; what remains is policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from canastra_py import Pool

from canastra_train import genome as genome_mod
from canastra_train import policy


@dataclass
class PairReport:
    pairs: int
    mean_diff: float
    ci95: float
    wins_a: int
    wins_b: int
    unfinished: int


def evaluate_pair(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    arch: genome_mod.Arch,
    seeds: list[int],
    cap: int = 200_000,
) -> PairReport:
    """A vs B over `seeds`, each seed played in both seatings."""
    count = len(seeds)
    pool_seeds = seeds + seeds  # first half: A in seats 0/2; second half: swapped
    pool = Pool(pool_seeds, max_actions_per_game=cap)
    trunk_a, head_a = genome_mod.to_modules(vec_a, arch)
    trunk_b, head_b = genome_mod.to_modules(vec_b, arch)

    while pool.has_live():
        obs, acts, mask, rows = pool.encode()
        obs_t = torch.from_numpy(obs)
        acts_t = torch.from_numpy(acts)
        mask_t = torch.from_numpy(mask)
        scores_a = policy.logits(trunk_a, head_a, obs_t, acts_t, mask_t)
        scores_b = policy.logits(trunk_b, head_b, obs_t, acts_t, mask_t)
        # Route each row to its genome: in the first half of the pool, genome A
        # owns the even seats (team 0); in the second half, the odd ones.
        use_a = torch.zeros(mask_t.shape[0], dtype=torch.bool)
        for row, (game, seat) in enumerate(rows):
            a_is_team_zero = game < count
            owns = (seat % 2 == 0) == a_is_team_zero
            use_a[row] = bool(owns)
        scores = torch.where(use_a.unsqueeze(1), scores_a, scores_b)
        pool.apply(policy.pick_argmax(scores))

    results = pool.results()
    assert len(results) == 2 * count

    diffs: list[float] = []
    unfinished = 0
    wins_a = wins_b = 0
    by_seed: dict[int, list[float]] = {}
    for index, (_seed, pair_scores, winner, _hands, is_unfinished) in enumerate(results):
        a_is_team_zero = index < count
        a_score = pair_scores[0] if a_is_team_zero else pair_scores[1]
        b_score = pair_scores[1] if a_is_team_zero else pair_scores[0]
        by_seed.setdefault(_seed, []).append(a_score - b_score)
        if is_unfinished:
            unfinished += 1
        elif winner is not None:
            won_a = (winner == 0) == a_is_team_zero
            if won_a:
                wins_a += 1
            else:
                wins_b += 1

    for seed in seeds:
        pair = by_seed[seed]
        assert len(pair) == 2, f"seed {seed} did not produce both seatings"
        diffs.append((pair[0] + pair[1]) / 2)

    arr = np.asarray(diffs)
    mean = float(arr.mean())
    ci95 = float(1.96 * arr.std(ddof=1) / np.sqrt(count)) if count > 1 else float("inf")
    return PairReport(count, mean, ci95, wins_a, wins_b, unfinished)
