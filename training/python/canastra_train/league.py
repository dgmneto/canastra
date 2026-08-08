"""One generation of self-play, batched into a single pool.

Every directed pairing (genome, opponent) contributes 2 x len(seeds) games —
duplicate deals, swapped seatings — and the pool's row metadata routes each
decision back to the genome that owns that seat. Fitness is the mean
duplicate-deal score differential (spec §G): score diff only, no win bonus,
unfinished matches at the cap count as-is.
"""

from __future__ import annotations

import numpy as np
import torch
from canastra_py import Pool

from canastra_train import ga, policy
from canastra_train import genome as genome_mod


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


def evaluate_generation(
    pop: np.ndarray,
    hof: ga.HallOfFame,
    pairings: list[tuple[int, int]],
    arch: genome_mod.Arch,
    seeds: list[int],
    cap: int,
    device: str,
) -> np.ndarray:
    """Mean duplicate-deal differential for every population genome."""
    roster = [pop[i] for i in range(len(pop))] + list(hof.genomes)
    modules = []
    for vec in roster:
        trunk, head = genome_mod.to_modules(vec, arch)
        modules.append((trunk.to(device), head.to(device)))

    game_seeds: list[int] = []
    game_meta: list[tuple[int, int, int]] = []  # (a, b, seating)
    for a, b in pairings:
        for seed in seeds:
            game_seeds.extend([seed, seed])
            game_meta.append((a, b, 0))
            game_meta.append((a, b, 1))

    pool = Pool(game_seeds, max_actions_per_game=cap)
    per_game = 2 * len(seeds)

    while pool.has_live():
        obs, acts, mask, rows = pool.encode()
        games = rows[:, 0]
        seats = rows[:, 1]
        meta = np.asarray([game_meta[g] for g in games])
        a_idx, b_idx, seat_info = meta[:, 0], meta[:, 1], meta[:, 2]
        # Genome owning this row: in seating 0, genome A holds the even seats
        # (team 0); in seating 1 the sides are swapped.
        genome_idx = np.where((seats % 2 == 0) == (seat_info == 0), a_idx, b_idx)

        picks = np.empty(len(games), dtype=np.int64)
        for gid in np.unique(genome_idx):
            sel = genome_idx == gid
            trunk, head = modules[int(gid)]
            scores = policy.logits(
                trunk,
                head,
                torch.from_numpy(obs[sel]).to(device),
                torch.from_numpy(acts[sel]).to(device),
                torch.from_numpy(mask[sel]).to(device),
            )
            picks[sel] = np.asarray(policy.pick_argmax(scores.cpu()))
        pool.apply(picks.tolist())

    # Aggregate duplicate-deal differentials per pairing, then per genome.
    results = pool.results()
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
