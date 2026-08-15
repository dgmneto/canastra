"""Dump one generation's deterministic outputs for Rust equivalence tests.

Run from `training/`:

    .venv/bin/python ../training-rs/tests/reference/dump_generation.py

Writes to `training-rs/tests/reference/`:
  - gen_pairings.json     — the pairings (stochastic, but fixed by the seed)
  - gen_results.json       — match results (scores per game) for ELO testing
  - gen_elo_after.json     — ELO ratings after the generation (EXACT reference)
  - gen_elites.json        — elite indices given the ELO ratings (EXACT reference)
  - gen_pop.npy            — the initial population (for reproducibility)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from canastra_train import elo as elo_mod
from canastra_train import ga, league, seedstream
from canastra_train import genome as genome_mod

ARCH = {"obs": 2002, "act": 101, "trunk": [512, 256], "head": [128], "activation": "tanh"}

OUT = Path(__file__).resolve().parent

# Small config for fast testing.
POPULATION = 8
OPPONENTS = 2
SEEDS = 2
RUN_SEED = 7


def main() -> None:
    cfg = ga.GAConfig(population=POPULATION, elites=2, tournament=4, sigma=0.02)
    pop = ga.initial_population(ARCH, cfg, RUN_SEED)
    np.save(OUT / "gen_pop.npy", pop)

    hof = ga.HallOfFame()
    elo = elo_mod.EloTracker(POPULATION)

    gen_rng = np.random.default_rng(seedstream.splitmix64(RUN_SEED + 0))
    gen_seeds = seedstream.generation_seeds(RUN_SEED, 0, SEEDS)
    pairings = league.schedule_pairings(POPULATION, OPPONENTS, hof, gen_rng)

    (OUT / "gen_pairings.json").write_text(json.dumps(pairings))

    elo = league.evaluate_generation(
        pop, hof, pairings, ARCH, gen_seeds, max_hands=1, device="cpu", elo=elo,
    )

    # Dump ELO ratings after evaluation (EXACT reference).
    elo_after = elo.ratings[:POPULATION].tolist()
    (OUT / "gen_elo_after.json").write_text(json.dumps(elo_after))

    # Dump elite indices (EXACT reference — deterministic from ELO ratings).
    order = np.argsort(elo.ratings[:POPULATION])[::-1]
    elites = order[: cfg.elites].tolist()
    (OUT / "gen_elites.json").write_text(json.dumps(elites))

    print(f"pairings: {len(pairings)}")
    print(f"elo_after: {elo_after}")
    print(f"elites: {elites}")


if __name__ == "__main__":
    main()