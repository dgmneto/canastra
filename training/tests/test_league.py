"""The league driver at smoke scale."""

import numpy as np
from canastra_train import elo as elo_mod
from canastra_train import ga, league

ARCH = {"obs": 2002, "act": 101, "trunk": [16], "head": [], "activation": "tanh"}


def test_a_generation_updates_elo_ratings() -> None:
    cfg = ga.GAConfig(population=4, elites=1, tournament=2)
    pop = ga.initial_population(ARCH, cfg, run_seed=3)
    hof = ga.HallOfFame()
    rng = np.random.default_rng(4)
    pairings = league.schedule_pairings(len(pop), opponents=1, hof=hof, rng=rng)
    assert len(pairings) == 4
    elo = elo_mod.EloTracker(len(pop))
    metrics: list[league.DriveMetrics] = []
    elo_out = league.evaluate_generation(
        pop,
        hof,
        pairings,
        ARCH,
        seeds=[5],
        cap=6000,
        device="cpu",
        elo=elo,
        metrics_out=metrics,
    )
    assert elo_out is elo, "ELO tracker updated in-place"
    assert len(elo.ratings) == 4
    assert np.isfinite(elo.ratings).all()
    assert len(metrics) == 1
    assert metrics[0].batch_rounds > 0
    assert metrics[0].individual_actions >= metrics[0].batch_rounds
    assert metrics[0].max_actions_per_game > 0


def test_self_pairing_elo_stays_near_base() -> None:
    # The same genome on both sides of a pairing: ELO should not move much
    # (wins and losses cancel).
    pop = ga.initial_population(ARCH, ga.GAConfig(population=1), run_seed=6)
    hof = ga.HallOfFame()
    elo = elo_mod.EloTracker(1)
    league.evaluate_generation(
        pop, hof, [(0, 0)], ARCH, seeds=[7], cap=6000, device="cpu", elo=elo
    )
    assert abs(elo.ratings[0] - 1200.0) < 100, f"ELO drifted {elo.ratings[0]} from base 1200"
