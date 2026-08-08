"""The league driver at smoke scale."""

import numpy as np
from canastra_train import ga, league

ARCH = {"obs": 2002, "act": 101, "trunk": [16], "head": [], "activation": "tanh"}


def test_a_generation_produces_one_fitness_per_genome() -> None:
    cfg = ga.GAConfig(population=4, elites=1, tournament=2)
    pop = ga.initial_population(ARCH, cfg, run_seed=3)
    hof = ga.HallOfFame()
    rng = np.random.default_rng(4)
    pairings = league.schedule_pairings(len(pop), opponents=1, hof=hof, rng=rng)
    assert len(pairings) == 4
    fitness = league.evaluate_generation(
        pop, hof, pairings, ARCH, seeds=[5], cap=6000, device="cpu"
    )
    assert fitness.shape == (4,)
    assert np.isfinite(fitness).all()


def test_self_pairing_reads_near_zero() -> None:
    # The same genome on both sides of a pairing must cancel (M2's self-null,
    # here at smoke scale — loose bound, one seed).
    pop = ga.initial_population(ARCH, ga.GAConfig(population=1), run_seed=6)
    hof = ga.HallOfFame()
    fitness = league.evaluate_generation(
        pop, hof, [(0, 0)], ARCH, seeds=[7], cap=6000, device="cpu"
    )
    assert abs(fitness[0]) < 5000
