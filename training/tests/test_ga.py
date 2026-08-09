"""GA mechanics: elitism, tournaments, mutation, hall of fame, checkpoints."""

from pathlib import Path

import numpy as np
import pytest
from canastra_train import ga, genome

ARCH = {"obs": 2002, "act": 101, "trunk": [16], "head": [], "activation": "tanh"}


def cfg(**overrides: object) -> ga.GAConfig:
    base: dict[str, object] = {
        "population": 8,
        "elites": 2,
        "tournament": 3,
        "sigma": 0.02,
        "sigma_decay": 0.995,
        "sigma_floor": 0.002,
        "hof_interval": 5,
    }
    base.update(overrides)
    return ga.GAConfig(**base)  # type: ignore[arg-type]


def test_initial_population_is_deterministic_and_shaped() -> None:
    a = ga.initial_population(ARCH, cfg(), run_seed=7)
    b = ga.initial_population(ARCH, cfg(), run_seed=7)
    assert a.shape == (8, genome.genome_size(ARCH))
    assert np.array_equal(a, b)


def test_elites_survive_unchanged() -> None:
    c = cfg()
    pop = np.random.default_rng(0).normal(0, 1, (8, 50)).astype(np.float32)
    fitness = np.arange(8, dtype=np.float32)  # genome 7 best, then 6, ...
    rng = np.random.default_rng(1)
    nxt = ga.next_generation(pop, fitness, c, generation=1, rng=rng)
    assert np.array_equal(nxt[0], pop[7]), "best genome is elite slot 0"
    assert np.array_equal(nxt[1], pop[6]), "second-best is elite slot 1"


def test_mutation_scale_tracks_sigma() -> None:
    c = cfg(sigma=0.05)
    pop = np.zeros((8, 4000), dtype=np.float32)
    fitness = np.arange(8, dtype=np.float32)
    rng = np.random.default_rng(2)
    nxt = ga.next_generation(pop, fitness, c, generation=1, rng=rng)
    mutants = nxt[c.elites :]
    std = float(mutants.std())
    assert 0.03 < std < 0.07, f"mutation std {std} should track sigma 0.05"


def test_sigma_decays_to_the_floor() -> None:
    c = cfg(sigma=0.02, sigma_decay=0.5, sigma_floor=0.01)
    assert ga.sigma_for(c, 0) == pytest.approx(0.02)
    assert ga.sigma_for(c, 1) == pytest.approx(0.01)
    assert ga.sigma_for(c, 99) == pytest.approx(0.01)


def test_hall_of_fame_archives_on_cadence() -> None:
    hof = ga.HallOfFame()
    genome_vec = np.ones(10, dtype=np.float32)
    for generation in range(10):
        if generation % 5 == 0:
            hof.archive(genome_vec + generation, fitness=float(genome_vec.sum()), generation=generation)
    assert len(hof) == 2
    sample = hof.sample(np.random.default_rng(3))
    assert sample.shape == (10,)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    pop = np.random.default_rng(4).normal(0, 1, (8, 50)).astype(np.float32)
    fitness = np.arange(8, dtype=np.float32)
    hof = ga.HallOfFame()
    hof.archive(pop[0], fitness=1.0, generation=0)
    ga.save_checkpoint(tmp_path, generation=5, pop=pop, fitness=fitness, hof=hof, seeds=[1, 2, 3])
    state = ga.load_checkpoint(tmp_path)
    assert state["generation"] == 5
    assert np.array_equal(state["pop"], pop)
    assert np.array_equal(state["fitness"], fitness)
    assert state["seeds"] == [1, 2, 3]
    assert len(state["hof"]) == 1
