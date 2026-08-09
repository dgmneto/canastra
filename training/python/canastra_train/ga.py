"""The GA core: selection, mutation, elitism, hall of fame, checkpoints.

Tabula rasa (spec §G): random init, self-play fitness, scalar reward is the
match result. No shaped rewards, no heuristic features.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np

from canastra_train import genome as genome_mod


@dataclass(frozen=True)
class GAConfig:
    population: int = 96
    elites: int = 8
    tournament: int = 4
    sigma: float = 0.02
    sigma_decay: float = 0.995
    sigma_floor: float = 0.002
    hof_interval: int = 5
    crossover: bool = False  # spec default off; the flag exists, nothing uses it yet


def initial_population(arch: genome_mod.Arch, cfg: GAConfig, run_seed: int) -> np.ndarray:
    size = genome_mod.genome_size(arch)
    rng = np.random.default_rng(run_seed)
    return rng.normal(0.0, 0.1, (cfg.population, size)).astype(np.float32)


def sigma_for(cfg: GAConfig, generation: int) -> float:
    return max(cfg.sigma * (cfg.sigma_decay**generation), cfg.sigma_floor)


def _tournament(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    contenders = rng.integers(0, len(fitness), size=k)
    return int(contenders[np.argmax(fitness[contenders])])


def next_generation(
    pop: np.ndarray,
    fitness: np.ndarray,
    cfg: GAConfig,
    generation: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Elites carried unchanged; the rest are mutated tournament winners."""
    order = np.argsort(fitness)[::-1]
    elites = pop[order[: cfg.elites]]
    sigma = sigma_for(cfg, generation)
    children = np.empty((0, pop.shape[1]), dtype=np.float32)
    for _ in range(cfg.population - cfg.elites):
        parent = pop[_tournament(fitness, cfg.tournament, rng)]
        child = parent + rng.normal(0.0, sigma, size=parent.shape).astype(np.float32)
        children = np.vstack([children, child[None, :]])
    return np.vstack([elites, children])


class HallOfFame:
    """Archived champions, sampled as opponents to prevent cycling (spec §G)."""

    def __init__(self) -> None:
        self.genomes: list[np.ndarray] = []
        self.fitnesses: list[float] = []
        self.generations: list[int] = []

    def archive(self, genome: np.ndarray, fitness: float, generation: int) -> None:
        self.genomes.append(genome.astype(np.float32).copy())
        self.fitnesses.append(float(fitness))
        self.generations.append(int(generation))

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return self.genomes[int(rng.integers(0, len(self.genomes)))]

    def __len__(self) -> int:
        return len(self.genomes)


def save_checkpoint(
    directory: Path,
    generation: int,
    pop: np.ndarray,
    fitness: np.ndarray,
    hof: HallOfFame,
    seeds: list[int],
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"gen-{generation:05d}.npz"
    np.savez_compressed(
        path,
        pop=pop,
        fitness=fitness,
        seeds=np.asarray(seeds, dtype=np.uint64),  # splitmix64 spans the full u64 range
        hof=np.vstack(hof.genomes) if len(hof) else np.zeros((0, pop.shape[1]), dtype=np.float32),
        hof_fitness=np.asarray(hof.fitnesses, dtype=np.float32),
        hof_generations=np.asarray(hof.generations, dtype=np.int64),
        generation=np.int64(generation),
    )
    _prune(directory, keep=10)
    return path


def _prune(directory: Path, keep: int) -> None:
    checkpoints = sorted(directory.glob("gen-*.npz"))
    for stale in checkpoints[:-keep]:
        stale.unlink()


class _Checkpoint(TypedDict):
    generation: int
    pop: np.ndarray
    fitness: np.ndarray
    seeds: list[int]
    hof: HallOfFame


def load_checkpoint(directory: Path) -> _Checkpoint:
    checkpoints = sorted(directory.glob("gen-*.npz"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint in {directory}")
    with np.load(checkpoints[-1]) as data:
        hof = HallOfFame()
        for genome_row, fit, gen in zip(data["hof"], data["hof_fitness"], data["hof_generations"]):
            if genome_row.size:
                hof.archive(genome_row, fitness=float(fit), generation=int(gen))
        return {
            "generation": int(data["generation"]),
            "pop": data["pop"],
            "fitness": data["fitness"],
            "seeds": [int(s) for s in data["seeds"]],
            "hof": hof,
        }
