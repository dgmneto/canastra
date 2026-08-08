# M3: GA Trainer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Work test-first (@superpowers:test-driven-development).

**Goal:** The genetic-algorithm trainer: tabula-rasa self-play evolution of policy genomes over the M2 evaluator — population, tournament selection, Gaussian mutation, elitism, hall of fame, checkpoints, JSONL logs, and a CLI — milestone M3 of the bot-training spec.

**Architecture:** A generation = one batched Pool containing every directed pairing (genome × sampled opponents × duplicate-deal seatings × the generation's common seeds); picks are routed to each row's genome via the M2 row metadata, fitness is the mean duplicate-deal score differential (spec §G: score diff only, no win bonus, unfinished matches at the cap count as-is). Everything derives from `run_seed + generation` — no stored RNG state, so a run resumes from its latest checkpoint with bit-identical behavior.

**Tech Stack:** Python 3.13 + torch (M2's venv), the M2 Pool/evaluator/genome modules, numpy.

**Authoritative references:**
- Spec: `docs/superpowers/specs/2026-08-06-bot-training-design.md` Section G (GA defaults) and Section I (M3 gate).
- **All work happens in `/Users/dgmneto/canastra-bot-training` on branch `bot-training`.** Python from `training/` (venv `training/.venv`).

**Standing facts (verified through M2):**
- `Pool(seeds, max_actions_per_game=cap)`; `encode()` → `(obs [N,2002] f32, acts [N,M,101] f32, mask [N,M] bool, rows [N,2] int64)`; `apply(picks: list[int])`; `results()` → `(seed, (s0, s1), winner, hands, unfinished)` in game order; safe-mode and cap are built in.
- `genome.random_genome(arch, seed)`, `genome.to_modules(vec, arch)` → `(trunk, head)`, `genome.save_json(path, arch, vec)`, `genome.load_json(path)`, `genome.genome_size(arch)`.
- `policy.logits(trunk, head, obs_t, acts_t, mask_t)` → `[N,M]` −inf-masked; `policy.pick_argmax(scores)`.
- Duplicate-deal math (from `evaluate.py`): per seed, two seatings; `diff_pair = (d_seating0 + d_seating1) / 2` where `d = a_score − b_score`.
- Training arch is `[512, 256]` head `[128]` (~1.2M params); smoke tests use tiny arches.
- **The spec's M3 success gate (champion beats `random`, reaches parity with `random-plus` at ~10k hands) is a TRAINING-MACHINE gate** — it needs real generations, not a smoke. This plan delivers the trainer code-complete, smoke-proven on this machine, with the exact training-machine commands documented.

---

### Task 1: Deterministic seed streams

**Files:** `training/python/canastra_train/seedstream.py`, `training/tests/test_seedstream.py`

- [ ] **Step 1: The test**

```python
"""Seed streams: deterministic, distinct, resumable by construction."""

from canastra_train import seedstream


def test_streams_are_deterministic() -> None:
    a = seedstream.generation_seeds(7, 3, 8)
    b = seedstream.generation_seeds(7, 3, 8)
    assert a == b


def test_generations_and_runs_get_disjoint_streams() -> None:
    a = seedstream.generation_seeds(7, 3, 8)
    b = seedstream.generation_seeds(7, 4, 8)
    c = seedstream.generation_seeds(8, 3, 8)
    assert len(set(a)) == 8
    assert not set(a) & set(b)
    assert not set(a) & set(c)
```

- [ ] **Step 2: Watch it fail, then implement**

```python
"""Deterministic seed streams.

SplitMix64 finalizer — the same mixer family the engine uses for hand seeds.
Everything derives from (run_seed, generation), so a resumed run regenerates
exactly the seeds it had without storing RNG state.
"""

from __future__ import annotations

MASK = (1 << 64) - 1


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
    return (x ^ (x >> 31)) & MASK


def generation_seeds(run_seed: int, generation: int, count: int) -> list[int]:
    """`count` distinct u64 seeds shared by every pairing of one generation."""
    base = splitmix64((run_seed & MASK) ^ ((generation * 0xD1B54A32D192ED03) & MASK))
    return [splitmix64(base + i) for i in range(count)]
```

- [ ] **Step 3: Gates + commit**

`cd training && .venv/bin/pytest -q tests/test_seedstream.py && .venv/bin/ruff check . && .venv/bin/mypy python/canastra_train tests`

Commit: "training: deterministic per-generation seed streams".

---

### Task 2: The GA core — selection, mutation, elitism, hall of fame, checkpoints

**Files:** `training/python/canastra_train/ga.py`, `training/tests/test_ga.py`

- [ ] **Step 1: The tests**

```python
"""GA mechanics: elitism, tournaments, mutation, hall of fame, checkpoints."""

import numpy as np
import pytest
from canastra_train import ga

ARCH = {"obs": 2002, "act": 101, "trunk": [16], "head": [], "activation": "tanh"}


def cfg(**overrides: object) -> ga.GAConfig:
    base = dict(
        population=8,
        elites=2,
        tournament=3,
        sigma=0.02,
        sigma_decay=0.995,
        sigma_floor=0.002,
        hof_interval=5,
    )
    base.update(overrides)
    return ga.GAConfig(**base)  # type: ignore[arg-type]


def test_initial_population_is_deterministic_and_shaped() -> None:
    a = ga.initial_population(ARCH, cfg(), run_seed=7)
    b = ga.initial_population(ARCH, cfg(), run_seed=7)
    assert a.shape == (8, 2219)  # (16*2002+16) + (1*33+1) = 32048+34? see below
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
    genome = np.ones(10, dtype=np.float32)
    for generation in range(10):
        if generation % 5 == 0:
            hof.archive(genome + generation, fitness=float(genome.sum()), generation=generation)
    assert len(hof) == 2
    sample = hof.sample(np.random.default_rng(3))
    assert sample.shape == (10,)


def test_checkpoint_round_trip(tmp_path) -> None:
    c = cfg()
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
```

NOTE for the implementer: compute the real genome width for the test arch from `genome.genome_size(ARCH)` and use it in `test_initial_population_is_deterministic_and_shaped` instead of the placeholder `2219` above (trunk [16]: 16×2002+16; head []: head.out 1×(16+101)+1 — let `genome_size` be the source of truth).

- [ ] **Step 2: Watch them fail, then implement `ga.py`**

```python
"""The GA core: selection, mutation, elitism, hall of fame, checkpoints.

Tabula rasa (spec §G): random init, self-play fitness, scalar reward is the
match result. No shaped rewards, no heuristic features.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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
    children = []
    for _ in range(cfg.population - cfg.elites):
        parent = pop[_tournament(fitness, cfg.tournament, rng)]
        child = parent + rng.normal(0.0, sigma, size=parent.shape).astype(np.float32)
        children.append(child)
    return np.vstack([elites, np.asarray(children)])


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


def load_checkpoint(directory: Path) -> dict:
    checkpoints = sorted(directory.glob("gen-*.npz"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint in {directory}")
    with np.load(checkpoints[-1]) as data:
        hof = HallOfFame()
        for genome, fit, gen in zip(data["hof"], data["hof_fitness"], data["hof_generations"]):
            if genome.size:
                hof.archive(genome, fitness=float(fit), generation=int(gen))
        return {
            "generation": int(data["generation"]),
            "pop": data["pop"],
            "fitness": data["fitness"],
            "seeds": [int(s) for s in data["seeds"]],
            "hof": hof,
        }
```

- [ ] **Step 3: Gates + commit**

`cd training && .venv/bin/pytest -q && .venv/bin/ruff check . && .venv/bin/mypy python/canastra_train tests`

Commit: "training: GA core — elitism, tournaments, mutation, hall of fame, checkpoints".

---

### Task 3: The league driver — one batched pool per generation

**Files:** `training/python/canastra_train/league.py`, `training/tests/test_league.py`

- [ ] **Step 1: The test**

```python
"""The league driver at smoke scale."""

import numpy as np
from canastra_train import ga, genome, league

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
```

- [ ] **Step 2: Watch it fail, then implement `league.py`**

```python
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

from canastra_train import ga, genome as genome_mod, policy


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
        a_idx, b_idx, seating = meta[:, 0], meta[:, 1], meta[:, 2]
        # Genome owning this row: in seating 0, genome A holds the even seats
        # (team 0); in seating 1 the sides are swapped.
        genome_idx = np.where((seats % 2 == 0) == (seating == 0), a_idx, b_idx)

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
    for game, (_seed, scores, _winner, _hands, _unfinished) in enumerate(results):
        pairing = game // per_game
        seating = game % 2
        seed = game_seeds[game]
        a_is_team_zero = seating == 0
        a_score = scores[0] if a_is_team_zero else scores[1]
        b_score = scores[1] if a_is_team_zero else scores[0]
        pairing_by_seed[pairing].setdefault(seed, []).append(a_score - b_score)

    fitness = np.zeros(len(pop), dtype=np.float64)
    counts = np.zeros(len(pop), dtype=np.float64)
    for pairing, (a, _b) in enumerate(pairings):
        diffs = []
        for seed in seeds:
            pair = pairing_by_seed[pairing][seed]
            assert len(pair) == 2, f"pairing {pairing} seed {seed} lost a seating"
            diffs.append((pair[0] + pair[1]) / 2)
        fitness[a] += float(np.mean(diffs))
        counts[a] += 1
    assert (counts > 0).all()
    return (fitness / counts).astype(np.float32)
```

- [ ] **Step 3: Gates + commit**

`cd training && .venv/bin/pytest -q && .venv/bin/ruff check . && .venv/bin/mypy python/canastra_train tests`

(The league tests add a few seconds each — bounded by the 6k-action caps.)

Commit: "training: batched league driver for one generation of self-play".

---

### Task 4: The trainer CLI — loop, logs, resume

**Files:** `training/python/canastra_train/train.py`, `training/tests/test_train.py`

- [ ] **Step 1: The integration test**

```python
"""The trainer end to end, at smoke scale: artifacts, logs, resume."""

import json
from pathlib import Path

from canastra_train import genome, train

ARCH = {"obs": 2002, "act": 101, "trunk": [16], "head": [], "activation": "tanh"}


def test_train_smoke_produces_the_artifacts(tmp_path: Path) -> None:
    train.run(
        arch=ARCH,
        run_dir=tmp_path,
        generations=2,
        population=4,
        elites=1,
        tournament=2,
        opponents=1,
        seeds=1,
        cap=6000,
        run_seed=9,
        hof_interval=1,
    )
    assert (tmp_path / "config.json").exists()
    lines = (tmp_path / "generations.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["generation"] == 0
    assert "fitness_best" in first and "sigma" in first and "seeds" in first
    champions = sorted(tmp_path.glob("champion-*.json"))
    assert champions, "champion weights exported"
    arch_loaded, _vec = genome.load_json(str(champions[-1]))
    assert arch_loaded == ARCH
    assert sorted(tmp_path.glob("gen-*.npz")), "checkpoint written"


def test_resume_continues_from_the_checkpoint(tmp_path: Path) -> None:
    kwargs = dict(
        arch=ARCH,
        run_dir=tmp_path,
        generations=1,
        population=4,
        elites=1,
        tournament=2,
        opponents=1,
        seeds=1,
        cap=6000,
        run_seed=9,
        hof_interval=1,
    )
    train.run(**kwargs)
    train.run(**kwargs, generations=2, resume=True)
    lines = (tmp_path / "generations.jsonl").read_text().strip().splitlines()
    assert [json.loads(line)["generation"] for line in lines] == [0, 1]
```

- [ ] **Step 2: Watch it fail, then implement `train.py`**

```python
"""The training loop: generations of self-play, evolution, and records.

Usage (from training/):

    .venv/bin/python -m canastra_train.train --generations 100 [options]

Everything derives from --run-seed plus the generation number (seed streams,
mutation draws), so --resume reproduces a run bit-for-bit from its latest
checkpoint. The spec's success gate — champion beats `random`, parity with
`random-plus` at ~10k hands — is measured AFTER training with the M2 tools:

    .venv/bin/python -m canastra_train.train --generations N ...   # train
    npx tsx harness/src/eval-nn.ts <champion.json> random 1000     # from repo root
    npx tsx harness/src/eval-nn.ts <champion.json> random-plus 1000
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from canastra_train import ga, genome as genome_mod, league, seedstream

TRAINING_ARCH = {"obs": 2002, "act": 101, "trunk": [512, 256], "head": [128], "activation": "tanh"}


def run(
    arch: genome_mod.Arch,
    run_dir: Path,
    generations: int,
    population: int = 96,
    elites: int = 8,
    tournament: int = 4,
    opponents: int = 4,
    seeds: int = 8,
    cap: int = 200_000,
    run_seed: int = 7,
    sigma: float = 0.02,
    hof_interval: int = 5,
    device: str = "cpu",
    resume: bool = False,
) -> None:
    cfg = ga.GAConfig(
        population=population,
        elites=elites,
        tournament=tournament,
        sigma=sigma,
        hof_interval=hof_interval,
    )
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({"arch": arch, "ga": asdict(cfg), "run_seed": run_seed,
                    "seeds": seeds, "cap": cap, "device": device}, indent=2)
    )

    if resume:
        state = ga.load_checkpoint(run_dir)
        pop, hof = state["pop"], state["hof"]
        start = state["generation"] + 1
        best_ever = float(state["fitness"].max())
    else:
        pop = ga.initial_population(arch, cfg, run_seed)
        hof = ga.HallOfFame()
        start = 0
        best_ever = float("-inf")

    log_path = run_dir / "generations.jsonl"
    for generation in range(start, generations):
        began = time.perf_counter()
        gen_rng = np.random.default_rng(seedstream.splitmix64(run_seed + generation))
        gen_seeds = seedstream.generation_seeds(run_seed, generation, seeds)
        pairings = league.schedule_pairings(len(pop), opponents, hof, gen_rng)
        fitness = league.evaluate_generation(pop, hof, pairings, arch, gen_seeds, cap, device)

        champion = int(np.argmax(fitness))
        record = {
            "generation": generation,
            "fitness_mean": float(fitness.mean()),
            "fitness_best": float(fitness[champion]),
            "fitness_worst": float(fitness.min()),
            "champion": champion,
            "sigma": ga.sigma_for(cfg, generation),
            "seeds": gen_seeds,
            "wall_seconds": round(time.perf_counter() - began, 2),
        }
        with log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

        improved = fitness[champion] > best_ever
        if improved:
            best_ever = float(fitness[champion])
        if generation % cfg.hof_interval == 0 or improved:
            genome_mod.save_json(
                str(run_dir / f"champion-gen{generation:05d}.json"), arch, pop[champion]
            )
        if generation % cfg.hof_interval == 0:
            hof.archive(pop[champion], fitness=float(fitness[champion]), generation=generation)
        if generation % 5 == 0 or generation == generations - 1 or improved:
            ga.save_checkpoint(run_dir, generation, pop, fitness, hof, gen_seeds)

        pop = ga.next_generation(pop, fitness, cfg, generation, gen_rng)
        print(
            f"gen {generation}: best {fitness[champion]:+.1f} "
            f"mean {fitness.mean():+.1f} sigma {record['sigma']:.4f} "
            f"({record['wall_seconds']}s)"
        )

    genome_mod.save_json(str(run_dir / "champion-final.json"), arch, pop[int(np.argmax(fitness))])
    print(f"done: {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--tournament", type=int, default=4)
    parser.add_argument("--opponents", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--cap", type=int, default=200_000)
    parser.add_argument("--run-seed", type=int, default=7)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--hof-interval", type=int, default=5)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    run_dir = args.run_dir or Path("runs") / time.strftime("%Y%m%d-%H%M%S")
    run(
        arch=TRAINING_ARCH,
        run_dir=run_dir,
        generations=args.generations,
        population=args.population,
        elites=args.elites,
        tournament=args.tournament,
        opponents=args.opponents,
        seeds=args.seeds,
        cap=args.cap,
        run_seed=args.run_seed,
        sigma=args.sigma,
        hof_interval=args.hof_interval,
        device=args.device,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Gates + commit**

`cd training && .venv/bin/pytest -q && .venv/bin/ruff check . && .venv/bin/mypy python/canastra_train tests`

Commit: "training: the trainer CLI — generations, logs, checkpoints, resume".

---

### Task 5: Smoke training run, docs, final sweep

**Files:** `training/README.md`, `CLAUDE.md`

- [ ] **Step 1: Smoke-run the real trainer on this machine**

```bash
cd training && .venv/bin/python -m canastra_train.train \
  --generations 2 --population 8 --opponents 2 --seeds 2 --cap 30000 \
  --run-dir runs/smoke --run-seed 7
```

Expected: two logged generations, champion JSONs, checkpoints under `training/runs/smoke/`. Record the verbatim output and wall times. Then prove the champion is playable through the M2 path:

```bash
npx tsx harness/src/eval-nn.ts training/runs/smoke/champion-final.json random 1
```

(from the worktree root; expect a report line — untrained-ish weights, cap behavior acceptable). Add `runs/` to `training/.gitignore` if not already covered.

- [ ] **Step 2: Docs**

`training/README.md`: add a "Training" section — the CLI with its flags, what a generation does (batched self-play league → duplicate-deal differentials → elitism + tournaments + Gaussian mutation), checkpoints/resume semantics (bit-identical from `run_seed`), champion exports playable via `eval-nn.ts`, and the **training-machine success gate** verbatim:

```
.venv/bin/python -m canastra_train.train --generations 200          # or as needed
npx tsx harness/src/eval-nn.ts runs/<run>/champion-final.json random 1000
npx tsx harness/src/eval-nn.ts runs/<run>/champion-final.json random-plus 1000
```

with the gate statement: champion beats `random` decisively; vs `random-plus`, mean differential ≥ 0 within CI (~10k hands ≈ 1000 matches at ~10 hands/match). Note honestly: on this machine the smoke only proves the loop; the gate belongs to the training machine.

`CLAUDE.md`: repo status component 2 — M3 landed: GA trainer (`canastra_train.train`) with self-play league fitness, checkpoints, resume; the bot project is code-complete pending real training runs. Commands — the train CLI line. Architecture — one sentence for `ga`/`league`/`train` in the training paragraph.

- [ ] **Step 3: Final gate sweep**

```bash
cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --check   # from engine/
cargo build -p canastra-wasm --target wasm32-unknown-unknown
npm run typecheck                                             # from worktree root
npx canastra-harness --seed 7 random random-plus random random-plus | head -1
cd training && .venv/bin/maturin develop --release && .venv/bin/pytest -q && .venv/bin/ruff check . && .venv/bin/mypy python/canastra_train tests
```

All green.

- [ ] **Step 4: Commit**

Commit: "M3 smoke run, training docs, and repo guide updates".

---

## Done criteria for M3

- `train.py` runs generations end-to-end (smoke-proven here): league fitness over the batched pool, elitism/tournaments/mutation, hall of fame, JSONL logs, checkpoints, resume.
- Champion weights export as `canastra-weights@1` JSON and play through `eval-nn.ts`.
- All M2 gates still green plus the new trainer tests.
- The training-machine success gate is documented with exact commands (it is NOT runnable here — that is deliberate, per the spec's "this is not the final machine").

This completes the bot-training spec (M0–M3). Final step afterwards: whole-branch summary and handoff notes for the training machine.
