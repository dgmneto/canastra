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
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from canastra_train import ga, league, seedstream
from canastra_train import genome as genome_mod
from canastra_train.tui import Dashboard

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
    sigma_decay: float = 0.995,
    sigma_floor: float = 0.002,
    hof_interval: int = 5,
    crossover: bool = False,
    device: str = "cpu",
    resume: bool = False,
    no_tui: bool = False,
    shards: int = 1,
) -> None:
    cfg = ga.GAConfig(
        population=population,
        elites=elites,
        tournament=tournament,
        sigma=sigma,
        sigma_decay=sigma_decay,
        sigma_floor=sigma_floor,
        hof_interval=hof_interval,
        crossover=crossover,
    )
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if not resume and (
        (run_dir / "generations.jsonl").exists()
        or list(run_dir.glob("gen-*.npz"))
    ):
        raise RuntimeError(
            f"{run_dir} already contains training artifacts (generations.jsonl or gen-*.npz); "
            "refusing a fresh start that would append duplicate records. "
            "Use --resume to continue the run, or pick a new --run-dir (or delete it)."
        )
    (run_dir / "config.json").write_text(
        json.dumps({"arch": arch, "ga": asdict(cfg), "run_seed": run_seed,
                    "seeds": seeds, "cap": cap, "device": device,
                    "shards": shards}, indent=2)
    )

    if resume:
        state = ga.load_checkpoint(run_dir)
        pop, hof = state["pop"], state["hof"]
        start = state["generation"] + 1
        best_ever = max(
            float(state["fitness"].max()),
            max(hof.fitnesses) if len(hof) else float(state["fitness"].max()),
        )
    else:
        pop = ga.initial_population(arch, cfg, run_seed)
        hof = ga.HallOfFame()
        start = 0
        best_ever = float("-inf")

    log_path = run_dir / "generations.jsonl"
    last_pop = pop
    last_fitness: np.ndarray | None = None
    dash = Dashboard(
        run_dir,
        generations,
        start,
        no_tui=no_tui or not sys.stdout.isatty(),
        device=device,
        sigma=sigma,
        games_total=0,
    )
    dash.start()
    if best_ever != float("-inf"):
        dash.status.best_ever = float(best_ever)
    for generation in range(start, generations):
        began = time.perf_counter()
        gen_rng = np.random.default_rng(seedstream.splitmix64(run_seed + generation))
        gen_seeds = seedstream.generation_seeds(run_seed, generation, seeds)
        pairings = league.schedule_pairings(len(pop), opponents, hof, gen_rng)
        dash.status.games_total = len(pairings) * 2 * len(gen_seeds)
        dash.set_phase("evaluating")
        fitness = league.evaluate_generation(
            pop, hof, pairings, arch, gen_seeds, cap, device,
            progress=dash.on_progress, shards=shards,
        )

        last_pop = pop
        last_fitness = fitness

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
            dash.status.best_ever = float(best_ever)
        dash.on_generation(record)
        if improved:
            dash.on_event(
                "best", f"new best-ever {fitness[champion]:+.1f} (gen {generation})"
            )
        if generation % cfg.hof_interval == 0 or improved:
            genome_mod.save_json(
                str(run_dir / f"champion-gen{generation:05d}.json"), arch, pop[champion]
            )
            dash.on_event(
                "export",
                f"champion-gen{generation:05d}.json ({fitness[champion]:+.1f})",
            )
        if generation % cfg.hof_interval == 0:
            hof.archive(pop[champion], fitness=float(fitness[champion]), generation=generation)
            dash.on_event(
                "hof",
                f"archived gen-{generation} champion ({fitness[champion]:+.1f})",
            )

        dash.set_phase("evolving")
        pop = ga.next_generation(pop, fitness, cfg, generation, gen_rng)
        # Checkpoint the EVOLVED population (what generation+1 evaluates) so a
        # resumed run starts from the exact hand-off — bit identical. The saved
        # fitness pairs with the population that was evaluated this generation.
        if generation % 5 == 0 or generation == generations - 1 or improved:
            ga.save_checkpoint(run_dir, generation, pop, fitness, hof, gen_seeds)

    assert last_fitness is not None, "no generation was evaluated"
    final = int(np.argmax(last_fitness))
    genome_mod.save_json(str(run_dir / "champion-final.json"), arch, last_pop[final])
    dash.stop()
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
    parser.add_argument("--sigma-decay", type=float, default=0.995)
    parser.add_argument("--sigma-floor", type=float, default=0.002)
    parser.add_argument("--hof-interval", type=int, default=5)
    parser.add_argument("--crossover", action="store_true", help="flag exists per spec; unused until crossover lands")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="disable the live dashboard and print one plain line per generation",
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=1,
        help="split the generation's games across this many worker processes "
        "(bit-identical results; the per-ply cost is GIL-serial glue, so this "
        "scales with your cores). E.g. 8 on this machine.",
    )
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
        sigma_decay=args.sigma_decay,
        sigma_floor=args.sigma_floor,
        hof_interval=args.hof_interval,
        crossover=args.crossover,
        device=args.device,
        resume=args.resume,
        no_tui=args.no_tui,
        shards=args.shards,
    )


if __name__ == "__main__":
    main()
