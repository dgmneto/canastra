"""REINFORCE policy-gradient training CLI.

Usage (from training/):

    .venv/bin/python -m canastra_train.train_pg --episodes 100 [options]

One gradient update per episode: `--games-per-update` games via duplicate deal,
split into `--mini-batch`-sized rollouts with gradient accumulation, one Adam
step. The EMA baseline lowers variance; bf16 autocast and `torch.compile` fill
the GPU. Sharded rollouts (`--shards N`) parallelize the CPU-bound per-ply glue
across cores.

The spec's success gate is measured AFTER training with the M2 tools:

    .venv/bin/python -m canastra_train.train_pg --episodes N ...   # train
    npx tsx harness/src/eval-nn.ts <champion.json> random 1000     # from repo root
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from canastra_train import evaluate
from canastra_train import genome as genome_mod
from canastra_train.pg import PGConfig, Trainer, unwrap
from canastra_train.tui_pg import PGDashboard

# The bigger net for the 16GB 5060 Ti training machine. The `canastra-weights@1`
# format supports any tanh MLP arch, so the deployment path (`bots/`, `harness/
# src/eval-nn.ts`) needs no changes — it's arch-agnostic.
TRAINING_ARCH = {"obs": 2002, "act": 101, "trunk": [1024, 512, 256], "head": [256], "activation": "tanh"}

# Default fixed baseline for the periodic growth eval: the toy random-init
# weights shipped with the repo. Passing `--eval-opponent` overrides this.
_DEFAULT_EVAL_OPPONENT = str(Path(__file__).resolve().parents[3] / "bots/src/fixtures/random-init.json")


def run(
    arch: genome_mod.Arch,
    run_dir: Path,
    episodes: int,
    games_per_update: int = 512,
    mini_batch: int = 64,
    lr: float = 1e-3,
    baseline_decay: float = 0.95,
    entropy_coef: float = 0.0,
    grad_clip: float = 0.0,
    cap: int = 200_000,
    run_seed: int = 7,
    device: str = "cpu",
    amp: bool = True,
    compile: bool = False,
    shards: int = 1,
    opponent: str | None = None,
    opponent_refresh: int = 0,
    resume: bool = False,
    no_tui: bool = False,
    log_interval: int = 1,
    ckpt_interval: int = 10,
    eval_interval: int = 10,
    eval_opponent: str | None = None,
    eval_seeds: int = 32,
) -> None:
    cfg = PGConfig(
        games_per_update=games_per_update,
        mini_batch=mini_batch,
        lr=lr,
        baseline_decay=baseline_decay,
        entropy_coef=entropy_coef,
        grad_clip=grad_clip,
        cap=cap,
        device=device,
        amp=amp,
        compile=compile,
    )
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if not resume and (
        (run_dir / "updates.jsonl").exists()
        or list(run_dir.glob("model-*.pt"))
    ):
        raise RuntimeError(
            f"{run_dir} already contains training artifacts (updates.jsonl or model-*.pt); "
            "refusing a fresh start that would append duplicate records. "
            "Use --resume to continue the run, or pick a new --run-dir (or delete it)."
        )

    trainer = Trainer(arch, cfg, run_seed, shards=shards)
    start = 0
    if opponent:
        trainer.load_opponent(opponent)
    if resume:
        trainer.load_checkpoint(run_dir)
        start = trainer.update_step

    (run_dir / "config.json").write_text(
        json.dumps({
            "arch": arch, "pg": _cfg_dict(cfg), "run_seed": run_seed,
            "shards": shards, "opponent": opponent, "opponent_refresh": opponent_refresh,
        }, indent=2)
    )

    log_path = run_dir / "updates.jsonl"
    eval_log_path = run_dir / "evals.jsonl"

    # Load the fixed baseline for the periodic growth eval.
    eval_opp_path = eval_opponent or _DEFAULT_EVAL_OPPONENT
    _eval_arch, eval_opp_vec = genome_mod.load_json(eval_opp_path)
    if _eval_arch != arch:
        # The default fixture uses the full training arch; a smaller test arch
        # needs its own baseline. Generate a random one with the right shape so
        # the growth eval still runs (it's a fixed, reproducible baseline).
        eval_opp_vec = genome_mod.random_genome(arch, seed=run_seed ^ 0xE0A1)
    eval_seeds_list = [run_seed * 7919 + i + 1 for i in range(eval_seeds)]

    dash = PGDashboard(
        run_dir, episodes, start,
        no_tui=no_tui, device=device, lr=lr, games_per_update=games_per_update,
    )
    dash.start()
    if resume and trainer.best_ever != float("-inf"):
        dash.status.best_ever = float(trainer.best_ever)

    for episode in range(start, episodes):
        # Frozen-self-play: periodically snapshot the learner as the opponent.
        if opponent is None and opponent_refresh > 0 and episode % opponent_refresh == 0:
            trainer.freeze_self_as_opponent()
            dash.on_event("opp-refresh", f"frozen-self opponent at update {episode}")

        record = trainer.step(progress=dash.on_rollout_progress)
        dash.status.phase = "updating"

        if episode % log_interval == 0:
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")

        if episode % ckpt_interval == 0 or episode == episodes - 1 or record["improved"]:
            trainer.save_checkpoint(run_dir)
            trainer.export_champion(str(run_dir / f"champion-update{episode:06d}.json"))
            dash.on_event("export", f"champion-update{episode:06d}.json (reward {record['mean_reward']:+.1f})")

        dash.on_update(record)

        # Periodic growth eval: play the current learner against a fixed baseline
        # via the duplicate-deal evaluator. This is the "is the bot actually
        # getting stronger?" signal, separate from the noisy training reward.
        if eval_interval > 0 and episode % eval_interval == 0:
            eval_began = time.perf_counter()
            champion_vec = unwrap(trainer.net).to_genome_vec()
            report = evaluate.evaluate_pair(champion_vec, eval_opp_vec, arch, seeds=eval_seeds_list, cap=cap)
            eval_record = {
                "update": episode,
                "mean_diff": report.mean_diff,
                "ci95": report.ci95,
                "wins": report.wins_a,
                "losses": report.wins_b,
                "pairs": report.pairs,
                "unfinished": report.unfinished,
                "wall_seconds": round(time.perf_counter() - eval_began, 2),
            }
            with eval_log_path.open("a") as handle:
                handle.write(json.dumps(eval_record) + "\n")
            dash.on_eval(eval_record)

    trainer.export_champion(str(run_dir / "champion-final.json"))
    dash.stop()
    print(f"done: {run_dir}")


def _cfg_dict(cfg: PGConfig) -> dict[str, Any]:
    from dataclasses import asdict
    return asdict(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--games-per-update", type=int, default=512)
    parser.add_argument("--mini-batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--baseline-decay", type=float, default=0.95)
    parser.add_argument("--entropy", type=float, default=0.0, help="entropy bonus coefficient")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="0 = no clipping")
    parser.add_argument("--cap", type=int, default=200_000)
    parser.add_argument("--run-seed", type=int, default=7)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--no-amp", action="store_true", help="disable bf16 autocast")
    parser.add_argument("--no-compile", action="store_true", help="disable torch.compile (default off; harmful without triton)")
    parser.add_argument("--compile", action="store_true", help="enable torch.compile (needs triton; slow on Windows without it)")
    parser.add_argument("--shards", type=int, default=1, help="worker processes for parallel rollouts")
    parser.add_argument("--opponent", type=str, default=None,
                        help="path to a canastra-weights@1 JSON (fixed opponent) or 'self' for frozen-self-play")
    parser.add_argument("--opponent-refresh", type=int, default=0,
                        help="with --opponent self: refresh the frozen opponent every N updates")
    parser.add_argument("--no-tui", action="store_true", help="print plain lines instead of a dashboard")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--ckpt-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=10,
                        help="evaluate the learner vs the fixed baseline every N updates (0 disables)")
    parser.add_argument("--eval-opponent", type=str, default=None,
                        help="path to a canastra-weights@1 JSON for the growth-eval baseline "
                             "(default: bots/src/fixtures/random-init.json)")
    parser.add_argument("--eval-seeds", type=int, default=32,
                        help="deals per growth eval (each played in both seatings)")
    args = parser.parse_args()

    opponent = args.opponent
    if opponent == "self":
        opponent = None  # internal: no fixed file, use frozen-self-play

    run_dir = args.run_dir or Path("runs") / f"pg-{time.strftime('%Y%m%d-%H%M%S')}"
    run(
        arch=TRAINING_ARCH,
        run_dir=run_dir,
        episodes=args.episodes,
        games_per_update=args.games_per_update,
        mini_batch=args.mini_batch,
        lr=args.lr,
        baseline_decay=args.baseline_decay,
        entropy_coef=args.entropy,
        grad_clip=args.grad_clip,
        cap=args.cap,
        run_seed=args.run_seed,
        device=args.device,
        amp=not args.no_amp,
        compile=args.compile and not args.no_compile,
        shards=args.shards,
        opponent=opponent,
        opponent_refresh=args.opponent_refresh,
        resume=args.resume,
        no_tui=args.no_tui,
        log_interval=args.log_interval,
        ckpt_interval=args.ckpt_interval,
        eval_interval=args.eval_interval,
        eval_opponent=args.eval_opponent,
        eval_seeds=args.eval_seeds,
    )


if __name__ == "__main__":
    main()
