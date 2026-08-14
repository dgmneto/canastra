"""Sharded PG: grads match single-process within fp32 tolerance; resume exact.

Sharding partitions seeds by unique deal (keeping dup-deal pairs together),
each worker runs its own rollouts + backward, and the parent sums flat grads.
With fp32 + no compile, same-config (`--shards N`) resume reproduces parameters
exactly. The bit-identity contract is intentionally relaxed vs the GA (bf16
drifts ~1e-3), but in fp32 mode the sharded grad equals the single-process grad
within float tolerance.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from canastra_train import pg, train_pg

ARCH = {"obs": 2002, "act": 101, "trunk": [16], "head": [], "activation": "tanh"}


def _cfg_dict(**overrides: Any) -> dict[str, Any]:
    base = {
        "games_per_update": 8, "mini_batch": 4, "lr": 1e-3,
        "cap": 6000, "device": "cpu", "amp": False, "compile": False,
    }
    base.update(overrides)
    return base


def test_sharded_grad_matches_single_process(tmp_path: Path) -> None:
    """With shards=1, the sharded grad (through the spawn + reduction path) equals
    the single-process grad exactly — same seeds, same generator, same scaling.
    This verifies the grad reduction (sum + divide) is correct."""
    from canastra_train import seedstream

    cfg = pg.PGConfig(**_cfg_dict())
    n_deals = cfg.games_per_update // 2
    seeds = seedstream.generation_seeds(11, 0, n_deals)

    # Single-process: run rollouts, capture the accumulated grad before optimizer.step().
    single_trainer = pg.Trainer(ARCH, cfg, run_seed=11, shards=1)
    single_trainer.optimizer.zero_grad(set_to_none=True)
    single_trainer._local_rollouts(seeds)
    single_flat = _extract_flat_grad(single_trainer)

    # Sharded with 1 shard: same generator seed, same seed partition → same grad.
    sharded_trainer = pg.Trainer(ARCH, cfg, run_seed=11, shards=1)
    sharded_trainer.optimizer.zero_grad(set_to_none=True)
    from canastra_train import shards_pg
    shards_pg.run_sharded_step(trainer=sharded_trainer, all_seeds=seeds)
    sharded_flat = _extract_flat_grad(sharded_trainer)

    assert np.allclose(single_flat, sharded_flat, atol=1e-5), (
        f"sharded grad diverges from single: max diff {np.max(np.abs(single_flat - sharded_flat))}"
    )


def test_sharded_resume_reproduces_weights(tmp_path: Path) -> None:
    """Same-config sharded resume reproduces champion-final.json bit-for-bit (fp32)."""
    def run_into(directory: Path, episodes: int, resume: bool = False) -> None:
        train_pg.run(
            arch=ARCH, run_dir=directory, episodes=episodes,
            games_per_update=8, mini_batch=4, cap=6000,
            run_seed=13, device="cpu", amp=False, compile=False,
            shards=2, log_interval=1, ckpt_interval=1, resume=resume,
        )

    continuous = tmp_path / "continuous"
    resumed = tmp_path / "resumed"
    run_into(continuous, episodes=2)
    run_into(resumed, episodes=1)
    run_into(resumed, episodes=2, resume=True)

    from canastra_train import genome
    cont_arch, cont_vec = genome.load_json(str(continuous / "champion-final.json"))
    res_arch, res_vec = genome.load_json(str(resumed / "champion-final.json"))
    assert cont_arch == res_arch == ARCH
    assert np.array_equal(cont_vec, res_vec), "sharded resume diverges from continuous"


def test_sharded_smoke_produces_artifacts(tmp_path: Path) -> None:
    train_pg.run(
        arch=ARCH, run_dir=tmp_path, episodes=2,
        games_per_update=8, mini_batch=4, cap=6000,
        run_seed=17, device="cpu", amp=False, compile=False,
        shards=2, log_interval=1, ckpt_interval=1,
    )
    lines = (tmp_path / "updates.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        rec = json.loads(line)
        assert rec["games"] >= 8


def _extract_flat_grad(trainer: pg.Trainer) -> np.ndarray:
    """Flatten the trainer's current parameter grads in to_genome_vec order."""
    from canastra_train.pg import unwrap
    parts = []
    for param in unwrap(trainer.net).parameters():
        if param.grad is None:
            parts.append(np.zeros(param.numel(), dtype=np.float32))
        else:
            parts.append(param.grad.detach().cpu().to(torch.float32).numpy().ravel())
    return np.concatenate(parts).astype(np.float32)
