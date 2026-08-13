"""Sharded evaluation is bit-identical to the single-process path.

Per-game play is deterministic and the workers return results in global game
order, so splitting a generation across processes must not change a single
record — otherwise every resumed run and every comparison in the gate breaks.
"""

import json
from pathlib import Path

from canastra_train import train

ARCH = {"obs": 2002, "act": 101, "trunk": [16], "head": [], "activation": "tanh"}


def run_into(directory: Path, shards: int) -> None:
    train.run(
        arch=ARCH,
        run_dir=directory,
        generations=2,
        population=8,
        elites=2,
        tournament=2,
        opponents=2,
        seeds=2,
        max_hands=1,
        run_seed=13,
        hof_interval=1,
        shards=shards,
    )


def test_shards_are_bit_identical_to_single_process(tmp_path: Path) -> None:
    single = tmp_path / "single"
    sharded = tmp_path / "sharded"
    run_into(single, shards=1)
    run_into(sharded, shards=2)

    single_lines = (single / "generations.jsonl").read_text().strip().splitlines()
    sharded_lines = (sharded / "generations.jsonl").read_text().strip().splitlines()
    assert len(single_lines) == len(sharded_lines) == 2
    for one, two in zip(single_lines, sharded_lines):
        one_rec = json.loads(one)
        two_rec = json.loads(two)
        for key in ("generation", "elo_mean", "elo_best", "elo_worst",
                    "champion", "sigma", "seeds"):
            assert one_rec[key] == two_rec[key], f"{key} diverges: {one} vs {two}"

    assert (tmp_path / "sharded" / "status.json").exists()