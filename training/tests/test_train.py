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
        max_hands=1,
        run_seed=9,
        hof_interval=1,
    )
    assert (tmp_path / "config.json").exists()
    lines = (tmp_path / "generations.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["generation"] == 0
    assert "elo_best" in first and "sigma" in first and "seeds" in first
    champions = sorted(tmp_path.glob("champion-*.json"))
    assert champions, "champion weights exported"
    arch_loaded, _vec = genome.load_json(str(champions[-1]))
    assert arch_loaded == ARCH
    assert sorted(tmp_path.glob("gen-*.npz")), "checkpoint written"


def test_resume_continues_from_the_checkpoint(tmp_path: Path) -> None:
    def run_once(generations: int, resume: bool = False) -> None:
        train.run(
            arch=ARCH,
            run_dir=tmp_path,
            generations=generations,
            population=4,
            elites=1,
            tournament=2,
            opponents=1,
            seeds=1,
            max_hands=1,
            run_seed=9,
            hof_interval=1,
            resume=resume,
        )

    run_once(1)
    run_once(2, resume=True)
    lines = (tmp_path / "generations.jsonl").read_text().strip().splitlines()
    assert [json.loads(line)["generation"] for line in lines] == [0, 1]


def test_resume_is_bit_identical_to_an_uninterrupted_run(tmp_path: Path) -> None:
    def run_into(directory: Path, generations: int, resume: bool = False) -> None:
        train.run(
            arch=ARCH,
            run_dir=directory,
            generations=generations,
            population=4,
            elites=1,
            tournament=2,
            opponents=1,
            seeds=1,
            max_hands=1,
            run_seed=9,
            hof_interval=1,
            resume=resume,
        )

    continuous = tmp_path / "continuous"
    resumed = tmp_path / "resumed"
    run_into(continuous, generations=2)
    run_into(resumed, generations=1)
    run_into(resumed, generations=2, resume=True)

    def record(directory: Path, generation: int) -> dict[str, object]:
        for line in (directory / "generations.jsonl").read_text().strip().splitlines():
            entry: dict[str, object] = json.loads(line)
            if entry["generation"] == generation:
                return entry
        raise AssertionError(f"no generation {generation} record in {directory}")

    keys = ("elo_mean", "elo_best", "champion", "seeds")
    cont = record(continuous, 1)
    res = record(resumed, 1)
    for key in keys:
        assert cont[key] == res[key], f"resumed generation-1 {key} diverges: {cont[key]} != {res[key]}"
