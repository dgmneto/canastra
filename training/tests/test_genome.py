"""Genome round-trips and the pinned JSON format."""

from pathlib import Path

import numpy as np
from canastra_train import genome

ARCH = {"obs": 2002, "act": 101, "trunk": [64, 32], "head": [16], "activation": "tanh"}


def test_round_trip_through_modules_is_exact() -> None:
    vec = genome.random_genome(ARCH, seed=1)
    trunk, head = genome.to_modules(vec, ARCH)
    assert np.array_equal(genome.from_modules(trunk, head), vec)


def test_round_trip_through_json_survives_the_precision_cut(tmp_path: Path) -> None:
    vec = genome.random_genome(ARCH, seed=2)
    path = tmp_path / "weights.json"
    genome.save_json(str(path), ARCH, vec)
    arch, loaded = genome.load_json(str(path))
    assert arch == ARCH
    assert np.allclose(loaded, vec, atol=1e-6)


def test_load_rejects_foreign_formats(tmp_path: Path) -> None:
    vec = genome.random_genome(ARCH, seed=3)
    path = tmp_path / "weights.json"
    genome.save_json(str(path), ARCH, vec)
    raw = path.read_text().replace(genome.FORMAT, "something-else@9")
    path.write_text(raw)
    try:
        genome.load_json(str(path))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass