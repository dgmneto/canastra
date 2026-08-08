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
