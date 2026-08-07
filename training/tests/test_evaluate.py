"""The evaluator, at smoke scale: random genomes, a handful of pairs."""

from canastra_train import evaluate, genome

ARCH = {"obs": 2002, "act": 101, "trunk": [32], "head": [16], "activation": "tanh"}


def test_random_genomes_are_indistinguishable_at_smoke_scale() -> None:
    vec_a = genome.random_genome(ARCH, seed=11)
    vec_b = genome.random_genome(ARCH, seed=22)
    report = evaluate.evaluate_pair(vec_a, vec_b, ARCH, seeds=[1, 2, 3], cap=200_000)
    assert report.pairs == 3
    assert report.unfinished == 0
    # Three pairs cannot prove equality; they can prove the machinery runs
    # and produces finite, paired differentials.
    assert abs(report.mean_diff) < 5000
