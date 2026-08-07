"""Sanity gate: two random genomes must be indistinguishable.

Same architecture, different seeds — neither has learned anything, so the
duplicate-deal differential must sit inside its confidence interval around
zero. If it does not, the evaluator (not the genomes) is broken.

Spec default is 1000 paired seeds; run that on the training machine. The
`--pairs` flag keeps local runs affordable.
"""

from __future__ import annotations

import argparse

from canastra_train import evaluate, genome

ARCH = {"obs": 2002, "act": 101, "trunk": [512, 256], "head": [128], "activation": "tanh"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=1000)
    parser.add_argument("--first-seed", type=int, default=1)
    args = parser.parse_args()

    vec_a = genome.random_genome(ARCH, seed=101)
    vec_b = genome.random_genome(ARCH, seed=202)
    seeds = list(range(args.first_seed, args.first_seed + args.pairs))
    report = evaluate.evaluate_pair(vec_a, vec_b, ARCH, seeds)
    print(
        f"{report.pairs} pairs: mean diff {report.mean_diff:+.1f} "
        f"(95% CI ±{report.ci95:.1f}), wins {report.wins_a}/{report.wins_b}, "
        f"unfinished {report.unfinished}"
    )
    if abs(report.mean_diff) > report.ci95:
        raise SystemExit("FAIL: random genomes separated — the evaluator is biased")
    print("OK: random genomes are indistinguishable")


if __name__ == "__main__":
    main()
