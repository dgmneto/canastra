"""Sanity gate: the EVALUATOR, not the genomes, must be unbiased.

The original gate ("random init ⇒ equal strength") was false: argmax over a
randomly-initialized network is a specific DETERMINISTIC policy, and two such
policies can genuinely differ in strength (like "random-plus" historically
beats "random" at ~75%). The M2 measurement proved it: seeds 101 vs 202 read
+1828 ± 287 head-to-head, flipped to −1938 ± 291 under swap, and collapsed to
+48 ± 150 self-vs-self — the evaluator was fair; the premise was wrong.

So instead of asserting that two random genomes are indistinguishable, we test
the evaluator's two structural properties directly:

1. **Self-null** — `evaluate_pair(vec, vec, ...)` must read ≈ 0. Duplicate
   deals cancel the shuffle; identical policies cancel everything else. Any
   nonzero reading is routing/pairing bias.
2. **Antisymmetry** — with two different genomes, `evaluate_pair(a, b)` and
   `evaluate_pair(b, a)` must flip sign. Any same-sign reading is slot bias.

A large |A vs B| differential is EXPECTED and fine here — it means the two
untrained policies genuinely differ in strength, which is exactly what training
will exploit.

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

    self_report = evaluate.evaluate_pair(vec_a, vec_a, ARCH, seeds)
    print(
        f"self-null: {self_report.pairs} pairs, mean diff {self_report.mean_diff:+.1f} "
        f"(95% CI ±{self_report.ci95:.1f})"
    )
    if abs(self_report.mean_diff) > self_report.ci95:
        raise SystemExit("FAIL: self-vs-self nonzero — routing or pairing bias")

    ab = evaluate.evaluate_pair(vec_a, vec_b, ARCH, seeds)
    ba = evaluate.evaluate_pair(vec_b, vec_a, ARCH, seeds)
    print(
        f"A vs B: mean diff {ab.mean_diff:+.1f} (±{ab.ci95:.1f}); "
        f"B vs A: {ba.mean_diff:+.1f} (±{ba.ci95:.1f})"
    )
    if abs(ab.mean_diff + ba.mean_diff) > ab.ci95 + ba.ci95:
        raise SystemExit("FAIL: differential does not flip with the slots — slot bias")

    print("OK: evaluator is unbiased (self-null and antisymmetry hold)")


if __name__ == "__main__":
    main()
