# canastra-train

Training harness for Canastra policy networks. This directory is **training-only**:
all game rules live in the Rust engine (`engine/crates/canastra-engine`) and are reached
through the `canastra_py` PyO3 bindings, so training and evaluation can never
drift from the rules the deployed bots play.

The harness is a separate maturin project, deliberately outside the engine Cargo
workspace — the engine's build and test gates stay Python-free and fast.

## Layout

- `src/lib.rs` — the PyO3 extension (`canastra_py`). `Pool` owns one engine per seed and
  drives whole batches with **two batched crossings per ply (encode, apply)** — Rust fills numpy buffers for
  observations, per-action features, and a legal-action mask, and consumes the picks back.
  Safe-mode menus mirror the `canastra-harness` driver — a dead-ended turn is backed out
  and retried with melding/pile-taking withheld.
- `python/canastra_py/` — the Rust extension package (the `.pyi` stub is for `mypy`).
- `python/canastra_train/` — the pure-Python training package (re-exports `canastra_py`
  and holds the benchmark).
- `tests/test_pool.py` — integration test driving whole matches to completion.

## Dev loop

From this directory, with a virtualenv:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"          # installs numpy + maturin, pytest, ruff, mypy
.venv/bin/maturin develop --release        # rebuild the Rust extension into the venv
.venv/bin/pytest                           # run the tests
.venv/bin/ruff check .                     # lint
.venv/bin/mypy python/canastra_train tests # type check
```

Use a **release build** (`maturin develop --release`, ~10–50× faster). A debug build is only
good for hunting a Rust panic; the benchmark and the test suite are meaningless on it. Re-run
`maturin develop --release` after any change to `src/lib.rs`.

## Benchmark

Env stepping is the training loop's ceiling (the forward pass is batched elsewhere), so
this number is the one to watch when touching the pool or the encoder:

```bash
.venv/bin/python -m canastra_train.bench
```

Measured on this machine on a **release build** (a random legal policy over a 64-game pool):
**104,998 plies in 219.16s ≈ 479 plies/s.** Here "plies" counts batch rounds — one `encode`
+ `apply` across every live game — not individual game-steps. The pool sustains
an order of magnitude more real game-steps per second at full depth; the rounds
figure is dominated by the tail, where a few long matches straggle with little parallelism.
This is expected for uniform-random play, and the action cap (`Pool(seeds, max_actions_per_game)`)
is what keeps a non-converging match from hanging a generation.

## Evaluation

Three evaluation paths, two sides of the same spec:

- `python/canastra_train/evaluate.py` — **duplicate-deal paired evaluation.** Canastra's
  variance is dominated by the deal, so two genomes are compared by playing each seed twice —
  once with genome A in seats 0/2, once with the seats swapped — and averaging A's score
  differential over the pair. The deal cancels; what remains is policy. `evaluate_pair` drives
  a `Pool` of `2 * len(seeds)` games and routes each row's pick to the correct genome based on
  the `(game, seat)` the pool reports. The `PairReport` carries the mean differential, a 95% CI,
  per-side wins, and how many games hit the action cap.
- `python/canastra_train/sanity.py` — the **evaluator-bias gate.** It tests the evaluator's
  structural properties rather than asserting anything about genomes: **self-null** —
  `evaluate_pair(vec, vec, ...)` reads ≈ 0 (duplicate deals cancel the shuffle, identical
  policies cancel everything else, so any nonzero reading is routing/pairing bias); and
  **antisymmetry** — `evaluate_pair(a, b)` and `evaluate_pair(b, a)` must flip sign (a same-sign
  reading is slot bias). The original gate ("two random genomes must be indistinguishable") was
  abandoned because its premise was false: argmax over a random-init network is a specific
  DETERMINISTIC policy, and two such policies can genuinely differ in strength. M2 measured it —
  seeds 101 vs 202 read +1828 ± 287 head-to-head, flipped to −1938 ± 291 under swap, and
  collapsed to +48 ± 150 self-vs-self: the evaluator was fair, the premise was wrong. A large
  |A vs B| differential is expected and fine; it is exactly what training will exploit.
  `--pairs` controls how many paired seeds run — local sanity uses **100** pairs (~1–3 min),
  the spec default of **1000** is the training-machine version of the same command.
- `harness/src/eval-nn.ts` — the **TS-side external validation.** It plays a weights file
  against a registered heuristic bot through the same harness everyone else uses, both seatings,
  and prints a head-to-head report. This is the mirror of the Python evaluator: there, genomes
  play genomes; here, a concrete weights file faces the heuristic bots.

The Python `Pool` returns rows of `(obs, acts, mask, rows)` where `rows` is per-row
`(game, seat)` — that is what lets the caller route each pick to a per-seat policy (the
duplicate-deal evaluator uses it to hand seat 0/2 to genome A and 1/3 to genome B).

Two behaviours worth holding in mind until training lands:

- **`turn_start` is re-captured on refusal phases.** The pool deliberately re-captures the turn
  checkpoint on `AwaitingRefusalChoice` as well as `AwaitingDraw`, so a red-3 replacement turn
  replays from after the draw with the red 3 already on the table — a documented divergence from
  the harness driver's checkpoint rule (which only re-captures on `AwaitingDraw`).
- **Untrained weights are degenerate.** Random genomes still finish matches (the evaluator's pool
  of two genomes per team runs ~20k plies), but a single weights file playing all four seats
  typically runs to the action cap and reports `unfinished:true`. That is expected before
  training pulls the policies away from uniform-random play, not a bug.

## Coming later

CUDA support is a configuration flag for a later milestone — `torch` arrives in M2, and
`canastra_py` expects to hand tensors to it unchanged.
