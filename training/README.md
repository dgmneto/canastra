# canastra-train

Training harness for Canastra policy networks. This directory is **training-only**:
all game rules live in the Rust engine (`engine/crates/canastra-engine`) and are reached
through the `canastra_py` PyO3 bindings, so training and evaluation can never
drift from the rules the deployed bots play.

The harness is a separate maturin project, deliberately outside the engine Cargo
workspace — the engine's build and test gates stay Python-free and fast.

## Layout

- `src/lib.rs` — the PyO3 extension (`canastra_py`). `Pool` owns one engine per seed and
  drives whole batches with **one FFI crossing per ply**: Rust fills numpy buffers for
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
**79,280 plies in 220.20s ≈ 360 plies/s.** Here "plies" counts batch rounds — one `encode`
+ `apply` across every live game — not individual game-steps. The pool sustains roughly an
order of magnitude more real game-steps per second (~9k actions/s at full depth); the rounds
figure is dominated by the tail, where a few long matches straggle with little parallelism.
This is expected for uniform-random play, and the action cap (`Pool(seeds, max_actions_per_game)`)
is what keeps a non-converging match from hanging a generation.

## Coming later

CUDA support is a configuration flag for a later milestone — `torch` arrives in M2, and
`canastra_py` expects to hand tensors to it unchanged.
