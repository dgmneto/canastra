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
.venv/bin/pip install -e ".[dev]"        # installs numpy + maturin, pytest, ruff, mypy
.venv/bin/maturin develop                # rebuild the Rust extension into the venv
.venv/bin/pytest                         # run the tests
.venv/bin/ruff check .                   # lint
.venv/bin/mypy python/canastra_train tests   # type check
```

Re-run `maturin develop` after any change to `src/lib.rs`; the Python gates only see the
rebuilt extension.

## Benchmark

Env stepping is the training loop's ceiling (the forward pass is batched elsewhere), so
this number is the one to watch when touching the pool or the encoder:

```bash
.venv/bin/python -m canastra_train.bench
```

Measured on this machine (a random legal policy over a 64-game pool):
**79,280 plies in 794.18s ≈ 100 plies/s.**

## Coming later

CUDA support is a configuration flag for a later milestone — `torch` arrives in M2, and
`canastra_py` expects to hand tensors to it unchanged.
