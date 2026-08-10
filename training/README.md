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

Env stepping is the training loop's ceiling (the forward pass is batched elsewhere),
so this number is the one to watch when touching the pool or the encoder:

```bash
.venv/bin/python -m canastra_train.bench                      # raw pool loop
.venv/bin/python -m canastra_train.bench --shape league --device cpu   # the real generation loop
```

Measured on this machine on a **release build**:

- **`--shape pool`** (a random legal policy over a 64-game pool): **104,998 plies in
  219.16s ≈ 479 plies/s.** Here "plies" counts batch rounds — one `encode` + `apply`
  across every live game — not individual game-steps. The pool sustains
  an order of magnitude more real game-steps per second at full depth; the rounds
  figure is dominated by the tail, where a few long matches straggle with little parallelism.
  This is expected for uniform-random play, and the action cap (`Pool(seeds, max_actions_per_game)`)
  is what keeps a non-converging match from hanging a generation.
- **`--shape league`** (the real loop through `drive_pool`, 64 games, pop 8, cap 30k):
  **332 plies/s on `cpu`** (forward 1.8 ms/ply, encode 1.1, apply 0.04) vs **378 plies/s on
  `cuda`** (forward 1.6, encode 1.0, apply 0.04). At the full training shape (pop 96, one
  pool of 768 games) the picker **buckets rows by real menu size** — real menus are tiny
  (p50 ≈ 2 actions), so each bucket's acts tensor is trimmed to the bucket's max width
  instead of the one rogue 600-action meld that pads every row to the ply's global max.
  That cut the cuda forward from 57 ms/ply to **27 ms/ply** (encode 17, apply 1.4) and the
  end-to-end pop-96 × 8-shard run from ~16 plies/s to ~33 plies/s. On `cpu` the same
  trimming does not pay off (per-ply forward ~165 ms — the smaller per-bucket ops beat the
  transfer saving), so keep `--device cpu` only on machines without a big GPU; the default
  on this laptop's RTX stays `cpu`, but pass `--device cuda` for training runs. The gap
  between the league figure and the pool figure is the torch effort per ply — the league
  number is the one to watch when touching the policy or the picker.

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

## Training

The real GA trainer is a pure-Python loop that ships an evolution over the
`canastra_py` pool; the forward pass and the search live in CPython/torch, only
the game itself lives in Rust. Run it from this directory:

```bash
.venv/bin/python -m canastra_train.train --generations N --run-dir runs/<run>
```

Flags:

- `--generations` (required) — how many generations to run (or, with
  `--resume`, how many total to have run).
- `--population` (default 96), `--elites` (8), `--tournament` (4) — the GA shape.
- `--opponents` (4) — how many opponents each genome faces per generation.
- `--seeds` (8) — deals per opponent pairing in the self-play league.
- `--cap` (200_000) — actions per game; a non-converging match hangs a
  generation otherwise.
- `--run-seed` (7) — derives every seed stream and every mutation draw, so a
  run is reproducible.
- `--sigma` (0.02), `--sigma-decay` (0.995), `--sigma-floor` (0.002) — Gaussian
  mutation scale and its decay across generations.
- `--hof-interval` (5) — how often the current champion is archived to the hall
  of fame and exported.
- `--crossover` — accepted but unused (the flag exists per spec; nothing
  implements crossover yet).
- `--device` (`cpu`|`cuda`|`mps`) — where the torch forward pass runs.
  **Default `cpu`** — on this machine the two devices are close (see Benchmark),
  and CPU keeps a second process (the dashboard writer, checkpointing, and the
  per-generation GA glue) off the PCIe bus. On the training machine, pass
  `--device cuda`; the stacked whole-roster forward is one batched pass that
  uses the GPU at full depth.
- `--run-dir` — output directory (default `runs/<timestamp>`).
- `--resume` — continue from the latest checkpoint in `--run-dir`.
- `--no-tui` — disable the live dashboard; print one plain line per
  generation instead (automatic when stdout is not a TTY, e.g. piped to a
  file).

### Live dashboard

On a TTY the trainer renders a watch-only `rich` dashboard while it runs:
generation count and phase, games finished / total with a progress bar,
plies and plies/s, ETA for the current generation and the whole run, sigma,
fitness best/mean with a sparkline, the last generations' table, and a live
events feed for the promotion moments — new best-erves, champion exports and
hall-of-fame archivals. It also writes a throttled, atomically-updated
`status.json` into the run directory (same shape as the in-memory status) so
a separate terminal or a future web view can tail a run without touching
training itself. The dashboard is exit-free — stop a run with Ctrl+C as
usual, and restart with `--resume`.

One generation does three things:

1. **Batched self-play league.** The population is paired against itself (and a
   sample of hall-of-fame champions, so an evolved policy cannot cycle against a
   single fixed target) and played out on the `Pool` with this generation's
   deterministic deal seeds.
2. **Duplicate-deal differentials.** Each pairing is scored by the paired
   evaluator — both seatings on the same deal, so the deal cancels and what
   remains is policy (`evaluate_pair`).
3. **Elitism + tournaments + Gaussian mutation.** The `--elites` fittest survive
   unchanged; the rest of the next population is tournament-selected parents
   perturbed by `--sigma` noise.

Checkpoints (`gen-*.npz` under `--run-dir`) store the population, fitness, hall
of fame and the generation's deal seeds. Every source of randomness derives from
`--run-seed` and the generation number (seed streams, mutation draws), so
`--resume` replays a run **bit-identically** from its latest checkpoint — no RNG
state is stored, because none needs to be. Champion weights export as plain JSON
(`champion-gen*.json`, `champion-final.json`) in the `canastra-weights@1` format,
playable directly through the TS evaluator:

```bash
npx tsx harness/src/eval-nn.ts runs/<run>/champion-final.json random 1   # from repo root
```

### Success gate

The training-machine gate for M3, measured *after* the run with the M2 tools:

```bash
.venv/bin/python -m canastra_train.train --generations 200          # or as needed
npx tsx harness/src/eval-nn.ts runs/<run>/champion-final.json random 1000
npx tsx harness/src/eval-nn.ts runs/<run>/champion-final.json random-plus 1000
```

The champion must beat `random` decisively; against `random-plus`, the mean
differential must be ≥ 0 within CI (~10k hands ≈ 1000 matches). Note honestly:
on an ordinary laptop this smoke run only proves the loop end to end — the gate
belongs to the training machine, where a handful of shallow generations is
nowhere near enough to pull a policy off uniform-random play.

## Coming later

CUDA support is a configuration flag for a later milestone — `torch` arrives in M2, and
`canastra_py` expects to hand tensors to it unchanged.
