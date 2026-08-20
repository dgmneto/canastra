# Optimization harness

An automated **benchmark → optimize → evaluate → repeat** loop for the
`training-rs` lockstep self-play loop. Loop control is **deterministic** and
lives entirely in `scripts/optimize-loop.sh`; only hypothesis generation is done
by a model (the `/optimize-iteration` subagent). The harness works by hand with
zero agent involvement (`--dry-run`, `--no-model`).

This task built the **infrastructure only**. Nothing in the training hot path
was optimized. Candidate hypotheses are listed at the bottom.

## Quick start

```bash
# 0. Build the three binaries (one-time). From training-rs/:
#    timing (CUDA):   cmd /c '...vcvars64.bat... && set CUDA_COMPUTE_CAP=120 && \
#                      cargo build --release --features cuda --bins'
#    profile (CUDA):   cargo build --release --features "cuda,profile" \
#                      --target-dir target/profile --bin canastra-bench
#    gate (CPU only):  cargo build --release --target-dir target/gate --bin gate

# 1. (Re)generate the golden correctness fixture from current HEAD:
./training-rs/target/gate/release/gate --gen -o bench/golden/generation.json

# 2. One benchmark run → JSON on stdout, diagnostics on stderr:
./bench/run.sh                       # 30 samples, ~6 min
./bench/run.sh --samples 3 --warmup 1        # quick smoke
./bench/run.sh --profile              # also populate profile_top (one extra run)

# 3. Dry-run the loop (no model, no benchmark — evaluates stop conditions):
./scripts/optimize-loop.sh --dry-run

# 4. Run the loop (model-driven). Set OPTIMIZER_CMD to your headless Kilo CLI,
#    or run /optimize-iteration by hand in a fresh session per iteration:
export OPTIMIZER_CMD="kilo run --prompt-file"
./scripts/optimize-loop.sh
```

> Run the `.sh` files under **Git Bash** (`C:\Program Files\Git\bin\bash.exe`).
> Native-exe stdout is captured through Python `subprocess` — do not call the
> bench `.exe` directly from bash expecting to read its stdout (Git Bash's
> redirection does not capture MSVC-runtime output).

## What the metric is

- **Metric:** `games_per_s` — games of self-play per wall-second, one ES
  generation through the production lockstep path
  (`canastra-bench --population 96 --opponents 4 --seeds 8 --device cuda
  --dtype bf16`). 6,144 games, ~10–17 s on this GPU.
- **Why pop=96:** throughput is flat in population (599→584→572 across a 10×
  pop rise, per `docs/benchmarks.md`), so pop=96 is representative without
  paying pop=1000's ~110 s cost. The bench is the league-only path; the
  *holdout* (below) covers the code it omits.

## How to read a run JSON

```jsonc
{
  "commit": "1e4185e",            // git short HEAD at run time
  "dirty": false,                 // uncommitted changes in the tree?
  "correctness": "pass",          // golden gate result — must be "pass" to benchmark
  "correctness_detail": "",       // failure reason if "fail"
  "metric": "games_per_s",
  "value": 604.27,                // bootstrap mean of the samples
  "ci_low": 599.73, "ci_high": 609.17,  // 95% bootstrap CI of the mean
  "samples": 30,
  "wall_seconds": 347.5,          // harness wall (all samples + warmup)
  "profile_top": [ {"symbol":"argmax","pct":38.2}, ... ]  // top-5 phases
}
```

`profile_top` is the **"where the time goes" map** — the reason the harness
exists. Phases are non-overlapping leaves of the lockstep critical path
(argmax + encode + apply + h2d + trunk/head/mask/gather ≈ 100% of wall):
- **argmax** — the per-ply device→host score download + argmax sync. The single
  dominant hotspot (~38–48% of wall). This is the per-ply `to_vec1::<f32>()`
  full-matrix download in `policy.rs`.
- **encode / apply** — the engine stepping (`Pool::encode` / `Pool::apply`),
  rayon-parallel, on the lockstep critical path (~13–24% combined). No overlap
  with GPU in the lockstep design.
- **h2d** — host→device uploads (obs/acts/mask tensors).
- **trunk/head/mask/gather** — the grouped matmul forward stages within GPU.

The profile build runs slower (atomics + spans per ply, ~415 vs 600 games/s),
which is why it is a separate gated run (`--profile`) and never on the timing
path.

## How to read the ledger

`bench/LEDGER.jsonl` — one JSON object per iteration, append-only:

```jsonc
{"iter":1,"ts":"…","parent_commit":"…","commit":"…",
 "hypothesis":"argmax: download argmax index only, not full score matrix",
 "target_symbol":"argmax",
 "predicted_speedup":1.20,   // the model's BEFORE-edit prediction (mandatory)
 "actual_speedup":1.18,      // value/baseline_value
 "verdict":"accept",         // accept | reject | correctness_fail | harness_fail
 "lines_changed":7,
 "notes":"605→713 games/s; ci [708,718] vs base [600,609]",
 "profile_top":[...]}
```

Queries (`python bench/ledger.py <cmd>`): `recent N`, `rejected`,
`patience`, `predcorr N`, `last-profile`, `targets`.

The **accept rule is the CI non-overlap test**: a change is accepted only if
`correctness == pass` AND `new.ci_low > baseline.ci_high` (the new CI sits
entirely above the baseline CI). Overlapping CIs are rejected *always*,
regardless of the point estimate — a 2% point gain whose CI overlaps the
baseline is noise, not a win. This is why the harness never emits a bare mean.

## Stopping conditions (all in `scripts/optimize-loop.sh`, all named constants)

| Constant | Default | Catches |
|---|---|---|
| `TARGET_VALUE` | unset | absolute goal reached |
| `PATIENCE` | 3 | consecutive non-significant iterations |
| `MIN_IMPROVEMENT` | 0.5 | significance = gain ≥ 0.5× baseline CI width (ties the bar to the noise floor, not a fixed %) |
| `AMDAHL_FLOOR` | 1.15 | stop (with patience) if zeroing the top hotspot can't reach this — `1/(1−top_pct) < 1.15` |
| `COMPLEXITY_RATIO` | 4 | stop if lines-changed-per-% > 4× the median of the first 5 accepted (the change is getting too expensive per unit of speedup) |
| `PRED_CORR_MIN` | 0.3 | stop if predicted-vs-actual Pearson r over last 5 accepted decays below this (the model's predictions are no longer informative) |
| `MAX_ITERS` | 50 | hard cap |
| `MAX_WALL_SECONDS` | 3600 | hard wall cap |
| `MAX_COST_USD` | 5.0 | hard cost cap (not metered by default — wire a cost source into `stop_max_cost` to enable) |

Master rule:
```
stop = target_reached
  OR (patience_exhausted AND amdahl_ceiling < AMDAHL_FLOOR)
  OR complexity_exceeded
  OR prediction_decayed
  OR any hard budget hit
```
Every stop is attributable — on exit the script prints which condition fired and
why. `--dry-run` evaluates every condition against the current ledger without
running the model or the benchmark.

## The correctness gate

`training-rs/src/bin/gate.rs` runs ONE fixed-seed generation (seed 7, pop 8,
opponents 2, seeds 2, max_hands 1) on **CPU fp32** (exact, deterministic) and
compares the full per-game result vector byte-for-byte against
`bench/golden/generation.json`. It is independent of the timing path and runs
FIRST in every `run.sh`. A change that speeds the loop up by altering behaviour
flips at least one argmax pick → a different score → a different result vector →
`correctness: "fail"`, non-zero exit, **no benchmark runs**. Regenerate the
golden from a known-good HEAD with `gate --gen`.

The bench also carries a **degenerate-forward guard**: if every game ends level
(the f16-masking-bug signature — both teams on the flat −300, games over
instantly), `_lib.py` rejects the sample as a harness failure, so a collapsed
policy can never be recorded as a speedup. See
`training-rs/docs/decision-ranking-metric.md` for the bug this guards against.

## The holdout

`bench/holdout.sh` runs one full `canastra-train` generation — a **different
path** through the same target: ES population materialisation + anchor
evaluation + the Adam step, which the league-only bench omits (see
`docs/benchmarks.md` "Whole-generation cost"). The loop never optimizes against
this number. `optimize-loop.sh` runs it once at the very end and compares to
`bench/baseline-holdout.json`. If the main metric improved materially but the
holdout did not move proportionally (gain < 0.5× the main gain), the script
prints a loud **OVERFIT** warning — the wins did not generalize.

Generate the holdout baseline before the loop:
```bash
./bench/holdout.sh --baseline bench/baseline-holdout.json
```

## Machine-state assumptions & noise floor

- **Hardware:** RTX 5060 Ti (16 GB, Blackwell sm_120), driver 610.88, CUDA 13.3,
  8 logical CPU cores. CUDA build via `vcvars64.bat` + `CUDA_COMPUTE_CAP=120`.
- **Dominant noise:** GPU thermals / clock-boost, not CPU scheduling. The first
  process after a cold GPU reads ~40% low (measured 359 vs 600 games/s). `run.sh`
  discards `WARMUP=2` runs before sampling — mandatory.
- **CPU pinning:** Windows has no `taskset`. Set `PIN_CORES` to a hex mask (e.g.
  `FF`) to pin via `start /affinity`. Default unset = no pinning. GPU work is
  unaffected; this only stabilizes the rayon encode/apply phases (~13% of wall).
- **Measured noise floor (acceptance item 4):** 10 control runs gave
  **mean 602.5 games/s, stddev 8.8, CV ≈ 1.46%** (min 588, max 618). With 30
  samples the SEM ≈ 1.6 games/s and a 95% CI half-width ≈ ±3.2 games/s. Two
  non-overlapping CIs (the accept rule) need point estimates separated by
  ≈ **6.4 games/s ≈ 1.06%**. **A 2% improvement is therefore detectable** —
  but only because the harness uses 30-sample CIs, never single runs.
- **Consequence:** the loop's 1% control-drift guard sits *below* single-run
  noise (±1.46%), so the control run is itself multi-sample (`CONTROL_SAMPLES=10`)
  and compared as a value-against-baseline; a single-shot control would
  false-abort constantly.

## Tuning the thresholds

- If your machine is noisier (CV > 2%), raise `MIN_IMPROVEMENT` to 0.75–1.0 and
  `SAMPLES`/`EXPERIMENT_SAMPLES` to 50 — the CI widens, so the non-overlap bar
  rises with it. Do not lower `MIN_IMPROVEMENT` below 0.5 on this hardware.
- `AMDAHL_FLOOR=1.15` means "stop once even zeroing the #1 hotspot couldn't
  reach a 15% speedup." Lower it to chase smaller wins; raise it to stop sooner.
- `PRED_CORR_MIN=0.3` over 5 accepted points — if the model can't predict
  direction better than chance, its hypotheses aren't worth running. Raise to
  0.5 for a stricter bar.
- `COMPLEXITY_RATIO=4` — if a change costs >4× the per-% cost of the early wins,
  you're paying too much per unit speedup. Tune to taste.

## Candidate hypotheses (NOT implemented — for the optimizer to pick up)

These are the obvious wins the profiler surfaces; they are noted here, not
done, per the task's "do not optimize" constraint.

1. **argmax download (38–48% of wall):** the per-ply `forward_scores` path
   downloads the full `[rows, width]` score matrix to CPU
   (`to_vec1::<f32>()`) just to take an argmax. An argmax-only download (index
   + value per row, or a GPU-side argmax with a single `[rows]` download) would
   cut D2H volume by ~width× (width up to 250 mid-game). This is the
   `argmax` span in `policy.rs`. Highest leverage.
2. **`WeightStack::from_roster` cast spike:** the transient f32→bf16 cast causes
   a VRAM spike (the pop=2000 OOM wall). De-spiking it is both a stability and
   a throughput win at large pop. Out of scope for pop=96 but profiled.
3. **encode/apply on the critical path (~24%):** the lockstep design
   sacrificed encode/apply↔GPU overlap that the old coalesced path had. A
   double-buffered / stream-overlap pipeline could hide encode behind the GPU
   forward. (Phase-1 finding in `docs/profile-baseline.md`.)
4. **`acts` H2D volume:** `[N, width, ACT_DIM]` is large; a genome-grouped grid
   layout would make the gather a no-op reshape. Lower priority at pop=96.

Dead-ends already tried (do NOT re-attempt): sparse embedding-bag input layer,
ES grouped-GEMM split, bmm transpose, f16 dtype — see `docs/benchmarks.md` and
`docs/decision-ranking-metric.md`.
