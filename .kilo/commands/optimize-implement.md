---
description: Optimisation loop - implement ONE idea, then STOP.
---
# /optimize-implement — implement ONE idea, then STOP

You are **one of N PARALLEL implementation sessions**. The driver spawns N
sessions at once, each in its own git worktree, each given ONE idea. You
implement exactly that ONE idea and then STOP.

The DRIVER (`scripts/optimize-loop.sh`) owns ALL control flow, ALL benchmarks,
ALL correctness-gate builds, ALL ledger writes, and ALL merge decisions. You do
NOT benchmark, do NOT build the gate, do NOT append to the ledger, do NOT merge.

## Your idea
Read the idea you must implement from `bench/.idea.txt` (the driver wrote it into
your worktree). It contains: `idea_index`, `target_symbol`,
`predicted_speedup`, `hypothesis`, `rationale`. You implement THAT idea only.

## Target
The optimisation target is the **training-rs lockstep self-play loop**, measured
by `canastra-bench` throughput (`games_per_s`, CUDA BF16 on RTX 5060 Ti). The
profiler breaks the wall into non-overlapping phases: `argmax` (per-ply
device→host score download + argmax sync — the dominant hotspot), `encode`,
`apply` (rayon engine stepping on the lockstep critical path), `h2d`, and the
grouped-matmul stages `trunk`/`head`/`mask`/`gather`. See
`training-rs/docs/profile-baseline.md` and `training-rs/docs/benchmarks.md`,
including the **dead-ends already tried** (sparse embedding-bag, ES grouped-GEMM
split, bmm transpose, f16 dtype) — do NOT re-attempt those.

## Step 1 — read context (no edits yet)
- `cat bench/.idea.txt` — YOUR assigned idea.
- `python bench/ledger.py recent 10` — recent iterations.
- `python bench/ledger.py rejected` — do not duplicate a rejected approach.
- `python bench/ledger.py last-profile` — the hotspot your idea targets.

## Step 2 — implement the ONE idea (smallest change)
Edit ONLY under `training-rs/src/**`. One minimal change that tests exactly the
hypothesis. Do not refactor unrelated code. Do not touch `bench/**`, the golden
fixtures (`bench/golden/**`), `training-rs/tests/reference/**`,
`training-rs/src/bin/gate.rs`, or any timing/metric code. If the idea would alter
behaviour, that is fine — the gate will catch it; do NOT "fix" the golden.

## Step 3 — verify by reasoning (NO builds, NO benchmarks)
Confirm your change is complete, minimal, and compiling-by-construction by
reading the code carefully. **Do not run any build or benchmark** (see hard
rules) — the driver compiles (correctness gate) and benchmarks your change
**serially** during validation. If you cannot produce a viable implementation
after reasonable effort, mark the idea **DROPPED** (`status: "dropped"`) and
stop.

## HARD RULE — do NOT run expensive builds or benchmarks
You are one of N sessions running in **parallel**. The machine has ONE GPU and
limited CPU; running builds/benchmarks in every parallel session would thrash
the environment. Therefore you MUST NOT run any of:
- `./bench/run.sh` — the GPU timing benchmark. **The driver owns this.**
- `cargo build --features cuda` — the GPU build.
- `cargo build --release` / `cargo build` — full builds.
- `cargo check` — even check is heavy when N run at once; reason instead.
- `training-rs/target/.../canastra-bench.exe` or `gate.exe` directly.

The driver rebuilds the (CPU) correctness gate and runs the GPU benchmark
**serially**, one candidate at a time, during validation. Your only job is a
complete, minimal implementation.

## Step 4 — write the response-format output
Write EXACTLY this JSON to `bench/.impl-result.json` (the response_format
contract the driver parses — `--format json` on the session is only an audit
stream; this file is the authoritative output):
```json
{
  "idea_index": <int from bench/.idea.txt>,
  "status": "implemented",
  "hypothesis": "<one sentence>",
  "target_symbol": "<symbol>",
  "predicted_speedup": <number from .idea.txt>,
  "lines_changed": <int>,
  "compiles": null,
  "notes": "<short: what you changed, or why dropped>"
}
```
- `status` is `"implemented"` or `"dropped"`.
- `lines_changed`: `git diff --stat training-rs/src | tail -1` (count the
  changed lines); `0` if dropped.
- `compiles`: `null` (you did not build — the driver determines this).
- Validate it parses as JSON before stopping
  (`python -c "import json;json.load(open('bench/.impl-result.json'))"`).

## Step 5 — commit and STOP
Commit your change on your worktree branch (the driver created it):
```bash
git add training-rs/src && git commit -m "impl idea #<index>: <hypothesis>"
```
Then STOP. Do NOT run benchmarks. Do NOT append to the ledger. Do NOT merge or
delete your branch. Do NOT start another idea.

## Hard rules
- Edit only `training-rs/src/**`. One minimal change.
- No builds, no benchmarks, no gate runs, no ledger writes.
- Commit the change. Stop.
