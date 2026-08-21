---
description: Optimisation loop - generate N candidate ideas, then STOP.
---
# /optimize-generate-ideas — produce N optimisation ideas, then STOP

You are the **IDEA-GENERATION** step inside a deterministic optimisation loop.
The DRIVER (`scripts/optimize-loop.sh`) owns ALL control flow and EVERY stopping
condition. You produce exactly N ideas and then STOP. You do NOT edit code. You
do NOT run benchmarks. You do NOT run the driver or append to the ledger.

`$1` is the number of ideas to produce (default **8** if empty). Call it N.

## Target
The optimisation target is the **training-rs lockstep self-play loop**, measured
by `canastra-bench` throughput (`games_per_s`, CUDA BF16 on RTX 5060 Ti). The
profiler (`bench/run.sh --profile`) breaks the wall into non-overlapping phases:
`argmax` (per-ply device→host score download + argmax sync — the dominant
hotspot, ~38–48% of wall), `encode`, `apply` (rayon engine stepping on the
lockstep critical path), `h2d` (host→device uploads), and the grouped-matmul
stages `trunk`/`head`/`mask`/`gather`. See `training-rs/docs/profile-baseline.md`
and `training-rs/docs/benchmarks.md` for the architecture and the **dead-ends
already tried** (sparse embedding-bag, ES grouped-GEMM split, bmm transpose, f16
dtype) — do NOT re-attempt those.

## Step 1 — read state (no edits)
Run, in this order, and read the output:
- `python bench/ledger.py recent 10` — the last 10 iterations.
- `python bench/ledger.py rejected` — every rejected / correctness_fail /
  harness_fail entry.
- `python bench/ledger.py targets` — the `target_symbol`s already rejected.
  **Never re-propose a symbol in this list.**
- `python bench/ledger.py last-profile` — the most recent `profile_top`. Pick
  hotspots you have a concrete, specific plan for.

## Step 2 — produce exactly N DISTINCT ideas
- Each idea targets a **different `target_symbol`** — spread across hotspots,
  no two ideas share a symbol. Diversifying the portfolio is the whole point of
  the parallel validation step.
- Each idea is a **concrete, specific, implementable change** — not "optimize X".
  Name the function/span and the transformation (e.g. "replace the per-ply
  `to_vec1::<f32>()` full score download in `policy.rs` with a GPU-side argmax +
  single `[rows]` index download").
- A numeric `predicted_speedup` is **mandatory** on every idea. Reason from the
  profiler percentages (e.g. "argmax is 48% of wall; cutting the D2H volume by
  ~width× → predicted_speedup 1.20"). The loop tracks predicted-vs-actual
  correlation and stops if it decays — "unknown"/1.0 placeholders are rejected.
- Do NOT re-propose any `target_symbol` from the rejected list.

## Step 3 — write the response-format output
Write EXACTLY this JSON to `bench/.ideas.json` (the response_format contract the
driver parses — `--format json` on the session is only an audit stream; this
file is the authoritative output):
```json
{
  "ideas": [
    {"index":0,"hypothesis":"<one sentence>","target_symbol":"<symbol>",
     "predicted_speedup":1.20,"rationale":"<why, tied to the profile pct>"},
    {"index":1,"hypothesis":"...","target_symbol":"...","predicted_speedup":1.05,"rationale":"..."}
  ]
}
```
- `index` runs `0..N-1`.
- Validate it parses as JSON and has exactly N entries before stopping (run
  `python -c "import json;d=json.load(open('bench/.ideas.json'));print(len(d['ideas']))"`).

## Step 4 — STOP
Report the N hypotheses + target_symbols + predicted_speedups and STOP. Do not
edit code. Do not run benchmarks. Do not append to the ledger.

## Hard rules
- No code edits. No benchmark runs. No ledger writes. No cargo builds.
- N distinct `target_symbol`s; none already in the rejected list.
- Numeric `predicted_speedup` on every idea.
