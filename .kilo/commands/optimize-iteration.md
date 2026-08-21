---
description: Optimisation loop - run EXACTLY ONE iteration, then stop.
---
# /optimize-iteration - one optimisation iteration, then STOP

You are the hypothesis-generator inside a deterministic benchmark-optimize-evaluate
loop. The DRIVER (scripts/optimize-loop.sh) owns all control flow and every
stopping condition. You do EXACTLY ONE iteration and then stop. Do not loop.
Do not run the driver. Do not edit anything under `bench/`, the golden fixtures,
or the metric.

## Target
The optimisation target is the **training-rs lockstep self-play loop**, measured
by `canastra-bench` throughput (`games_per_s`, CUDA BF16 on RTX 5060 Ti). The
profiler (run.sh `--profile`) breaks the wall into: `argmax` (the D2H
score-download + argmax sync - the dominant hotspot), `encode`, `apply`, `h2d`,
`trunk`, `head`, `mask`, `gather`. See `training-rs/docs/profile-baseline.md`
and `training-rs/docs/benchmarks.md` for the architecture and the dead-ends
already tried (sparse embedding-bag, ES grouped-GEMM split, bmm transpose,
f16 dtype) - do NOT re-attempt those.

## Step 1 - read state (no edits yet)
Run, in this order, and read the output:
- `python bench/ledger.py recent 10` - the last 10 iterations.
- `python bench/ledger.py rejected` - every rejected/correctness_fail/harness_fail
  entry. **Never re-propose a hypothesis whose `target_symbol` already appears
  here.**
- `python bench/ledger.py last-profile` - the most recent `profile_top`. Pick
  the **single largest hotspot you have a concrete, specific plan for**. That
  symbol is your `target_symbol`.

## Step 2 - state the hypothesis BEFORE editing
You MUST record a prediction. "Unknown" / "1.0" placeholders are NOT acceptable
- a numeric `predicted_speedup` is mandatory (the loop tracks
predicted-vs-actual correlation and stops if it decays). Reason from the
profiler: e.g. "argmax is 48% of wall; the per-ply `to_vec1::<f32>()` full
score download can be replaced by an argmax-only download, cutting the download
volume by ~width x -> predicted_speedup 1.20". Write this to
`bench/.hypothesis.json`:
```json
{"hypothesis":"<one sentence>","target_symbol":"<symbol>",
 "predicted_speedup":1.20,"lines_changed":0}
```
(`lines_changed` you fill in after editing.)

## Step 3 - make ONE smallest change that tests exactly that hypothesis
Edit ONLY under `training-rs/src/**`. One change. Do not refactor unrelated
code. Do not touch `bench/**`, the golden fixtures (`bench/golden/**`),
`training-rs/tests/reference/**`, or any timing/metric code. If the gate fails,
that is the result - do not "fix" the golden to make it pass.

## Step 4 - run the benchmark
```bash
./bench/run.sh --samples 30 --warmup 2 --profile > bench/.experiment.json
```
ENCODING WARNING: if your shell tool is PowerShell, do NOT use `>` redirection
for this command - Windows PowerShell writes redirections as UTF-16LE, which
the driver's parser cannot read and which has mislabelled whole iterations as
correctness_fail. Instead capture the output and write it as UTF-8, e.g.:
```
python -c "import subprocess; r = subprocess.run(['C:/Program Files/Git/bin/bash.exe', '-lc', './bench/run.sh --samples 30 --warmup 2 --profile'], capture_output=True, text=True); open('bench/.experiment.json', 'w', encoding='utf-8').write(r.stdout); print(r.stderr[-2000:])"
```
The driver re-verifies the artifact anyway and re-runs the bench itself if it
does not parse, but a readable artifact saves a full benchmark run.

`run.sh` runs the correctness gate FIRST (byte-identical golden on CPU). If the
gate fails, `.experiment.json` will contain `correctness: "fail"` and a non-zero
exit - leave it, the driver records `correctness_fail`. **Do not retry.** A
behaviour-altering speedup MUST fail the gate; that is the whole point.

Fill in `lines_changed` in `bench/.hypothesis.json` (use
`git diff --stat training-rs/src | tail -1` or count the changed lines).

## Step 5 - stop
Do NOT append to the ledger - the driver owns that. Do NOT merge or delete the
branch. Do NOT start another iteration. Report:
- the hypothesis + target_symbol + predicted_speedup,
- `lines_changed`,
- the `correctness`, `value`, `ci_low`, `ci_high` from `.experiment.json`,
and STOP.

## Hard rules
- Never edit `bench/**`, `bench/golden/**`, `training-rs/tests/reference/**`,
  `docs/OPTIMIZATION.md`, or the metric.
- Never adjust the golden fixtures or the gate.
- One edit per invocation. If you cannot make the change small, do not make it.
- If `profile_top` is empty (no `--profile` run yet), STOP and ask the driver
  to re-run with `--profile` - never guess a hotspot.
