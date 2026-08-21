#!/usr/bin/env bash
# scripts/optimize-loop.sh — the optimisation driver.
#
# ALL control flow and EVERY stopping condition lives in this file, never in a
# prompt. Each generation runs TWO agentic steps, then a serial validation:
#   A. idea generation  — one session produces PARALLELISM ideas (bench/.ideas.json)
#   B. implementation   — PARALLELISM parallel sessions (capped by MAX_CONCURRENT),
#                          each in its own worktree, implement ONE idea (bench/.impl-result.json)
#   C. validate + merge  — the harness cherry-picks each 'implemented' candidate onto
#                          the current best, rebuilds the gate + CUDA bench binary, and
#                          benchmarks it SERIALLY (one GPU), merging accepted changes 1-by-1.
# Agents never build or benchmark — only the harness touches the GPU. The model only
# generates ideas + implementations; this script validates, merges, and decides when to
# stop. The harness works with zero agent involvement: --dry-run evaluates every stop
# condition against the ledger without invoking the model, and --no-model runs the loop
# body skipping the agentic steps (useful for testing the plumbing).
#
# Run (Git Bash):  ./scripts/optimize-loop.sh
# Dry run:          ./scripts/optimize-loop.sh --dry-run
#
# ─── Named constants (every threshold lives here) ──────────────────────────
TARGET_VALUE=""            # absolute games/s goal; "" = unset; stop on reach
PATIENCE="${PATIENCE:-3}"                 # consecutive non-significant iterations
MIN_IMPROVEMENT=0.5        # significance = gain >= MIN_IMPROVEMENT × baseline CI width
AMDAHL_FLOOR="${AMDAHL_FLOOR:-1.15}"          # stop (with patience) if zeroing top hotspot can't reach this
COMPLEXITY_RATIO="${COMPLEXITY_RATIO:-4}"         # stop if lines/% exceeds this × median of first 5 accepted
PRED_CORR_MIN="${PRED_CORR_MIN:-0.3}"          # stop if predicted-vs-actual corr over last 5 decays below this
MAX_ITERS="${MAX_ITERS:-50}"               # hard cap on iterations
MAX_WALL_SECONDS="${MAX_WALL_SECONDS:-3600}"      # hard wall cap
MAX_COST_USD="${MAX_COST_USD:-5.0}"          # hard cost cap (placeholder; see docs)
CONTROL_SAMPLES="${CONTROL_SAMPLES:-10}"        # per-iteration control run sample count (multi-sample!)
CONTROL_WARMUP="${CONTROL_WARMUP:-3}"          # control warmup (cold-GPU guard)
EXPERIMENT_SAMPLES="${EXPERIMENT_SAMPLES:-30}"     # sample count for the experiment run.sh
# The control run is the WITHIN-SESSION reference, not baseline.json. Control
# and experiment run back-to-back so they share the same GPU thermal/clock
# state. Accept = experiment.ci_low > control.ci_high. baseline.json is only
# the initial reference + the final total-gain anchor, never a per-iteration
# gate — between-session GPU clock shifts (measured ~2%) would otherwise
# false-abort every iteration.
PATHOLOGICAL_CV_PCT="${PATHOLOGICAL_CV_PCT:-5.0}"   # abort iteration if the control's own CV exceeds this
                          # (a wildly unstable control means the machine itself
                          # is thrashing, not just shifted)


# Model selection. Override on the CLI: --model <provider/model> [--variant <effort>].
# Default: the Kilo config default (unset = kilo picks one). The model is VALIDATED
# at startup: `kilo models` must list it, then `kilo run` must respond to a trivial
# probe (Reply with the single word OK) before the loop starts -- catches auth,
# billing, and connectivity failures before any GPU time is spent.
KILO_MODEL="${KILO_MODEL:-}"
KILO_VARIANT="${KILO_VARIANT:-}"
MODEL_PROBE_TIMEOUT=60   # seconds; a model that can't answer in 60s is unusable for
                          # a tight hypothesis-edit-test loop anyway

# ─── Parallel agentic loop config ───────────────────────────────────────────
# Each generation runs TWO agentic steps, both via the Kilo CLI headless
# (`kilo run --command <name> --auto --format json`):
#   A. idea generation  — ONE session produces PARALLELISM distinct ideas,
#      writing bench/.ideas.json (response_format contract).
#   B. implementation   — PARALLELISM sessions run concurrently (capped by
#      MAX_CONCURRENT), each in its own git worktree, implementing ONE idea and
#      writing bench/.impl-result.json (response_format contract). Sessions are
#      FORBIDDEN from running builds/benchmarks (one GPU; N parallel runs would
#      thrash). The harness rebuilds the gate + CUDA bench binary and benchmarks
#      each candidate SERIALLY during validation.
# Override IDEA_CMD / IMPL_CMD to use a different headless runner.
IDEA_CMD="${IDEA_CMD:-kilo run --command optimize-generate-ideas --auto --format json}"
IMPL_CMD="${IMPL_CMD:-kilo run --command optimize-implement --auto --format json}"
PARALLELISM="${PARALLELISM:-8}"        # ideas produced == impl sessions spawned
MAX_CONCURRENT="${MAX_CONCURRENT:-$PARALLELISM}"  # cap concurrent impl sessions
# Serial validation (harness-owned, one GPU): rebuild the (CPU) gate and the
# (CUDA) bench binary per candidate so the experiment times the candidate's
# code, not a stale binary. Set REBUILD_BENCH=0 only if you rebuild the bench
# binary per candidate externally.
REBUILD_BENCH="${REBUILD_BENCH:-1}"
VCVARS="${VCVARS:-C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat}"
CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP:-120}"
EXPERIMENT_PROFILE="${EXPERIMENT_PROFILE:-0}"  # 1 = add --profile to the per-candidate experiment run
# Legacy single-iteration path (not used by the parallel loop; kept for the
# manual /optimize-iteration flow). Ignored unless PARALLELISM=1.
OPTIMIZER_CMD="${OPTIMIZER_CMD:-kilo run --command optimize-iteration --auto}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
BENCH="$REPO/bench"
TR="$REPO/training-rs"
PY="${PYTHON:-python}"
LEDGER="$BENCH/LEDGER.jsonl"
BASELINE="$BENCH/baseline.json"
HOLDOUT_BASELINE="$BENCH/baseline-holdout.json"
LIB="$BENCH/_lib.py"
LED="$BENCH/ledger.py"   # ledger queries live in ledger.py, not _lib.py
# Branch naming: impl sessions -> opt/impl-<gen>-<i>; validation -> opt/cand-<gen>-<pid>-<i>.

DRY_RUN=0
NO_MODEL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)  DRY_RUN=1; shift;;
    --no-model) NO_MODEL=1; shift;;
    --model)    KILO_MODEL="$2"; shift 2;;
    --variant)  KILO_VARIANT="$2"; shift 2;;
    -h|--help)
      sed -n '2,30p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# ─── model validation (before any work: catch auth/billing/connectivity) ────
MODEL_FLAGS=()
if [ -n "$KILO_MODEL" ]; then
  MODEL_FLAGS+=(--model "$KILO_MODEL")
  if [ -n "$KILO_VARIANT" ]; then MODEL_FLAGS+=(--variant "$KILO_VARIANT"); fi
fi
if [ "$DRY_RUN" -eq 0 ] && [ "$NO_MODEL" -eq 0 ]; then
  echo "validating model..." >&2
  if [ -n "$KILO_MODEL" ]; then
    if ! kilo models 2>/dev/null | grep -qxF "$KILO_MODEL"; then
      echo "model '$KILO_MODEL' not found in kilo models. First 20 available:" >&2
      kilo models 2>/dev/null | head -20 >&2
      exit 2
    fi
  fi
  echo "probing model ${KILO_MODEL:-<default>}..." >&2
  probe_out="$(kilo run "Reply with the single word OK and nothing else."     ${MODEL_FLAGS[*]} --dir "$REPO" 2>&1)" || true
  if ! echo "$probe_out" | grep -qi "OK"; then
    echo "model probe FAILED -- no OK in response. First 500 chars:" >&2
    echo "$probe_out" | head -c 500 >&2
    echo "" >&2
    echo "the model may be down, misconfigured, or out of credits. Aborting." >&2
    exit 2
  fi
  echo "model probe OK" >&2
fi

# ─── helpers (python as a calculator; control flow stays in bash) ──────────
py() { "$PY" -c "$1"; }
# boolean python expr -> lowercase "true"/"false" (bash compares are
# case-sensitive; Python prints "True"/"False", which would silently never
# match). Use bq for every boolean predicate.
bq() { "$PY" -c "print('true' if ($1) else 'false')"; }

# read a field from a JSON file: jfield <file> <key> / jnum <file> <key>.
# Delegates to _lib.py getfield: fails soft ('' on missing/empty/unparseable —
# never a traceback) and tolerates UTF-8/BOM/UTF-16 encodings. A parallel impl
# session whose shell is PowerShell writes '>' redirections as UTF-16LE; plain
# json.load(open(...)) chokes on that at char 0, which would mislabel a real
# implementation as a bogus correctness_fail.
jfield() { "$PY" "$LIB" getfield "$1" "$2" 2>/dev/null; }
jnum() { jfield "$1" "$2"; }

git_ok() { git -C "$REPO" "$@" >/dev/null 2>&1; }

# ─── build/benchmark helpers (validation phase, SERIAL — one GPU) ───────────
# Rebuild the CPU correctness gate for the current working tree's source.
# Returns non-zero on compile failure (caller records correctness_fail, no GPU).
rebuild_gate() {
  ( cd "$TR" && cargo build --release --target-dir target/gate --bin gate ) \
    >"$BENCH/.gatebuild.stderr" 2>&1
}

# Rebuild the CUDA bench binary for the current working tree's source so the
# experiment times the candidate's code, not a stale binary. Serial by design:
# only the harness calls this (sessions are forbidden from building). Override
# VCVARS / CUDA_COMPUTE_CAP / REBUILD_BENCH for your toolchain.
rebuild_bench() {
  [ "$REBUILD_BENCH" = "1" ] || return 0
  local inner="\"${VCVARS}\" >nul 2>&1"
  inner="${inner} && set NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor"
  inner="${inner} && set CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP}"
  inner="${inner} && cargo build --release --features cuda --bin canastra-bench"
  ( cd "$TR" && cmd /c "$inner" ) >"$BENCH/.benchbuild.stderr" 2>&1
}

# Run a control (within-session reference) benchmark against the current best
# tree. $1 = output json path. Sets ctrl_val/ctrl_low/ctrl_high/ctrl_cv_proxy
# globally. Returns non-zero if the run failed or the control's CV is
# pathological (machine thrashing) — caller skips the candidate.
run_control() {
  local out="$1"
  if ! "$BENCH/run.sh" --samples "$CONTROL_SAMPLES" --warmup "$CONTROL_WARMUP" >"$out" 2>"$BENCH/.control.stderr"; then
    echo "  control run failed:" >&2; tail -20 "$BENCH/.control.stderr" >&2 2>/dev/null
    return 1
  fi
  ctrl_val="$(jnum "$out" value)"; ctrl_low="$(jnum "$out" ci_low)"; ctrl_high="$(jnum "$out" ci_high)"
  ctrl_ci_width="$(py "print(${ctrl_high:-0} - ${ctrl_low:-0})")"
  ctrl_cv_proxy="$(py "print(${ctrl_ci_width} / (2.0 * ${ctrl_val:-1}) * 100 if ${ctrl_val} else 0)")"
  if bq "${ctrl_cv_proxy} > ${PATHOLOGICAL_CV_PCT}" | grep -q true; then
    echo "  ABORT candidate: control CV proxy ${ctrl_cv_proxy}% > ${PATHOLOGICAL_CV_PCT}% — machine thrashing" >&2
    return 1
  fi
  echo "  control=$ctrl_val [${ctrl_low},${ctrl_high}] (cv~${ctrl_cv_proxy}%)" >&2
  return 0
}

# Merge an experiment result into baseline.json, preserving the existing
# profile_top (the per-candidate experiment run omits --profile by default, so
# baseline.json would otherwise lose its hotspot map on every accept).
# $1 = experiment json, $2 = baseline json.
merge_baseline() {
  "$PY" - "$1" "$2" <<'PYEOF'
import json, sys
exp, bl = sys.argv[1], sys.argv[2]
e = json.load(open(exp, encoding="utf-8"))
try:
    b = json.load(open(bl, encoding="utf-8"))
except Exception:
    b = {}
if "profile_top" not in e and "profile_top" in b:
    e["profile_top"] = b["profile_top"]
json.dump(e, open(bl, "w", encoding="utf-8"), indent=2)
PYEOF
}

# ─── stop-condition evaluators (each prints "true"/"false") ────────────────
stop_target_reached() {
  [ -z "$TARGET_VALUE" ] && { echo false; return; }
  cur="$(jnum "$BASELINE" value)"
  [ -n "$cur" ] && bq "${cur} >= ${TARGET_VALUE}" || echo false
}

# patience: consecutive non-significant iterations at the ledger tail.
# "non-significant" = verdict != accept, OR (accept but gain < MIN_IMPROVEMENT×ci_width)
stop_patience_exhausted() {
  "$PY" "$LED" patience >/dev/null 2>&1
  p="$("$PY" "$LED" patience 2>/dev/null | "$PY" -c 'import json,sys
raw=sys.stdin.read().strip()
print(json.loads(raw)["patience"] if raw else 0)' 2>/dev/null)"
  [ "${p:-0}" -ge "$PATIENCE" ] && echo true || echo false
}

# amdahl ceiling from the most recent profile_top's largest hotspot.
amdahl_ceiling() {
  prof="$("$PY" "$LED" last-profile 2>/dev/null)"
  top="$(printf '%s' "$prof" | "$PY" -c 'import json,sys
raw=sys.stdin.read().strip()
d=json.loads(raw) if raw else []
print(d[0]["pct"] if d else 0)' 2>/dev/null)"
  [ -z "$top" ] && top=0
  # max speedup if that hotspot is zeroed = 1/(1 - p/100)
  py "p=$top/100.0; print(round(1.0/(1.0-p),3) if p<1 else 9999)"
}
stop_amdahl_below_floor() {
  prof="$("$PY" "$LED" last-profile 2>/dev/null)"
  # No profile yet => no data; do NOT claim the ceiling is below floor.
  if [ -z "$prof" ] || [ "$prof" = "[]" ]; then echo false; return; fi
  ceil="$(amdahl_ceiling)"
  bq "${ceil} < ${AMDAHL_FLOOR}"
}

# complexity: lines-changed-per-percent of the latest accepted entry vs median
# of the first five accepted. If > COMPLEXITY_RATIO × median → stop.
stop_complexity_exceeded() {
  "$PY" - "$COMPLEXITY_RATIO" <<'PYEOF' "$BENCH"
import json, os, sys
ratio = float(sys.argv[1])
led = os.path.join(sys.argv[2], "LEDGER.jsonl")
acc = []
if os.path.exists(led):
    for line in open(led, encoding="utf-8"):
        line=line.strip()
        if not line: continue
        try: e=json.loads(line)
        except: continue
        if e.get("verdict")=="accept":
            sp=e.get("actual_speedup",1.0)
            gain=(sp-1.0)*100.0
            if gain<=0: continue
            acc.append(e.get("lines_changed",0)/gain)
if len(acc)<5:
    print("false"); sys.exit()
import statistics
med=statistics.median(acc[:5])
if med<=0: print("false"); sys.exit()
latest=acc[-1]
print("true" if latest > ratio*med else "false")
PYEOF
}

# prediction decay: Pearson r over last n accepted < PRED_CORR_MIN.
# Returns false when there is too little data to define a correlation (<2
# accepted points) — a NaN correlation is "no signal yet", not "decayed".
stop_prediction_decayed() {
  raw="$("$PY" "$LED" predcorr 5 2>/dev/null)"
  r="$(printf '%s' "$raw" | "$PY" -c 'import json,sys,math
raw=sys.stdin.read().strip()
if not raw: print(""); sys.exit()
d=json.loads(raw)
r=d["prediction_correlation"]
print("" if (r is None or math.isnan(r)) else r)' 2>/dev/null)"
  [ -z "$r" ] && { echo false; return; }
  bq "${r} < ${PRED_CORR_MIN}"
}

# hard budgets. NB: pass paths as standalone argv, never interpolated into the
# -c script — MSYS path conversion only rewrites standalone arguments, so an
# embedded /c/... path makes os.path.exists() silently false on Windows.
iter_count() { "$PY" -c "import os,sys
n=0
if os.path.exists(sys.argv[1]):
    for l in open(sys.argv[1],encoding='utf-8'):
        if l.strip(): n+=1
print(n)" "$LEDGER" 2>/dev/null; }

stop_max_iters() { n="$(iter_count)"; [ "${n:-0}" -ge "$MAX_ITERS" ] && echo true || echo false; }
loop_start=$(date +%s)
stop_max_wall() { now=$(date +%s); elapsed=$((now-loop_start)); [ "$elapsed" -ge "$MAX_WALL_SECONDS" ] && echo true || echo false; }
stop_max_cost() { echo false; }  # placeholder: no cost meter wired (see docs)

# ─── the master stop check ──────────────────────────────────────────────────
evaluate_stop() {
  local reason="" hit=false
  if [ "$(stop_target_reached)" = "true" ]; then reason="target_reached (baseline value >= TARGET_VALUE=${TARGET_VALUE})"; hit=true; fi
  if [ "$hit" = "false" ] && [ "$(stop_patience_exhausted)" = "true" ] && [ "$(stop_amdahl_below_floor)" = "true" ]; then
    reason="patience_exhausted (${PATIENCE} non-significant) AND amdahl_ceiling=$(amdahl_ceiling) < AMDAHL_FLOOR=${AMDAHL_FLOOR}"; hit=true
  fi
  if [ "$hit" = "false" ] && [ "$(stop_complexity_exceeded)" = "true" ]; then reason="complexity_exceeded (lines/% > ${COMPLEXITY_RATIO}× first-5 median)"; hit=true; fi
  if [ "$hit" = "false" ] && [ "$(stop_prediction_decayed)" = "true" ]; then reason="prediction_decayed (predcorr(5) < ${PRED_CORR_MIN})"; hit=true; fi
  if [ "$hit" = "false" ] && [ "$(stop_max_iters)" = "true" ]; then reason="max_iters (${MAX_ITERS})"; hit=true; fi
  if [ "$hit" = "false" ] && [ "$(stop_max_wall)" = "true" ]; then reason="max_wall_seconds (${MAX_WALL_SECONDS})"; hit=true; fi
  if [ "$hit" = "false" ] && [ "$(stop_max_cost)" = "true" ]; then reason="max_cost_usd (${MAX_COST_USD})"; hit=true; fi
  echo "$hit|$reason"
}

# ─── DRY RUN: print the plan + evaluate every condition, no model, no bench ──
if [ "$DRY_RUN" -eq 1 ]; then
  echo "=== optimize-loop DRY RUN ===" >&2
  echo "constants: TARGET_VALUE=${TARGET_VALUE:-<unset>} PATIENCE=$PATIENCE MIN_IMPROVEMENT=$MIN_IMPROVEMENT AMDAHL_FLOOR=$AMDAHL_FLOOR COMPLEXITY_RATIO=$COMPLEXITY_RATIO PRED_CORR_MIN=$PRED_CORR_MIN MAX_ITERS=$MAX_ITERS MAX_WALL_SECONDS=$MAX_WALL_SECONDS MAX_COST_USD=$MAX_COST_USD PARALLELISM=$PARALLELISM MAX_CONCURRENT=$MAX_CONCURRENT REBUILD_BENCH=$REBUILD_BENCH" >&2
  echo "" >&2
  echo "generation plan (per generation):" >&2
  echo "  A. idea generation: 1 session produces $PARALLELISM ideas -> bench/.ideas.json" >&2
  echo "  B. implementation: $PARALLELISM parallel sessions (cap $MAX_CONCURRENT), each in its own" >&2
  echo "     git worktree, implement ONE idea -> bench/.impl-result.json. Sessions run NO" >&2
  echo "     builds/benchmarks (one GPU; N parallel runs would thrash)." >&2
  echo "  C. validate+merge (SERIAL, harness-owned): for each 'implemented' candidate," >&2
  echo "     cherry-pick onto current best, rebuild gate (CPU) + CUDA bench binary," >&2
  echo "     run ./bench/run.sh --samples $EXPERIMENT_SAMPLES (control reuse: re-run only" >&2
  echo "     when the best tree changed). accept iff correctness=pass AND" >&2
  echo "     experiment.ci_low > control.ci_high (non-overlapping CIs)." >&2
  echo "     accept => merge into best + update baseline.json; reject/correctness_fail" >&2
  echo "     => record + delete branch. Candidates merged 1-by-1 (baseline shifts)." >&2
  echo "" >&2
  echo "ledger state:" >&2
  n="$(iter_count)"; echo "  iterations logged: ${n:-0}" >&2
  "$PY" "$LED" recent 5 >&2 2>&1
  echo "" >&2
  pat="$("$PY" "$LED" patience 2>/dev/null | "$PY" -c 'import json,sys;print(json.load(sys.stdin)["patience"])' 2>/dev/null)"
  pcorr="$("$PY" "$LED" predcorr 5 2>/dev/null | "$PY" -c 'import json,sys,math
d=json.load(sys.stdin); r=d["prediction_correlation"]; print("n/a" if (r is None or math.isnan(r)) else f"{r:.3f}")' 2>/dev/null)"
  echo "stop-condition evaluation (against current ledger):" >&2
  echo "  target_reached       = $(stop_target_reached)   [TARGET_VALUE=${TARGET_VALUE:-<unset>}]" >&2
  echo "  patience_exhausted   = $(stop_patience_exhausted)   [count=${pat:-0}, PATIENCE=$PATIENCE]" >&2
  echo "  amdahl_ceiling       = $(amdahl_ceiling)   [top hotspot zeroed; floor=$AMDAHL_FLOOR]" >&2
  echo "  amdahl_below_floor   = $(stop_amdahl_below_floor)" >&2
  echo "  complexity_exceeded  = $(stop_complexity_exceeded)   [ratio=$COMPLEXITY_RATIO× first-5 median]" >&2
  echo "  prediction_decayed   = $(stop_prediction_decayed)   [predcorr(5)=${pcorr:-n/a}, min=$PRED_CORR_MIN]" >&2
  echo "  max_iters            = $(stop_max_iters)   [count=${n:-0}/$MAX_ITERS]" >&2
  echo "  max_wall_seconds     = $(stop_max_wall)   [cap=$MAX_WALL_SECONDS]" >&2
  echo "  max_cost_usd         = $(stop_max_cost)   [cap=$MAX_COST_USD, not metered]" >&2
  echo "" >&2
  hit_reason="$(evaluate_stop)"
  hit="${hit_reason%%|*}"; reason="${hit_reason#*|}"
  if [ "$hit" = "true" ]; then
    echo "STOP: would stop now — $reason" >&2
  else
    echo "CONTINUE: no stop condition fires; loop would proceed to the next model iteration." >&2
  fi
  # Emit a machine-readable JSON summary to stdout.
  "$PY" - "$hit" "$reason" <<'PYEOF'
import json,sys
hit=sys.argv[1]; reason=sys.argv[2]
print(json.dumps({"dry_run":True,"stop":hit=="true","reason":reason}, indent=2))
PYEOF
  exit 0
fi

# ─── live loop ──────────────────────────────────────────────────────────────
[ -f "$BASELINE" ] || { echo "missing $BASELINE — run ./bench/run.sh first" >&2; exit 1; }
# Snapshot the original baseline so the final report can compute total gain vs
# the starting point (baseline.json is overwritten on each accept).
[ -f "$BENCH/baseline.orig.json" ] || cp "$BASELINE" "$BENCH/baseline.orig.json"

# A crashed earlier run can leave HEAD parked on a stale opt/* branch with its
# edit still dirty in the working tree; generations must start from the trunk or
# the control run silently measures leftover experiments.
case "$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null)" in
  opt/impl-*|opt/cand-*|opt/iter-*)
    echo "HEAD is on a stale $(git -C "$REPO" rev-parse --abbrev-ref HEAD) — checking out main first" >&2
    git_ok checkout main || { echo "cannot return to main — fix git state and re-run" >&2; exit 1; } ;;
esac
if [ -n "$(git -C "$REPO" status --porcelain=v1 -- training-rs/src 2>/dev/null)" ]; then
  echo "WARNING: training-rs/src has uncommitted changes — control and experiment will measure this dirty tree" >&2
fi

# Seed the ledger with an iteration-0 baseline entry so the optimizer's first
# call to `recent 10` and `last-profile` has data. Without this, the first
# iteration deadlocks: the optimizer sees an empty ledger, no profile_top, and
# (correctly) stops without proposing anything.
if [ ! -s "$LEDGER" ]; then
  bl_prof="$("$PY" -c 'import json,sys;print(json.dumps(json.load(open(sys.argv[1])).get("profile_top",[])))' "$BASELINE" 2>/dev/null)"
  bl_val="$(jnum "$BASELINE" value)"
  bl_commit="$(jfield "$BASELINE" commit)"
  "$PY" "$LED" append "{\"parent_commit\":\"\",\"commit\":\"$bl_commit\",\"hypothesis\":\"baseline\",\"target_symbol\":\"\",\"predicted_speedup\":1.0,\"actual_speedup\":1.0,\"verdict\":\"accept\",\"lines_changed\":0,\"notes\":\"seeded baseline: ${bl_val} games/s\",\"profile_top\":${bl_prof:-[]}}" >/dev/null 2>&1
  echo "seeded ledger with baseline entry (iter 0)" >&2
fi

loop_start=$(date +%s)
iter=0
while true; do
  iter=$((iter+1))
  echo "=== generation $iter ===" >&2

  # 0. master stop check BEFORE doing any work
  hit_reason="$(evaluate_stop)"; hit="${hit_reason%%|*}"; reason="${hit_reason#*|}"
  if [ "$hit" = "true" ]; then echo "STOP at gen $iter: $reason" >&2; break; fi

  # All implementation worktrees branch from the current best HEAD.
  base_commit="$(git -C "$REPO" rev-parse HEAD)"
  best_branch="$(git -C "$REPO" rev-parse --abbrev-ref HEAD)"

  # ── STEP A: idea generation (ONE session) ────────────────────────────────
  # Produces PARALLELISM distinct ideas into bench/.ideas.json (response_format
  # contract). --format json is also captured as an audit stream.
  rm -f "$BENCH/.ideas.json" "$BENCH"/.idea-*.txt
  if [ "$NO_MODEL" -eq 1 ]; then
    echo "gen $iter A: --no-model, skipping idea generation" >&2
    echo '{"ideas":[]}' >"$BENCH/.ideas.json"
  else
    echo "gen $iter A: generating $PARALLELISM ideas..." >&2
    $IDEA_CMD "$PARALLELISM" ${MODEL_FLAGS[*]} --dir "$REPO" \
      >"$BENCH/.session-gen-$iter.jsonl" 2>"$BENCH/.session-gen-$iter.stderr" \
      || echo "idea-gen command failed (rc=$?)" >&2
  fi

  # Split .ideas.json into per-idea text files for the implementation worktrees.
  split_out="$("$PY" - "$BENCH" "$PARALLELISM" 2>"$BENCH/.ideas.split.stderr" <<'PYEOF'
import json, os, sys
bench, n = sys.argv[1], int(sys.argv[2])
p = os.path.join(bench, ".ideas.json")
try:
    d = json.load(open(p, encoding="utf-8"))
except Exception as e:
    print("ERR " + str(e)); sys.exit(0)
ideas = d.get("ideas", [])
for it in ideas:
    i = it.get("index", 0)
    txt = (f"idea_index: {i}\n"
           f"target_symbol: {it.get('target_symbol','')}\n"
           f"predicted_speedup: {it.get('predicted_speedup',1.0)}\n"
           f"hypothesis: {it.get('hypothesis','')}\n"
           f"rationale: {it.get('rationale','')}\n")
    open(os.path.join(bench, f".idea-{i}.txt"), "w", encoding="utf-8").write(txt)
print("OK " + str(len(ideas)))
PYEOF
)"
  idea_count="$(printf '%s' "$split_out" | sed -n 's/^OK //p')"
  if [ -z "$idea_count" ] || [ "$idea_count" -lt 1 ]; then
    echo "gen $iter A: no ideas produced ($(printf '%s' "$split_out" | head -1)); recording harness_fail" >&2
    "$PY" "$LED" append "{\"parent_commit\":\"$(git -C "$REPO" rev-parse --short "$base_commit")\",\"commit\":\"$(git -C "$REPO" rev-parse --short HEAD)\",\"hypothesis\":\"(no ideas)\",\"target_symbol\":\"\",\"predicted_speedup\":1.0,\"actual_speedup\":1.0,\"verdict\":\"harness_fail\",\"lines_changed\":0,\"notes\":\"idea generation produced no ideas\"}" >/dev/null
    continue
  fi
  n_impl="$idea_count"; [ "$n_impl" -gt "$PARALLELISM" ] && n_impl="$PARALLELISM"
  echo "gen $iter A: $n_impl ideas to implement" >&2

  # ── STEP B: parallel implementation (N sessions, each its own worktree) ───
  # Each session implements ONE idea (read from bench/.idea.txt in its worktree)
  # and writes bench/.impl-result.json (response_format contract). Sessions run
  # NO builds/benchmarks — the harness owns all of that, serially, in STEP C.
  WT_ROOT="$(mktemp -d -t kilo-opt-XXXXXX)"
  echo "$WT_ROOT" >"$BENCH/.worktrees-$iter.txt"
  echo "gen $iter B: spawning $n_impl impl sessions (max $MAX_CONCURRENT concurrent) in $WT_ROOT" >&2
  impl_wts=()
  i=0
  while [ "$i" -lt "$n_impl" ]; do
    # throttle concurrency to MAX_CONCURRENT
    while [ "$(jobs -rp 2>/dev/null | wc -l)" -ge "$MAX_CONCURRENT" ]; do sleep 2; done
    wt="$WT_ROOT/impl-$iter-$i"
    impl_branch="opt/impl-$iter-$i"
    if ! git -C "$REPO" worktree add --force -b "$impl_branch" "$wt" "$base_commit" >/dev/null 2>"$BENCH/.wt-$iter-$i.stderr"; then
      echo "  worktree add failed for idea $i: $(cat "$BENCH/.wt-$iter-$i.stderr" 2>/dev/null)" >&2
      i=$((i+1)); continue
    fi
    impl_wts+=("$wt")
    # seed the idea + the (uncommitted) implement command into the worktree so
    # `kilo run --command optimize-implement --dir <wt>` resolves it there.
    cp "$BENCH/.idea-$i.txt" "$wt/bench/.idea.txt" 2>/dev/null || true
    mkdir -p "$wt/.kilo/commands"
    cp "$REPO/.kilo/commands/optimize-implement.md" "$wt/.kilo/commands/" 2>/dev/null || true
    # seed current ledger + baseline so the session sees the live rejection
    # history and current hotspots (the worktree checkout has the committed —
    # possibly stale — copies). These are read-only context; the session stages
    # only training-rs/src, so they are never cherry-picked into the best branch.
    cp "$LEDGER" "$wt/bench/LEDGER.jsonl" 2>/dev/null || true
    cp "$BASELINE" "$wt/bench/baseline.json" 2>/dev/null || true
    if [ "$NO_MODEL" -eq 1 ]; then
      echo "  --no-model: skipping impl session $i" >&2
      echo "{\"idea_index\":$i,\"status\":\"dropped\",\"hypothesis\":\"(no-model)\",\"target_symbol\":\"\",\"predicted_speedup\":1.0,\"lines_changed\":0,\"compiles\":null,\"notes\":\"no-model\"}" >"$wt/bench/.impl-result.json"
    else
      ( $IMPL_CMD ${MODEL_FLAGS[*]} --dir "$wt" \
          >"$BENCH/.session-impl-$iter-$i.jsonl" 2>"$BENCH/.session-impl-$iter-$i.stderr" ) &
    fi
    i=$((i+1))
  done
  wait  # all implementation sessions
  echo "gen $iter B: all impl sessions finished" >&2

  # Safety net: a session that implemented its idea in the working tree but
  # forgot to `git commit` would be dropped by the n_commits check below and
  # then lost when the worktree is removed. Commit any leftover training-rs/src
  # edits on each impl branch harness-side (the harness owns the commit, never
  # the model — same philosophy as the single-iteration path). Sessions that
  # already committed are a no-op.
  i=0
  while [ "$i" -lt "$n_impl" ]; do
    wt_i="$WT_ROOT/impl-$iter-$i"
    if [ -d "$wt_i" ] && [ -n "$(git -C "$wt_i" status --porcelain=v1 -- training-rs/src 2>/dev/null)" ]; then
      git -C "$wt_i" add -- training-rs/src >/dev/null 2>&1 \
        && git -C "$wt_i" commit -m "opt impl $iter-$i: harness-committed (session left it uncommitted)" >/dev/null 2>&1 \
        || echo "  warning: could not harness-commit leftover edits in $wt_i" >&2
    fi
    i=$((i+1))
  done

  # ── collect implemented candidates (sorted by predicted_speedup desc) ─────
  cand_file="$BENCH/.candidates.$$.tsv"
  "$PY" - "$WT_ROOT" "$iter" "$n_impl" >"$cand_file" 2>"$BENCH/.candidates.stderr" <<'PYEOF'
import json, os, sys
wtroot, iter, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = []
for i in range(n):
    p = os.path.join(wtroot, f"impl-{iter}-{i}", "bench", ".impl-result.json")
    if not os.path.exists(p):
        continue
    try:
        r = json.load(open(p, encoding="utf-8"))
    except Exception:
        continue
    if r.get("status") == "implemented":
        r["_branch"] = f"opt/impl-{iter}-{r.get('idea_index', i)}"
        rows.append(r)
rows.sort(key=lambda r: float(r.get("predicted_speedup", 1.0)), reverse=True)
def esc(s): return str(s).replace("\t", " ").replace("\n", " ")
for r in rows:
    print("\t".join([esc(r.get("_branch", "")), esc(r.get("hypothesis", "")),
                     esc(r.get("target_symbol", "")), esc(r.get("predicted_speedup", 1.0)),
                     esc(r.get("lines_changed", 0)), esc(r.get("notes", ""))]))
PYEOF
  n_cand="$(grep -c . "$cand_file" 2>/dev/null || echo 0)"
  echo "gen $iter C: $n_cand candidates to validate+merge (serially)" >&2

  # ── STEP C: serial validation + 1-by-1 merge ──────────────────────────────
  # The harness owns the GPU: one candidate at a time. For each 'implemented'
  # candidate: cherry-pick onto current best, rebuild gate (CPU) + CUDA bench
  # binary, run the experiment benchmark, accept iff correctness=pass AND
  # exp.ci_low > control.ci_high. Control is the within-session reference,
  # reused while the best tree is unchanged and re-run only after an accept
  # (the tree moved). Accepted changes merge 1-by-1 — the baseline shifts, so
  # later candidates cherry-pick onto the advanced best.
  ctrl_json=""; ctrl_commit=""
  accept_count=0
  git -C "$REPO" checkout -q "$best_branch" 2>/dev/null || true
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    c_branch="$(printf '%s' "$line" | cut -f1)"
    c_hyp="$(printf '%s' "$line" | cut -f2)"
    c_sym="$(printf '%s' "$line" | cut -f3)"
    c_pred="$(printf '%s' "$line" | cut -f4)"; [ -z "$c_pred" ] && c_pred=1.0
    c_lch="$(printf '%s' "$line" | cut -f5)"; [ -z "$c_lch" ] && c_lch=0
    c_notes="$(printf '%s' "$line" | cut -f6)"
    parent="$(git -C "$REPO" rev-parse --short HEAD)"
    echo "gen $iter C: validating $c_branch ($c_sym, pred=$c_pred)" >&2

    # control: reuse while best unchanged, re-run when it moved (after an accept)
    cur_head="$(git -C "$REPO" rev-parse HEAD)"
    if [ "$cur_head" != "$ctrl_commit" ]; then
      ctrl_json="$BENCH/.control.$$.json"
      if ! run_control "$ctrl_json"; then rm -f "$ctrl_json"; continue; fi
      ctrl_commit="$cur_head"
    fi

    # the candidate must have committed its implementation
    n_commits="$(git -C "$REPO" rev-list --count "$base_commit..$c_branch" 2>/dev/null || echo 0)"
    if [ "${n_commits:-0}" -lt 1 ]; then
      echo "  no commits on $c_branch — dropping (session did not commit)" >&2
      "$PY" "$LED" append "{\"parent_commit\":\"$parent\",\"commit\":\"$parent\",\"hypothesis\":\"$c_hyp\",\"target_symbol\":\"$c_sym\",\"predicted_speedup\":${c_pred},\"actual_speedup\":1.0,\"verdict\":\"harness_fail\",\"lines_changed\":0,\"notes\":\"session reported implemented but committed nothing\"}" >/dev/null
      continue
    fi

    # branch from current best, cherry-pick the candidate's implementation.
    # $$ (PID) in the name makes the candidate branch unique per run, so a
    # crashed prior run's stale branch never blocks checkout -b.
    cand_branch="opt/cand-$iter-$$-$(printf '%s' "$c_branch" | sed 's#opt/impl-##')"
    git -C "$REPO" checkout -q "$best_branch" 2>/dev/null || true
    git -C "$REPO" branch -q -D "$cand_branch" 2>/dev/null || true
    if ! git -C "$REPO" checkout -q -b "$cand_branch" 2>/dev/null; then
      echo "  cand branch create failed" >&2; continue
    fi
    if ! git -C "$REPO" cherry-pick "$base_commit..$c_branch" >/dev/null 2>"$BENCH/.cp.stderr"; then
      git -C "$REPO" cherry-pick --abort 2>/dev/null || true
      echo "  MERGE CONFLICT cherry-picking $c_branch — dropping candidate" >&2
      "$PY" "$LED" append "{\"parent_commit\":\"$parent\",\"commit\":\"$(git -C "$REPO" rev-parse --short HEAD)\",\"hypothesis\":\"$c_hyp\",\"target_symbol\":\"$c_sym\",\"predicted_speedup\":${c_pred},\"actual_speedup\":1.0,\"verdict\":\"harness_fail\",\"lines_changed\":${c_lch},\"notes\":\"merge conflict\"}" >/dev/null
      git -C "$REPO" checkout -q "$best_branch" 2>/dev/null || true
      git -C "$REPO" branch -q -D "$cand_branch" 2>/dev/null || true
      continue
    fi

    # rebuild the (CPU) correctness gate for this candidate's source. A build
    # failure => correctness_fail, no GPU time spent. Then rebuild the CUDA
    # bench binary so the experiment times the candidate's code.
    if ! rebuild_gate; then
      echo "  gate BUILD FAILED — correctness_fail (no benchmark)" >&2
      "$PY" "$LED" append "{\"parent_commit\":\"$parent\",\"commit\":\"$(git -C "$REPO" rev-parse --short HEAD)\",\"hypothesis\":\"$c_hyp\",\"target_symbol\":\"$c_sym\",\"predicted_speedup\":${c_pred},\"actual_speedup\":1.0,\"verdict\":\"correctness_fail\",\"lines_changed\":${c_lch},\"notes\":\"gate build failed\"}" >/dev/null
      git -C "$REPO" checkout -q "$best_branch" 2>/dev/null || true
      git -C "$REPO" branch -q -D "$cand_branch" 2>/dev/null || true
      continue
    fi
    if ! rebuild_bench; then
      echo "  CUDA bench BUILD FAILED — harness_fail (no benchmark)" >&2
      "$PY" "$LED" append "{\"parent_commit\":\"$parent\",\"commit\":\"$(git -C "$REPO" rev-parse --short HEAD)\",\"hypothesis\":\"$c_hyp\",\"target_symbol\":\"$c_sym\",\"predicted_speedup\":${c_pred},\"actual_speedup\":1.0,\"verdict\":\"harness_fail\",\"lines_changed\":${c_lch},\"notes\":\"cuda bench build failed\"}" >/dev/null
      git -C "$REPO" checkout -q "$best_branch" 2>/dev/null || true
      git -C "$REPO" branch -q -D "$cand_branch" 2>/dev/null || true
      continue
    fi

    # experiment benchmark (GPU, serial — the driver owns this). The artifact
    # is trusted only once it actually parses: run.sh may fail (GPU/build issue)
    # and leave no/unparseable JSON; an empty correctness => harness_fail, never
    # a bogus correctness_fail that would mislabel real work.
    exp_json="$BENCH/.experiment.$$.json"
    rm -f "$exp_json"
    exp_flags="--samples $EXPERIMENT_SAMPLES --warmup 2"
    [ "$EXPERIMENT_PROFILE" = "1" ] && exp_flags="$exp_flags --profile"
    "$BENCH/run.sh" $exp_flags >"$exp_json" 2>"$BENCH/.exp.stderr" \
      || echo "  experiment run.sh failed (rc=$?; see $BENCH/.exp.stderr)" >&2
    exp_correct="$(jfield "$exp_json" correctness)"
    exp_low="$(jnum "$exp_json" ci_low)"; exp_high="$(jnum "$exp_json" ci_high)"; exp_val="$(jnum "$exp_json" value)"
    actual="$(py "print(round((${exp_val:-0})/${ctrl_val:-1},4))")"

    # ACCEPT RULE: correctness passes AND experiment.ci_low > control.ci_high
    accept=false
    if [ "$exp_correct" = "pass" ] && [ -n "$exp_low" ] && [ -n "$ctrl_high" ]; then
      if bq "${exp_low} > ${ctrl_high}" | grep -q true; then accept=true; fi
    fi

    if [ "$accept" = "true" ]; then
      git -C "$REPO" checkout -q "$best_branch" 2>/dev/null || true
      git -C "$REPO" merge --no-ff "$cand_branch" -m "opt gen $iter: accept ${ctrl_val} -> ${exp_val} games/s ($c_sym)" >/dev/null 2>&1
      merge_baseline "$exp_json" "$BASELINE"
      "$PY" "$LED" append "{\"parent_commit\":\"$parent\",\"commit\":\"$(git -C "$REPO" rev-parse --short HEAD)\",\"hypothesis\":\"$c_hyp\",\"target_symbol\":\"$c_sym\",\"predicted_speedup\":${c_pred},\"actual_speedup\":${actual},\"verdict\":\"accept\",\"lines_changed\":${c_lch},\"notes\":\"control ${ctrl_val}->exp ${exp_val}; exp_ci [${exp_low},${exp_high}] vs ctrl_ci [${ctrl_low},${ctrl_high}]\"}" >/dev/null
      git -C "$REPO" branch -q -D "$cand_branch" 2>/dev/null || true
      ctrl_commit=""   # best moved -> force a fresh control before the next candidate
      accept_count=$((accept_count+1))
      echo "  ACCEPT: control ${ctrl_val} -> experiment ${exp_val} games/s (speedup=${actual})" >&2
    else
      # reject: empty correctness => harness_fail; !=pass => correctness_fail;
      # else (passed but CI overlap) => reject. Preserve the diff as a patch so a
      # discarded candidate stays inspectable without leaking into the next run.
      if [ -z "$exp_correct" ]; then v="harness_fail"
      elif [ "$exp_correct" != "pass" ]; then v="correctness_fail"
      else v="reject"; fi
      git -C "$REPO" diff "$best_branch" "$cand_branch" -- training-rs/src >"$BENCH/.rejected.gen-$iter-$$-$(printf '%s' "$c_sym" | tr -c 'A-Za-z0-9' '_').patch" 2>/dev/null
      "$PY" "$LED" append "{\"parent_commit\":\"$parent\",\"commit\":\"$(git -C "$REPO" rev-parse --short HEAD)\",\"hypothesis\":\"$c_hyp\",\"target_symbol\":\"$c_sym\",\"predicted_speedup\":${c_pred},\"actual_speedup\":${actual},\"verdict\":\"$v\",\"lines_changed\":${c_lch},\"notes\":\"correctness=${exp_correct:-<none>}; exp_ci [${exp_low:-0},${exp_high:-0}] vs ctrl_ci [${ctrl_low},${ctrl_high}]\"}" >/dev/null
      git -C "$REPO" checkout -q "$best_branch" 2>/dev/null || true
      git -C "$REPO" branch -q -D "$cand_branch" 2>/dev/null || true
      if [ "$v" = "harness_fail" ]; then
        echo "  REJECT (verdict=harness_fail: no parsable experiment JSON after driver rerun; see $BENCH/.exp.stderr)" >&2
      else
        echo "  REJECT (verdict=$v, correctness=$exp_correct; exp=${exp_val:-0} vs ctrl=${ctrl_val})" >&2
      fi
    fi
    rm -f "$exp_json"
  done < "$cand_file"
  echo "gen $iter: $accept_count accepted this generation" >&2

  # ── cleanup: remove impl worktrees + branches + temp files ───────────────
  for wt in "${impl_wts[@]}"; do
    [ -z "$wt" ] && continue
    git -C "$REPO" worktree remove --force "$wt" 2>/dev/null || rm -rf "$wt"
  done
  i=0
  while [ "$i" -lt "$n_impl" ]; do
    git -C "$REPO" branch -q -D "opt/impl-$iter-$i" 2>/dev/null || true
    i=$((i+1))
  done
  rm -f "$cand_file" "$BENCH"/.idea-*.txt "$BENCH"/.session-impl-$iter-*.jsonl "$BENCH"/.session-impl-$iter-*.stderr \
        "$BENCH"/.session-gen-$iter.* "$BENCH"/.control.* "$BENCH"/.experiment.*.json \
        "$BENCH"/.candidates.stderr "$BENCH"/.ideas.split.stderr "$BENCH"/.wt-$iter-*.stderr \
        "$BENCH"/.cp.stderr "$BENCH"/.gatebuild.stderr "$BENCH"/.benchbuild.stderr \
        "$BENCH"/.control.stderr "$BENCH"/.exp.stderr "$BENCH"/.worktrees-$iter.txt 2>/dev/null
  rmdir "$WT_ROOT" 2>/dev/null || true
done

# ─── final report + holdout ─────────────────────────────────────────────────
echo "" >&2
echo "=== FINAL ===" >&2
hit_reason="$(evaluate_stop)"; reason="${hit_reason#*|}"
echo "stop reason: $reason" >&2
"$PY" "$LED" recent "$iter" >&2 2>&1
echo "" >&2
echo "holdout check (different path: one full canastra-train generation):" >&2
if [ ! -f "$HOLDOUT_BASELINE" ]; then
  echo "  no $HOLDOUT_BASELINE — run ./bench/holdout.sh --baseline $HOLDOUT_BASELINE before the loop; skipping proportional check" >&2
else
  "$BENCH/holdout.sh" --samples 3 --warmup 1 >"$BENCH/.holdout.final.json" 2>"$BENCH/.holdout.stderr" || true
  hb="$(jnum "$HOLDOUT_BASELINE" value)"; hf="$(jnum "$BENCH/.holdout.final.json" value)"
  orig_val="$(jnum "$BENCH/baseline.orig.json" value 2>/dev/null)"
  base_val="$(jnum "$BASELINE" value)"
  if [ -n "$orig_val" ] && [ -n "$base_val" ]; then
    main_gain="$(py "print(round((${base_val}-${orig_val})/${orig_val}*100,1))")"
  else
    main_gain=0
  fi
  if [ -n "$hb" ] && [ -n "$hf" ]; then
    hold_gain="$(py "print(round((${hf}-${hb})/${hb}*100,1))")"
    echo "  main metric:  orig=$orig_val final=$base_val games/s (+${main_gain}%)" >&2
    echo "  holdout:       baseline=$hb final=$hf games/s (+${hold_gain}%)" >&2
    # Overfit guard: if the main metric gained materially but the holdout did
    # not gain at least half as much (proportionally), flag overfit.
    if bq "${main_gain} > 2.0 and ${hold_gain} < 0.5*${main_gain}" | grep -q true; then
      echo "  !! HOLDOUT DID NOT IMPROVE PROPORTIONALLY (main +${main_gain}% vs holdout +${hold_gain}%)." >&2
      echo "  !! The loop likely OVERFIT the harness — the wins did not generalise to the full-generation path." >&2
    else
      echo "  holdout moved proportionally — gains appear to generalise." >&2
    fi
  fi
  rm -f "$BENCH/.holdout.final.json"
fi
