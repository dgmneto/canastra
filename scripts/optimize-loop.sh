#!/usr/bin/env bash
# scripts/optimize-loop.sh — the optimisation driver.
#
# ALL control flow and EVERY stopping condition lives in this file, never in a
# prompt. The model only generates a hypothesis + one edit; this script decides
# whether to keep it, and when to stop. The harness works with zero agent
# involvement: --dry-run evaluates every stop condition against the ledger
# without invoking the model, and --no-model runs the loop body skipping the
# model step (useful for testing the plumbing).
#
# Run (Git Bash):  ./scripts/optimize-loop.sh
# Dry run:          ./scripts/optimize-loop.sh --dry-run
#
# ─── Named constants (every threshold lives here) ──────────────────────────
TARGET_VALUE=""            # absolute games/s goal; "" = unset; stop on reach
PATIENCE=3                 # consecutive non-significant iterations
MIN_IMPROVEMENT=0.5        # significance = gain >= MIN_IMPROVEMENT × baseline CI width
AMDAHL_FLOOR=1.15          # stop (with patience) if zeroing top hotspot can't reach this
COMPLEXITY_RATIO=4         # stop if lines/% exceeds this × median of first 5 accepted
PRED_CORR_MIN=0.3          # stop if predicted-vs-actual corr over last 5 decays below this
MAX_ITERS=50               # hard cap on iterations
MAX_WALL_SECONDS=3600      # hard wall cap
MAX_COST_USD=5.0          # hard cost cap (placeholder; see docs)
CONTROL_SAMPLES=10        # per-iteration control run sample count (multi-sample!)
CONTROL_DRIFT_PCT=1.0     # abort iteration if control moved > this % from baseline
EXPERIMENT_SAMPLES=30     # sample count for the experiment run.sh
DRIFT_GUARD_PCT=1.0       # machine-state drift guard for the control

# Optimiser invocation. Default: the Kilo CLI headless (`kilo run`), which runs
# the /optimize-iteration slash command non-interactively with auto-approve on
# the allowed edit paths (src/**). Override with OPTIMIZER_CMD if you want a
# different runner; set to "" to fall back to manual (it prints the prompt and
# waits for Enter).
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
BRANCH_PREFIX="opt/iter"

DRY_RUN=0
NO_MODEL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)  DRY_RUN=1; shift;;
    --no-model) NO_MODEL=1; shift;;
    -h|--help)
      sed -n '2,30p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# ─── helpers (python as a calculator; control flow stays in bash) ──────────
py() { "$PY" -c "$1"; }
# boolean python expr -> lowercase "true"/"false" (bash compares are
# case-sensitive; Python prints "True"/"False", which would silently never
# match). Use bq for every boolean predicate.
bq() { "$PY" -c "print('true' if ($1) else 'false')"; }

# read a field from a JSON file: jfield <file> <key>
jfield() { "$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2],''))" "$1" "$2"; }
# read a nested numeric field
jnum() { "$PY" -c "import json,sys;v=json.load(open(sys.argv[1])).get(sys.argv[2]);print('' if v is None else v)" "$1" "$2"; }

git_ok() { git -C "$REPO" "$@" >/dev/null 2>&1; }

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
  p="$("$PY" "$LED" patience 2>/dev/null | "$PY" -c 'import json,sys;print(json.load(sys.stdin)["patience"])')"
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

# hard budgets
iter_count() { "$PY" -c "import os;led=r'$LEDGER';n=0
import os
if os.path.exists(led):
    for l in open(led,encoding='utf-8'):
        if l.strip(): n+=1
print(n)" 2>/dev/null; }

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
  echo "constants: TARGET_VALUE=${TARGET_VALUE:-<unset>} PATIENCE=$PATIENCE MIN_IMPROVEMENT=$MIN_IMPROVEMENT AMDAHL_FLOOR=$AMDAHL_FLOOR COMPLEXITY_RATIO=$COMPLEXITY_RATIO PRED_CORR_MIN=$PRED_CORR_MIN MAX_ITERS=$MAX_ITERS MAX_WALL_SECONDS=$MAX_WALL_SECONDS MAX_COST_USD=$MAX_COST_USD" >&2
  echo "" >&2
  echo "iteration plan (per iteration):" >&2
  echo "  1. control: ./bench/run.sh --samples $CONTROL_SAMPLES --warmup 2  (unchanged baseline)" >&2
  echo "     abort iter if |control.value - baseline.value|/baseline.value > ${CONTROL_DRIFT_PCT}%" >&2
  echo "  2. branch from current best: git checkout -b ${BRANCH_PREFIX}-<N>" >&2
  echo "  3. invoke optimiser subagent (fresh context, one iteration): \$OPTIMIZER_CMD or /optimize-iteration" >&2
  echo "  4. read last ledger entry; accept iff correctness=pass AND new.ci_low > baseline.ci_high" >&2
  echo "     (non-overlapping CIs in the improvement direction; overlap => reject)" >&2
  echo "  5. accept => merge + update baseline.json; reject => record + delete branch" >&2
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
loop_start=$(date +%s)
iter=0
while true; do
  iter=$((iter+1))
  echo "=== iteration $iter ===" >&2

  # 0. master stop check BEFORE doing any work
  hit_reason="$(evaluate_stop)"; hit="${hit_reason%%|*}"; reason="${hit_reason#*|}"
  if [ "$hit" = "true" ]; then echo "STOP at iter $iter: $reason" >&2; break; fi

  # 1. control run on unchanged baseline
  ctrl_json="$BENCH/.control.$$.json"
  if ! "$BENCH/run.sh" --samples "$CONTROL_SAMPLES" --warmup 2 >"$ctrl_json" 2>"$BENCH/.control.stderr"; then
    echo "control run failed — aborting iteration" >&2; rm -f "$ctrl_json"; continue
  fi
  ctrl_val="$(jnum "$ctrl_json" value)"; base_val="$(jnum "$BASELINE" value)"
  rm -f "$ctrl_json"
  drift="$(py "print(abs(${ctrl_val}-${base_val})/${base_val}*100 if ${base_val} else 0)")"
  echo "control=$ctrl_val baseline=$base_val drift=${drift}%" >&2
  if bq "${drift} > ${CONTROL_DRIFT_PCT}" | grep -q true; then
    echo "ABORT iter $iter: control drifted ${drift}% > ${CONTROL_DRIFT_PCT}% — machine state untrustworthy" >&2
    continue
  fi

  # 2. branch from current best
  branch="${BRANCH_PREFIX}-${iter}"
  cur_branch="$(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
  git_ok checkout -b "$branch" || { echo "branch create failed" >&2; continue; }

  # 3. invoke optimiser (fresh context, one iteration). Contract: the subagent
  #    edits training-rs/src/**, runs `./bench/run.sh --samples $EXPERIMENT_SAMPLES
  #    --warmup 2 --profile` (writing bench/.experiment.json), writes
  #    bench/.hypothesis.json (hypothesis/target_symbol/predicted_speedup/
  #    lines_changed), then stops. The DRIVER owns the ledger append.
  rm -f "$BENCH/.experiment.json" "$BENCH/.hypothesis.json"
  if [ "$NO_MODEL" -eq 1 ]; then
    echo "--no-model: skipping model step (no edit; ledger will record harness_fail)" >&2
    echo '{"hypothesis":"(no-model)","target_symbol":"","predicted_speedup":1.0,"lines_changed":0}' >"$BENCH/.hypothesis.json"
    cp "$BASELINE" "$BENCH/.experiment.json" 2>/dev/null || true
  elif [ -n "$OPTIMIZER_CMD" ]; then
    # Headless: run /optimize-iteration in the repo dir with auto-approve.
    # kilo run --command <name> --auto --dir <path>
    echo "invoking optimiser: $OPTIMIZER_CMD --dir \"$REPO\"" >&2
    $OPTIMIZER_CMD --dir "$REPO" >&2 2>&1 || echo "optimiser command failed (rc=$?)" >&2
  else
    echo "=== OPTIMISER (run /optimize-iteration in a FRESH Kilo session on branch $branch) ===" >&2
    echo "iter=$iter parent=$(git -C "$REPO" rev-parse --short HEAD) experiment_samples=$EXPERIMENT_SAMPLES" >&2
    echo "press Enter once .experiment.json + .hypothesis.json are written..." >&2
    read -r _ </dev/tty
  fi

  # 4. evaluate the experiment. If the subagent didn't produce .experiment.json,
  #    run the experiment ourselves (defensive — the accept decision is the
  #    driver's, never the model's).
  exp_json="$BENCH/.experiment.$$.json"
  if [ -f "$BENCH/.experiment.json" ]; then cp "$BENCH/.experiment.json" "$exp_json"; fi
  if [ ! -f "$exp_json" ]; then
    "$BENCH/run.sh" --samples "$EXPERIMENT_SAMPLES" --warmup 2 --profile >"$exp_json" 2>"$BENCH/.exp.stderr" \
      || echo "experiment run.sh failed" >&2
  fi
  exp_correct="$(jfield "$exp_json" correctness)"
  exp_low="$(jnum "$exp_json" ci_low)"; exp_high="$(jnum "$exp_json" ci_high)"; exp_val="$(jnum "$exp_json" value)"
  base_high="$(jnum "$BASELINE" ci_high)"; base_low="$(jnum "$BASELINE" ci_low)"

  # accept iff correctness passes AND the new CI sits entirely ABOVE the
  # baseline CI (new.ci_low > base.ci_high). Overlap or regression => reject.
  accept=false
  if [ "$exp_correct" = "pass" ] && [ -n "$exp_low" ] && [ -n "$base_high" ]; then
    if bq "${exp_low} > ${base_high}" | grep -q true; then accept=true; fi
  fi

  # hypothesis metadata (from the subagent) for the ledger entry.
  hyp="$(jfield "$BENCH/.hypothesis.json" hypothesis 2>/dev/null)"
  tsym="$(jfield "$BENCH/.hypothesis.json" target_symbol 2>/dev/null)"
  pred="$(jnum "$BENCH/.hypothesis.json" predicted_speedup 2>/dev/null)"; [ -z "$pred" ] && pred=1.0
  lch="$(jnum "$BENCH/.hypothesis.json" lines_changed 2>/dev/null)"; [ -z "$lch" ] && lch=0
  parent="$(git -C "$REPO" rev-parse --short HEAD)"
  actual="$(py "print(round((${exp_val:-0})/${base_val:-1},4))")"

  if [ "$accept" = "true" ]; then
    # 5a. accept: merge into best branch, promote experiment JSON to baseline.
    git_ok checkout "$cur_branch" && git_ok merge --no-ff "$branch" -m "opt iter $iter: accept ${base_val} -> ${exp_val} games/s"
    cp "$exp_json" "$BASELINE"   # experiment JSON becomes the new baseline reference
    [ -f "$BENCH/.experiment.json" ] && rm -f "$BENCH/.experiment.json"
    "$PY" "$LED" append "{\"parent_commit\":\"$parent\",\"commit\":\"$(git -C "$REPO" rev-parse --short HEAD)\",\"hypothesis\":\"$hyp\",\"target_symbol\":\"$tsym\",\"predicted_speedup\":${pred},\"actual_speedup\":${actual},\"verdict\":\"accept\",\"lines_changed\":${lch},\"notes\":\"${base_val}->${exp_val} games/s; ci [${exp_low},${exp_high}] vs base [${base_low},${base_high}]\"}" >/dev/null
    git_ok branch -D "$branch"
    echo "ACCEPT iter $iter: ${base_val} -> ${exp_val} games/s (actual_speedup=${actual})" >&2
  else
    # 5b. reject: record the failure mode, delete the branch.
    if [ "$exp_correct" != "pass" ]; then v="correctness_fail"; else v="reject"; fi
    "$PY" "$LED" append "{\"parent_commit\":\"$parent\",\"commit\":\"$(git -C "$REPO" rev-parse --short HEAD)\",\"hypothesis\":\"$hyp\",\"target_symbol\":\"$tsym\",\"predicted_speedup\":${pred},\"actual_speedup\":${actual},\"verdict\":\"$v\",\"lines_changed\":${lch},\"notes\":\"correctness=${exp_correct}; ci_overlap=$(bq "(${exp_low:-0} <= ${base_high:-0}) and (${exp_high:-0} >= ${base_low:-0})")\"}" >/dev/null
    git_ok checkout "$cur_branch" && git_ok branch -D "$branch"
    echo "REJECT iter $iter (verdict=$v, correctness=$exp_correct)" >&2
  fi
  rm -f "$exp_json" "$BENCH/.hypothesis.json" "$BENCH/.experiment.json"
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
