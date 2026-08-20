#!/usr/bin/env bash
# bench/holdout.sh — the holdout benchmark. Exercises the SAME lockstep target
# code along a DIFFERENT path: one full `canastra-train` generation, which
# additionally runs ES population materialisation + anchor evaluation + the Adam
# step — work the league-only `canastra-bench` omits (see docs/benchmarks.md
# "Whole-generation cost"). The optimisation loop never optimises against this
# number; scripts/optimize-loop.sh runs it once at the very end and compares to
# bench/baseline-holdout.json. If the holdout didn't improve proportionally,
# the loop overfit the harness — reported loudly.
#
# Emits the run-JSON schema (metric=holdout_games_per_s).
set -euo pipefail

# ─── Named constants ────────────────────────────────────────────────────────
# Match the bench's population (pop=96) so the holdout is stable (~10-15s/gen)
# and comparable; the full-generation wall adds materialise + anchors + Adam.
N_PERTURBATIONS=48     # population = 2 × this = 96
OPPONENTS=4
SEEDS=8
MAXHANDS=1
DEVICE=cuda
WARMUP=1
SAMPLES=5
METRIC="holdout_games_per_s"
OUT_FILE=""
# games per generation = 2 × n_perturbations × opponents × seeds
GAMES=$(( 2 * N_PERTURBATIONS * OPPONENTS * SEEDS ))
FLAGS=(--generations 1 --n-perturbations ${N_PERTURBATIONS} --opponents ${OPPONENTS}
       --seeds ${SEEDS} --device ${DEVICE} --max-hands ${MAXHANDS}
       --anchor-interval 0 --run-dir)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
TR="$REPO/training-rs"
TRAIN_BIN="$TR/target/release/canastra-train.exe"
LIB="$HERE/_lib.py"
PY="${PYTHON:-python}"

while [ $# -gt 0 ]; do
  case "$1" in
    --samples) SAMPLES="$2"; shift 2;;
    --warmup)  WARMUP="$2"; shift 2;;
    --baseline) OUT_FILE="$2"; shift 2;;
    -h|--help) echo "usage: $0 [--samples N] [--warmup N] [--baseline FILE]"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

if [ ! -x "$TRAIN_BIN" ]; then
  echo "holdout.sh: missing $TRAIN_BIN — build: (cd training-rs && cargo build --release --features cuda --bin canastra-train)" >&2
  exit 1
fi

run_tmp="$HERE/.holdout.run.$$"
mkdir -p "$run_tmp"
trap 'rm -rf "$run_tmp"' EXIT

samples=()
wall_start=$(date +%s.%N)
for i in $(seq 1 "$WARMUP"); do
  echo "holdout: warmup $i (discarded)..." >&2
  rm -rf "$run_tmp"; mkdir -p "$run_tmp"
  "$PY" "$LIB" train-once "$TRAIN_BIN" "$GAMES" "${FLAGS[@]}" "$run_tmp" >/dev/null 2>&1 || true
done
for i in $(seq 1 "$SAMPLES"); do
  rm -rf "$run_tmp"; mkdir -p "$run_tmp"
  out="$("$PY" "$LIB" train-once "$TRAIN_BIN" "$GAMES" "${FLAGS[@]}" "$run_tmp" 2>&1)" || {
    echo "holdout: sample $i failed: $out" >&2; samples+=("0"); continue
  }
  v="$(printf '%s' "$out" | "$PY" -c 'import json,sys;print(json.load(sys.stdin)["games_per_s"])' 2>/dev/null)"
  samples+=("${v:-0}")
  echo "holdout: sample $i = ${v:-0} games/s (full-gen)" >&2
done
wall_end=$(date +%s.%N)
wall_seconds="$("$PY" -c "print(round(${wall_end}-${wall_start},3))")"

commit="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
dirty=false; git -C "$REPO" diff --quiet 2>/dev/null || dirty=true
json="$("$PY" "$LIB" assemble "$commit" "$dirty" "pass" "" "$METRIC" "$wall_seconds" - "${samples[@]}")"
if [ -n "$OUT_FILE" ]; then printf '%s\n' "$json" >"$OUT_FILE"; else printf '%s\n' "$json"; fi
