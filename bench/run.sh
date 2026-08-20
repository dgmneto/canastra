#!/usr/bin/env bash
# bench/run.sh — one command, no required args. Emits the run JSON to stdout
# and nothing else; diagnostics to stderr. Exit 0 on success, non-zero on
# harness failure.
#
# ─── Machine-state assumptions (READ BEFORE TRUSTING A NUMBER) ───────────────
#  * Target: the training-rs lockstep self-play loop (canastra-bench, CUDA BF16
#    on an RTX 5060 Ti / 16 GB, driver 610.88, CUDA 13.3). Metric = games/s.
#    CPU fp32 is the deterministic correctness path only (~20x slower, not the
#    optimisation target).
#  * Dominant noise source is GPU thermals/clock-boost, not CPU scheduling. A
#    discarded warmup is MANDATORY: the first process after a cold GPU can read
#    ~40% low (measured 359 vs 600 games/s). run.sh always discards WARMUP.
#  * CPU pinning: Windows has no taskset. Set PIN_CORES to a hex affinity mask
#    (e.g. "FF" for cores 0-7) to pin the bench via `start /affinity`. Default
#    unset = no pinning. GPU work is unaffected; this only stabilises the rayon
#    encode/apply phases. Documented in docs/OPTIMIZATION.md.
#  * Native-exe stdout is NOT capturable by Git Bash redirection (MSVC runtime);
#    all bench invocation goes through Python subprocess, which captures it
#    reliably. Do not call the .exe directly from bash expecting stdout.
#
# ─── Named constants (every threshold lives here, never inline) ─────────────
POPULATION=96
OPPONENTS=4
SEEDS=8
DEVICE=cuda
DTYPE=bf16
WARMUP=2          # discarded warmup runs (cold-GPU guard)
SAMPLES=30        # timed samples; CI is a bootstrap of the mean
METRIC="games_per_s"
BENCH_FLAGS=(--population ${POPULATION} --opponents ${OPPONENTS} --seeds ${SEEDS} --device ${DEVICE} --dtype ${DTYPE})
DO_PROFILE=0

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
TR="$REPO/training-rs"
BENCH_BIN="$TR/target/release/canastra-bench.exe"
PROFILE_BIN="$TR/target/profile/release/canastra-bench.exe"
GATE_BIN="$TR/target/gate/release/gate.exe"
GOLDEN="$HERE/golden/generation.json"
LIB="$HERE/_lib.py"
PY="${PYTHON:-python}"

usage() { echo "usage: $0 [--profile] [--samples N] [--warmup N]" >&2; }
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) DO_PROFILE=1; shift;;
    --samples) SAMPLES="$2"; shift 2;;
    --warmup)  WARMUP="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "unknown arg: $1" >&2; usage; exit 2;;
  esac
done

emit_fail() {  # $1=correctness_detail
  commit="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  dirty=false; git -C "$REPO" diff --quiet 2>/dev/null || dirty=true
  "$PY" "$LIB" assemble "$commit" "$dirty" "fail" "$1" "$METRIC" 0.0 - 0.0
  exit 1
}

# ─── Precondition: binaries present ─────────────────────────────────────────
if [ ! -x "$BENCH_BIN" ]; then
  echo "run.sh: missing $BENCH_BIN — see docs/OPTIMIZATION.md build step" >&2
  emit_fail "missing bench binary"
fi
if [ ! -x "$GATE_BIN" ]; then
  echo "run.sh: missing $GATE_BIN — build: (cd training-rs && cargo build --release --target-dir target/gate --bin gate)" >&2
  emit_fail "missing gate binary"
fi

# ─── 1. Correctness gate FIRST ──────────────────────────────────────────────
# Byte-identical golden on CPU. A behaviour-altering speedup flips a pick ->
# different score -> FAIL. Independent of the timing path.
gate_out="$("$GATE_BIN" --check -i "$GOLDEN" 2>&1)"
gate_rc=$?
if [ $gate_rc -ne 0 ]; then
  echo "run.sh: correctness gate FAILED:" >&2
  echo "$gate_out" >&2
  emit_fail "golden generation mismatch"
fi
echo "run.sh: correctness gate PASS" >&2

# ─── 2. Warmup (discarded) + timed samples ─────────────────────────────────
# Bench invocation is routed through Python so native-exe stdout is captured.
run_bench_py() { "$PY" "$LIB" bench-once "$BENCH_BIN" "${BENCH_FLAGS[@]}"; }

samples=()
wall_start=$(date +%s.%N)
for i in $(seq 1 "$WARMUP"); do
  echo "run.sh: warmup $i (discarded)..." >&2
  run_bench_py >/dev/null || emit_fail "warmup run failed"
done
for i in $(seq 1 "$SAMPLES"); do
  out="$(run_bench_py)"
  v="$(echo "$out" | "$PY" -c 'import json,sys;print(json.load(sys.stdin).get("games_per_s",""))' 2>/dev/null)"
  if [ -z "$v" ]; then
    echo "run.sh: sample $i failed: $out" >&2
    emit_fail "bench sample $i failed: $out"
  fi
  samples+=("$v")
  lvl="$(echo "$out" | "$PY" -c 'import json,sys;print(json.load(sys.stdin).get("level_pct",""))' 2>/dev/null)"
  echo "run.sh: sample $i = $v games/s (level ${lvl}%)" >&2
done
wall_end=$(date +%s.%N)
wall_seconds="$("$PY" -c "print(round(${wall_end}-${wall_start},3))")"

# ─── 3. Optional profile (separate, slower run) ─────────────────────────────
prof_file="-"
if [ "$DO_PROFILE" -eq 1 ]; then
  if [ ! -x "$PROFILE_BIN" ]; then
    echo "run.sh: --profile requested but $PROFILE_BIN missing — building" >&2
    (cd "$TR" && cargo build --release --features "cuda,profile" --target-dir target/profile --bin canastra-bench) >&2
  fi
  prof_file="$HERE/.profile.$$.json"
  "$PY" "$LIB" profile-once "$PROFILE_BIN" "${BENCH_FLAGS[@]}" >"$prof_file" 2>/dev/null \
    || { echo "run.sh: profile run failed" >&2; prof_file="-"; }
  echo "run.sh: profile run captured -> $prof_file" >&2
fi

# ─── 4. Emit JSON ───────────────────────────────────────────────────────────
commit="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
dirty=false; git -C "$REPO" diff --quiet 2>/dev/null || dirty=true
"$PY" "$LIB" assemble "$commit" "$dirty" "pass" "" "$METRIC" "$wall_seconds" "$prof_file" "${samples[@]}"
rc=$?
[ -f "$prof_file" ] && [ "$prof_file" != "-" ] && rm -f "$prof_file"
exit $rc
