"""Shared helpers for the optimisation harness: bench/profile output parsing,
sample statistics, and a bootstrap confidence interval.

Pure stdlib — no jq, no numpy. Used by bench/run.sh and scripts/optimize-loop.sh
(via `python bench/_lib.py <cmd> ...`).
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from random import Random

# ─── bench stdout parsing ──────────────────────────────────────────────────
# Example line:
#   pop=96 games=6144 device=cuda dtype=BF16: 17.1s = 359 games/s | level 0.7% | mean |diff| 270 pts
_BENCH_RE = re.compile(
    r"games=(\d+).*?=\s*([\d.]+)\s+games/s\s*\|\s*level\s+([\d.]+)%\s*\|\s*"
    r"mean\s*\|diff\|\s+([\d.]+)\s+pts"
)


def parse_bench(stdout: str) -> dict:
    """Parse one canastra-bench stdout line into a dict.

    Returns games, games_per_s, level_pct, mean_abs_diff. Raises on no match
    (the caller treats that as a harness failure).
    """
    m = _BENCH_RE.search(stdout)
    if not m:
        raise ValueError(f"could not parse bench line: {stdout!r}")
    return {
        "games": int(m.group(1)),
        "games_per_s": float(m.group(2)),
        "level_pct": float(m.group(3)),
        "mean_abs_diff": float(m.group(4)),
    }


# ─── profile breakdown parsing ──────────────────────────────────────────────
# The crate's `#[cfg(feature="profile")]` report prints a table to stderr. We
# extract the per-phase pct of wall time. The phases are non-overlapping leaves
# of the lockstep critical path (encode + apply + h2d + the forward stages
# {gather,trunk,head,mask,argmax} + idle), so they sum to ~100%.
_PHASE_RE = re.compile(r"(\w[\w-]*)\s+([\d.]+)s\s+([\d.]+)%")
# the busy line carries the H2D figure inside parentheses
_H2D_RE = re.compile(r"busy\s+[\d.]+s\s+[\d.]+%\s+\(H2D\s+([\d.]+)s")


def parse_profile(stderr: str) -> list[dict]:
    """Parse the profile breakdown into [{"symbol":..,"pct":..},...] sorted desc."""
    phases: dict[str, float] = {}
    h2d = _H2D_RE.search(stderr)
    if h2d:
        # Derive H2D's pct from its share of wall (wall is on the 'wall' line).
        pass  # handled below via explicit extraction
    for m in _PHASE_RE.finditer(stderr):
        name, _secs, pct = m.group(1), float(m.group(2)), float(m.group(3))
        # skip the aggregate 'busy'/'fwd-wait' rows to avoid double counting;
        # the leaves (gather/trunk/head/mask/argmax) already cover GPU time.
        if name in ("busy",):
            continue
        phases[name] = pct
    # Pull H2D out of the busy line explicitly so it appears as its own symbol.
    wall_m = re.search(r"wall\s+([\d.]+)s", stderr)
    if h2d and wall_m:
        wall = float(wall_m.group(1))
        phases["h2d"] = float(h2d.group(1)) / wall * 100.0 if wall > 0 else 0.0
    out = [{"symbol": k, "pct": round(v, 2)} for k, v in phases.items() if v > 0]
    out.sort(key=lambda d: d["pct"], reverse=True)
    return out


# ─── statistics ─────────────────────────────────────────────────────────────


def mean_std(samples: list[float]) -> tuple[float, float]:
    n = len(samples)
    if n == 0:
        return 0.0, 0.0
    m = statistics.fmean(samples)
    if n == 1:
        return m, 0.0
    var = statistics.variance(samples, m)  # sample variance (n-1)
    return m, var**0.5


def bootstrap_ci(samples: list[float], conf: float = 0.95, n_boot: int = 2000,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean. Deterministic (fixed seed) so two
    runs on identical samples yield identical bounds — required for the loop's
    non-overlap accept rule to be reproducible."""
    n = len(samples)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return samples[0], samples[0]
    rng = Random(seed)
    means = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        means.append(statistics.fmean(samples[i] for i in idx))
    means.sort()
    lo_idx = int((1 - conf) / 2 * n_boot)
    hi_idx = int((1 - (1 - conf) / 2) * (n_boot - 1))
    return means[lo_idx], means[hi_idx]


def cv(samples: list[float]) -> float:
    m, s = mean_std(samples)
    return s / m * 100.0 if m != 0 else 0.0


def getfield(path: str, key: str) -> str:
    """Value of `key` in JSON file `path`, or '' on any failure.

    Fails soft on purpose: the optimisation loop reads agent-written artifacts
    (.experiment.json / .hypothesis.json) and must never crash on a missing,
    empty, or unparseable file. Decoding tolerates UTF-8 (with/without BOM),
    UTF-16, and cp1252 — an agent whose shell is PowerShell writes '>'
    redirections as UTF-16LE, which a plain json.load(open(...)) cannot parse
    and which used to surface as a bogus correctness_fail.
    """
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError:
        return ""
    text = None
    for enc in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except (UnicodeDecodeError, ValueError):
            continue
    if text is None:
        return ""
    try:
        v = json.loads(text).get(key, "")
    except (json.JSONDecodeError, AttributeError):
        return ""
    return "" if v is None else str(v)


# ─── CLI shim ───────────────────────────────────────────────────────────────
# `python bench/_lib.py ci  'v1 v2 v3'`        -> {"mean":..,"sd":..,"ci_low":..,"ci_high":..}
# `python bench/_lib.py parse-bench '<line>'`  -> bench dict (json)
# `python bench/_lib.py parse-profile '<file>'`-> profile list (json)
# `python bench/_lib.py cv 'v1 v2 ...'`         -> {"cv":..}
# `python bench/_lib.py getfield <file> <key>`  -> field value or '' (fail-soft)
# `python bench/_lib.py bench-once <bin> <flag>...`   -> run bench, parse, json
# `python bench/_lib.py profile-once <bin> <flag>...`  -> run profile build, json
# `python bench/_lib.py assemble ...`                  -> full run JSON
import subprocess

# Degenerate-forward guard: if every game ends level (both teams on the §13.3
# flat -300), the forward collapsed to "first legal action" and finishes games
# near-instantly — a fake speedup. This is exactly the f16 masking bug (recorded
# as 3.9x). The harness must reject such a "win" as a correctness failure.
DEGENERATE_LEVEL_PCT = 99.0


def run_native(bin_path: str, flags: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a native Windows .exe and capture stdout/stderr reliably.

    Git Bash's redirection does NOT capture MSVC-runtime stdout from native
    exes, so all bench invocation goes through Python subprocess (which uses
    the real pipe handles and captures it). Optional CPU affinity via the
    PIN_CORES env var (a hex mask accepted by `start /affinity`).
    """
    import os
    args = [bin_path] + flags
    pin = os.environ.get("PIN_CORES")
    if pin:
        # Affinity is best-effort; `start` doesn't route child stdout to a
        # capturable pipe, so when pinning we forgo capture (the bench still
        # writes its result line; we re-read via a temp file fallback).
        cmd = ["cmd", "/c", "start", "/affinity", pin, "/wait", "", bin_path] + flags
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              errors="replace")
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                          errors="replace")


def bench_once(bin_path: str, flags: list[str]) -> dict:
    r = run_native(bin_path, flags)
    if r.returncode != 0:
        raise RuntimeError(f"bench exit {r.returncode}: {r.stderr.strip()[:300]}")
    d = parse_bench(r.stdout)
    if d["level_pct"] >= DEGENERATE_LEVEL_PCT:
        raise RuntimeError(
            f"degenerate forward: level {d['level_pct']}% — every game ended "
            f"level; throughput {d['games_per_s']} is a collapsed-policy artefact, "
            f"not a speedup (see docs/decision-ranking-metric.md)")
    return d


def profile_once(bin_path: str, flags: list[str]) -> list[dict]:
    r = run_native(bin_path, flags)
    if r.returncode != 0:
        raise RuntimeError(f"profile exit {r.returncode}: {r.stderr.strip()[:300]}")
    return parse_profile(r.stderr)


_TRAIN_RE = re.compile(r"gen\s+\d+:.*?\(([\d.]+)s\)")


def train_once(bin_path: str, games: int, flags: list[str]) -> dict:
    """Run one canastra-train generation (a different path: materialise +
    league + anchors + Adam, not the league-only bench) and return its
    full-generation throughput. The holdout exercises code the bench omits.

    Wall is timed externally (subprocess wall) for full precision — the train
    binary prints only `{:.1}s`, which quantises a ~10s run to ~0.5% steps and
    makes the CI degenerate. External wall includes ~1-2s CUDA init, a roughly
    constant offset that dampens the % slightly but preserves the variance
    component the overfit check needs."""
    import time
    t0 = time.perf_counter()
    r = run_native(bin_path, flags, timeout=600)
    wall = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(f"train exit {r.returncode}: {r.stderr.strip()[:300]}")
    # Sanity: the train binary must have printed a gen line (confirms it ran a
    # generation, not just exited). Not used for the number (external wall is).
    if not _TRAIN_RE.search(r.stdout):
        raise RuntimeError(f"could not parse train gen line: {r.stdout[-300:]!r}")
    return {"wall_seconds": round(wall, 4), "games": games,
            "games_per_s": round(games / wall, 4) if wall > 0 else 0.0}


def _main() -> int:
    if len(sys.argv) < 2:
        print("usage: _lib.py {ci|cv|parse-bench|parse-profile} ...", file=sys.stderr)
        return 2
    cmd = sys.argv[1]
    if cmd == "ci":
        vals = [float(x) for x in sys.argv[2].split()]
        m, s = mean_std(vals)
        lo, hi = bootstrap_ci(vals)
        print(json.dumps({"mean": round(m, 6), "sd": round(s, 6),
                          "ci_low": round(lo, 6), "ci_high": round(hi, 6),
                          "n": len(vals)}))
    elif cmd == "cv":
        vals = [float(x) for x in sys.argv[2].split()]
        print(json.dumps({"cv": round(cv(vals), 4)}))
    elif cmd == "getfield":
        print(getfield(sys.argv[2], sys.argv[3]))
    elif cmd == "bench-once":
        try:
            print(json.dumps(bench_once(sys.argv[2], sys.argv[3:])))
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            return 1
    elif cmd == "profile-once":
        try:
            print(json.dumps(profile_once(sys.argv[2], sys.argv[3:])))
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            return 1
    elif cmd == "train-once":
        try:
            print(json.dumps(train_once(sys.argv[2], int(sys.argv[3]), sys.argv[4:])))
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            return 1
    elif cmd == "parse-bench":
        print(json.dumps(parse_bench(sys.argv[2])))
    elif cmd == "parse-profile":
        text = sys.argv[2]
        if text.startswith("@"):
            with open(text[1:], "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        print(json.dumps(parse_profile(text)))
    elif cmd == "assemble":
        # python _lib.py assemble <commit> <dirty> <correctness> <detail> <metric>
        #                        <wall_seconds> <profile_file|-> <v1> <v2> ...
        commit, dirty, correctness = sys.argv[2], sys.argv[3] == "true", sys.argv[4]
        detail, metric = sys.argv[5], sys.argv[6]
        wall = float(sys.argv[7])
        prof_file = sys.argv[8]
        vals = [float(x) for x in sys.argv[9:]] if len(sys.argv) > 9 else []
        if correctness != "pass" or not vals:
            m = vals[0] if len(vals) == 1 else (statistics.fmean(vals) if vals else 0.0)
            lo, hi = (m, m) if vals else (0.0, 0.0)
        else:
            m, _ = mean_std(vals)
            lo, hi = bootstrap_ci(vals)
        prof = []
        if prof_file and prof_file != "-":
            try:
                with open(prof_file, "r", encoding="utf-8", errors="replace") as f:
                    raw = f.read()
                try:
                    decoded = json.loads(raw)
                    prof = decoded[:5] if isinstance(decoded, list) else parse_profile(raw)[:5]
                except json.JSONDecodeError:
                    prof = parse_profile(raw)[:5]
            except OSError:
                prof = []
        out = {
            "commit": commit,
            "dirty": dirty,
            "correctness": correctness,
            "correctness_detail": detail,
            "metric": metric,
            "value": round(m, 4),
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
            "samples": len(vals),
            "wall_seconds": round(wall, 3),
            "profile_top": prof,
        }
        print(json.dumps(out, indent=2))
    else:
        print(f"unknown cmd {cmd}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_main())
