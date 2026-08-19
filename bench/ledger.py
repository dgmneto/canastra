"""bench/ledger.py — append-only ledger of optimisation iterations.

Each line of bench/LEDGER.jsonl is one JSON object:
  {"iter":1,"ts":"","parent_commit":"","commit":"","hypothesis":"",
   "target_symbol":"","predicted_speedup":1.0,"actual_speedup":1.0,
   "verdict":"accept|reject|correctness_fail|harness_fail",
   "lines_changed":0,"notes":""}

Queries used by the driver and the optimizer subagent:
  append(...)            — append one entry (auto iter + ts)
  recent(n)              — the last n entries (oldest→newest)
  rejected_hypotheses()  — every rejected/correctness-fail/harness-fail entry
  patience_count()       — consecutive non-accept (reject|correctness_fail|
                           harness_fail) at the tail
  prediction_correlation(n) — Pearson r of predicted vs actual speedup over the
                           last n *accepted* entries; NaN if <2 points
  last_profile_top()     — profile_top of the most recent entry (for the
                           optimizer's hotspot pick)

Pure stdlib. CLI: `python bench/ledger.py <cmd> ...`
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "LEDGER.jsonl")

VERDICTS = {"accept", "reject", "correctness_fail", "harness_fail"}
NON_ACCEPT = {"reject", "correctness_fail", "harness_fail"}


def _read_all() -> list[dict]:
    if not os.path.exists(LEDGER):
        return []
    out = []
    with open(LEDGER, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _next_iter(entries: list[dict]) -> int:
    return (max((e.get("iter", 0) for e in entries), default=0) + 1) if entries else 1


def append(*, parent_commit: str, commit: str, hypothesis: str,
           target_symbol: str, predicted_speedup: float, actual_speedup: float,
           verdict: str, lines_changed: int = 0, notes: str = "",
           profile_top: list | None = None) -> dict:
    if verdict not in VERDICTS:
        raise ValueError(f"verdict {verdict!r} not in {VERDICTS}")
    entries = _read_all()
    entry = {
        "iter": _next_iter(entries),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parent_commit": parent_commit,
        "commit": commit,
        "hypothesis": hypothesis,
        "target_symbol": target_symbol,
        "predicted_speedup": round(float(predicted_speedup), 4),
        "actual_speedup": round(float(actual_speedup), 4),
        "verdict": verdict,
        "lines_changed": int(lines_changed),
        "notes": notes,
    }
    if profile_top is not None:
        entry["profile_top"] = profile_top
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def recent(n: int = 10) -> list[dict]:
    return _read_all()[-n:]


def rejected_hypotheses() -> list[dict]:
    return [e for e in _read_all() if e.get("verdict") in NON_ACCEPT]


def patience_count() -> int:
    """Consecutive non-accept entries at the tail of the ledger."""
    entries = _read_all()
    c = 0
    for e in reversed(entries):
        if e.get("verdict") in NON_ACCEPT:
            c += 1
        else:
            break
    return c


def accepted() -> list[dict]:
    return [e for e in _read_all() if e.get("verdict") == "accept"]


def prediction_correlation(n: int = 5) -> float:
    """Pearson r of predicted_speedup vs actual_speedup over the last n accepted
    entries. NaN if fewer than 2 points (insufficient to define a line)."""
    acc = accepted()[-n:]
    if len(acc) < 2:
        return float("nan")
    xs = [e["predicted_speedup"] for e in acc]
    ys = [e["actual_speedup"] for e in acc]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def last_profile_top() -> list[dict]:
    for e in reversed(_read_all()):
        if "profile_top" in e:
            return e["profile_top"]
    return []


def rejected_targets() -> list[str]:
    """target_symbols of every non-accept entry — the optimizer must not
    re-propose these."""
    return [e.get("target_symbol", "") for e in rejected_hypotheses() if e.get("target_symbol")]


def _main() -> int:
    if len(sys.argv) < 2:
        print("usage: ledger.py {recent|rejected|patience|predcorr|last-profile|"
              "targets|append ...}", file=sys.stderr)
        return 2
    cmd = sys.argv[1]
    if cmd == "recent":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        print(json.dumps(recent(n), indent=2))
    elif cmd == "rejected":
        print(json.dumps(rejected_hypotheses(), indent=2))
    elif cmd == "patience":
        print(json.dumps({"patience": patience_count()}))
    elif cmd == "predcorr":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        print(json.dumps({"prediction_correlation": prediction_correlation(n), "n": n}))
    elif cmd == "last-profile":
        print(json.dumps(last_profile_top()))
    elif cmd == "targets":
        print(json.dumps({"rejected_targets": rejected_targets()}))
    elif cmd == "append":
        # `ledger.py append '{json}'` — read a JSON object from argv[2]
        e = json.loads(sys.argv[2])
        out = append(**e)
        print(json.dumps(out))
    else:
        print(f"unknown cmd {cmd}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_main())
