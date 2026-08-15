# Benchmarks

Population sweep: `opponents=4 seeds=8 max_hands=1 workers=8`, CUDA build, RTX
5060 Ti. One generation per run via `canastra-bench`. Wall time and games/s as
reported by the bench. Re-run after every phase that touches the rollout.

## Baseline (Phase 0) — commit `665555f` (pre-instrumentation)

The pre-existing architecture: `GpuServer` + 8 worker threads, `forward_picks`
with per-forward `index_select` weight re-slicing and CPU argmax (device sync per
batch). Measured here with the Phase-0 `profile` feature compiled in (negligible
overhead — a few `Instant` spans + atomics).

| Population | Games  | Wall (s) | games/s | Server busy | rows/req |
|-----------:|-------:|---------:|--------:|------------:|--------:|
| 96         | 6,144  | 23.7     | 259     | 99.2%       | 449     |
| 500        | 32,000 | 257.8    | 124     | 99.6%       | 2331    |
| 1,000      | 64,000 | 607.7    | 105     | 99.6%       | 4348    |

Scaling is negative (259 → 105 games/s as pop grows 10x). Per-forward GPU time
grows 14x for a 5x population rise. See `docs/profile-baseline.md`.
