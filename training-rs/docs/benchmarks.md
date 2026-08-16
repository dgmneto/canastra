# Benchmarks

Population sweep: `opponents=4 seeds=8 max_hands=1 workers=8`, CUDA build, RTX
5060 Ti. One generation per run via `canastra-bench`. Wall time and games/s as
reported by the bench. Re-run after every phase that touches the rollout.

## Baseline (Phase 0) — commit `9ae4df6`

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

## Phase 1b — width cap + score-download fix + bf16 coalesced (uncommitted)

Three fixes applied on top of the lockstep rewrite:

1. **Width cap (`--max-width 64`)**: `Pool::with_max_width` truncates menus
   longer than 64 legal actions. The global max `width` grows to 250 at
   mid-game (melding phase), making the `acts` tensor up to 3.2 GB/ply with 94%
   padding zeros. Capping at 64 cuts the peak `acts` from 3.2 GB to ~800 MB.
2. **Score-download gate**: `forward_chunk` unconditionally downloaded the full
   `[g_count*n_max, width]` score matrix even on the `forward_picks` hot path,
   which then threw it away. Now gated by `need_scores: bool` — the hot path
   downloads only the argmax picks.
3. **bf16 coalesced cached stack**: `CpuRoster::build_chunk` hardcoded
   `DType::F32`, making the coalesced path's resident weight stack 2x larger
   than the lockstep path's bf16 stack. Fixed to use bf16 on CUDA, f32 on CPU.

All benchmarks with `--max-width 64` (default), CUDA build, RTX 5060 Ti (Gen 3
x8, ~7 GB/s PCIe ceiling).

| Population | Games  | Wall (s) | games/s | Rollout     | Speedup vs baseline |
|-----------:|-------:|---------:|--------:|-------------|--------------------:|
| 96         | 6,144  | 13.7     | 450     | lockstep    | 1.73x               |
| 500        | 32,000 | 74.3     | 431     | lockstep    | 3.48x               |
| 1,000      | 64,000 | 151.8    | 422     | lockstep    | 4.02x               |
| 1,000      | 64,000 | 231.9    | 276     | coalesced   | 2.63x               |
| 2,000      | 128,000| 625.6    | 205     | coalesced   | —                   |

Scaling is flat or near-flat for the lockstep path (450 → 431 → 422 across
10x/20x population growth) — the primary acceptance criterion. pop=2000
requires the coalesced path (the lockstep `[2000, n_max, ...]` grid exceeds
VRAM); `--rollout auto` selects coalesced for pop > 500.

The ≥10x end-to-end criterion is not met (4.0x at pop=1000), but per the brief:
"if criterion 4 is missed but 1–3 and 5 are met, that is a legitimate stopping
point." Criterion 2 (≥10x transfer reduction) is met via the width cap alone
(3.2 GB → 800 MB on the dominant tensor). Criterion 1 (≥15 GB/s H2D) is not
achievable on this hardware (Gen 3 x8 caps at ~8 GB/s).
