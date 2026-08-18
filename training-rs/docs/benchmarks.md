# Benchmarks

> **The "Phase 2 — u8 transfer + F16 dtype" table below is void.** Its speedups
> are the f16 masking bug, not work done. See
> [decision-ranking-metric.md](decision-ranking-metric.md).
>
> Switching `DType::BF16` → `DType::F16` also changed what the `-1e9` masking
> sentinel does. BF16 carries f32's exponent range, so `-1e9` is finite there
> and `(1 - mask) * -1e9` is `0` for legal actions. F16 tops out at 65504, so
> the sentinel overflows to `-inf` and the offset becomes `0 × -inf = NaN` for
> every *legal* action. Every score row went NaN, argmax returned index 0, and
> every policy collapsed to "take the first legal action": no melds, no opening,
> both partnerships at §13.3's flat −300, every hand over almost immediately.
> The 3.6–3.9x "speedup" is games ending instantly.
>
> **Earlier sections are unaffected** — Baseline and Phase 1b were measured with
> BF16, before the switch, as were `task4-ksweep.md` and `runs/es-smoke`. CPU
> figures were never affected (dtype `F32`).

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

## Phase 2 — u8 transfer + F16 dtype — **TABLE VOID, RE-MEASURE**

> The F16 switch broke action masking (`-1e9` → `-inf` → `0 × -inf = NaN`), so
> every game below ended instantly at −300 to −300. The table times degenerate
> games. The masking is fixed (`policy::mask_illegal_f32`) and
> `tests/fitness_signal.rs` guards it, but **these rows have not been
> re-measured**. Note the reasoning recorded below — "the F16 path's correctness
> is verified by the existing forward-pass test (CPU F32 vs Python) and the GPU
> BF16 vs CPU agreement test" — names the gap exactly: neither test exercises
> F16. The u8 transfer change is independent and unaffected.

Two changes applied on top of Phase 1b:

1. **u8 dtype for obs/acts/mask transfer**: both observation and action
   tensors are 100% binary (verified in `docs/task3a-sparsity.md`).
   `EncodedPly` now stores `Vec<u8>` instead of `Vec<f32>`; the forward
   path casts device-side via `to_dtype` (u8→F16 on CUDA, u8→F32 on CPU).
   This cuts H2D transfer volume 4x with zero precision loss.

2. **F16 dtype on CUDA (was bf16)**: the RTX 5060 Ti (Blackwell, sm_120)
   has dramatically faster F16 tensor cores than BF16. Changing
   `DType::BF16` → `DType::F16` in the league's dtype selection gave a
   ~2.5x speedup on the grouped bmm — the dominant compute cost. The
   CPU path remains F32 (exact, for correctness tests); the GPU
   equivalence test still uses BF16 (unchanged). The F16 path's
   correctness is verified by the existing forward-pass test (CPU F32
   vs Python) and the GPU BF16 vs CPU agreement test.

Also explored but **not shipped** (dead ends, documented for the record):
- **Sparse embedding-bag input layer** (`src/sparse.rs`, `forward_picks_sparse`):
  implemented per `task3a-sparsity.md`'s split design. The `index_select` +
  `sum` gather was 2.4x *slower* than the dense bmm — candle lacks a fused
  gather+segment-sum CUDA kernel, and random-access gathers on this GPU
  achieve much lower effective bandwidth than the coalesced dense matmul.
- **ES grouped-GEMM split** (`forward_picks_es`, `forward_pass_es`):
  splitting trunk.0 into base GEMM (large M) + perturbation bmm (half FLOPs).
  The split doubles total FLOPs (base + perturbation = 2× original), and the
  perturbation bmm at M=128 has similar efficiency to the original at M=64.
  Net effect: negligible.
- **Bmm transpose** (making M the larger dimension): no improvement — the
  cuBLAS batched GEMM efficiency is not M-limited on this GPU/architecture.

All benchmarks with `--max-width 64` (default), CUDA build, RTX 5060 Ti
(Gen 3 x8, ~7 GB/s PCIe ceiling), F16 dtype on CUDA. The bench uses ES
populations (θ ± σε pairs) via `ESState::materialise_population` to match
the production training path.

| Population | Games  | Wall (s) | games/s | Speedup vs baseline | Speedup vs Phase 1b |
|-----------:|-------:|---------:|--------:|--------------------:|--------------------:|
| 96         | 6,144  | 3.8      | 1,627   | 6.28x               | 3.61x               |
| 500        | 32,000 | 19.0     | 1,681   | 13.56x              | 3.90x               |
| 1,000      | 64,000 | 38.9     | 1,646   | 6.35x               | 3.90x               |

Scaling is flat (1,627 → 1,681 → 1,646 across 10x population growth) — the
lockstep architecture scales well. The 2x target (≥844 games/s at pop=1000)
is exceeded by 1.95x (1,646 / 844).
