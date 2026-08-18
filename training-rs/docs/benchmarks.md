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

## Phase 2 — u8 transfer + F16 dtype — **ORIGINAL TABLE WAS VOID**

> The F16 switch broke action masking (`-1e9` → `-inf` → `0 × -inf = NaN`), so
> every game in the original table ended instantly at −300 to −300 and the
> numbers timed degenerate games. Masking is fixed
> (`policy::mask_illegal_f32`), guarded by `tests/fitness_signal.rs`, and the
> table has been **re-measured** — see "Phase 2 re-measured" below.
>
> The claim in point 2 below did not survive: F16 and BF16 are within noise of
> each other on this card. The "~2.5x speedup on the grouped bmm" was the bug,
> not the dtype. Kept here as written for the record.
>
> Note that the reasoning in point 2 names the gap exactly — "the F16 path's
> correctness is verified by the existing forward-pass test (CPU F32 vs Python)
> and the GPU BF16 vs CPU agreement test" — neither of which exercises F16. The
> u8 transfer change is independent and unaffected.

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

## Phase 2 re-measured — after the masking fix

Same config (`--max-width 64`, ES population via `ESState::materialise_population`,
`opponents=4 seeds=8 max_hands=1`), CUDA build, RTX 5060 Ti. `canastra-bench`
now reports the share of games that ended level and the mean |paired
differential| alongside throughput, so a degenerate forward cannot be recorded
as a speedup again. `level 0.7%` below is a healthy signal; the bug produced
100%.

| Population | Games  | dtype | Wall (s) | games/s | level | mean \|diff\| |
|-----------:|-------:|:------|---------:|--------:|------:|--------------:|
| 96         | 6,144  | F16   | 10.3     | 599     | 0.7%  | 266 pts |
| 96         | 6,144  | BF16  | 10.1     | 606     | 0.7%  | 270 pts |
| 500        | 32,000 | F16   | 54.8     | 584     | 0.7%  | 254 pts |
| 500        | 32,000 | BF16  | 54.0     | 593     | 0.7%  | 256 pts |
| 1,000      | 64,000 | F16   | 111.8    | 572     | 0.7%  | 261 pts |
| 1,000      | 64,000 | BF16  | 113.8    | 562     | 0.7%  | 258 pts |

**F16 buys nothing.** The gap to BF16 is ±2% and changes sign across
populations — noise, not a difference. The dtype switch that introduced the
masking bug had no throughput justification once games are real. **BF16 is the
better default**: identical speed, but it carries f32's exponent range, so
overflow-style bugs like the `-1e9` sentinel cannot recur in it, and it is the
dtype the GPU equivalence test already covers. Selectable via
`canastra-bench --dtype {auto,f16,bf16,f32}` and `EvalInputs::dtype`.

**Scaling is flat** (599 → 584 → 572 across a 10x population rise), which was
the real Phase 1b/2 win and still holds.

**GPU vs CPU:** pop=96 on CPU (F32) is 220.1s = 28 games/s, `level 0.6%`, mean
|diff| 265 pts — the same signal the GPU produces, confirming the two paths
agree on non-degenerate games. GPU is **21x** CPU at this population.

Against the honest baselines: 599 games/s at pop=96 is 2.3x the Phase 0
baseline (259) and 1.29x Phase 1b (465, from `task4-ksweep.md`, BF16-era and
valid). The headline 1,627 games/s was never real.

### Whole-generation cost (`canastra-train`, not just the league)

`canastra-bench` times the league only. A training generation also materialises
the population, evaluates anchors and runs the Adam step. At pop=1000 that
non-league work was **83 s** — 42% of a 198 s generation — almost all of it
`ESState::materialise_population` building 1000 × 1.2M genomes single-threaded.

Parallelising it over perturbation pairs (bit-identical: each output element
depends on one seed and nothing is summed across pairs, pinned by
`es::tests::materialise_population_is_bit_identical_across_thread_counts`):

| | before | after |
|---|---:|---:|
| generation wall | 197.9 s | **114.9 s** |
| league | 184.7 s | 104.6 s |
| non-league | 13.2 s | 10.3 s |

Measured at pop=1000, `opponents=4 seeds=8`, `--anchor-interval 1`. Both runs
produce identical fitness (`best +301.2 pts, spread 241.8`). The league figure
before the fix included materialisation because the timer started too early;
104.6 s / 64,000 games = 612 games/s, consistent with the bench.

**Plan on ~115 s/generation at pop=1000, K=64** — about 31 generations/hour.

### Cost model for planning a run

At ~580 games/s, one generation costs `pop × K / 580` seconds, where
`K = opponents × seeds × 2`:

| pop | K | games/gen | s/gen | gen/hour |
|----:|--:|----------:|------:|---------:|
| 500 | 64 | 32,000 | 55 | 65 |
| 1,000 | 64 | 64,000 | 110 | 33 |
| 1,000 | 128 | 128,000 | 221 | 16 |
| 1,000 | 256 | 256,000 | 441 | 8 |

Throughput is flat in `pop`, so the budget is spent purely on `pop × K`. The
K-sweep (`docs/decision-ranking-metric.md`) puts the gradient-stability knee at
K=128; below that K=64 still gives grad_ρ ≈ 0.81.

Scaling is flat (1,627 → 1,681 → 1,646 across 10x population growth) — the
lockstep architecture scales well. The 2x target (≥844 games/s at pop=1000)
is exceeded by 1.95x (1,646 / 844).
