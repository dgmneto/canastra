# Decision: GA vs ES, lockstep vs coalesced — removal of the losing paths

Date: 2026-08-17. Context: productionizing `training-rs/` for the first real
training run after the code-review/hardening pass.

## TL;DR

The GA optimiser and the coalesced rollout were **removed**. The repo now has
one optimiser (ES) and one rollout (lockstep). The historical benchmark docs
(`benchmarks.md`, `profile-baseline.md`, `task4-ksweep.md`,
`task1-transfer-breakdown.md`, `task3a-sparsity.md`) are kept as the evidence
trail for *why*.

## Why ES over GA

Two structural wins, both documented in `task4-ksweep.md` Task 4a:

1. **Memory.** GA stores `G × genome_size × 4B` = 9.6 GB at pop=2000. ES stores
   a base policy (~4.8 MB) + `Vec<(seed, sigma)>` (kilobytes), materialising
   the population on demand. Populations of 10k+ become trivial.
2. **Noise tolerance.** Mirrored (antithetic) sampling + rank-normalised
   fitness shaping averages the gradient estimate over the whole population.
   Noisy per-genome fitness is fine because nothing is selected — everything
   is weighted by rank.

GA's VRAM cliff at pop=2000 (a transient f32→bf16 cast spike in
`WeightStack::from_roster`, diagnosed in `task4-ksweep.md` Task 4a — peak
12.3 GB for trunk.0 alone during the cast) was the hard wall. ES dissolves
it.

## Why lockstep over coalesced

`benchmarks.md` Phase 1b measured both paths at pop=1000:

| Path       | games/s | Speedup vs baseline |
|------------|--------:|--------------------:|
| lockstep   | 422     | 4.02x               |
| coalesced  | 276     | 2.63x               |

Lockstep is 1.53x faster at the production population. The coalesced path's
`GpuServer` + N worker threads + `CpuRoster` was the original architecture;
lockstep replaced it (one `Pool`, one forward per ply, no channels/mutex) and
won. `auto` used to pick coalesced above pop=500, but at pop=1000 lockstep is
both faster and measured-working — `auto` and `coalesced` were dead weight.

## Scaling wall (flagged, not silently deleted)

Lockstep was measured up to pop=1000 (422 games/s). Above that,
`task4-ksweep.md` Task 4a documents a transient VRAM allocation spike in
`WeightStack::from_roster`: it builds a CPU-side `Vec<f32>` of all weights per
layer (trunk.0 at pop=2000: 2000 × 1,025,024 × 4B = 8.2 GB), uploads as f32,
then casts to bf16 — the f32 (8.2 GB) and bf16 (4.1 GB) coexist briefly →
12.3 GB for trunk.0 alone, exceeding 16 GB with subsequent layers + activations.

The production config (pop=1000) stays well clear. A future run targeting
pop>2000 would need either:
- a coalesced/stream-overlap path restored, or
- the f32→bf16 upload de-spiked (e.g. cast in chunks, or upload as bf16
  directly via a fused path — see `task1-transfer-breakdown.md` "u8 upload"
  and `profile-baseline.md` Phase 1 findings).

This is a known wall, not a regression. It is flagged in `league.rs`'s module
doc and in `CLAUDE.md`'s `training-rs/` paragraph.

## What was removed

- `src/ga.rs` — deleted entirely (`GAConfig`, `initial_population`,
  `sigma_for`, `tournament`, `next_generation`, GA checkpoint
  save/load/rotation).
- `src/league.rs` — `Rollout` enum, `Rollout::default_for`, `GpuServer`,
  `drive_worker`, `rollout_coalesced`, `rollout_coalesced_public`,
  `EvalInputs.n_workers`, `EvalInputs.rollout`. `evaluate_generation` now
  calls `rollout_lockstep` directly.
- `src/policy.rs` — `CpuRoster` (struct + impl, including `build_chunk`).
- `src/anchors.rs` — `Rollout` import and the `rollout` parameter on
  `AnchorSet::evaluate` (now always lockstep).
- `src/bin/train.rs` — `--optimiser`, `--rollout`, `--workers`, `--elites`,
  `--tournament` flags; the entire GA branch (`run_es` is now the only path,
  inlined into `main`).
- `src/bin/bench.rs`, `src/bin/ksweep.rs` — `--rollout`, `--workers` flags;
  GA deps replaced with `genome::random_population`.
- `tests/equivalence.rs` — GA-specific tests removed (`elite_selection`,
  `mutation_distribution`, GA `checkpoint_resume`). Self-determinism test
  retargeted to league-only (ELO across rayon thread counts). ES checkpoint
  round-trip is covered by `es_checkpoint_round_trips` in `src/es.rs`.

## What was kept and why

- **`HallOfFame`** (moved `src/ga.rs` → `src/hof.rs`) — shared infrastructure
  used by ES (`es.rs`), the league (`league.rs`), and anchors (`anchors.rs`).
  Not optimizer-specific.
- **`genome::random_population`** (added to `src/genome.rs`) — the random-init
  routine the GA had as `initial_population`, now neutral infrastructure
  needed by the diagnostic bins (`bench`, `ksweep`) and not tied to any
  optimizer.
- **`ESConfig`, `ESState`, ES checkpoint save/load/rotation** (`src/es.rs`) —
  the surviving optimizer.
- **All `docs/*.md`** — historical evidence trail for the decisions above.
- **`src/bin/sparsity.rs`, `src/bin/h2d_bench.rs`** — neither exercises GA or
  coalesced; both are optimizer/rollout-agnostic diagnostics.
- **`--max-width`** — still load-bearing (cuts the `acts` tensor at peak
  plies, `benchmarks.md` Phase 1b).

## Gates

`cargo build --release`, `cargo clippy --all-targets -- -D warnings`,
`cargo fmt --check`, and `cargo test --test equivalence` (9/9 passed) all
green from `training-rs/`.
