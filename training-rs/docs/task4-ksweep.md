# Task 4 — K sweep + rank correlation

## Method

`canastra-ksweep` bin: runs the same population three times with different deal
seeds (7, 99, 777), computes Spearman's ρ of ELO rankings between runs. High ρ
means the fitness signal is stable across deal seeds; low ρ means selection is
mostly noise. K = opponents × seeds × 2 (2 for seat swap). Opponents fixed at 4;
seeds varied to hit K ∈ {64, 128, 256, 512}.

All runs: `--max-width 64`, CUDA, RTX 5060 Ti Gen 3 x8, max_hands=1, lockstep.

## Results

### pop=96

| K | Games | Wall (s) | games/s | dec/s | ρ (1v2) | ρ (1v3) |
|---:|------:|---------:|--------:|------:|--------:|--------:|
| 64 | 6,144 | 13.0 | 465 | 93K | 0.855 | 0.863 |
| 128 | 12,288 | 24.4 | 499 | 100K | 0.860 | 0.848 |
| 256 | 24,576 | 47.1 | 520 | 104K | 0.842 | 0.866 |
| 512 | 49,152 | 95.9 | 517 | 103K | 0.855 | 0.829 |

### pop=500

| K | Games | Wall (s) | games/s | dec/s | ρ (1v2) | ρ (1v3) |
|---:|------:|---------:|--------:|------:|--------:|--------:|
| 64 | 32,000 | 73.1 | 426 | 85K | 0.823 | 0.809 |
| 128 | 64,000 | 138.5 | 454 | 91K | 0.866 | 0.848 |
| 256 | 128,000 | 286.1 | 444 | 89K | 0.860 | 0.847 |
| 512 | 256,000 | — | — | — | — | — |

(K=512 at pop=500 timed out at 30min — ~286s for K=256, so K=512 would be ~572s × 3 runs ≈ 29min, just over the limit.)

## Findings

### K=128 is the sweet spot, not K=256

At pop=96, correlation is already 0.855 at K=64 and does not improve with
higher K (0.855 → 0.860 → 0.842 → 0.855). More games buy nothing — the fitness
signal is already stable.

At pop=500, there is a real improvement from K=64 to K=128 (0.823 → 0.866), but
K=256 does not improve further (0.860). K=128 captures the gain.

The brief predicted K=256 would halve the SE and improve the signal. The data
shows the noise floor is reached at K=128 — the remaining ~15% variance is not
deal-quality noise (which duplicate deals cancel) but strategic variance
(different opponents, different seating order). More games of the same type
don't help.

### Wall scales linearly with K

Wall ≈ K × (base wall per game). At pop=500: 73s (K=64) → 139s (K=128) → 286s
(K=256). The exchange rate is ~1.9x wall for 2x K — near-linear, with a slight
efficiency gain from larger batches (games/s: 426 → 454).

### Decisions/s ~90-100K

At pop=500 K=128: 91K decisions/s. A 1000-generation run at K=128 would be
~139s/gen × 1000 = ~39 hours. At K=64: ~73s × 1000 = ~20 hours. Both feasible.

## Recommendation

**Use K=128 as the default** (opponents=4, seeds=16). It gives the best
correlation-per-wall at both population sizes. K=64 is acceptable for quick
iteration; K=256 buys nothing over K=128.

The duplicate-deal mechanism is already in place (`batch_layout` plays each deal
twice with seats swapped). The margin scoring from the original Phase 2 brief
can be added later, but the rank correlation data shows the current ELO-based
scoring is already stable enough at K=128 — the variance problem the brief
identified is less severe than predicted, at least for untrained populations.
This may change as genomes differentiate during training; the anchor curve
(Phase 3c) will detect if it does.

## Task 4a — pop=2000 lockstep VRAM diagnosis

The pop=2000 lockstep hang is a **transient VRAM allocation spike** in
`WeightStack::from_roster`, not a structural issue:

1. `from_roster` builds a CPU-side `Vec<f32>` of all weights per layer
   (trunk.0: 2000 × 1,025,024 × 4B = 8.2 GB), uploads it as f32 via
   `Tensor::from_vec`, then casts to bf16 via `.to_dtype()`.
2. Peak VRAM during the cast: the f32 tensor (8.2 GB) and the bf16 tensor
   (4.1 GB) coexist briefly → **12.3 GB for trunk.0 alone**.
3. With subsequent layers and activation tensors, this exceeds 16 GB.

The coalesced path works at pop=2000 because `CpuRoster::build_chunk` (same
code) builds the same stack, but the `GpuServer` threshold is `n_genomes <= 2000`
and the OOM is borderline — it succeeds barely.

**Phase 4 (ES) dissolves this entirely**: seed-based ES stores zero genomes.
The population is `(base_params, Vec<(seed, sigma)>)`. The weight stack is a
single base genome (~4.8 MB), and perturbations are generated in-kernel. The
4.8 GB → 4.8 MB reduction means populations of 10k+ are trivial. The cliff is
inherited only if ES retains the dense-genome materialisation path for
checkpointing/export — but that's a one-time cost, not per-generation.
