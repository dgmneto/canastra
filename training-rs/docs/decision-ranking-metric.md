# Decision — replacing ELO as the ES selection signal

## Summary

ELO was doing the wrong job in the training loop and has been replaced by the
**paired duplicate-deal score differential** (`src/fitness.rs`). ELO is retained
where it is genuinely the right tool: anchored evaluation (`src/anchors.rs`),
which accumulates a rating against *fixed, frozen* opponents across generations.

Two changes shipped together, plus one bug that the investigation uncovered:

1. **Metric** — fitness is now the mean paired score differential, not an ELO
   rating (`src/fitness.rs`).
2. **Matchmaking** — mirrored ES twins now share an opponent list, so `f⁺ − f⁻`
   is a like-for-like comparison (`league::schedule_pairings_mirrored`).
3. **Bug** — the recent `BF16 → F16` switch made the CUDA forward produce NaN
   scores for every legal action, collapsing all play to "take the first legal
   action". See [The f16 masking bug](#the-f16-masking-bug). Introduced in
   commit `942bbc0`; anything measured on CUDA after it is void, anything before
   it is fine.

## Why ELO was the wrong tool here

ELO earns its keep by accumulating a rating over a long history against
opponents whose strength is itself being estimated. The training loop met none
of those conditions.

**It was reset every generation.** `EloTracker::new(pop_size)` was constructed
inside the generation loop, so every genome started from a flat 1200 and played
~K games. Being zero-sum, the population mean is then pinned to *exactly* 1200
by construction — visible in `runs/es-smoke/generations.jsonl`, where `elo_mean`
reads `1200.0` for every generation of the run. `elo_best` and
`best_ever` were compared across generations despite being measured on a scale
that restarts each time, so the `improved` flag in the log was noise.

**It was order-dependent.** `batch_update` folds results in sequentially, so two
genomes with identical records get different ratings purely from the order their
games were scheduled. That is variance injected into fitness with nothing bought
in return.

**It discarded the margin.** Results were thresholded to win/draw/loss, throwing
away the point differential — the richest, lowest-variance signal Canastra
offers. Losing by 20 and losing by 800 were the same number.

**ES rank-normalises anyway.** `es::centred_ranks` reduces fitness to its
ordering before the gradient step, so the one thing ELO adds over a plain win
rate — scaling a result by opponent strength — is discarded downstream. Only
ELO's noise survived into the gradient.

**The duplicate deals were being thrown away.** `batch_layout` already plays
every deal twice with the seats swapped, and the project uses that pairing for
evaluation on both the Python and TypeScript sides (`sanity.py`, `eval-nn.ts`).
The training loop fed the two seatings to ELO as two *independent* games, which
discards the entire point of dealing them twice.

## What replaced it

For each pairing and deal, the two seatings are folded into one comparison:

```
diff(a, b, seed) = [ (score_a − score_b)│seating 0 + (score_a − score_b)│seating 1 ] / 2
```

The deal's luck enters the two seatings with opposite sign and cancels to first
order. `fitness[i]` is the mean of `diff` over every comparison genome `i` took
part in. It is antisymmetric, so the population mean sits at zero and a positive
value means "beats its opponents on shared deals". Heavy tails (§13.3's flat
−300, a 1000-point ace canastra) need no special handling, because ES consumes
ranks rather than magnitudes.

The halving makes the unit "mean points per game", matching
`evaluate::PairReport::mean_diff` so a training log and an A/B evaluation are
directly comparable.

**The project already had this.** `src/evaluate.rs` — "Duplicate-deal paired
evaluation" — has computed exactly this quantity in Rust the whole time, and has
**no callers**. The right metric was written, then the training loop reached for
ELO instead. `evaluate.rs` is kept rather than merged because the two have
different shapes: it owns a rollout and compares two specific genomes with a 95%
CI, while `fitness::score_generation` is a pure fold over a whole generation's
results.

## Mirrored common random numbers

ES's gradient is `Σⱼ (f⁺ⱼ − f⁻ⱼ) εⱼ`, so its variance is driven by
`Var(f⁺ − f⁻)`. Conditions the twins *share* cancel out of that difference;
conditions that differ survive as noise. `es.rs` claimed this cancellation, but
`schedule_pairings` drew a fresh opponent list per genome, so the twins were
measured against different opponents and the cancellation never happened. At
σ=0.02 the twins are near-identical policies, so the opponent draw dominated.

`schedule_pairings_mirrored` gives genome `2j` and `2j+1` the same opponent
list. Combined with the shared deal seeds, the only difference between the
twins' conditions is the sign of `εⱼ` — exactly the quantity being measured.

## Measurements

`canastra-ksweep` computes both metrics from the *same* games, so the
difference is purely how results are folded into a per-genome number. Three runs
over different deal seeds; ρ is Spearman between runs. RTX 5060 Ti, CUDA,
pop=96, `max_width=64`, `max_hands=1`.

### Independent random genomes (σ=0) — the historical Task 4 setting

| K | games/s | elo ρ12 | elo ρ13 | diff ρ12 | diff ρ13 |
|---:|--------:|--------:|--------:|---------:|---------:|
| 64 | 228 | 0.826 | 0.835 | 0.795 | 0.874 |
| 128 | 398 | 0.826 | 0.836 | 0.823 | 0.863 |

Roughly a tie. With wildly different random genomes the signal is strong enough
that both metrics find it — which is why Task 4's original sweep saw ρ≈0.85 and
concluded ELO was "already stable enough". **That conclusion did not transfer to
the regime training actually runs in.**

### ES-shaped population (σ=0.02) — the real training regime

| K | elo ρ12 | elo ρ13 | diff ρ12 | diff ρ13 |
|---:|--------:|--------:|---------:|---------:|
| 64 | 0.434 | 0.388 | 0.588 | 0.559 |
| 128 | 0.549 | 0.520 | 0.695 | 0.658 |
| 256 | 0.558 | 0.517 | 0.793 | 0.778 |

Three things to read off this:

- ELO is far worse here (0.39–0.56) than on a random population (0.83). Task 4's
  caveat that its finding might not survive as genomes converge was correct, and
  ES converges them *by construction* — every genome is θ±σε off one base.
- The differential beats ELO at every K, by 0.15–0.24 absolute.
- **ELO plateaus at ~0.55 by K=128 and stops improving** (0.549 → 0.558), while
  the differential keeps converting games into signal (0.588 → 0.695 → 0.793).
  ELO has a noise floor — its own order-dependence and margin-discarding — that
  more compute cannot buy through.

The differential at K=64 already matches ELO at K=128, so the metric change
alone is worth roughly 2× the games.

### Mirrored CRN, measured on the quantity it targets

Population-wide ρ is the wrong instrument for CRN — it can look stable while
every *pairwise* difference is noise. `grad_ρ` correlates the per-pair
`f⁺ − f⁻` across deal seeds, which is what the gradient is built from.

| K | grad ρ (CRN on) | grad ρ (CRN off) |
|---:|----------------:|-----------------:|
| 64 | 0.858 / 0.758 | 0.702 / 0.782 |
| 128 | 0.936 / 0.852 | 0.801 / 0.839 |
| 256 | 0.950 / 0.925 | 0.858 / 0.905 |

A consistent +0.06–0.07 at every K. CRN at K=64 is about level with CRN-off at
K=128 — another ~2× on the gradient estimate, for no extra compute.

Read this one more cautiously than the metric comparison above: it is 48 pairs
per cell, and the two ρ estimates within a cell differ by as much as 0.10
(0.858 vs 0.758 at K=64), so the per-cell noise is the same order as the effect.
What carries it is that the sign is the same in all six cells and the gap does
not shrink with K. The theoretical case is the stronger argument — `f⁺ − f⁻` is
a difference of two measurements, and sharing conditions between them removes
variance that is otherwise irreducible — and the measurement is consistent with
it rather than proof on its own.

## The f16 masking bug

Found while trying to reproduce the K-sweep: on CUDA, **all 6144 games ended
level at −300 to −300** — §13.3's "never opened" penalty for both partnerships —
while the identical genomes on CPU scored −1015 to +2480.

`forward_pass*` masked illegal actions with `scores + (1 - mask) * (-1e9)`,
computed **in the stack dtype**. That was safe while the CUDA dtype was `BF16`,
which carries f32's exponent range: `-1e9` is finite, and `(1 - mask) * -1e9` is
`0` for legal actions.

Commit `942bbc0` switched the CUDA dtype to `F16` for its faster tensor cores on
Blackwell. `F16` tops out at 65504, so `-1e9` overflows to `-inf` on the cast.
For every **legal** action `(1 - mask)` is `0`, making the offset
`0 × -inf = NaN`. One NaN per legal action poisons the whole score row; `argmax`
over an all-NaN row returns index 0; every policy degenerates to "always take
the first legal action", never melds, never meets §6's opening minimum, and both
partnerships take the flat −300. Every game a draw, fitness uniformly zero, ES
gradient exactly zero.

The dtype switch and the masking sentinel are a hundred lines apart and nothing
connected them — the change was reasoned about purely as a throughput knob.

It stayed invisible because:

- **CPU is unaffected** — dtype `F32`, where `-1e9` is finite and `0 × -1e9 = 0`.
- **No test reached it.** Every test path builds its stack through
  `single_genome_weights`, which hard-codes `DType::F32`. `F16` is reached only
  from `league::rollout_lockstep` — the production training path and nothing
  else. `benchmarks.md` states this outright while arguing the switch was safe:
  "the F16 path's correctness is verified by the existing forward-pass test
  (CPU F32 vs Python) and the GPU BF16 vs CPU agreement test". Neither test runs
  in F16. The gap was written down; it just did not read as a gap.
- **It made the GPU look fast.** Degenerate games end almost immediately, and
  the resulting 3.6–3.9x jump was recorded as the payoff for the switch. A
  correctness bug that *raises* your headline metric is one nobody goes looking
  for.

**Fix:** `policy::mask_illegal_f32` does the masking in f32 and returns f32. The
callers cast to f32 for the argmax on the very next line regardless, so this
costs nothing. Three duplicated copies of the block collapsed into it.

**Blast radius:** the bug was introduced in `942bbc0` and is bounded by it.

- Void: the "Phase 2 — u8 transfer + F16 dtype" table in `benchmarks.md`, and
  any CUDA run made between `942bbc0` and this fix.
- Fine: everything BF16-era — the Baseline and Phase 1b benchmark tables,
  `task4-ksweep.md`, and `runs/es-smoke` (whose ELO spread of 1006–1385 shows a
  live signal; had the bug been active every rating would read exactly 1200).
- Never affected: all CPU runs and the whole test suite.

**Regression cover:** `tests/fitness_signal.rs` asserts that games produce
non-level scores, and runs on CUDA under `FITNESS_SIGNAL_DEVICE=cuda`:

```
$env:FITNESS_SIGNAL_DEVICE="cuda"
cargo test --release --features cuda --test fitness_signal -- --ignored --nocapture
```

## What was kept

- `elo.rs` is untouched. The Python-equivalence test pins its arithmetic
  (max-abs-diff 2.3e-13) and `anchors.rs` uses it for what it is good at.
- `league::evaluate_generation` still exists with the same signature, because
  `tests/equivalence.rs` pins its output against `gen_elo_after.json`. It is now
  a thin wrapper over `play_generation` + `elo_updates`; training calls
  `play_generation` + `fitness::score_generation` instead.
- HOF entries and frozen anchors are now archived with the **anchor** rating
  rather than the internal self-play ELO, which was on a scale that reset every
  generation and therefore meant nothing as a frozen reference.

## Two smaller corrections in the same area

**Anchors now rate the base policy θ, not `pop[champion]`.** The freeze site
already argued that "the base policy θ is the actual policy being trained; the
champion is an ephemeral probe whose advantage is mostly noise" — and then the
evaluation site rated the champion anyway. Rating a different random
perturbation every generation injects exactly that noise into the one metric
meant to be clean. The anchor evaluation also moved to *before* `es_state.update`
so that "generation N's anchor rating" is the θ that produced generation N's
population.

**`improved` / `best_ever` now track the anchor rating.** They previously
tracked `elo_best`, and my first pass moved them to `fitness_best` — which has
the identical defect: fitness is antisymmetric and re-measured against a fresh
population each generation, so it says how much the twins differ, not how strong
they are. Only the anchor rating is comparable across generations, so only it
can support a claim of improvement. Generations without an anchor evaluation now
make no claim either way.

## Follow-ups worth considering

**A fixed shared opponent set.** The opponent list is still a random draw from
the population, so CRN holds within a mirrored pair but not across pairs. Making
every genome play the *same* opponents — the base policy θ plus HOF entries —
would extend CRN population-wide, so `f_i − f_j` becomes a paired comparison for
any i,j rather than only for twins. The `grad_ρ` numbers above suggest that is
where the remaining variance lives. It is a bigger change to the training regime
than a metric swap (the population would no longer play each other at all), so
it is left as a separate decision.

**~~Re-measure the Phase 2 benchmarks.~~ Done** — see `benchmarks.md`, "Phase 2
re-measured". F16 and BF16 came out within ±2% of each other, so the dtype
switch that introduced the masking bug bought nothing; **BF16 is now the
default** (`league::default_dtype`) because it carries f32's exponent range and
cannot reproduce that class of overflow. `canastra-bench` prints the level-game
percentage and mean |differential| next to games/s, so a degenerate forward can
no longer be recorded as a speedup.

**Consider whether `max_hands=1` is the right training unit.** With a one-hand
cap, ~5% of games end level on CPU and both partnerships frequently fail to
open, which compresses the signal. Full matches separate far more strongly
(mean |diff| ~7000 vs ~780 in `tests/fitness_signal.rs`) at proportionally more
compute. Not investigated here.
