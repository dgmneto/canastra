# Production training run

The measured config for training a real bot on this machine (RTX 5060 Ti,
16 GB, CUDA 13.3). Numbers come from `benchmarks.md` ("Phase 2 re-measured")
and `decision-ranking-metric.md`.

## Build

CUDA needs MSVC on PATH, so the build goes through `vcvars64.bat`:

```powershell
cd training-rs
cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && set NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor && set CUDA_COMPUTE_CAP=120 && cargo build --release --features cuda'
```

## Launch

```powershell
cd training-rs
.\target\release\canastra-train.exe `
  --generations 500 `
  --n-perturbations 500 `
  --opponents 4 `
  --seeds 8 `
  --device cuda `
  --run-seed 7 `
  --checkpoint-interval 10 `
  --anchor-interval 5 `
  --anchor-freeze-interval 50 `
  --hof-interval 10 `
  --hof-capacity 20 `
  --run-dir runs\prod-01
```

`--n-perturbations 500` means **population 1000** (mirrored θ±σε pairs).
K = `opponents × seeds × 2` = 64, so 64,000 games per generation. Measured at
**115 s/generation** end to end (104.6 s league at 612 games/s, plus ~10 s for
materialising the population, anchors and the Adam step) — about 31
generations/hour, so 500 generations ≈ **16 hours**.

## Resume

The run checkpoints every 10 generations and is bit-identical on resume:

```powershell
.\target\release\canastra-train.exe --generations 500 --resume runs\prod-01 --run-dir runs\prod-01 <same flags as above>
```

Pass the same hyper-parameters — `--resume` restores θ, the Adam moments, σ and
the hall of fame, but the flags still govern everything else.

## Where the config came from

**pop=1000 (500 perturbation pairs).** The genome is ~1.2M parameters, so 500
sampled directions is deeply under-determined either way, and ES's rank
normalisation makes it robust to noisy per-pair estimates — which is why
directions are worth more than precision per direction. pop=1000 is also the
largest measured-working population: `WeightStack::from_roster` has a transient
f32→bf16 cast spike that OOMs near pop=2000 on 16 GB
(`task4-ksweep.md` Task 4a).

**K=64 (opponents=4, seeds=8).** Throughput is flat in population, so the budget
is spent entirely on `pop × K` and the real choice is pop against K. At a fixed
64,000 games/generation the options are pop=1000/K=64 (500 directions,
grad_ρ ≈ 0.81) or pop=500/K=128 (250 directions, grad_ρ ≈ 0.89). More directions
wins in a 1.2M-dimensional space. If you would rather buy gradient quality than
directions, `--seeds 16` gives K=128 at 221 s/generation.

**BF16.** Measured within ±2% of F16 across pop 96/500/1000, and it carries
f32's exponent range. See `benchmarks.md`.

**`--hof-capacity 20`.** Every archived genome is cloned into the roster,
re-uploaded to the GPU each generation, and written into every checkpoint.
Unbounded at `--hof-interval 10` over 500 generations that would be 50 entries
(~240 MB per checkpoint, growing). Over capacity the archive is thinned to stay
spread across training history rather than collapsing onto recent generations —
which under ES are near-identical to current θ and useless as opponents.

**`--anchor-interval 5`.** The anchor rating is the only cross-generation
progress metric; the within-generation fitness is antisymmetric and says nothing
about absolute strength. It is cheap (a handful of games) but every 5
generations is plenty of resolution.

## Reading the run

Per-generation lines look like:

```
gen 12: best +412.5 pts (win 66%) spread 268.1 sigma 0.0189 (110.4s)
  anchor ELO: 1243.6
```

- `spread` — mean |paired differential| across the population. **If this
  collapses toward 0, stop**: the population has stopped differentiating and
  the gradient is noise. It is also the signal that caught the f16 masking bug.
- `anchor ELO` — the progress metric. Should trend up over tens of generations.
  Flat or cycling means the run is stuck.
- `best` / `win %` — the best perturbation this generation. Ephemeral; do not
  read a trend into it.

`generations.jsonl` in the run directory has the same data plus fitness
percentiles.

## Output

- `champion-final.json` — the trained policy, in `canastra-weights@1` format.
- `champion-gen<N>.json` — periodic snapshots.
- `gen-<N>.es.bin` — checkpoints (rotated: last 10 plus every 50th).

Play the result against a heuristic bot on the TS side, both seatings:

```bash
npx tsx harness/src/eval-nn.ts training-rs/runs/prod-01/champion-final.json random-plus 1
```
