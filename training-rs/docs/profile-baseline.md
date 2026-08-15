# Profile baseline (Phase 0)

Instrumentation: `#[cfg(feature = "profile")]` spans in `league.rs` (server thread
+ worker loop), accumulated into atomics, printed once per generation. `nsys`
is unavailable on this machine, so this is coarse `Instant::now()` attribution
plus a `CUDA_LAUNCH_BLOCKING=1` run to separate launch overhead from execution.

All runs: `opponents=4 seeds=8 max_hands=1 workers=8`, CUDA build (`--features
"cuda,profile"`), RTX 5060 Ti. `bench.rs` config; one generation per run.

## Results

| Pop | Games | Wall (s) | games/s | Server busy | Server idle | H2D (s) | GPU (s) | reqs | rows/req | plies |
|----:|------:|--------:|--------:|-----------:|------------:|--------:|--------:|-----:|---------:|-----:|
| 96  | 6,144 | 23.7    | 259     | 99.2%      | 0.0%        | 10.3    | 13.2    | 1530 | 449      | 1530 |
| 500 | 32,000| 257.8   | 124     | 99.6%      | 0.0%        | 70.4    | 186.4   | 1559 | 2331     | 1559 |

Worker side (summed across 8 workers; each worker's critical path ≈ wall):

| Pop | encode (s) | apply (s) | fwd-wait (s) | fwd-wait /worker |
|----:|----------:|----------:|-------------:|----------------:|
| 96  | 3.4       | 1.2       | 184.3        | 23.0s ≈ wall    |
| 500 | 15.4      | 6.1       | 2037.5       | 254.7s ≈ wall   |

`CUDA_LAUNCH_BLOCKING=1` run at pop=96 (isolates launch overhead from execution):

| Pop | Wall (s) | games/s | H2D (s) | GPU (s) | GPU/blocking vs async |
|----:|---------:|--------:|--------:|--------:|----------------------:|
| 96  | 47.1     | 130     | 10.2    | 36.7    | 36.7 vs 13.2 → 2.8x    |

## What this shows

1. **The server (GPU) thread is saturated, not starving.** Server busy is
   99.2–99.6% at both sizes; idle is effectively zero. Workers feed the GPU as
   fast as it can consume — the `Arc<Mutex>` channel round-trip is **not** the
   bottleneck. This **contradicts the brief's prediction** ("server idle >80%",
   "mean batch of ~8 rows").

2. **The mean batch is ~449 rows (pop 96) and ~2331 rows (pop 500), not ~8.**
   `drive_coalesced` already coalesces each worker's live games into one forward
   per ply, so a batch is hundreds-to-thousands of rows. The brief's "a batch is
   ~8 rows against a 1.2M-param network" premise is wrong.

3. **The negative scaling is in the forward pass, not the channel.** Per-forward
   GPU time grows from 8.6 ms (pop 96) to 120 ms (pop 500) — a 14x rise for a 5x
   population rise. The cause is `forward_picks`'s per-forward work:
   - `stack.slice(&present)` does an `index_select` gather over the full
     `[pop, out, in]` weight tensors on every forward (and again per 1024-row
     batch), re-transposes, and forces `.contiguous()`. This gather scales with
     population and is paid every ply.
   - `forward_scores_chunk` downloads the masked scores to CPU
     (`flatten_all().to_vec1::<f32>()`) for the argmax — a full device→host sync
     per batch, every ply. This is the "per-forward device sync from the CPU
     argmax" the brief names, and `CUDA_LAUNCH_BLOCKING` confirms it: blocking
     inflates GPU time 2.8x (13.2s → 36.7s), exposing the many small kernels
     (slices, transposes, per-layer bmms, mask arithmetic) the async queue hides
     behind the argmax sync.

4. **Engine stepping is negligible.** encode+apply is 4.6s summed (pop 96) and
   21.5s summed (pop 500) — i.e. ~0.6s and ~2.7s per worker on the critical path,
   well under 5% of wall. The brief's stop-gate ("engine stepping >50% of wall")
   does **not** trigger.

## Verdict

The pipeline is **GPU-forward-overhead-bound**, not latency/starving-bound and
not engine-bound. The brief's *symptom* (negative scaling, ~0.1% of the card) is
real, but the *mechanism* it hypothesised (idle GPU, ~8-row batches, channel
round-trip) is not what is happening. The real mechanism is:

> Each ply, `forward_picks` re-slices the cached weight stack with
> `index_select` (cost scales with population) and downloads scores to CPU for
> argmax (a device sync per batch). Both grow super-linearly with population and
> force many small kernels per ply.

Phase 1's fix addresses exactly this by a different route than the brief assumed:
the lockstep matmul reads weights once per genome per ply (no `index_select`
gather), one sync per ply (no per-batch CPU argmax), bf16 halves the weight
reads, and the embedding-bag input layer removes the dominant `2119×512` (here
`2002×512`) matmul entirely. The brief's design is still correct; only its
diagnosis of *why* was off. Proceeding to Phase 1.

## Build/run

```bash
cmd /c '"...\vcvars64.bat" ... && set CUDA_COMPUTE_CAP=120 && \
  cargo build --release --features "cuda,profile"'
target\release\canastra-bench.exe --population 500 --device cuda
```
