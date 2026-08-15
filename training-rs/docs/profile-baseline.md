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

---

# Phase 1 findings (lockstep rewrite) — below the 10x gate, stopped to report

Phase 1 deleted `GpuServer`/channels/workers and replaced them with a single
`Pool` driving all games in lockstep: one forward per ply, weights read once
per present genome, GPU argmax (deterministic first-max via an index penalty),
bf16 on CUDA, fp32 on CPU, genome-chunked to bound activation memory. The
per-forward `index_select` weight re-slicing and the per-batch CPU-argmax sync
(the Phase-0 bottleneck) are gone. All correctness tests pass (CPU exact, GPU
bf16 ≥98% argmax agreement, self-determinism across rayon thread counts).

But the **realised speedup is well below the brief's 10x gate**, and pop=1000
**regressed** below baseline:

| Pop | Baseline games/s | Phase-1 games/s | Ratio |
|----:|----------------:|----------------:|------:|
| 96  | 259              | 253             | 0.98x |
| 500 | 124              | 198             | 1.60x |
| 1000| 105              | >1500s (timeout)| <0.07x (regression) |

Scaling did **not** flatten (253 → 198 → timeout); the primary acceptance
criterion (`games/s at pop=1000 ≥ games/s at pop=96`) is **not met**.

## Why — three things the brief's diagnosis did not account for

1. **candle's batched matmul is inefficient for the grouped small-M shapes.**
   The lockstep grouped bmm is `[G, n_max, obs] × [G, obs, hidden]` with
   batch=G, **M=n_max=64** (games per genome). cuBLAS with M=64 is severely
   underutilised: the forward runs at ~350 GFLOP/s ≈ **1.5% of fp32 peak**
   (measured via the argmax sync absorbing async matmul time; the async matmul
   spans report ~0.5s but the real execution fills ~117s of the argmax wait at
   pop=500/2GB-chunks). bf16 is faster than fp32 but both are slow — this is a
   candle-backend small-M issue, not a precision issue. The trunk.0 input
   layer (2002×512, 85% of params, M=64) is the worst offender.

2. **The `acts` tensor dominates data movement and the PCIe link is slow.**
   `acts` is `[N, width, ACT_DIM]` = `[64K, 200, 101]` f32 = **5.2 GB/ply at
   pop=1000**, uploaded every ply. Measured host→device throughput is ~3.7 GB/s
   (likely a limited PCIe link on this laptop GPU — RTX 5060 Ti mobile often
   runs x4). That is a **311 s floor at pop=1000** before any compute. bf16
   upload would halve the bandwidth, but the CPU-side cast path in candle
   (`from_vec(f32, Cpu).to_dtype(BF16).to_device`) adds copies that made it
   *slower* (146 s vs 88 s at pop=500) — a fused/async bf16 upload is needed.

3. **The lockstep sacrifices encode/apply overlap.** The old coalesced design
   had 8 workers with separate game sets: while the GPU forwarded one worker's
   batch, the others encoded/applied in parallel — encode+apply hid behind the
   449 s GPU bottleneck. The single-Pool lockstep is strictly sequential per ply
   (encode→upload→forward→apply), so encode (30 s) + apply (13 s) at pop=1000
   land on the critical path with no overlap. The old design was GPU-bound; the
   new one is H2D+encode-bound, and at pop=1000 the H2D floor alone (311 s)
   approaches the old total (608 s).

## What would actually unlock the speedup

- **Embedding-bag input layer (Phase 1d) — now load-bearing, not optional.**
  The obs is 100% binary (verified: one-hot/thermometer/census, no continuous
  features). Replacing the trunk.0 dense `[G, 64, 2002]×[G, 2002, 512]` (M=64,
  the inefficient matmul) with a sparse gather+sum over active features makes it
  a big-M=N op and removes the dominant inefficient matmul. This is the single
  highest-leverage change, but needs a fused gather+segment-sum candle doesn't
  cleanly expose (a `[N, active, hidden]` materialisation is too large).
- **Reduce `acts` volume:** upload bf16 via a fused path, and/or stop padding to
  the global max width (encode in genome-grouped grid layout so the gather is a
  no-op reshape). The 5.2 GB/ply at pop=1000 is the floor.
- **Overlap H2D / encode with GPU** via CUDA streams (candle's sync `from_vec`
  blocks; needs stream-aware upload or a double-buffered pipeline).

## Recommendation

The lockstep architecture is correct and the per-forward overhead is genuinely
eliminated (GPU forward dropped 3.6x at pop=500: 186 s → 52 s). But without 1d
(embedding-bag), fused bf16 H2D, and stream overlap, the lockstep **regresses
pop=1000** because it trades a GPU-bound pipeline for an H2D+encode-bound one on
hardware with a slow PCIe link. Three options:

1. **Implement 1d + fused bf16 H2D + stream overlap** before committing Phase 1
   (deeper candle work; targets the real bottlenecks).
2. **Keep the old coalesced design** and only port the GPU argmax + bf16 +
   smaller chunks (incremental, lower risk, ~1.5-2x, no regression).
3. **Accept Phase 1 as-is for pop ≤ 500** (1.6x) and move to Phases 2–3
   (variance/anchoring), returning to throughput later.

Awaiting direction.

