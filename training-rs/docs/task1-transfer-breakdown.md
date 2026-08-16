# Task 1 — Per-ply transfer breakdown

## Method

Instrumented `pool.encode()` behind `#[cfg(feature = "profile")]` to print
`n_rows`, `width`, `mean_menu`, `max_menu`, `total_items`, `acts_bytes`, and
`obs_bytes` per ply. Ran the lockstep benchmark at pop=96 and pop=500
(max_hands=1, opponents=4, seeds=8). Checked the PCIe link via
`nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current,...`.

## PCIe link — the root cause

```
pcie.link.gen.current = 1
pcie.link.width.current = 8
pcie.link.gen.max = 3
pcie.link.width.max = 8
```

The link is negotiated at **Gen 1 x8** (250 MB/s per lane × 8 = **2 GB/s
theoretical**, ~1 GB/s practical with pageable memory). The GPU supports up to
**Gen 3 x8** (8 GB/s theoretical, ~6 GB/s with pinned memory). This is a
mobile/laptop RTX 5060 Ti with a limited PCIe link, not the desktop PCIe 5.0 x8
the brief assumed (25–30 GB/s).

**This is a configuration issue, not a fundamental hardware limit.** Forcing
Gen 3 would 4x the bandwidth; pinned memory would add another ~1.5x. Combined:
~6x, turning the 88s H2D floor into ~15s.

## Byte-level breakdown (pop=500, representative large ply)

| Tensor | Shape | Dtype | Bytes | Per-row | % of ply |
|--------|-------|-------|------:|--------:|---------:|
| `obs`  | [32000, 2002] | f32 | 256 MB | 8.0 KB | 35% |
| `acts` | [32000, 37, 101] | f32 | 478 MB | 15.0 KB | 65% |
| `mask` | [32000, 37] | u32 | 5 MB | 148 B | <1% |
| **Total** | | | **739 MB** | **23 KB** | 100% |

(Width=37 is the global max menu size at pop=500 with max_hands=1. The reference
`forward_meta.json` showed width=203, but that was from a *full game* — with
max_hands=1, menus are smaller.)

## Where the 5.2 GB went wrong

The earlier "5.2 GB/ply" estimate assumed `width=200` (from the reference data).
Actual width at pop=500/max_hands=1 is **37**. The per-ply transfer is ~740 MB,
not 5.2 GB. The 81 KB/row the brief computed was `200 × 101 × 4 = 80.8 KB` for
acts alone — with the real width=37, acts are `37 × 101 × 4 = 15.0 KB/row`.

However, the **cumulative** transfer across all 222 plies is large: 3.6M total
rows × 23 KB/row ≈ **84 GB**. At 1 GB/s (Gen 1 + pageable), that is **84s ≈ the
measured 88.5s H2D**. The bandwidth wall is real, just at a different scale than
assumed.

## Reducibility analysis

### 1. Width padding (2.5x waste on acts)

`width` is the **global max** menu size across all rows. The mean menu is ~15,
but all rows pad to max=37. This wastes `(37-15)/37 ≈ 60%` of the acts tensor.
A jagged/per-row-width layout would cut acts by ~2.5x. Not trivial in the
current tensor layout but possible.

### 2. Binary features → u8 (4x cut, free)

Both `obs` and `acts` are **100% binary** (one-hot/thermometer/bits — verified
in Phase 0 by reading the encoder source). Uploading as `u8` and casting
device-side is a 4x cut on the entire transfer with no precision loss:
740 MB → 185 MB per ply, 84 GB → 21 GB total.

### 3. Static data re-uploaded (none found)

The weight stack is already resident (built once per generation). The obs/acts/
mask are genuinely per-ply data — nothing static is being re-uploaded. The
`mask` tensor is technically derivable from `acts` (zero rows = illegal), but
it's only 5 MB (<1%) — not worth optimising.

### 4. Acts constructible device-side? (partially)

The action encoding is derived from the `PlayerView` + the legal action list.
The observation already encodes the view. If the action features could be
reconstructed from the observation + a compact action descriptor (kind, card,
meld target — ~3 bytes per action), the upload would be `32000 × 15 × 3 = 1.4
MB` instead of 478 MB — a 340x cut. This is a deeper redesign (Task 3 territory)
and not needed if u8 + width de-padding suffice.

### 5. Per-candidate observation expansion? (no)

The observation is encoded **once per state** (`encode_observation`), not once
per action candidate. The `acts` tensor is the per-candidate expansion, but it
encodes *action features*, not the observation. No redundancy found here.

## Irreducible minimum

With the current encoder and no architectural change:
- **u8 upload + device cast**: 84 GB → 21 GB. At Gen 3 + pinned (6 GB/s): 3.5s.
- **+ width de-padding (2.5x on acts)**: 21 GB → 12 GB. At 6 GB/s: 2.0s.
- **+ sparse obs (Task 3, ~85x on obs)**: 12 GB → 5 GB. At 6 GB/s: 0.8s.

The **irreducible minimum** without a device-side action encoder is ~5 GB
(sparse obs indices + compact acts). At Gen 3 + pinned, that's <1s — no longer
the bottleneck.

## Verdict

The 5.2 GB/ply was wrong (actual ~740 MB/ply), but the cumulative 84 GB at 1
GB/s (Gen 1 PCIe) is the real wall. Three fixes, in priority order:

1. **Fix the PCIe link to Gen 3** (BIOS/driver) — 4x bandwidth, 0 code change.
2. **Pinned memory** (Task 2) — 1.5x on top of Gen 3.
3. **u8 upload for binary features** — 4x data volume cut, trivial to implement.
4. **Sparse input layer** (Task 3) — 85x on obs, the biggest single cut.

The transfer is **not** irreducible — 90%+ is reducible with known techniques.
Proceeding to Task 2 (pinned memory + PCIe link fix).

---

# Revision — per-tensor timing reveals the real bottleneck

After the microbenchmark (all approaches at 7 GB/s, no pinning/stream benefit)
and per-tensor H2D timing in the training pipeline, the picture changed again:

## PCIe link is fine

- **Gen 1 at idle is normal** — the link negotiates to **Gen 3 x8 under load**
  (confirmed by polling `nvidia-smi -l 1` during the microbenchmark). No
  configuration issue.
- **Max is Gen 3 x8** — the platform (motherboard/slot) caps at Gen 3 despite
  the Blackwell GPU being natively Gen 5. This permanently limits the ceiling
  to ~8 GB/s. The brief's 15 GB/s acceptance criterion is **not achievable on
  this hardware**.
- **Pinned = pageable = pre-alloc = transfer-stream: all 7.1 GB/s** for 256 MB
  transfers. No approach improves raw bandwidth. `has_async_alloc: true`.

## The `acts` tensor is the real wall — 94% padding

Per-tensor timing at pop=500 (32,000 rows/ply):

| Ply | obs (MB) | obs time | acts (MB) | acts time | mask (MB) | width |
|----:|---------:|---------:|----------:|----------:|----------:|------:|
| 1   | 256      | 43 ms    | 12        | 2 ms      | 0         | 1     |
| 51  | 255      | 44 ms    | **1833**  | **604 ms**| 18        | ~142  |
| 101 | 206      | 36 ms    | **3237**  | **859 ms**| 32        | ~250  |
| 151 | 9        | 2 ms     | 151       | 38 ms     | 1         | ~12   |

`width` is the **global max menu size** across all 32,000 rows. It grows from
1 (draw phase) to **250** (mid-game melding phase) as players accumulate cards
and meld options. With a mean menu of ~15 actions, width=250 means **94% of the
acts tensor is padding zeros**. At ply 101, the acts tensor is **3.2 GB** — a
single tensor upload taking 859 ms (3.8 GB/s, half the link speed due to the
massive allocation).

The `acts` bandwidth (3.0-3.8 GB/s) is lower than `obs` bandwidth (5.8-6.0
GB/s) despite larger transfers — likely because `cuMemAllocAsync` for 1.8-3.2
GB triggers memory pool pressure or fragmentation.

## Revised breakdown

The original 23 KB/row estimate was wrong because it used the early-game width
(37). At mid-game, the per-row transfer is:

| Tensor | Per-row (width=250) | Per-row (width=15, mean) |
|--------|-------------------:|-------------------------:|
| obs    | 8.0 KB             | 8.0 KB                   |
| acts   | **101 KB**         | 6.1 KB                   |
| mask   | 1.0 KB             | 0.06 KB                  |
| Total  | **110 KB**         | 14.2 KB                  |

At width=250, the acts tensor dominates (92% of transfer volume). The obs is
only 7% — making the sparse input layer (Task 3) a secondary concern compared
to acts width de-padding.

## Revised reducibility

1. **Width de-padding (17x on acts)**: upload only each row's actual menu
   entries, not the global-max-padded dense tensor. This is the single biggest
   cut — from 3.2 GB to ~0.2 GB per peak ply. Requires a jagged layout or
   per-row encoding.
2. **u8 for binary features (4x on everything)**: both obs and acts are 100%
   binary. Upload as u8, cast device-side. Free, no precision loss.
3. **Sparse input layer (85x on obs)**: obs is 100% binary and sparse (~2%
   density). Embedding-bag cuts obs from 256 MB to ~3 MB. Secondary to acts.
4. **Pinned memory + streams**: no benefit measured (7.1 GB/s either way). The
   link is at Gen 3 x8 ceiling. Skip.

## Revised irreducible minimum

With width de-padding + u8 + sparse obs:
- obs: 3 MB/ply (sparse indices, u16)
- acts: ~0.2 GB/ply (mean 15 actions × 101 features × u8 = 1.5 KB/row × 32K = 48 MB)
- Total: ~51 MB/ply, ~11 GB total. At 7 GB/s: **1.6s**. Not the bottleneck.

The brief's 15 GB/s criterion is not achievable (Gen 3 x8 caps at 8 GB/s), but
with the data volume cuts, the H2D drops from 88s to <2s regardless. Proceeding
to Task 3a (sparsity measurement) ahead of u8, as instructed.
