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
