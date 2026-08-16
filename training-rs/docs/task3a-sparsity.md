# Task 3a — Observation sparsity measurement

## Method

`canastra-sparsity` bin: plays 2000 single-hand games with random actions,
collects 10,000 observation vectors (OBS_DIM=2002), and reports density,
binary fraction, non-zeros per row, run structure, and per-segment density.

## Results

```
OBS_DIM:              2002
Total elements:       20,020,000
Total non-zeros:      1,249,394
Density (mean):       6.24%
Binary non-zeros:     1,249,394 (100.0%)

Non-zeros per row:
  min:   50
  p25:   91
  p50:   128
  p75:   157
  p95:   188
  p99:   204
  max:   236
  mean:  124.9

Contiguous runs of non-zeros: 721,901
Mean run length:              1.7
Max run length:               23
Distinct non-binary values:   0
```

## Gate: PASSED

Density is **6.24%** (well below the 10% gate). 100% binary (every non-zero
is exactly 1.0, zero non-binary values). Mean 125 non-zeros per 2002-dim row,
max 236.

## Structure: scattered one-hot blocks, not contiguous

Mean run length is 1.7 — non-zeros are scattered, not in one large contiguous
block. Max run is 23 (a one-hot block within a segment, e.g. a 13-rank
one-hot plus a 4-suit one-hot plus thermometers). The layout is a sequence of
small one-hot/thermometer segments, each contributing a few non-zeros.

## Per-segment density (first 100 rows)

```
         phase one-hot [   0:   5] (   5 wide): density   20.0%  mean_nz   1.0
      laid_value therm [   5:  18] (  13 wide): density    5.0%  mean_nz   0.7
             took_pile [  18:  19] (   1 wide): density    1.0%  mean_nz   0.0
     refusal_available [  19:  20] (   1 wide): density    4.0%  mean_nz   0.0
          pending card [  20:  73] (  53 wide): density    0.0%  mean_nz   0.0
           hand_number [  73:  79] (   6 wide): density    0.0%  mean_nz   0.0
        my hand census [  79: 183] ( 104 wide): density   10.6%  mean_nz  11.1
        my hand jokers [ 183: 187] (   4 wide): density    5.8%  mean_nz   0.2
         frozen census [ 187: 295] ( 108 wide): density    0.0%  mean_nz   0.0
          my hand size [ 295: 303] (   8 wide): density    2.2%  mean_nz   0.2
     other hand counts [ 303: 339] (  36 wide): density   43.5%  mean_nz  15.7
           stock count [ 339: 350] (  11 wide): density   53.5%  mean_nz   5.9
              my score [ 350: 370] (  20 wide): density    0.0%  mean_nz   0.0
           their score [ 370: 390] (  20 wide): density    0.0%  mean_nz   0.0
           >=2500 bits [ 390: 392] (   2 wide): density    0.0%  mean_nz   0.0
           opening min [ 392:  395] (   3 wide): density   33.3%  mean_nz   1.0
           opened bits [ 395:  397] (   2 wide): density   81.0%  mean_nz   1.6
        clean canastra [ 397:  399] (   2 wide): density   44.0%  mean_nz   0.9
            red threes [ 399: 407] (   8 wide): density   42.5%  mean_nz   3.4
              pile top [ 407: 460] (  53 wide): density    1.8%  mean_nz   0.9
             pile size [ 460: 475] (  15 wide): density   49.4%  mean_nz   7.4
           pile census [ 475: 583] ( 108 wide): density   14.6%  mean_nz  15.8
           meld tokens [ 583:2002] (1419 wide): density    4.1%  mean_nz  58.0
```

## Design implications for Task 3b

### Dense segments (keep as small dense matmul)

- `other hand counts` (303:339, 36 wide, 43.5%): a thermometer of 12 thresholds
  × 3 seats. Dense enough to keep dense.
- `stock count` (339:350, 11 wide, 53.5%): a thermometer of 11 thresholds.
- `pile size` (460:475, 15 wide, 49.4%): a thermometer of 15 thresholds.
- `opened bits` (395:397, 2 wide, 81%): two bits.
- `clean canastra` (397:399, 2 wide, 44%): two bits.
- `red threes` (399:407, 8 wide, 42.5%): two 4-thermometers.
- `opening min` (392:395, 3 wide, 33.3%): one-hot.

Total dense block: 36+11+15+2+2+8+3 = **77 features** (3.9% of OBS_DIM). These
are short thermometer/bit segments where the overhead of index lookup exceeds
the savings from skipping zeros.

### Sparse segments (embedding-bag)

- `meld tokens` (583:2002, 1419 wide, 4.1%): the largest segment, 71% of
  OBS_DIM. Mean 58 non-zeros out of 1419. **This is the primary embedding-bag
  target** — it alone cuts 1419 × 4B = 5.7 KB/row to ~58 × 2B = 116 B/row
  (49x cut).
- `my hand census` (79:183, 104 wide, 10.6%): mean 11 non-zeros.
- `pile census` (475:583, 108 wide, 14.6%): mean 16 non-zeros.
- `pending card` (20:73, 53 wide, 0.0%): one-hot, usually zero.
- `pile top` (407:460, 53 wide, 1.8%): one-hot, usually zero.
- `frozen census` (187:295, 108 wide, 0.0%): usually zero.
- `laid_value therm` (5:18, 13 wide, 5%): short thermometer.
- `phase one-hot` (0:5, 5 wide, 20%): one-hot, 1 non-zero.

### Split design

Expected outcome per the brief: embedding-bag over the sparse block (1925
features, ~6% density), small dense matmul over the dense block (77 features),
sum the results. The sparse block's 1419-wide meld tokens segment is the single
biggest win — replacing the 2002×512 dense matmul with a 1925-wide embedding-bag
+ 77-wide dense matmul.

### Index buffer sizing

Max non-zeros per row = 236. A fixed-width u16 index buffer of 256 entries
(rounded up from 236) per row covers all cases: 256 × 2B = 512 B/row vs 2002 ×
4B = 8008 B/row — a **15.6x cut** on the obs transfer volume.

## Verdict

Density 6.24%, 100% binary, scattered one-hot structure, max 236 non-zeros/row.
Gate passed. Proceeding to Task 3b (unfused embedding-bag fallback).
