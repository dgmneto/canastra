//! Sparse/dense column split for the observation vector.
//!
//! The observation is 2002-wide and 100% binary, but only 6.24% dense
//! (`docs/task3a-sparsity.md`). The trunk.0 dense matmul `[G, n_max, 2002] ×
//! [G, 2002, 512]` runs at ~1.5% of fp32 peak (small M, cuBLAS inefficiency).
//! Splitting trunk.0 into an embedding-bag (gather + segment-sum) over the
//! sparse block + a small dense matmul over the dense block eliminates the
//! dominant compute bottleneck.
//!
//! **Dense block (77 features):** short thermometer/bit segments where the
//! overhead of index lookup exceeds the savings from skipping zeros.
//! **Sparse block (1925 features):** one-hot/thermometer segments, mean 125
//! non-zeros/row, max 236. The `meld tokens` segment (583:2002, 1419 wide,
//! 4.1% density) is 71% of OBS_DIM and is the single biggest win.
//!
//! The sparse path pads indices to a fixed width of 256 (covers max 236
//! non-zeros) with a sentinel index (1925) pointing to a zero row in the
//! weight matrix, making the gather+sum correct without a validity mask.

use canastra_encode::OBS_DIM;
use std::sync::OnceLock;

/// Number of dense features (the genuinely dense block).
pub const DENSE_WIDTH: usize = 77;

/// Number of sparse features (everything not in the dense block).
pub const SPARSE_WIDTH: usize = 1925;

/// Total sparse width including the zero-padding sentinel row.
pub const SPARSE_PADDED_WIDTH: usize = SPARSE_WIDTH + 1; // 1926

/// Fixed-width index buffer per row (covers max 236 non-zeros, rounded up).
pub const MAX_NNZ_PER_ROW: usize = 256;

/// Dense column ranges in the 2002-wide observation, as (start, end) pairs.
const DENSE_RANGES: [(usize, usize); 7] = [
    (303, 339), // other hand counts, 36 wide
    (339, 350), // stock count, 11 wide
    (392, 395), // opening min, 3 wide
    (395, 397), // opened bits, 2 wide
    (397, 399), // clean canastra, 2 wide
    (399, 407), // red threes, 8 wide
    (460, 475), // pile size, 15 wide
];

/// Lookup table: for each obs column, the dense local index (0..76) or 255
/// if the column is sparse.
fn dense_local_table() -> [u8; OBS_DIM] {
    let mut table = [255u8; OBS_DIM];
    let mut dl = 0u8;
    for &(start, end) in &DENSE_RANGES {
        for entry in table.iter_mut().take(end).skip(start) {
            *entry = dl;
            dl += 1;
        }
    }
    assert_eq!(dl as usize, DENSE_WIDTH);
    table
}

/// Lookup table: for each obs column, the sparse local index (0..1924) or
/// 65535 if the column is dense.
fn sparse_local_table() -> [u16; OBS_DIM] {
    let dense = dense_local_table();
    let mut table = [65535u16; OBS_DIM]; // 65535 = dense sentinel
    let mut sl = 0u16;
    for col in 0..OBS_DIM {
        if dense[col] == 255 {
            table[col] = sl;
            sl += 1;
        }
    }
    assert_eq!(sl as usize, SPARSE_WIDTH);
    table
}

static DENSE_LOCAL: OnceLock<[u8; OBS_DIM]> = OnceLock::new();
static SPARSE_LOCAL: OnceLock<[u16; OBS_DIM]> = OnceLock::new();

/// Per-column dense local index (0..76), or 255 if sparse.
pub fn dense_local() -> &'static [u8; OBS_DIM] {
    DENSE_LOCAL.get_or_init(dense_local_table)
}

/// Per-column sparse local index (0..1924), or 65535 if dense.
pub fn sparse_local() -> &'static [u16; OBS_DIM] {
    SPARSE_LOCAL.get_or_init(sparse_local_table)
}

/// Convert a dense observation row (OBS_DIM f32 values, binary) to sparse
/// indices + dense values. The sparse indices are padded with the sentinel
/// index (SPARSE_WIDTH = 1925) pointing to a zero weight row.
pub fn dense_to_sparse_row(obs_f32: &[f32], sparse_idx_out: &mut [u16], dense_out: &mut [u8]) {
    assert_eq!(obs_f32.len(), OBS_DIM);
    assert_eq!(sparse_idx_out.len(), MAX_NNZ_PER_ROW);
    assert_eq!(dense_out.len(), DENSE_WIDTH);

    let dl = dense_local();
    let sl = sparse_local();

    dense_out.fill(0);
    let mut nnz = 0;
    for (col, &v) in obs_f32.iter().enumerate() {
        let d = dl[col];
        if d != 255 {
            dense_out[d as usize] = v as u8;
        } else if v != 0.0 && nnz < MAX_NNZ_PER_ROW {
            sparse_idx_out[nnz] = sl[col];
            nnz += 1;
        }
    }
    // Pad remaining slots with the sentinel (zero-row) index.
    for entry in sparse_idx_out.iter_mut().take(MAX_NNZ_PER_ROW).skip(nnz) {
        *entry = SPARSE_WIDTH as u16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_and_sparse_partition_is_complete() {
        let dl = dense_local();
        let n_dense = dl.iter().filter(|&&d| d != 255).count();
        assert_eq!(n_dense, DENSE_WIDTH);

        let sl = sparse_local();
        let n_sparse = sl.iter().filter(|&&s| s < SPARSE_WIDTH as u16).count();
        assert_eq!(n_sparse, SPARSE_WIDTH);

        // Every column is exactly one of dense or sparse.
        assert_eq!(n_dense + n_sparse, OBS_DIM);
    }

    #[test]
    fn sparse_local_indices_are_contiguous() {
        let sl = sparse_local();
        let mut indices: Vec<u16> = sl
            .iter()
            .filter(|&&s| s < SPARSE_WIDTH as u16)
            .copied()
            .collect();
        indices.sort();
        for (i, &idx) in indices.iter().enumerate() {
            assert_eq!(idx, i as u16, "sparse local index {} should be {}", idx, i);
        }
    }
}
