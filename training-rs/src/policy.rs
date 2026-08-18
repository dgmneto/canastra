//! Batched policy scoring via candle (pure Rust ML with CUDA via cudarc).
//!
//! Trunk layers all tanh, head hidden layers tanh, final layer linear. The
//! forward is a **lockstep grouped matmul**: every live row in a ply is gathered
//! into a `[G, n_max, ...]` grid (grouped by genome), then one batched matmul
//! per layer reads each genome's weights exactly once. There is one sync
//! point per ply (the argmax download), not thousands.
//!
//! Memory is bounded by chunking over the `n_max` (games-per-genome) axis:
//! the grid is processed `K` rows per genome at a time, so peak activation
//! memory is `[G, K, width, ...]` regardless of `n_max`. The weight stack stays
//! resident (no per-forward `index_select` gather); only present genomes are
//! forwarded, sliced once per ply.
//!
//! Precision: the stack and activations are bf16 on CUDA (tensor cores, half
//! the bandwidth), fp32 on CPU (exact, for the equivalence tests). Argmax is
//! done in fp32 — on CPU it is an exact first-max loop (matches numpy); on GPU
//! it uses a monotonically decreasing index penalty so ties break to the first
//! (lowest-index) action, making the pick deterministic.

use crate::genome::{Arch, Genome};
use crate::sparse::{DENSE_WIDTH, MAX_NNZ_PER_ROW, SPARSE_PADDED_WIDTH, SPARSE_WIDTH};
use candle_core::{DType, Device, Tensor};

#[cfg(feature = "profile")]
use crate::profile;

/// Per-layer weight tensors, stacked on a leading G (genome) dimension.
/// Built once per generation in the chosen dtype (bf16 on CUDA, f32 on CPU)
/// and kept resident for every ply. `slice` gathers the present genomes once
/// per ply — the only weight copy, and a small one relative to the matmuls.
pub struct WeightStack {
    pub trunk_w: Vec<Tensor>, // [G, out, in] — all trunk layers including trunk.0
    pub trunk_b: Vec<Tensor>, // [G, out]
    pub head_w: Vec<Tensor>,  // [G, out, in]
    pub head_b: Vec<Tensor>,  // [G, out]
    // Sparse components for the embedding-bag path (built from trunk_w[0]).
    pub trunk_w0_sparse_flat: Tensor,
    pub trunk_w0_dense: Tensor,
    // ES grouped-GEMM split for trunk.0. Under ES, genomes are θ±σε pairs.
    // The base GEMM (obs @ θ^T) runs at M=G*n_max (efficient), and the
    // perturbation bmm (obs @ (σε)^T) has n_pairs groups (half the FLOPs).
    pub trunk_w0_base: Option<Tensor>, // [1, 512, OBS_DIM] — shared base weight
    pub trunk_w0_pert: Option<Tensor>, // [n_pairs, 512, OBS_DIM] — per-pair perturbation
    pub trunk_b0_base: Option<Tensor>, // [1, 512] — shared base bias
    pub trunk_b0_pert: Option<Tensor>, // [n_pairs, 512] — per-pair perturbation bias
    pub n_pairs: usize,                // 0 if not ES
    pub device: Device,
    pub dtype: DType,
}

impl WeightStack {
    /// Stack the roster's flat genomes into per-layer `[G, out, in]` tensors.
    /// `dtype` is bf16 for CUDA training, f32 for CPU correctness tests.
    /// Also builds the sparse split of trunk.0 (embedding-bag + dense) from
    /// the same genome data, avoiding a separate GPU-side column gather.
    pub fn from_roster(roster: &[&Genome], arch: &Arch, device: &Device, dtype: DType) -> Self {
        let g = roster.len();
        let shapes = layer_shapes_owned(arch);

        let mut trunk_w = Vec::new();
        let mut trunk_b = Vec::new();
        let mut head_w = Vec::new();
        let mut head_b = Vec::new();
        let mut offset = 0usize;

        // The first trunk layer (trunk.0) is split into sparse + dense for the
        // embedding-bag path. We build the sparse flat weight directly from
        // the genome data: for each genome, gather the 1925 sparse columns
        // (transposed to [in, out] = [1925, 512]) plus a zero-padding row,
        // then flatten to [G * 1926, 512]. The dense columns are gathered
        // into [G, 512, 77] for a small bmm.
        let dense_local = crate::sparse::dense_local();
        let sparse_cols: Vec<usize> = (0..arch.obs).filter(|&c| dense_local[c] == 255).collect();
        let dense_cols: Vec<usize> = (0..arch.obs).filter(|&c| dense_local[c] != 255).collect();
        debug_assert_eq!(sparse_cols.len(), SPARSE_WIDTH);
        debug_assert_eq!(dense_cols.len(), DENSE_WIDTH);

        let mut trunk_w0_sparse_flat: Option<Tensor> = None;
        let mut trunk_w0_dense: Option<Tensor> = None;

        for (name, out, inn) in &shapes {
            let w_size = out * inn;
            let mut w_data = Vec::with_capacity(g * w_size);
            let mut b_data = Vec::with_capacity(g * out);
            for genome in roster {
                w_data.extend_from_slice(&genome[offset..offset + w_size]);
                b_data.extend_from_slice(&genome[offset + w_size..offset + w_size + out]);
            }
            offset += w_size + out;

            let w = Tensor::from_vec(w_data, (g, *out, *inn), device)
                .unwrap_or_else(|e| panic!("weight tensor: {e}"))
                .to_dtype(dtype)
                .unwrap_or_else(|e| panic!("weight cast: {e}"));
            let b = Tensor::from_vec(b_data, (g, *out), device)
                .unwrap_or_else(|e| panic!("bias tensor: {e}"))
                .to_dtype(dtype)
                .unwrap_or_else(|e| panic!("bias cast: {e}"));

            if name == "trunk.0" {
                // Build the sparse flat weight [G * SPARSE_PADDED_WIDTH, out]
                // by gathering the sparse columns from each genome's weight,
                // transposed to [in, out] layout, plus a zero sentinel row.
                // Genome stores [out, in] row-major: w[o * in + i].
                // Sparse gather: for sparse feature j (local), obs_col = sparse_cols[j],
                // weight column = w[o * in + obs_col] for o in 0..out.
                // We want [SPARSE_PADDED_WIDTH, out] per genome: [1925, 512] + [1, 512] zeros.
                let mut sparse_data = Vec::with_capacity(g * SPARSE_PADDED_WIDTH * out);
                for genome in roster {
                    let gw = &genome[offset - w_size - out..offset - out]; // trunk.0 weight
                    for &obs_col in &sparse_cols {
                        for o in 0..*out {
                            sparse_data.push(gw[o * inn + obs_col]);
                        }
                    }
                    // Zero-padding sentinel row.
                    sparse_data.resize(sparse_data.len() + *out, 0.0f32);
                }
                trunk_w0_sparse_flat = Some(
                    Tensor::from_vec(sparse_data, (g * SPARSE_PADDED_WIDTH, *out), device)
                        .unwrap_or_else(|e| panic!("sparse weight tensor: {e}"))
                        .to_dtype(dtype)
                        .unwrap_or_else(|e| panic!("sparse weight cast: {e}")),
                );

                // Build the dense weight [G, out, DENSE_WIDTH] by gathering
                // the dense columns. Layout: [out, DENSE_WIDTH] per genome
                // (same [out, in] orientation as the original, just fewer columns).
                let mut dense_data = Vec::with_capacity(g * out * DENSE_WIDTH);
                for genome in roster {
                    let gw = &genome[offset - w_size - out..offset - out];
                    for o in 0..*out {
                        for &obs_col in &dense_cols {
                            dense_data.push(gw[o * inn + obs_col]);
                        }
                    }
                }
                trunk_w0_dense = Some(
                    Tensor::from_vec(dense_data, (g, *out, DENSE_WIDTH), device)
                        .unwrap_or_else(|e| panic!("dense weight tensor: {e}"))
                        .to_dtype(dtype)
                        .unwrap_or_else(|e| panic!("dense weight cast: {e}")),
                );
            }

            if name.starts_with("trunk") {
                trunk_w.push(w);
                trunk_b.push(b);
            } else {
                head_w.push(w);
                head_b.push(b);
            }
        }

        // ES grouped-GEMM split: if the roster has an even number of genomes
        // (ES materialises pairs: genome 2j = θ+σε, 2j+1 = θ-σε), compute the
        // shared base weight and per-pair perturbation weights. The forward
        // can then split trunk.0 into a base GEMM (large M, efficient) + a
        // perturbation bmm (half the FLOPs, since ε is computed once per pair).
        let (trunk_w0_base, trunk_w0_pert, trunk_b0_base, trunk_b0_pert, n_pairs) =
            if g >= 2 && g.is_multiple_of(2) {
                let n_pairs = g / 2;
                let trunk_w0 = &trunk_w[0]; // [G, 512, OBS_DIM]
                let trunk_b0 = &trunk_b[0]; // [G, 512]

                // Base: average of genome 0 and 1 (a mirrored pair → θ).
                let base_w = trunk_w0
                    .narrow(0, 0, 2)
                    .unwrap()
                    .mean(0)
                    .unwrap() // [512, OBS_DIM]
                    .unsqueeze(0)
                    .unwrap(); // [1, 512, OBS_DIM]
                let base_b = trunk_b0
                    .narrow(0, 0, 2)
                    .unwrap()
                    .mean(0)
                    .unwrap()
                    .unsqueeze(0)
                    .unwrap(); // [1, 512]

                // Perturbation: genome[2j] - base = σε_j for each pair j.
                let pert_indices: Vec<u32> = (0..n_pairs).map(|j| (2 * j) as u32).collect();
                let pert_idx_t = Tensor::from_vec(pert_indices, (n_pairs,), device).unwrap();
                let pert_full = trunk_w0.index_select(&pert_idx_t, 0).unwrap(); // [n_pairs, 512, OBS_DIM]
                let pert_w = pert_full
                    .sub(&base_w.broadcast_as(pert_full.shape()).unwrap())
                    .unwrap(); // [n_pairs, 512, OBS_DIM]
                let pert_b_full = trunk_b0.index_select(&pert_idx_t, 0).unwrap();
                let pert_b = pert_b_full
                    .sub(&base_b.broadcast_as(pert_b_full.shape()).unwrap())
                    .unwrap(); // [n_pairs, 512]

                (
                    Some(base_w),
                    Some(pert_w),
                    Some(base_b),
                    Some(pert_b),
                    n_pairs,
                )
            } else {
                (None, None, None, None, 0)
            };

        WeightStack {
            trunk_w,
            trunk_b,
            head_w,
            head_b,
            trunk_w0_sparse_flat: trunk_w0_sparse_flat
                .expect("trunk.0 sparse weight was not built"),
            trunk_w0_dense: trunk_w0_dense.expect("trunk.0 dense weight was not built"),
            trunk_w0_base,
            trunk_w0_pert,
            trunk_b0_base,
            trunk_b0_pert,
            n_pairs,
            device: device.clone(),
            dtype,
        }
    }

    /// Slice to only the genomes present this ply (one gather per layer, once).
    pub fn slice(&self, present: &[usize]) -> Self {
        let idx = Tensor::from_vec(
            present.iter().map(|&i| i as u32).collect::<Vec<_>>(),
            present.len(),
            &self.device,
        )
        .unwrap_or_else(|e| panic!("slice index: {e}"));

        // For trunk_w0_sparse_flat [G * SPARSE_PADDED_WIDTH, 512], select rows
        // for present genomes: each genome occupies SPARSE_PADDED_WIDTH rows.
        let sparse_idx: Vec<u32> = present
            .iter()
            .flat_map(|&g| {
                let base = (g * SPARSE_PADDED_WIDTH) as u32;
                (0..SPARSE_PADDED_WIDTH as u32).map(move |j| base + j)
            })
            .collect();
        let sparse_idx_t = Tensor::from_vec(
            sparse_idx,
            (present.len() * SPARSE_PADDED_WIDTH,),
            &self.device,
        )
        .unwrap_or_else(|e| panic!("sparse slice index: {e}"));

        WeightStack {
            trunk_w: self
                .trunk_w
                .iter()
                .map(|w| w.index_select(&idx, 0).unwrap())
                .collect(),
            trunk_b: self
                .trunk_b
                .iter()
                .map(|b| b.index_select(&idx, 0).unwrap())
                .collect(),
            head_w: self
                .head_w
                .iter()
                .map(|w| w.index_select(&idx, 0).unwrap())
                .collect(),
            head_b: self
                .head_b
                .iter()
                .map(|b| b.index_select(&idx, 0).unwrap())
                .collect(),
            trunk_w0_sparse_flat: self
                .trunk_w0_sparse_flat
                .index_select(&sparse_idx_t, 0)
                .unwrap(),
            trunk_w0_dense: self.trunk_w0_dense.index_select(&idx, 0).unwrap(),
            // ES split: base is shared (no slicing). Perturbation is sliced
            // to present pairs. If n_pairs == 0, ES split is not available.
            trunk_w0_base: self.trunk_w0_base.clone(),
            trunk_w0_pert: if self.n_pairs > 0 {
                self.trunk_w0_pert.clone()
            } else {
                None
            },
            trunk_b0_base: self.trunk_b0_base.clone(),
            trunk_b0_pert: if self.n_pairs > 0 {
                self.trunk_b0_pert.clone()
            } else {
                None
            },
            n_pairs: self.n_pairs,
            device: self.device.clone(),
            dtype: self.dtype,
        }
    }

    pub fn n_groups(&self) -> usize {
        self.trunk_w[0].dims()[0]
    }
}

/// (name, out, in) for every layer, in genome order.
fn layer_shapes_owned(arch: &Arch) -> Vec<(String, usize, usize)> {
    let mut result = Vec::new();
    let mut prev = arch.obs;
    for (i, &width) in arch.trunk.iter().enumerate() {
        result.push((format!("trunk.{}", i), width, prev));
        prev = width;
    }
    prev += arch.act;
    for (i, &width) in arch.head.iter().enumerate() {
        result.push((format!("head.{}", i), width, prev));
        prev = width;
    }
    result.push(("head.out".to_string(), 1, prev));
    result
}

/// Stacked batched matmul: x [G, N, in] × w [G, out, in]^T → [G, N, out]
fn stacked_bmm(x: &Tensor, w: &Tensor) -> Tensor {
    let w_t = w.transpose(1, 2).unwrap().contiguous().unwrap(); // [G, in, out]
    let x = x.contiguous().unwrap_or_else(|_| x.clone());
    x.matmul(&w_t).unwrap()
}

/// Result of a forward pass: per-row argmax picks and the full masked score
/// matrix `[n_rows, width]` in original row order (padded columns = -1e9).
/// `scores_flat` is only downloaded by `forward_scores`; the training hot path
/// uses `forward_picks`, which argmaxes on-device and downloads only the picks.
pub struct ForwardOutput {
    pub picks: Vec<usize>,
    pub scores_flat: Vec<f32>,
    pub width: usize,
}

/// Per-ply head activation memory budget (bytes). The forward chunks over
/// `n_max` so peak activation memory is `[G, K, width, 128]` with
/// `G*K*width*128*dtype ≤ BUDGET`. 2 GB leaves room for the weight stack
/// (≤ 4.8 GB bf16 at pop 2000) and the per-chunk input gather within 16 GB.
const HEAD_ACTIVATION_BUDGET: usize = 200_000_000;

/// Sparse gather intermediate budget (bytes). The embedding-bag path
/// materialises `[g, n_max, MAX_NNZ, 512]` during the gather+sum. This budget
/// bounds that intermediate so it stays within VRAM alongside the weight
/// stack. At 1 GB and n_max=64/bf16, this allows ~60 genomes per chunk.
const SPARSE_GATHER_BUDGET: usize = 1_000_000_000;

fn dtype_size(dt: DType) -> usize {
    match dt {
        DType::BF16 | DType::F16 => 2,
        DType::F32 => 4,
        DType::F64 => 8,
        _ => 4,
    }
}

/// Shape of one forward chunk — bundled to keep `forward_pass` under the arg limit.
struct ChunkDims {
    n_present: usize,
    k: usize,
    width: usize,
    act_dim: usize,
}

/// The unified lockstep forward. Returns picks AND the full score matrix
/// (downloaded) — used by the correctness tests. The training hot path calls
/// [`forward_picks`] instead (argmax on-device, no score download).
///
/// `obs` is `[N, OBS_DIM]` f32, `acts` is `[N, width, ACT_DIM]` f32, `mask` is
/// `[N, width]` u32 (1 = legal), `genome_idx` is which genome owns each row.
/// Inputs are cast to the stack's dtype; argmax is done in f32.
pub fn forward_scores(
    stack: &WeightStack,
    obs: &Tensor,
    acts: &Tensor,
    mask: &Tensor,
    genome_idx: &[usize],
) -> ForwardOutput {
    let n_rows = obs.dims()[0];
    let width = acts.dims()[1];
    let act_dim = acts.dims()[2];

    if n_rows == 0 {
        return ForwardOutput {
            picks: Vec::new(),
            scores_flat: Vec::new(),
            width: 0,
        };
    }

    let (grid, ply_stack) = build_grid(stack, obs, acts, mask, genome_idx);
    let mut picks = vec![0usize; n_rows];
    let mut scores_flat = vec![-1e9f32; n_rows * width];

    for chunk in grid.chunks() {
        let out = forward_chunk(ply_stack.as_ref(), &grid, &chunk, width, act_dim, true);
        scatter(
            &grid,
            &chunk,
            &out.picks,
            &out.scores_flat,
            width,
            &mut picks,
            &mut scores_flat,
        );
    }

    ForwardOutput {
        picks,
        scores_flat,
        width,
    }
}

/// Forward + argmax, returning picks only. The training hot path: argmax on
/// the device (one small download per ply), no full score download.
pub fn forward_picks(
    stack: &WeightStack,
    obs: &Tensor,
    acts: &Tensor,
    mask: &Tensor,
    genome_idx: &[usize],
) -> Vec<usize> {
    let n_rows = obs.dims()[0];
    let width = acts.dims()[1];
    let act_dim = acts.dims()[2];

    if n_rows == 0 {
        return Vec::new();
    }

    let (grid, ply_stack) = build_grid(stack, obs, acts, mask, genome_idx);
    let mut picks = vec![0usize; n_rows];

    for chunk in grid.chunks() {
        let out = forward_chunk(ply_stack.as_ref(), &grid, &chunk, width, act_dim, false);
        scatter_picks(&grid, &chunk, &out.picks, &mut picks);
    }

    picks
}

// ─── ES grouped-GEMM forward path ────────────────────────────────────────
//
// Under ES, genomes are θ±σε pairs (genome 2j = θ+σε, 2j+1 = θ-σε). The
// trunk.0 dense bmm [G, n_max, OBS] × [G, OBS, 512] runs at M=n_max=64 —
// ~1.5% of fp32 peak. The ES split restructures this as:
//   1. Base GEMM: obs @ θ^T — one standard matmul with M=G*n_max (64000+),
//      ~50% of peak. Nearly free.
//   2. Perturbation bmm: obs_pair @ (σε)^T — grouped bmm with n_pairs
//      groups (half the FLOPs since ε is computed once per pair, not once
//      per genome). M=2*n_max=128 per group.
//   3. Sign: first n_max rows of each pair +1, last n_max rows -1.
//   4. emb = emb_base + sign * emb_pert + bias. Then tanh.
//
// The remaining trunk layers and head use the per-genome weights (same as
// the dense path). Only trunk.0 is split — it's the dominant cost (85% of
// params, 2002×512 vs 512×256 for trunk.1).

/// Forward + argmax via the ES grouped-GEMM split. Falls back to the dense
/// path if the stack has no ES split or if a chunk has an odd genome count.
pub fn forward_picks_es(
    stack: &WeightStack,
    obs: &Tensor,
    acts: &Tensor,
    mask: &Tensor,
    genome_idx: &[usize],
) -> Vec<usize> {
    if stack.n_pairs == 0 {
        return forward_picks(stack, obs, acts, mask, genome_idx);
    }

    let n_rows = obs.dims()[0];
    let width = acts.dims()[1];
    let act_dim = acts.dims()[2];

    if n_rows == 0 {
        return Vec::new();
    }

    let (grid, ply_stack) = build_grid(stack, obs, acts, mask, genome_idx);
    let mut picks = vec![0usize; n_rows];

    for chunk in grid.chunks() {
        let out = if chunk.g_count % 2 == 0 {
            forward_chunk_es(ply_stack.as_ref(), &grid, &chunk, width, act_dim, false)
        } else {
            forward_chunk(ply_stack.as_ref(), &grid, &chunk, width, act_dim, false)
        };
        scatter_picks(&grid, &chunk, &out.picks, &mut picks);
    }

    picks
}
//
// The sparse path replaces trunk.0's dense `[G, n_max, 2002] × [G, 2002, 512]`
// matmul (which cuBLAS executes at ~1.5% of fp32 peak due to small M=n_max=64)
// with:
//   1. An embedding-bag (gather + segment-sum) over the 1925 sparse features.
//      Each row has ~125 non-zeros (max 236), so we gather ~125 rows of W and
//      sum — this is memory-bound, not compute-bound, and much faster than the
//      dense matmul at small M.
//   2. A small dense `[G, n_max, 77] × [G, 77, 512]` bmm for the 77 genuinely
//      dense features (thermometers/bit segments).
//   3. Sum the two + bias + tanh, then continue with trunk.1, head, etc.
//
// The sparse indices are padded to a fixed width of 256 with a sentinel index
// (1925) pointing to a zero weight row, making the gather+sum correct without
// a validity mask. The sparse weight is stored as [G * 1926, 512] (transposed,
// flattened) for a single index_select per chunk.

/// Forward + argmax via the sparse embedding-bag path. The training hot path.
///
/// `obs_sparse_idx` is `[N, MAX_NNZ]` u16 (sparse local indices, padded with
/// SPARSE_WIDTH), `obs_dense` is `[N, DENSE_WIDTH]` u8 (binary dense features),
/// `acts` is `[N, width, ACT_DIM]` u8, `mask` is `[N, width]` u8.
pub fn forward_picks_sparse(
    stack: &WeightStack,
    obs_sparse_idx: &Tensor,
    obs_dense: &Tensor,
    acts: &Tensor,
    mask: &Tensor,
    genome_idx: &[usize],
) -> Vec<usize> {
    let n_rows = obs_sparse_idx.dims()[0];
    let width = acts.dims()[1];
    let act_dim = acts.dims()[2];

    if n_rows == 0 {
        return Vec::new();
    }

    let (grid, ply_stack) =
        build_grid_sparse(stack, obs_sparse_idx, obs_dense, acts, mask, genome_idx);
    let mut picks = vec![0usize; n_rows];

    for chunk in grid.chunks() {
        let out = forward_chunk_sparse(ply_stack.as_ref(), &grid, &chunk, width, act_dim, false);
        scatter_picks(&grid, &chunk, &out.picks, &mut picks);
    }

    picks
}

/// Forward + argmax via the sparse path, returning picks AND the full score
/// matrix (downloaded). Used by the sparse correctness tests.
pub fn forward_scores_sparse(
    stack: &WeightStack,
    obs_sparse_idx: &Tensor,
    obs_dense: &Tensor,
    acts: &Tensor,
    mask: &Tensor,
    genome_idx: &[usize],
) -> ForwardOutput {
    let n_rows = obs_sparse_idx.dims()[0];
    let width = acts.dims()[1];
    let act_dim = acts.dims()[2];

    if n_rows == 0 {
        return ForwardOutput {
            picks: Vec::new(),
            scores_flat: Vec::new(),
            width: 0,
        };
    }

    let (grid, ply_stack) =
        build_grid_sparse(stack, obs_sparse_idx, obs_dense, acts, mask, genome_idx);
    let mut picks = vec![0usize; n_rows];
    let mut scores_flat = vec![-1e9f32; n_rows * width];

    for chunk in grid.chunks() {
        let out = forward_chunk_sparse(ply_stack.as_ref(), &grid, &chunk, width, act_dim, true);
        scatter(
            &grid,
            &chunk,
            &out.picks,
            &out.scores_flat,
            width,
            &mut picks,
            &mut scores_flat,
        );
    }

    ForwardOutput {
        picks,
        scores_flat,
        width,
    }
}
/// iterator over **genomes**. The forward is chunked over the genome axis so
/// peak activation memory is `[g, n_max, width, ...]` regardless of population
/// — a single `[G, n_max, ...]` grid at pop 1000 would be ~5 GB of acts alone
/// and blow past 16 GB. The weight stack is sliced to the present genomes once
/// per ply; each genome-chunk re-slices that to its `g` genomes (a small gather).
fn build_grid<'s>(
    stack: &'s WeightStack,
    obs: &Tensor,
    acts: &Tensor,
    mask: &Tensor,
    genome_idx: &[usize],
) -> (Grid, PlyStack<'s>) {
    let n_groups = stack.n_groups();
    let width = acts.dims()[1];

    let mut present_set = vec![false; n_groups];
    for &g in genome_idx {
        present_set[g] = true;
    }
    let present: Vec<usize> = (0..n_groups).filter(|&g| present_set[g]).collect();
    let n_present = present.len();

    let mut genome_to_pos = vec![usize::MAX; n_groups];
    for (i, &g) in present.iter().enumerate() {
        genome_to_pos[g] = i;
    }

    let mut counts = vec![0usize; n_present];
    for &g in genome_idx {
        counts[genome_to_pos[g]] += 1;
    }
    let n_max = counts.iter().copied().max().unwrap_or(0).max(1);

    let mut row_order = vec![u32::MAX; n_present * n_max];
    let mut pos = vec![0usize; n_present];
    for (row, &g) in genome_idx.iter().enumerate() {
        let p = genome_to_pos[g];
        row_order[p * n_max + pos[p]] = row as u32;
        pos[p] += 1;
    }

    // Genome-chunk size g: bound head.0 output [g, n_max, width, 128].
    let bytes_per_genome = n_max * width * 128 * dtype_size(stack.dtype);
    let g = (HEAD_ACTIVATION_BUDGET / bytes_per_genome.max(1))
        .max(1)
        .min(n_present);

    let ply_stack = if n_present < n_groups {
        PlyStack::Owned(stack.slice(&present))
    } else {
        PlyStack::Borrowed(stack)
    };

    let grid = Grid {
        n_present,
        n_max,
        g,
        row_order,
        counts,
        obs_src: obs.clone(),
        obs_sparse_src: Tensor::new(0u32, &stack.device).unwrap(),
        obs_dense_src: Tensor::new(0u8, &stack.device).unwrap(),
        acts_src: acts.clone(),
        mask_src: mask.clone(),
        device: stack.device.clone(),
    };
    (grid, ply_stack)
}

/// Build the grid for the sparse path. Same chunking logic as `build_grid`
/// but with an additional budget for the sparse gather intermediate.
fn build_grid_sparse<'s>(
    stack: &'s WeightStack,
    obs_sparse_idx: &Tensor,
    obs_dense: &Tensor,
    acts: &Tensor,
    mask: &Tensor,
    genome_idx: &[usize],
) -> (Grid, PlyStack<'s>) {
    let n_groups = stack.n_groups();
    let width = acts.dims()[1];

    let mut present_set = vec![false; n_groups];
    for &g in genome_idx {
        present_set[g] = true;
    }
    let present: Vec<usize> = (0..n_groups).filter(|&g| present_set[g]).collect();
    let n_present = present.len();

    let mut genome_to_pos = vec![usize::MAX; n_groups];
    for (i, &g) in present.iter().enumerate() {
        genome_to_pos[g] = i;
    }

    let mut counts = vec![0usize; n_present];
    for &g in genome_idx {
        counts[genome_to_pos[g]] += 1;
    }
    let n_max = counts.iter().copied().max().unwrap_or(0).max(1);

    let mut row_order = vec![u32::MAX; n_present * n_max];
    let mut pos = vec![0usize; n_present];
    for (row, &g) in genome_idx.iter().enumerate() {
        let p = genome_to_pos[g];
        row_order[p * n_max + pos[p]] = row as u32;
        pos[p] += 1;
    }

    // Genome-chunk size: bounded by both the head activation and the sparse
    // gather intermediate (the embedding-bag materialises [g, n_max, 256, 512]).
    let bytes_per_genome_head = n_max * width * 128 * dtype_size(stack.dtype);
    let g_head = (HEAD_ACTIVATION_BUDGET / bytes_per_genome_head.max(1))
        .max(1)
        .min(n_present);
    let bytes_per_genome_sparse = n_max * MAX_NNZ_PER_ROW * 512 * dtype_size(stack.dtype);
    let g_sparse = (SPARSE_GATHER_BUDGET / bytes_per_genome_sparse.max(1))
        .max(1)
        .min(n_present);
    let g = g_head.min(g_sparse);

    let ply_stack = if n_present < n_groups {
        PlyStack::Owned(stack.slice(&present))
    } else {
        PlyStack::Borrowed(stack)
    };

    let grid = Grid {
        n_present,
        n_max,
        g,
        row_order,
        counts,
        obs_src: Tensor::new(0f32, &stack.device).unwrap(),
        obs_sparse_src: obs_sparse_idx.clone(),
        obs_dense_src: obs_dense.clone(),
        acts_src: acts.clone(),
        mask_src: mask.clone(),
        device: stack.device.clone(),
    };
    (grid, ply_stack)
}

/// The stack for this ply: either the original (all genomes present) or a
/// once-sliced copy (only present genomes).
enum PlyStack<'s> {
    Borrowed(&'s WeightStack),
    Owned(WeightStack),
}

impl PlyStack<'_> {
    fn as_ref(&self) -> &WeightStack {
        match self {
            PlyStack::Borrowed(s) => s,
            PlyStack::Owned(s) => s,
        }
    }
}

/// The per-ply grid: indexing data + genome-chunk iterator.
struct Grid {
    n_present: usize,
    n_max: usize,
    g: usize,               // genomes per chunk
    row_order: Vec<u32>,    // [n_present, n_max] global row indices (u32::MAX = padding)
    counts: Vec<usize>,     // rows per present genome
    obs_src: Tensor,        // dense path: [N, OBS_DIM]
    obs_sparse_src: Tensor, // sparse path: [N, MAX_NNZ] u16
    obs_dense_src: Tensor,  // sparse path: [N, DENSE_WIDTH] u8
    acts_src: Tensor,
    mask_src: Tensor,
    device: Device,
}

impl Grid {
    fn chunks(&self) -> GridChunkIter<'_> {
        let n_chunks = self.n_present.div_ceil(self.g);
        GridChunkIter {
            grid: self,
            idx: 0,
            n_chunks,
        }
    }
}

struct GridChunkIter<'a> {
    grid: &'a Grid,
    idx: usize,
    n_chunks: usize,
}

impl<'a> Iterator for GridChunkIter<'a> {
    type Item = Chunk;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.n_chunks {
            return None;
        }
        let g_start = self.idx * self.grid.g;
        let g_count = self.grid.g.min(self.grid.n_present - g_start);
        self.idx += 1;
        Some(Chunk { g_start, g_count })
    }
}

struct Chunk {
    g_start: usize,
    g_count: usize,
}

/// Forward one genome-chunk: slice the present-stack to these `g_count`
/// genomes, gather `[g_count, n_max, ...]` rows, grouped bmm, mask, argmax.
/// Returns picks `[g_count*n_max]` and optionally scores `[g_count*n_max, width]`.
///
/// When `need_scores` is false (the `forward_picks` hot path), the full score
/// matrix D2H download is skipped — only the argmax picks are downloaded. This
/// avoids paying for a `[g_count*n_max, width]` transfer that `scatter_picks`
/// would throw away.
fn forward_chunk(
    present_stack: &WeightStack,
    grid: &Grid,
    chunk: &Chunk,
    width: usize,
    act_dim: usize,
    need_scores: bool,
) -> ChunkOutput {
    let g_count = chunk.g_count;
    let n_max = grid.n_max;
    let obs_dim = grid.obs_src.dims()[1];

    // Slice the present-stack to this chunk's genomes: [g_count, ...].
    let chunk_genomes: Vec<u32> = (0..g_count).map(|i| (chunk.g_start + i) as u32).collect();
    let genome_idx_t = Tensor::from_vec(chunk_genomes, (g_count,), &grid.device).unwrap();
    let stack = WeightStack {
        trunk_w: present_stack
            .trunk_w
            .iter()
            .map(|w| w.index_select(&genome_idx_t, 0).unwrap())
            .collect(),
        trunk_b: present_stack
            .trunk_b
            .iter()
            .map(|b| b.index_select(&genome_idx_t, 0).unwrap())
            .collect(),
        head_w: present_stack
            .head_w
            .iter()
            .map(|w| w.index_select(&genome_idx_t, 0).unwrap())
            .collect(),
        head_b: present_stack
            .head_b
            .iter()
            .map(|b| b.index_select(&genome_idx_t, 0).unwrap())
            .collect(),
        trunk_w0_sparse_flat: present_stack
            .trunk_w0_sparse_flat
            .narrow(
                0,
                chunk.g_start * SPARSE_PADDED_WIDTH,
                g_count * SPARSE_PADDED_WIDTH,
            )
            .unwrap(),
        trunk_w0_dense: present_stack
            .trunk_w0_dense
            .index_select(&genome_idx_t, 0)
            .unwrap(),
        trunk_w0_base: None,
        trunk_w0_pert: None,
        trunk_b0_base: None,
        trunk_b0_pert: None,
        n_pairs: 0,
        device: present_stack.device.clone(),
        dtype: present_stack.dtype,
    };

    // Gather [g_count, n_max] rows for these genomes.
    let (chunk_rows, chunk_valid) = chunk_indices(grid, chunk, n_max);
    let safe_rows: Vec<u32> = chunk_rows
        .iter()
        .map(|&r| if r == u32::MAX { 0 } else { r })
        .collect();
    let idx_t = Tensor::from_vec(safe_rows, (g_count * n_max,), &grid.device).unwrap();

    let obs_g = grid
        .obs_src
        .index_select(&idx_t, 0)
        .unwrap()
        .reshape((g_count, n_max, obs_dim))
        .unwrap();
    let acts_g = grid
        .acts_src
        .index_select(&idx_t, 0)
        .unwrap()
        .reshape((g_count, n_max, width, act_dim))
        .unwrap();
    let mask_g = grid
        .mask_src
        .index_select(&idx_t, 0)
        .unwrap()
        .reshape((g_count, n_max, width))
        .unwrap();

    let valid_t = Tensor::from_vec(chunk_valid, (g_count, n_max, 1), &grid.device).unwrap();
    let valid_t = valid_t.to_dtype(mask_g.dtype()).unwrap();
    let mask_g = mask_g.broadcast_mul(&valid_t).unwrap();

    let dims = ChunkDims {
        n_present: g_count,
        k: n_max,
        width,
        act_dim,
    };
    let scores = forward_pass(&stack, &obs_g, &acts_g, &mask_g, &dims);
    #[cfg(feature = "profile")]
    profile::FWD_CHUNKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let scores_f32 = scores.to_dtype(DType::F32).unwrap();

    #[cfg(feature = "profile")]
    let _a = profile::Span::new(&profile::FWD_ARGMAX_NS);
    let (picks, _gpu) = argmax_first_max(&scores_f32, g_count, n_max, width, &stack.device);
    #[cfg(feature = "profile")]
    drop(_a);

    // Only download the full score matrix on the test path (forward_scores).
    // The hot path (forward_picks) skips this — scatter_picks only needs picks.
    let scores_flat = if need_scores {
        scores_f32.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    } else {
        Vec::new()
    };

    ChunkOutput { picks, scores_flat }
}

/// Forward one genome-chunk via the sparse embedding-bag path.
fn forward_chunk_sparse(
    present_stack: &WeightStack,
    grid: &Grid,
    chunk: &Chunk,
    width: usize,
    act_dim: usize,
    need_scores: bool,
) -> ChunkOutput {
    let g_count = chunk.g_count;
    let n_max = grid.n_max;
    let nnz_width = MAX_NNZ_PER_ROW;
    let dense_w = DENSE_WIDTH;

    // Slice the present-stack to this chunk's genomes (same as forward_chunk).
    let chunk_genomes: Vec<u32> = (0..g_count).map(|i| (chunk.g_start + i) as u32).collect();
    let genome_idx_t = Tensor::from_vec(chunk_genomes, (g_count,), &grid.device).unwrap();
    let stack = WeightStack {
        trunk_w: present_stack
            .trunk_w
            .iter()
            .map(|w| w.index_select(&genome_idx_t, 0).unwrap())
            .collect(),
        trunk_b: present_stack
            .trunk_b
            .iter()
            .map(|b| b.index_select(&genome_idx_t, 0).unwrap())
            .collect(),
        head_w: present_stack
            .head_w
            .iter()
            .map(|w| w.index_select(&genome_idx_t, 0).unwrap())
            .collect(),
        head_b: present_stack
            .head_b
            .iter()
            .map(|b| b.index_select(&genome_idx_t, 0).unwrap())
            .collect(),
        trunk_w0_sparse_flat: present_stack
            .trunk_w0_sparse_flat
            .narrow(
                0,
                chunk.g_start * SPARSE_PADDED_WIDTH,
                g_count * SPARSE_PADDED_WIDTH,
            )
            .unwrap(),
        trunk_w0_dense: present_stack
            .trunk_w0_dense
            .index_select(&genome_idx_t, 0)
            .unwrap(),
        trunk_w0_base: None,
        trunk_w0_pert: None,
        trunk_b0_base: None,
        trunk_b0_pert: None,
        n_pairs: 0,
        device: present_stack.device.clone(),
        dtype: present_stack.dtype,
    };

    // Gather [g_count, n_max] rows for these genomes.
    let (chunk_rows, chunk_valid) = chunk_indices(grid, chunk, n_max);
    let safe_rows: Vec<u32> = chunk_rows
        .iter()
        .map(|&r| if r == u32::MAX { 0 } else { r })
        .collect();
    let idx_t = Tensor::from_vec(safe_rows, (g_count * n_max,), &grid.device).unwrap();

    let obs_sparse_g = grid
        .obs_sparse_src
        .index_select(&idx_t, 0)
        .unwrap()
        .reshape((g_count, n_max, nnz_width))
        .unwrap();
    let obs_dense_g = grid
        .obs_dense_src
        .index_select(&idx_t, 0)
        .unwrap()
        .reshape((g_count, n_max, dense_w))
        .unwrap();
    let acts_g = grid
        .acts_src
        .index_select(&idx_t, 0)
        .unwrap()
        .reshape((g_count, n_max, width, act_dim))
        .unwrap();
    let mask_g = grid
        .mask_src
        .index_select(&idx_t, 0)
        .unwrap()
        .reshape((g_count, n_max, width))
        .unwrap();

    let valid_t = Tensor::from_vec(chunk_valid, (g_count, n_max, 1), &grid.device).unwrap();
    let valid_t = valid_t.to_dtype(mask_g.dtype()).unwrap();
    let mask_g = mask_g.broadcast_mul(&valid_t).unwrap();

    let dims = ChunkDims {
        n_present: g_count,
        k: n_max,
        width,
        act_dim,
    };
    let scores = forward_pass_sparse(&stack, &obs_sparse_g, &obs_dense_g, &acts_g, &mask_g, &dims);
    #[cfg(feature = "profile")]
    profile::FWD_CHUNKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let scores_f32 = scores.to_dtype(DType::F32).unwrap();

    #[cfg(feature = "profile")]
    let _a = profile::Span::new(&profile::FWD_ARGMAX_NS);
    let (picks, _gpu) = argmax_first_max(&scores_f32, g_count, n_max, width, &stack.device);
    #[cfg(feature = "profile")]
    drop(_a);

    let scores_flat = if need_scores {
        scores_f32.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    } else {
        Vec::new()
    };

    ChunkOutput { picks, scores_flat }
}

/// Forward one genome-chunk via the ES grouped-GEMM split.
fn forward_chunk_es(
    present_stack: &WeightStack,
    grid: &Grid,
    chunk: &Chunk,
    width: usize,
    act_dim: usize,
    need_scores: bool,
) -> ChunkOutput {
    let g_count = chunk.g_count;
    let n_max = grid.n_max;
    let obs_dim = grid.obs_src.dims()[1];

    // Slice the present-stack to this chunk's genomes.
    let chunk_genomes: Vec<u32> = (0..g_count).map(|i| (chunk.g_start + i) as u32).collect();
    let genome_idx_t = Tensor::from_vec(chunk_genomes, (g_count,), &grid.device).unwrap();
    let stack = WeightStack {
        trunk_w: present_stack
            .trunk_w
            .iter()
            .map(|w| w.index_select(&genome_idx_t, 0).unwrap())
            .collect(),
        trunk_b: present_stack
            .trunk_b
            .iter()
            .map(|b| b.index_select(&genome_idx_t, 0).unwrap())
            .collect(),
        head_w: present_stack
            .head_w
            .iter()
            .map(|w| w.index_select(&genome_idx_t, 0).unwrap())
            .collect(),
        head_b: present_stack
            .head_b
            .iter()
            .map(|b| b.index_select(&genome_idx_t, 0).unwrap())
            .collect(),
        trunk_w0_sparse_flat: present_stack
            .trunk_w0_sparse_flat
            .narrow(
                0,
                chunk.g_start * SPARSE_PADDED_WIDTH,
                g_count * SPARSE_PADDED_WIDTH,
            )
            .unwrap(),
        trunk_w0_dense: present_stack
            .trunk_w0_dense
            .index_select(&genome_idx_t, 0)
            .unwrap(),
        // ES split: base is shared (clone). Pert is narrowed to chunk's pairs.
        trunk_w0_base: present_stack.trunk_w0_base.clone(),
        trunk_w0_pert: present_stack.trunk_w0_pert.as_ref().map(|pert| {
            let chunk_pairs_start = chunk.g_start / 2;
            pert.narrow(0, chunk_pairs_start, g_count / 2).unwrap()
        }),
        trunk_b0_base: present_stack.trunk_b0_base.clone(),
        trunk_b0_pert: present_stack.trunk_b0_pert.as_ref().map(|pert| {
            let chunk_pairs_start = chunk.g_start / 2;
            pert.narrow(0, chunk_pairs_start, g_count / 2).unwrap()
        }),
        n_pairs: g_count / 2,
        device: present_stack.device.clone(),
        dtype: present_stack.dtype,
    };

    // Gather [g_count, n_max] rows for these genomes.
    let (chunk_rows, chunk_valid) = chunk_indices(grid, chunk, n_max);
    let safe_rows: Vec<u32> = chunk_rows
        .iter()
        .map(|&r| if r == u32::MAX { 0 } else { r })
        .collect();
    let idx_t = Tensor::from_vec(safe_rows, (g_count * n_max,), &grid.device).unwrap();

    let obs_g = grid
        .obs_src
        .index_select(&idx_t, 0)
        .unwrap()
        .reshape((g_count, n_max, obs_dim))
        .unwrap();
    let acts_g = grid
        .acts_src
        .index_select(&idx_t, 0)
        .unwrap()
        .reshape((g_count, n_max, width, act_dim))
        .unwrap();
    let mask_g = grid
        .mask_src
        .index_select(&idx_t, 0)
        .unwrap()
        .reshape((g_count, n_max, width))
        .unwrap();

    let valid_t = Tensor::from_vec(chunk_valid, (g_count, n_max, 1), &grid.device).unwrap();
    let valid_t = valid_t.to_dtype(mask_g.dtype()).unwrap();
    let mask_g = mask_g.broadcast_mul(&valid_t).unwrap();

    let dims = ChunkDims {
        n_present: g_count,
        k: n_max,
        width,
        act_dim,
    };
    let scores = forward_pass_es(&stack, &obs_g, &acts_g, &mask_g, &dims);
    #[cfg(feature = "profile")]
    profile::FWD_CHUNKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let scores_f32 = scores.to_dtype(DType::F32).unwrap();

    #[cfg(feature = "profile")]
    let _a = profile::Span::new(&profile::FWD_ARGMAX_NS);
    let (picks, _gpu) = argmax_first_max(&scores_f32, g_count, n_max, width, &stack.device);
    #[cfg(feature = "profile")]
    drop(_a);

    let scores_flat = if need_scores {
        scores_f32.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    } else {
        Vec::new()
    };

    ChunkOutput { picks, scores_flat }
}

struct ChunkOutput {
    picks: Vec<usize>,
    scores_flat: Vec<f32>,
}

/// The `[g_count*n_max]` global row indices and validity mask for one
/// genome-chunk (all `n_max` slots of each genome in the chunk).
fn chunk_indices(grid: &Grid, chunk: &Chunk, n_max: usize) -> (Vec<u32>, Vec<u8>) {
    let g_count = chunk.g_count;
    let mut rows = Vec::with_capacity(g_count * n_max);
    let mut valid = Vec::with_capacity(g_count * n_max);
    for gi in 0..g_count {
        let base = (chunk.g_start + gi) * grid.n_max;
        for j in 0..n_max {
            let r = grid.row_order[base + j];
            rows.push(r);
            valid.push(if r == u32::MAX { 0 } else { 1 });
        }
    }
    (rows, valid)
}

/// Scatter a genome-chunk's picks (and optionally scores) back to global row
/// order. Each genome's rows are packed at the front of its `n_max` slots.
fn scatter(
    grid: &Grid,
    chunk: &Chunk,
    picks: &[usize],
    scores: &[f32],
    width: usize,
    out_picks: &mut [usize],
    out_scores: &mut [f32],
) {
    let n_max = grid.n_max;
    for gi in 0..chunk.g_count {
        let cnt = grid.counts[chunk.g_start + gi];
        for slot in 0..cnt {
            let orig = grid.row_order[(chunk.g_start + gi) * grid.n_max + slot] as usize;
            out_picks[orig] = picks[gi * n_max + slot];
            let src = (gi * n_max + slot) * width;
            let dst = orig * width;
            out_scores[dst..dst + width].copy_from_slice(&scores[src..src + width]);
        }
    }
}

fn scatter_picks(grid: &Grid, chunk: &Chunk, picks: &[usize], out: &mut [usize]) {
    let n_max = grid.n_max;
    for gi in 0..chunk.g_count {
        let cnt = grid.counts[chunk.g_start + gi];
        for slot in 0..cnt {
            let orig = grid.row_order[(chunk.g_start + gi) * grid.n_max + slot] as usize;
            out[orig] = picks[gi * n_max + slot];
        }
    }
}

/// The grouped forward through trunk + head. Returns masked scores
/// `[G, k, width]` in the stack's dtype (illegal actions at -1e9).
fn forward_pass(
    stack: &WeightStack,
    obs_g: &Tensor,
    acts_g: &Tensor,
    mask_g: &Tensor,
    dims: &ChunkDims,
) -> Tensor {
    let ChunkDims {
        n_present,
        k,
        width,
        act_dim,
    } = *dims;
    let dtype = stack.dtype;

    #[cfg(feature = "profile")]
    let _g = profile::Span::new(&profile::FWD_GATHER_NS);
    let obs_g = obs_g.to_dtype(dtype).unwrap();
    let acts_g = acts_g.to_dtype(dtype).unwrap();

    #[cfg(feature = "profile")]
    drop(_g);
    // Trunk (all tanh): [G, k, obs] → [G, k, E]
    #[cfg(feature = "profile")]
    let _t = profile::Span::new(&profile::FWD_TRUNK_NS);
    let mut x = obs_g;
    for (w, b) in stack.trunk_w.iter().zip(&stack.trunk_b) {
        x = stacked_bmm(&x, w); // [G, k, out]
        let b_expanded = b.reshape((stack.n_groups(), 1, ())).unwrap();
        x = x.broadcast_add(&b_expanded).unwrap();
        x = x.tanh().unwrap();
    }
    let emb = x; // [G, k, E]

    #[cfg(feature = "profile")]
    drop(_t);
    #[cfg(feature = "profile")]
    let _h = profile::Span::new(&profile::FWD_HEAD_NS);
    // Head.0: linear over cat(emb, acts). Split weight on the `in` dim.
    let head_w0 = &stack.head_w[0];
    let head_b0 = &stack.head_b[0];
    let emb_dim = emb.dims()[2];
    let emb_w = head_w0.narrow(2, 0, emb_dim).unwrap();
    let act_w = head_w0.narrow(2, emb_dim, act_dim).unwrap();

    let emb_in = stacked_bmm(&emb, &emb_w); // [G, k, H]
    let emb_in = emb_in.unsqueeze(2).unwrap(); // [G, k, 1, H]

    let acts_flat = acts_g.reshape((n_present, k * width, act_dim)).unwrap();
    let act_w_t = act_w.transpose(1, 2).unwrap().contiguous().unwrap(); // [G, act_dim, H]
    let act_in = acts_flat.matmul(&act_w_t).unwrap(); // [G, k*width, H]
    let act_in = act_in.reshape((n_present, k, width, ())).unwrap();

    let mut x = emb_in.broadcast_add(&act_in).unwrap();
    let b0_expanded = head_b0.reshape((n_present, 1, 1, ())).unwrap();
    x = x.broadcast_add(&b0_expanded).unwrap();
    x = x.tanh().unwrap();
    x = x.reshape((n_present, k * width, ())).unwrap();

    // Remaining head layers (head.out is linear).
    let n_head = stack.head_w.len();
    for (i, (w, b)) in stack.head_w.iter().zip(&stack.head_b).enumerate().skip(1) {
        x = stacked_bmm(&x, w);
        let b_expanded = b.reshape((n_present, 1, ())).unwrap();
        x = x.broadcast_add(&b_expanded).unwrap();
        if i < n_head - 1 {
            x = x.tanh().unwrap();
        }
    }

    let scores = x.reshape((n_present, k, width)).unwrap();

    #[cfg(feature = "profile")]
    drop(_h);
    #[cfg(feature = "profile")]
    let _m = profile::Span::new(&profile::FWD_MASK_NS);
    // Mask illegal actions: scores + (1 - mask) * (-1e9), in the stack dtype.
    let mask_dt = mask_g.to_dtype(dtype).unwrap();
    let ones = Tensor::ones_like(&mask_dt).unwrap();
    let inv_mask = ones.sub(&mask_dt).unwrap();
    let neg = Tensor::new(-1e9f32, &stack.device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
        .broadcast_as(mask_dt.shape())
        .unwrap();
    let neg_offset = inv_mask.mul(&neg).unwrap();
    scores.broadcast_add(&neg_offset).unwrap()
}

/// The ES grouped-GEMM forward. trunk.0 is split into:
///   1. Base GEMM: obs @ θ^T (one big matmul, M=g_count*n_max, efficient)
///   2. Perturbation bmm: obs_pair @ (σε)^T (n_pairs groups, half the FLOPs)
///   3. Sign + add + bias + tanh
///
/// The remaining trunk layers and head use per-genome weights (same as dense).
fn forward_pass_es(
    stack: &WeightStack,
    obs_g: &Tensor, // [g_count, n_max, OBS_DIM]
    acts_g: &Tensor,
    mask_g: &Tensor,
    dims: &ChunkDims,
) -> Tensor {
    let ChunkDims {
        n_present: g_count,
        k: n_max,
        width,
        act_dim,
    } = *dims;
    let dtype = stack.dtype;
    let obs_dim = obs_g.dims()[2];
    let n_pairs = stack.n_pairs;

    #[cfg(feature = "profile")]
    let _g = profile::Span::new(&profile::FWD_GATHER_NS);
    let obs_g = obs_g.to_dtype(dtype).unwrap();
    let acts_g = acts_g.to_dtype(dtype).unwrap();
    #[cfg(feature = "profile")]
    drop(_g);

    // ── Trunk.0: base GEMM + perturbation bmm ──────────────────────────
    #[cfg(feature = "profile")]
    let _t = profile::Span::new(&profile::FWD_TRUNK_NS);

    // 1. Base GEMM: [g_count*n_max, OBS_DIM] × [OBS_DIM, 512] → [g_count*n_max, 512]
    let base_w = stack.trunk_w0_base.as_ref().expect("ES base weight");
    let base_b = stack.trunk_b0_base.as_ref().expect("ES base bias");
    let base_w_t = base_w
        .squeeze(0)
        .unwrap() // [512, OBS_DIM]
        .transpose(0, 1)
        .unwrap() // [OBS_DIM, 512]
        .contiguous()
        .unwrap();
    let obs_flat = obs_g.reshape((g_count * n_max, obs_dim)).unwrap();
    let emb_base = obs_flat.matmul(&base_w_t).unwrap(); // [g_count*n_max, 512]
    let emb_base = emb_base.reshape((g_count, n_max, ())).unwrap();

    // 2. Perturbation bmm: [n_pairs, 2*n_max, OBS_DIM] × [n_pairs, OBS_DIM, 512]
    //    = reshape obs_g from [g_count, n_max, OBS_DIM] to [n_pairs, 2*n_max, OBS_DIM]
    //    (consecutive genomes form pairs, so this is a no-op reshape).
    let pert_w = stack.trunk_w0_pert.as_ref().expect("ES pert weight");
    let pert_b = stack.trunk_b0_pert.as_ref().expect("ES pert bias");
    let obs_pair = obs_g.reshape((n_pairs, 2 * n_max, obs_dim)).unwrap();
    let emb_pert = stacked_bmm(&obs_pair, pert_w); // [n_pairs, 2*n_max, out_dim]
    let out_dim = emb_pert.dims()[2]; // trunk.0 output dim (512 for training arch)

    // 3. Sign: +1 for first n_max rows of each pair, -1 for last n_max.
    let sign_data: Vec<f32> = (0..n_pairs)
        .flat_map(|_| std::iter::repeat_n(1.0f32, n_max).chain(std::iter::repeat_n(-1.0f32, n_max)))
        .collect();
    let sign = Tensor::from_vec(sign_data, (n_pairs * 2 * n_max, 1), &stack.device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    // 4. emb = emb_base + sign * (emb_pert + pert_b) + base_b
    let pert_b_expanded = pert_b
        .unsqueeze(1)
        .unwrap() // [n_pairs, 1, out_dim]
        .broadcast_as((n_pairs, 2 * n_max, out_dim))
        .unwrap();
    let emb_pert_with_bias = emb_pert.broadcast_add(&pert_b_expanded).unwrap();
    let emb_pert_signed = emb_pert_with_bias
        .reshape((n_pairs * 2 * n_max, ()))
        .unwrap()
        .broadcast_mul(&sign)
        .unwrap();

    let mut x = emb_base
        .reshape((g_count * n_max, out_dim))
        .unwrap()
        .broadcast_add(&emb_pert_signed)
        .unwrap();
    x = x.broadcast_add(base_b).unwrap();
    x = x.reshape((g_count, n_max, out_dim)).unwrap();
    x = x.tanh().unwrap();

    // ── Remaining trunk layers (trunk.1, ...) ────────────────────────────
    for (w, b) in stack.trunk_w.iter().zip(&stack.trunk_b).skip(1) {
        x = stacked_bmm(&x, w);
        let b_expanded = b.reshape((g_count, 1, ())).unwrap();
        x = x.broadcast_add(&b_expanded).unwrap();
        x = x.tanh().unwrap();
    }
    let emb = x;

    #[cfg(feature = "profile")]
    drop(_t);
    #[cfg(feature = "profile")]
    let _h = profile::Span::new(&profile::FWD_HEAD_NS);

    // ── Head (same as forward_pass) ──────────────────────────────────────
    let head_w0 = &stack.head_w[0];
    let head_b0 = &stack.head_b[0];
    let emb_dim = emb.dims()[2];
    let emb_w = head_w0.narrow(2, 0, emb_dim).unwrap();
    let act_w = head_w0.narrow(2, emb_dim, act_dim).unwrap();

    let emb_in = stacked_bmm(&emb, &emb_w);
    let emb_in = emb_in.unsqueeze(2).unwrap();

    let acts_flat = acts_g.reshape((g_count, n_max * width, act_dim)).unwrap();
    let act_w_t = act_w.transpose(1, 2).unwrap().contiguous().unwrap();
    let act_in = acts_flat.matmul(&act_w_t).unwrap();
    let act_in = act_in.reshape((g_count, n_max, width, ())).unwrap();

    let mut x = emb_in.broadcast_add(&act_in).unwrap();
    let b0_expanded = head_b0.reshape((g_count, 1, 1, ())).unwrap();
    x = x.broadcast_add(&b0_expanded).unwrap();
    x = x.tanh().unwrap();
    x = x.reshape((g_count, n_max * width, ())).unwrap();

    let n_head = stack.head_w.len();
    for (i, (w, b)) in stack.head_w.iter().zip(&stack.head_b).enumerate().skip(1) {
        x = stacked_bmm(&x, w);
        let b_expanded = b.reshape((g_count, 1, ())).unwrap();
        x = x.broadcast_add(&b_expanded).unwrap();
        if i < n_head - 1 {
            x = x.tanh().unwrap();
        }
    }

    let scores = x.reshape((g_count, n_max, width)).unwrap();

    #[cfg(feature = "profile")]
    drop(_h);
    #[cfg(feature = "profile")]
    let _m = profile::Span::new(&profile::FWD_MASK_NS);
    // Mask illegal actions (same as forward_pass).
    let mask_dt = mask_g.to_dtype(dtype).unwrap();
    let ones = Tensor::ones_like(&mask_dt).unwrap();
    let inv_mask = ones.sub(&mask_dt).unwrap();
    let neg = Tensor::new(-1e9f32, &stack.device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
        .broadcast_as(mask_dt.shape())
        .unwrap();
    let neg_offset = inv_mask.mul(&neg).unwrap();
    scores.broadcast_add(&neg_offset).unwrap()
}

/// The sparse forward through trunk + head. trunk.0 is replaced by the
/// embedding-bag (gather + segment-sum) over sparse features + a small dense
/// matmul over the dense block, summed. The remaining trunk layers (trunk.1,
/// ...) and the head are the same as the dense `forward_pass`.
fn forward_pass_sparse(
    stack: &WeightStack,
    obs_sparse_g: &Tensor, // [g_count, n_max, MAX_NNZ] u16
    obs_dense_g: &Tensor,  // [g_count, n_max, DENSE_WIDTH] u8
    acts_g: &Tensor,       // [g_count, n_max, width, ACT_DIM] u8
    mask_g: &Tensor,       // [g_count, n_max, width] u8
    dims: &ChunkDims,
) -> Tensor {
    let ChunkDims {
        n_present,
        k,
        width,
        act_dim,
    } = *dims;
    let dtype = stack.dtype;

    #[cfg(feature = "profile")]
    let _g = profile::Span::new(&profile::FWD_GATHER_NS);
    let obs_dense_g = obs_dense_g.to_dtype(dtype).unwrap();
    let acts_g = acts_g.to_dtype(dtype).unwrap();

    #[cfg(feature = "profile")]
    drop(_g);

    // ── Trunk.0: embedding-bag (sparse) + dense matmul ──────────────────
    #[cfg(feature = "profile")]
    let _t = profile::Span::new(&profile::FWD_TRUNK_NS);

    // 1. Compute genome offsets: each row's local genome index (0..g_count-1)
    //    * SPARSE_PADDED_WIDTH. Rows are ordered by genome in the grid.
    let genome_offsets: Vec<u32> = (0..n_present)
        .flat_map(|g| std::iter::repeat_n((g * SPARSE_PADDED_WIDTH) as u32, k))
        .collect();
    let offsets_t = Tensor::from_vec(genome_offsets, (n_present * k, 1), &stack.device).unwrap();

    // 2. Add genome offsets to sparse indices and flatten for index_select.
    let indices = obs_sparse_g
        .to_dtype(DType::U32)
        .unwrap()
        .reshape((n_present * k, MAX_NNZ_PER_ROW))
        .unwrap()
        .broadcast_add(&offsets_t)
        .unwrap()
        .flatten_all()
        .unwrap(); // [g_count*n_max*MAX_NNZ]

    // 3. Gather: index_select from [g_count * SPARSE_PADDED_WIDTH, 512].
    let gathered = stack
        .trunk_w0_sparse_flat
        .index_select(&indices, 0)
        .unwrap(); // [g_count*n_max*MAX_NNZ, 512]

    // 4. Segment-sum over the MAX_NNZ slots → [g_count*n_max, 512].
    let emb_sparse = gathered
        .reshape((n_present * k, MAX_NNZ_PER_ROW, ()))
        .unwrap()
        .sum(1)
        .unwrap(); // [g_count*n_max, 512]

    // 5. Dense matmul: [g_count, n_max, 77] × [g_count, 77, 512] → [g_count, n_max, 512].
    let emb_dense = stacked_bmm(&obs_dense_g, &stack.trunk_w0_dense);

    // 6. Sum + bias + tanh.
    let mut x = emb_sparse
        .reshape((n_present, k, ()))
        .unwrap()
        .broadcast_add(&emb_dense)
        .unwrap();
    let b0_expanded = stack.trunk_b[0].reshape((n_present, 1, ())).unwrap();
    x = x.broadcast_add(&b0_expanded).unwrap();
    x = x.tanh().unwrap();

    // ── Remaining trunk layers (trunk.1, ...) ────────────────────────────
    for (w, b) in stack.trunk_w.iter().zip(&stack.trunk_b).skip(1) {
        x = stacked_bmm(&x, w);
        let b_expanded = b.reshape((n_present, 1, ())).unwrap();
        x = x.broadcast_add(&b_expanded).unwrap();
        x = x.tanh().unwrap();
    }
    let emb = x; // [g_count, n_max, trunk_out]

    #[cfg(feature = "profile")]
    drop(_t);
    #[cfg(feature = "profile")]
    let _h = profile::Span::new(&profile::FWD_HEAD_NS);

    // ── Head (same as forward_pass) ──────────────────────────────────────
    let head_w0 = &stack.head_w[0];
    let head_b0 = &stack.head_b[0];
    let emb_dim = emb.dims()[2];
    let emb_w = head_w0.narrow(2, 0, emb_dim).unwrap();
    let act_w = head_w0.narrow(2, emb_dim, act_dim).unwrap();

    let emb_in = stacked_bmm(&emb, &emb_w);
    let emb_in = emb_in.unsqueeze(2).unwrap();

    let acts_flat = acts_g.reshape((n_present, k * width, act_dim)).unwrap();
    let act_w_t = act_w.transpose(1, 2).unwrap().contiguous().unwrap();
    let act_in = acts_flat.matmul(&act_w_t).unwrap();
    let act_in = act_in.reshape((n_present, k, width, ())).unwrap();

    let mut x = emb_in.broadcast_add(&act_in).unwrap();
    let b0_expanded = head_b0.reshape((n_present, 1, 1, ())).unwrap();
    x = x.broadcast_add(&b0_expanded).unwrap();
    x = x.tanh().unwrap();
    x = x.reshape((n_present, k * width, ())).unwrap();

    let n_head = stack.head_w.len();
    for (i, (w, b)) in stack.head_w.iter().zip(&stack.head_b).enumerate().skip(1) {
        x = stacked_bmm(&x, w);
        let b_expanded = b.reshape((n_present, 1, ())).unwrap();
        x = x.broadcast_add(&b_expanded).unwrap();
        if i < n_head - 1 {
            x = x.tanh().unwrap();
        }
    }

    let scores = x.reshape((n_present, k, width)).unwrap();

    #[cfg(feature = "profile")]
    drop(_h);
    #[cfg(feature = "profile")]
    let _m = profile::Span::new(&profile::FWD_MASK_NS);
    // Mask illegal actions (same as forward_pass).
    let mask_dt = mask_g.to_dtype(dtype).unwrap();
    let ones = Tensor::ones_like(&mask_dt).unwrap();
    let inv_mask = ones.sub(&mask_dt).unwrap();
    let neg = Tensor::new(-1e9f32, &stack.device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
        .broadcast_as(mask_dt.shape())
        .unwrap();
    let neg_offset = inv_mask.mul(&neg).unwrap();
    scores.broadcast_add(&neg_offset).unwrap()
}
///
/// On CPU: an exact strict-`>` loop (matches numpy/torch first-max), so the
/// single-game replay and logit equivalence tests stay byte-exact.
///
/// On GPU: candle's `argmax` returns the last maximal index on some backends,
/// which diverges from numpy. We add a monotonically *decreasing* penalty
/// `-epsilon * index` (epsilon tiny) before the reduction so the lowest index
/// wins a tie, making the pick deterministic. epsilon is below any real logit
/// gap, so disagreements with the fp32 CPU path are confined to genuine
/// near-ties. Returns (picks, used_gpu).
fn argmax_first_max(
    scores_f32: &Tensor,
    n_present: usize,
    k: usize,
    width: usize,
    device: &Device,
) -> (Vec<usize>, bool) {
    let is_cuda = matches!(device, Device::Cuda(_));
    if !is_cuda {
        let host: Vec<f32> = scores_f32.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut picks = vec![0usize; n_present * k];
        for gi in 0..n_present {
            for ni in 0..k {
                let base = (gi * k + ni) * width;
                let mut best = 0usize;
                let mut best_val = f32::NEG_INFINITY;
                for j in 0..width {
                    let v = host[base + j];
                    if v > best_val {
                        best_val = v;
                        best = j;
                    }
                }
                picks[gi * k + ni] = best;
            }
        }
        return (picks, false);
    }

    let epsilon = 1e-4f32;
    let idx = Tensor::arange(0u32, width as u32, device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let neg_eps = Tensor::new(-epsilon, device).unwrap();
    let pen = (idx.broadcast_mul(&neg_eps.broadcast_as(idx.shape()).unwrap()))
        .unwrap()
        .broadcast_as(scores_f32.shape())
        .unwrap();
    let adjusted = scores_f32.add(&pen).unwrap();
    let picks_t = adjusted.argmax(2).unwrap(); // [G, k] u32
    let picks_host: Vec<u32> = picks_t.flatten_all().unwrap().to_vec1::<u32>().unwrap();
    (picks_host.into_iter().map(|v| v as usize).collect(), true)
}

/// Build a WeightStack from a single genome (for evaluation/tests), fp32.
pub fn single_genome_weights(genome: &Genome, arch: &Arch, device: &Device) -> WeightStack {
    WeightStack::from_roster(&[genome], arch, device, DType::F32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_first_max_breaks_ties_to_lowest_index() {
        // Hand-constructed tied logits: the first (lowest) index must win.
        let device = Device::Cpu;
        // [G=1, k=2, width=4]
        let logits = vec![
            5.0f32, 5.0, 1.0, 5.0, // row 0: indices 0,1,3 tie at 5.0 → pick 0
            -1e9, 0.0, -1e9, 0.0, // row 1: indices 1,3 tie at 0.0 → pick 1
        ];
        let scores = Tensor::from_vec(logits, (1, 2, 4), &device).unwrap();
        let (picks, _) = argmax_first_max(&scores, 1, 2, 4, &device);
        assert_eq!(picks[0], 0, "row 0: lowest of tied indices {{0,1,3}} is 0");
        assert_eq!(picks[1], 1, "row 1: lowest of tied indices {{1,3}} is 1");
    }

    #[test]
    fn argmax_epsilon_does_not_flip_real_gaps() {
        let device = Device::Cpu;
        let logits = vec![
            0.0f32, 1.0, 0.0, 0.0, // row 0: index 1 wins by 1.0 >> epsilon
            0.0, 0.0, 0.0, 0.5, // row 1: index 3 wins by 0.5 >> epsilon
        ];
        let scores = Tensor::from_vec(logits, (1, 2, 4), &device).unwrap();
        let (picks, _) = argmax_first_max(&scores, 1, 2, 4, &device);
        assert_eq!(picks[0], 1);
        assert_eq!(picks[1], 3);
    }
}
