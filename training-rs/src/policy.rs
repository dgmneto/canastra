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
use candle_core::{DType, Device, Tensor};

#[cfg(feature = "profile")]
use crate::profile;

/// Per-layer weight tensors, stacked on a leading G (genome) dimension.
/// Built once per generation in the chosen dtype (bf16 on CUDA, f32 on CPU)
/// and kept resident for every ply. `slice` gathers the present genomes once
/// per ply — the only weight copy, and a small one relative to the matmuls.
pub struct WeightStack {
    pub trunk_w: Vec<Tensor>, // [G, out, in]
    pub trunk_b: Vec<Tensor>, // [G, out]
    pub head_w: Vec<Tensor>,  // [G, out, in]
    pub head_b: Vec<Tensor>,  // [G, out]
    pub device: Device,
    pub dtype: DType,
}

impl WeightStack {
    /// Stack the roster's flat genomes into per-layer `[G, out, in]` tensors.
    /// `dtype` is bf16 for CUDA training, f32 for CPU correctness tests.
    pub fn from_roster(roster: &[&Genome], arch: &Arch, device: &Device, dtype: DType) -> Self {
        let g = roster.len();
        let shapes = layer_shapes_owned(arch);

        let mut trunk_w = Vec::new();
        let mut trunk_b = Vec::new();
        let mut head_w = Vec::new();
        let mut head_b = Vec::new();
        let mut offset = 0usize;

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

            if name.starts_with("trunk") {
                trunk_w.push(w);
                trunk_b.push(b);
            } else {
                head_w.push(w);
                head_b.push(b);
            }
        }

        WeightStack {
            trunk_w,
            trunk_b,
            head_w,
            head_b,
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
            device: self.device.clone(),
            dtype: self.dtype,
        }
    }

    pub fn n_groups(&self) -> usize {
        self.trunk_w[0].dims()[0]
    }
}

/// A roster of flat genomes kept on CPU. Used by the coalesced rollout path:
/// the GPU never holds the full roster — instead, `build_chunk` uploads a
/// subset to the device as a `WeightStack`. The lockstep path builds its own
/// `WeightStack` directly via `from_roster`.
pub struct CpuRoster {
    pub genomes: Vec<Genome>,
    pub arch: Arch,
}

impl CpuRoster {
    pub fn new(genomes: Vec<Genome>, arch: Arch) -> Self {
        Self { genomes, arch }
    }

    pub fn n_genomes(&self) -> usize {
        self.genomes.len()
    }

    /// Build a `WeightStack` on `device` for genomes at indices `which`.
    /// dtype is f32 — the coalesced path used fp32 throughout.
    pub fn build_chunk(
        &self,
        which: &[usize],
        device: &Device,
    ) -> candle_core::Result<WeightStack> {
        let refs: Vec<&Genome> = which.iter().map(|&i| &self.genomes[i]).collect();
        Ok(WeightStack::from_roster(
            &refs,
            &self.arch,
            device,
            DType::F32,
        ))
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
        let out = forward_chunk(ply_stack.as_ref(), &grid, &chunk, width, act_dim);
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
        let out = forward_chunk(ply_stack.as_ref(), &grid, &chunk, width, act_dim);
        scatter_picks(&grid, &chunk, &out.picks, &mut picks);
    }

    picks
}

/// The grouping of a ply's rows into a `[G, n_max, ...]` grid, plus the chunk
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
    g: usize,            // genomes per chunk
    row_order: Vec<u32>, // [n_present, n_max] global row indices (u32::MAX = padding)
    counts: Vec<usize>,  // rows per present genome
    obs_src: Tensor,
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
/// Returns picks `[g_count*n_max]` and scores `[g_count*n_max, width]`.
fn forward_chunk(
    present_stack: &WeightStack,
    grid: &Grid,
    chunk: &Chunk,
    width: usize,
    act_dim: usize,
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

    // Download scores (fp32) — only on the test path (forward_scores).
    let scores_host: Vec<f32> = scores_f32.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    ChunkOutput {
        picks,
        scores_flat: scores_host,
    }
}

struct ChunkOutput {
    picks: Vec<usize>,
    scores_flat: Vec<f32>,
}

/// The `[g_count*n_max]` global row indices and validity mask for one
/// genome-chunk (all `n_max` slots of each genome in the chunk).
fn chunk_indices(grid: &Grid, chunk: &Chunk, n_max: usize) -> (Vec<u32>, Vec<u32>) {
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

/// Argmax over the width dimension with first-max tie-breaking.
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
        // Hand-constructed tied logits: the first index must win.
        let device = Device::Cpu;
        // [G=1, k=2, width=4]
        let logits = vec![
            5.0f32, 5.0, 1.0, 5.0, // row 0: indices 1,3 tie at 5.0 → pick 1
            -1e9, 0.0, -1e9, 0.0, // row 1: indices 1,3 tie at 0.0 → pick 1
        ];
        let scores = Tensor::from_vec(logits, (1, 2, 4), &device).unwrap();
        let (picks, _) = argmax_first_max(&scores, 1, 2, 4, &device);
        assert_eq!(picks[0], 1, "row 0: lowest of tied indices {{1,3}} is 1");
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
