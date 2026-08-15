//! Batched policy scoring via candle (pure Rust ML with CUDA via cudarc).
//!
//! Mirrors `training/python/canastra_train/policy.py` exactly: trunk layers
//! all tanh, head hidden layers tanh, final layer linear. The stacked forward
//! processes the whole genome roster in one batched pass.

use crate::genome::{layer_shapes, Arch, Genome};
use candle_core::{DType, Device, Tensor};

/// A roster of flat genomes kept on CPU. The GPU never holds the full roster —
/// instead, `build_chunk` uploads only a small subset (typically 64 genomes) to
/// the device as a `WeightStack`. This bounds GPU memory to ~300 MB regardless
/// of population size, at the cost of a per-chunk host→device memcpy (~30 ms
/// for 64 genomes on PCIe 4.0).
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

    /// Build a small `WeightStack` on `device` for genomes at indices `which`.
    /// The weight data is gathered on CPU then uploaded in one batch per layer.
    pub fn build_chunk(
        &self,
        which: &[usize],
        device: &Device,
    ) -> candle_core::Result<WeightStack> {
        let g = which.len();
        let shapes = layer_shapes(&self.arch);
        let mut trunk_w = Vec::new();
        let mut trunk_b = Vec::new();
        let mut head_w = Vec::new();
        let mut head_b = Vec::new();
        let mut offset = 0usize;

        for (name, out, inn) in &shapes {
            let w_size = out * inn;
            let mut w_data = Vec::with_capacity(g * w_size);
            let mut b_data = Vec::with_capacity(g * out);
            for &gi in which {
                let genome = &self.genomes[gi];
                w_data.extend_from_slice(&genome[offset..offset + w_size]);
                b_data.extend_from_slice(&genome[offset + w_size..offset + w_size + out]);
            }
            offset += w_size + out;

            let w = Tensor::from_vec(w_data, (g, *out, *inn), device)?;
            let b = Tensor::from_vec(b_data, (g, *out), device)?;

            if name.starts_with("trunk") {
                trunk_w.push(w);
                trunk_b.push(b);
            } else {
                head_w.push(w);
                head_b.push(b);
            }
        }

        Ok(WeightStack {
            trunk_w,
            trunk_b,
            head_w,
            head_b,
            device: device.clone(),
        })
    }
}

/// Per-layer weight tensors, stacked on a leading G (genome) dimension.
/// Built once per generation, reused every ply.
///
/// Weights are stored as `[G, out, in]` (natural layout). The `slice` method
/// returns a `WeightStack` with the subset of genomes — the forward pass
/// transposes only the small sliced chunk, not the full roster. This bounds
/// GPU memory regardless of population size.
pub struct WeightStack {
    pub trunk_w: Vec<Tensor>, // [G, out, in]
    pub trunk_b: Vec<Tensor>, // [G, out]
    pub head_w: Vec<Tensor>,  // [G, out, in]
    pub head_b: Vec<Tensor>,  // [G, out]
    pub device: Device,
}

impl WeightStack {
    /// Stack the roster's flat genomes into per-layer [G, out, in] tensors.
    pub fn from_roster(roster: &[&Genome], arch: &Arch, device: &Device) -> Self {
        let g = roster.len();
        let shapes = layer_shapes_owned(arch);

        let mut trunk_w = Vec::new();
        let mut trunk_b = Vec::new();
        let mut head_w = Vec::new();
        let mut head_b = Vec::new();
        let mut offset = 0usize;

        for (name, out, inn) in &shapes {
            let w_size = out * inn;
            // Gather weights from all genomes for this layer.
            let mut w_data = Vec::with_capacity(g * w_size);
            let mut b_data = Vec::with_capacity(g * out);
            for genome in roster {
                w_data.extend_from_slice(&genome[offset..offset + w_size]);
                b_data.extend_from_slice(&genome[offset + w_size..offset + w_size + out]);
            }
            offset += w_size + out;

            let w = Tensor::from_vec(w_data, (g, *out, *inn), device)
                .unwrap_or_else(|e| panic!("weight tensor: {e}"));
            let b = Tensor::from_vec(b_data, (g, *out), device)
                .unwrap_or_else(|e| panic!("bias tensor: {e}"));

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
        }
    }

    /// Slice to only the genomes present in this ply.
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
        }
    }

    pub fn n_groups(&self) -> usize {
        self.trunk_w[0].dims()[0]
    }

    pub fn shallow_clone(&self) -> Self {
        WeightStack {
            trunk_w: self.trunk_w.clone(),
            trunk_b: self.trunk_b.clone(),
            head_w: self.head_w.clone(),
            head_b: self.head_b.clone(),
            device: self.device.clone(),
        }
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
///
/// Transposes the weight tensor (which is small — a sliced chunk) and ensures
/// both operands are contiguous for CUDA's matmul.
fn stacked_bmm(x: &Tensor, w: &Tensor) -> Tensor {
    let w_t = w.transpose(1, 2).unwrap().contiguous().unwrap(); // [G, in, out]
    let x = x.contiguous().unwrap_or_else(|_| x.clone());
    x.matmul(&w_t).unwrap()
}

/// Result of a forward pass: per-row argmax picks and the full masked score
/// matrix `[n_rows, width]` in original row order (padded columns = -1e9).
pub struct ForwardOutput {
    pub picks: Vec<usize>,
    pub scores_flat: Vec<f32>,
    pub width: usize,
}

/// Forward pass using a `CpuRoster`: builds small GPU `WeightStack`s per chunk,
/// forwards each, and reassembles picks. This bounds GPU memory to
/// `CHUNK_SIZE` genomes regardless of population size.
///
/// `genome_idx` values are indices into the roster (0..roster.n_genomes()).
pub fn forward_scores_roster(
    roster: &CpuRoster,
    obs: &Tensor,
    acts: &Tensor,
    mask: &Tensor,
    genome_idx: &[usize],
    device: &Device,
) -> ForwardOutput {
    const CHUNK_SIZE: usize = 64;
    let n_rows = obs.dims()[0];
    let width = acts.dims()[1];

    if n_rows == 0 {
        return ForwardOutput {
            picks: Vec::new(),
            scores_flat: Vec::new(),
            width: 0,
        };
    }

    // Distinct genomes present.
    let n_genomes = roster.n_genomes();
    let mut present_set = vec![false; n_genomes];
    for &g in genome_idx {
        present_set[g] = true;
    }
    let present: Vec<usize> = (0..n_genomes).filter(|&g| present_set[g]).collect();

    // If small enough, build one chunk and forward directly.
    if present.len() <= CHUNK_SIZE {
        let stack = roster.build_chunk(&present, device).unwrap();
        let local_present: Vec<usize> = (0..present.len()).collect();
        let mut gmap = std::collections::HashMap::new();
        for (i, &g) in present.iter().enumerate() {
            gmap.insert(g, i);
        }
        let local_gidx: Vec<usize> = genome_idx.iter().map(|&g| gmap[&g]).collect();
        let n_max = local_present
            .iter()
            .map(|&g| local_gidx.iter().filter(|&&gg| gg == g).count())
            .max()
            .unwrap_or(0)
            .max(4);
        return forward_scores_chunk(&stack, obs, acts, mask, &local_gidx, &local_present, n_max);
    }

    // Chunked: split present into groups of CHUNK_SIZE.
    let mut picks = vec![0usize; n_rows];
    let mut scores_flat = vec![-1e9f32; n_rows * width];

    for chunk_start in (0..present.len()).step_by(CHUNK_SIZE) {
        let chunk_end = (chunk_start + CHUNK_SIZE).min(present.len());
        let chunk_genomes: Vec<usize> = present[chunk_start..chunk_end].to_vec();

        // Find rows belonging to genomes in this chunk.
        let chunk_set: std::collections::HashSet<usize> = chunk_genomes.iter().copied().collect();
        let chunk_rows: Vec<usize> = (0..n_rows)
            .filter(|&r| chunk_set.contains(&genome_idx[r]))
            .collect();
        if chunk_rows.is_empty() {
            continue;
        }

        // Slice tensors to chunk rows.
        let row_idx: Vec<u32> = chunk_rows.iter().map(|&r| r as u32).collect();
        let row_idx_t = Tensor::from_vec(row_idx, chunk_rows.len(), device).unwrap();
        let chunk_obs = obs.index_select(&row_idx_t, 0).unwrap();
        let chunk_acts = acts.index_select(&row_idx_t, 0).unwrap();
        let chunk_mask = mask.index_select(&row_idx_t, 0).unwrap();

        // Remap genome_idx to chunk-local positions.
        let mut gmap = std::collections::HashMap::new();
        for (i, &g) in chunk_genomes.iter().enumerate() {
            gmap.insert(g, i);
        }
        let chunk_gidx: Vec<usize> = chunk_rows.iter().map(|&r| gmap[&genome_idx[r]]).collect();

        // Build the chunk's weight stack on the device.
        let chunk_stack = roster.build_chunk(&chunk_genomes, device).unwrap();

        let local_present: Vec<usize> = (0..chunk_genomes.len()).collect();
        let chunk_n_max = local_present
            .iter()
            .map(|&g| chunk_gidx.iter().filter(|&&gg| gg == g).count())
            .max()
            .unwrap_or(0)
            .max(4);

        let out = forward_scores_chunk(
            &chunk_stack,
            &chunk_obs,
            &chunk_acts,
            &chunk_mask,
            &chunk_gidx,
            &local_present,
            chunk_n_max,
        );

        // Scatter picks and scores back to global row order.
        for (local_i, &global_r) in chunk_rows.iter().enumerate() {
            picks[global_r] = out.picks[local_i];
            let src = local_i * width;
            let dst = global_r * width;
            scores_flat[dst..dst + width].copy_from_slice(&out.scores_flat[src..src + width]);
        }
    }

    ForwardOutput {
        picks,
        scores_flat,
        width,
    }
}

/// Convenience: forward + argmax picks only, computing `present` and `n_max`
/// internally from `genome_idx`. Used by the GpuServer fast path.
///
/// When the stack holds all genomes on GPU (cached path), processes in
/// row-batches of `BATCH_ROWS` to bound activation memory. Each batch forwards
/// against only the genomes present in those rows (sliced from the cached
/// stack — small, avoids copying the full stack). This keeps peak GPU memory
/// at ~stack_size + batch_activations, regardless of population or ply size.
pub fn forward_picks(
    stack: &WeightStack,
    obs: &Tensor,
    acts: &Tensor,
    mask: &Tensor,
    genome_idx: &[usize],
) -> Vec<usize> {
    const BATCH_ROWS: usize = 1024;
    let n_rows = obs.dims()[0];

    if n_rows == 0 {
        return Vec::new();
    }

    if n_rows <= BATCH_ROWS {
        // Small enough to forward in one pass.
        return forward_picks_onepass(stack, obs, acts, mask, genome_idx);
    }

    // Row-batched: split rows into batches, forward each against only the
    // genomes present in that batch. This bounds activation memory to
    // [BATCH_GENOMES, n_max, width, act_dim] instead of
    // [ALL_GENOMES, n_max, width, act_dim].
    let device = &stack.device;
    let mut picks = vec![0usize; n_rows];

    for batch_start in (0..n_rows).step_by(BATCH_ROWS) {
        let batch_end = (batch_start + BATCH_ROWS).min(n_rows);
        let batch_idx: Vec<u32> = (batch_start..batch_end).map(|i| i as u32).collect();
        let batch_idx_t = Tensor::from_vec(batch_idx, batch_end - batch_start, device).unwrap();
        let batch_obs = obs.index_select(&batch_idx_t, 0).unwrap();
        let batch_acts = acts.index_select(&batch_idx_t, 0).unwrap();
        let batch_mask = mask.index_select(&batch_idx_t, 0).unwrap();
        let batch_gidx: Vec<usize> = genome_idx[batch_start..batch_end].to_vec();

        let batch_picks =
            forward_picks_onepass(stack, &batch_obs, &batch_acts, &batch_mask, &batch_gidx);
        for (i, &p) in batch_picks.iter().enumerate() {
            picks[batch_start + i] = p;
        }
    }

    picks
}

/// One-pass forward: slice to present genomes, forward, return picks.
/// Activation memory is [present.len(), n_max, width, act_dim].
fn forward_picks_onepass(
    stack: &WeightStack,
    obs: &Tensor,
    acts: &Tensor,
    mask: &Tensor,
    genome_idx: &[usize],
) -> Vec<usize> {
    let n_groups = stack.n_groups();
    let mut present_set = vec![false; n_groups];
    for &g in genome_idx {
        present_set[g] = true;
    }
    let present: Vec<usize> = (0..n_groups).filter(|&g| present_set[g]).collect();
    let n_max = present
        .iter()
        .map(|&g| genome_idx.iter().filter(|&&gg| gg == g).count())
        .max()
        .unwrap_or(0)
        .max(4);

    let sub = if present.len() < n_groups {
        stack.slice(&present)
    } else {
        WeightStack {
            trunk_w: stack.trunk_w.clone(),
            trunk_b: stack.trunk_b.clone(),
            head_w: stack.head_w.clone(),
            head_b: stack.head_b.clone(),
            device: stack.device.clone(),
        }
    };
    let mut gmap = std::collections::HashMap::new();
    for (i, &g) in present.iter().enumerate() {
        gmap.insert(g, i);
    }
    let local_gidx: Vec<usize> = genome_idx.iter().map(|&g| gmap[&g]).collect();
    let local_present: Vec<usize> = (0..present.len()).collect();
    forward_scores_chunk(&sub, obs, acts, mask, &local_gidx, &local_present, n_max).picks
}

/// Forward pass: returns per-row argmax picks AND the full masked score
/// matrix `[n_rows, width]` (original row order, padded columns = -1e9).
///
/// obs: [N, OBS_DIM] on device
/// acts: [N, width, ACT_DIM] on device
/// mask: [N, width] (bool) on device
/// genome_idx: [N] which genome owns each row
/// present: distinct genome indices present
/// n_max: max rows per genome (padded to >=4)
///
/// For large rosters, this automatically chunks the forward over groups of
/// `CHUNK_SIZE` genomes to bound GPU memory. Each chunk is an independent
/// forward pass on a [CHUNK_SIZE, n_max, ...] tensor; results are reassembled
/// into the original row order. This is mathematically identical to the
/// un-chunked forward (each genome's forward is independent).
pub fn forward_scores(
    stack: &WeightStack,
    obs: &Tensor,
    acts: &Tensor,
    mask: &Tensor,
    genome_idx: &[usize],
    present: &[usize],
    n_max: usize,
) -> ForwardOutput {
    const CHUNK_SIZE: usize = 64;

    let n_rows = obs.dims()[0];
    let width = acts.dims()[1];

    if n_rows == 0 {
        return ForwardOutput {
            picks: Vec::new(),
            scores_flat: Vec::new(),
            width: 0,
        };
    }

    if present.len() <= CHUNK_SIZE {
        // Slice the stack to only present genomes (if not already), then
        // forward in one pass. Remap genome_idx to local positions.
        let n_groups = stack.n_groups();
        let sub = if present.len() < n_groups {
            stack.slice(present)
        } else {
            // All genomes present — use the stack as-is.
            // Can't shallow_clone without owning, so clone the tensor refs.
            WeightStack {
                trunk_w: stack.trunk_w.clone(),
                trunk_b: stack.trunk_b.clone(),
                head_w: stack.head_w.clone(),
                head_b: stack.head_b.clone(),
                device: stack.device.clone(),
            }
        };
        let mut gmap = std::collections::HashMap::new();
        for (i, &g) in present.iter().enumerate() {
            gmap.insert(g, i);
        }
        let local_gidx: Vec<usize> = genome_idx.iter().map(|&g| gmap[&g]).collect();
        let local_present: Vec<usize> = (0..present.len()).collect();
        return forward_scores_chunk(&sub, obs, acts, mask, &local_gidx, &local_present, n_max);
    }

    // Chunked: split present into groups of CHUNK_SIZE, forward each, combine.
    let mut picks = vec![0usize; n_rows];
    let mut scores_flat = vec![-1e9f32; n_rows * width];

    for chunk_start in (0..present.len()).step_by(CHUNK_SIZE) {
        let chunk_end = (chunk_start + CHUNK_SIZE).min(present.len());
        let chunk_present: Vec<usize> = present[chunk_start..chunk_end].to_vec();

        // Find rows belonging to genomes in this chunk.
        let chunk_set: std::collections::HashSet<usize> = chunk_present.iter().copied().collect();
        let chunk_rows: Vec<usize> = (0..n_rows)
            .filter(|&r| chunk_set.contains(&genome_idx[r]))
            .collect();
        if chunk_rows.is_empty() {
            continue;
        }

        // Slice tensors to chunk rows.
        let device = &stack.device;
        let row_idx: Vec<u32> = chunk_rows.iter().map(|&r| r as u32).collect();
        let row_idx_t = Tensor::from_vec(row_idx.clone(), chunk_rows.len(), device).unwrap();
        let chunk_obs = obs.index_select(&row_idx_t, 0).unwrap();
        let chunk_acts = acts.index_select(&row_idx_t, 0).unwrap();
        let chunk_mask = mask.index_select(&row_idx_t, 0).unwrap();

        // Remap genome_idx to chunk-local positions.
        let mut gmap = std::collections::HashMap::new();
        for (i, &g) in chunk_present.iter().enumerate() {
            gmap.insert(g, i);
        }
        let chunk_gidx: Vec<usize> = chunk_rows.iter().map(|&r| gmap[&genome_idx[r]]).collect();

        // Compute n_max for this chunk using local indices.
        let local_present: Vec<usize> = (0..chunk_present.len()).collect();
        let chunk_n_max = local_present
            .iter()
            .map(|&g| chunk_gidx.iter().filter(|&&gg| gg == g).count())
            .max()
            .unwrap_or(0)
            .max(4);

        // Slice the weight stack to this chunk.
        let chunk_stack = stack.slice(&chunk_present);

        let out = forward_scores_chunk(
            &chunk_stack,
            &chunk_obs,
            &chunk_acts,
            &chunk_mask,
            &chunk_gidx,
            &local_present,
            chunk_n_max,
        );

        // Scatter picks and scores back to global row order.
        for (local_i, &global_r) in chunk_rows.iter().enumerate() {
            picks[global_r] = out.picks[local_i];
            let src = local_i * width;
            let dst = global_r * width;
            scores_flat[dst..dst + width].copy_from_slice(&out.scores_flat[src..src + width]);
        }
    }

    ForwardOutput {
        picks,
        scores_flat,
        width,
    }
}

/// The actual forward pass (one chunk, no further splitting).
fn forward_scores_chunk(
    stack: &WeightStack,
    obs: &Tensor,
    acts: &Tensor,
    mask: &Tensor,
    genome_idx: &[usize],
    present: &[usize],
    n_max: usize,
) -> ForwardOutput {
    let n_rows = obs.dims()[0];
    let obs_dim = obs.dims()[1];
    let width = acts.dims()[1];
    let act_dim = acts.dims()[2];
    let n_present = present.len();

    if n_rows == 0 {
        return ForwardOutput {
            picks: Vec::new(),
            scores_flat: Vec::new(),
            width: 0,
        };
    }

    // Build the [G_present, n_max] padded index into global rows.
    let mut sorted: Vec<(usize, usize)> = genome_idx
        .iter()
        .enumerate()
        .map(|(row, &g)| (row, g))
        .collect();
    sorted.sort_by_key(|&(_, g)| g);

    let n_groups = stack.n_groups();
    let mut genome_to_pos = vec![usize::MAX; n_groups];
    for (i, &g) in present.iter().enumerate() {
        if g >= n_groups {
            eprintln!("ERROR: genome_idx {g} >= n_groups {n_groups} (present={present:?}, n_rows={n_rows})");
            panic!("genome index out of bounds");
        }
        genome_to_pos[g] = i;
    }

    let mut counts = vec![0usize; present.len()];
    let mut starts = vec![0usize; present.len()];
    let mut cum = 0;
    for (i, &g) in present.iter().enumerate() {
        let cnt = genome_idx.iter().filter(|&&gg| gg == g).count();
        starts[i] = cum;
        cum += cnt;
        counts[i] = cnt;
    }

    let mut sorted_rows = vec![0u32; n_rows];
    let mut pos = vec![0usize; present.len()];
    for &(row, g) in &sorted {
        let p = genome_to_pos[g];
        sorted_rows[starts[p] + pos[p]] = row as u32;
        pos[p] += 1;
    }

    // Fill padded [G_present, n_max].
    let mut row_order = vec![0u32; n_present * n_max];
    let mut valid = vec![false; n_present * n_max];
    for (i, &cnt) in counts.iter().enumerate() {
        for j in 0..cnt.min(n_max) {
            row_order[i * n_max + j] = sorted_rows[starts[i] + j];
            valid[i * n_max + j] = true;
        }
    }

    let row_order_t =
        Tensor::from_vec(row_order.clone(), (n_present * n_max,), &stack.device).unwrap();
    let valid_t = Tensor::from_vec(
        valid
            .iter()
            .map(|&v| if v { 1u32 } else { 0u32 })
            .collect::<Vec<_>>(),
        (n_present * n_max,),
        &stack.device,
    )
    .unwrap();

    // Gather: [G_present, n_max, ...]
    let obs_g = obs
        .index_select(&row_order_t, 0)
        .unwrap()
        .reshape((n_present, n_max, obs_dim))
        .unwrap();
    let acts_g = acts
        .index_select(&row_order_t, 0)
        .unwrap()
        .reshape((n_present, n_max, width, act_dim))
        .unwrap();
    let mask_g = mask
        .index_select(&row_order_t, 0)
        .unwrap()
        .reshape((n_present, n_max, width))
        .unwrap();

    // Mask out invalid (padded) rows — use U32 to match mask_g's dtype.
    let valid_reshaped = valid_t.reshape((n_present, n_max, 1)).unwrap();
    let mask_g = mask_g.broadcast_mul(&valid_reshaped).unwrap();

    // Stacked forward: trunk (all tanh)
    let mut x = obs_g; // [G, n_max, OBS]
    for (w, b) in stack.trunk_w.iter().zip(&stack.trunk_b) {
        x = stacked_bmm(&x, w); // [G, n_max, out]
                                // Add bias: [G, out] → [G, 1, out] broadcast
        let b_expanded = b.reshape((stack.n_groups(), 1, ())).unwrap();
        x = x.broadcast_add(&b_expanded).unwrap();
        x = x.tanh().unwrap();
    }
    let emb = x; // [G, n_max, E]

    // Head: first layer is linear over cat(emb, acts). Split weight.
    let head_w0 = &stack.head_w[0];
    let head_b0 = &stack.head_b[0];
    let emb_dim = emb.dims()[2];
    // head_w0 is [G, out, emb_dim + act_dim]. Split on dim=2 (the `in` dim)
    // into emb_w [G, out, emb_dim] and act_w [G, out, act_dim].
    let emb_w = head_w0.narrow(2, 0, emb_dim).unwrap();
    let act_w = head_w0.narrow(2, emb_dim, act_dim).unwrap();

    // emb_in: [G, n_max, H] via bmm([G, n_max, emb_dim] × [G, emb_dim, H])
    let emb_in = stacked_bmm(&emb, &emb_w); // [G, n_max, H]
    let emb_in = emb_in.unsqueeze(2).unwrap(); // [G, n_max, 1, H]

    // act_in: [G, n_max, width, H] via reshaped bmm.
    // act_w [G, out, act_dim] needs transpose to [G, act_dim, out] for bmm.
    let g_dim = n_present;
    let acts_flat = acts_g.reshape((g_dim, n_max * width, act_dim)).unwrap();
    let act_w_t = act_w.transpose(1, 2).unwrap().contiguous().unwrap(); // [G, act_dim, H]
    let act_in = acts_flat.matmul(&act_w_t).unwrap(); // [G, n_max*width, H]
    let act_in = act_in.reshape((g_dim, n_max, width, ())).unwrap();

    // Combine: emb_in + act_in + bias
    let mut x = emb_in.broadcast_add(&act_in).unwrap();
    let b0_expanded = head_b0.reshape((g_dim, 1, 1, ())).unwrap();
    x = x.broadcast_add(&b0_expanded).unwrap();
    x = x.tanh().unwrap();
    x = x.reshape((g_dim, n_max * width, ())).unwrap();

    // Remaining head layers
    let n_head = stack.head_w.len();
    for (i, (w, b)) in stack.head_w.iter().zip(&stack.head_b).enumerate().skip(1) {
        x = stacked_bmm(&x, w);
        let b_expanded = b.reshape((g_dim, 1, ())).unwrap();
        x = x.broadcast_add(&b_expanded).unwrap();
        if i < n_head - 1 {
            x = x.tanh().unwrap();
        }
    }

    // scores: [G, n_max, width]
    let scores = x.reshape((g_dim, n_max, width)).unwrap();

    // Mask: set invalid columns to -inf by adding (mask - 1) * large_negative.
    // candle's where_cond doesn't support F32 on CPU, so we use arithmetic:
    // masked = scores * mask + (1 - mask) * (-inf)
    // But -inf * 0 = NaN, so use: scores + (1 - mask) * (-1e9)
    let mask_f32 = mask_g.to_dtype(DType::F32).unwrap();
    let ones = Tensor::ones_like(&mask_f32).unwrap();
    let inv_mask = ones.sub(&mask_f32).unwrap();
    let neg_scalar = Tensor::new(-1e9f32, &stack.device)
        .unwrap()
        .broadcast_as(mask_f32.shape())
        .unwrap();
    let neg_offset = inv_mask.mul(&neg_scalar).unwrap();
    let masked = scores.broadcast_add(&neg_offset).unwrap();

    // Download masked scores to CPU for argmax. This guarantees first-max
    // tie-breaking on BOTH backends (matching numpy's argmax semantics).
    // Candle's GPU argmax does NOT return first-max on ties — it returns
    // the last-max, which diverges from numpy and the CPU path. Doing
    // argmax on the already-downloaded scores is free (we download them
    // for the scatter-back anyway) and eliminates the device argmax kernel.
    let scores_host: Vec<f32> = masked.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // CPU argmax: first-max wins ties (matches numpy/torch on CPU).
    let mut picks_host = vec![0u32; n_present * n_max];
    for gi in 0..n_present {
        for ni in 0..n_max {
            let base = (gi * n_max + ni) * width;
            let mut best = 0u32;
            let mut best_val = f32::NEG_INFINITY;
            for j in 0..width {
                let v = scores_host[base + j];
                if v > best_val {
                    best_val = v;
                    best = j as u32;
                }
            }
            picks_host[gi * n_max + ni] = best;
        }
    }

    let mut picks = vec![0usize; n_rows];
    for (i, &cnt) in counts.iter().enumerate() {
        for j in 0..cnt {
            let sorted_pos = starts[i] + j;
            let orig_row = sorted_rows[sorted_pos] as usize;
            picks[orig_row] = picks_host[i * n_max + j] as usize;
        }
    }

    // Scatter scores back to original row order, padded to the ply width.
    // Padded columns (and padded rows) carry -1e9 (the mask sentinel), so a
    // caller comparing against Python's [N, width] -inf logits sees the same
    // shape on both sides. `scores_host` is [G_present, n_max, width].
    let mut scores_flat = vec![-1e9f32; n_rows * width];
    for (i, &cnt) in counts.iter().enumerate() {
        for j in 0..cnt {
            let sorted_pos = starts[i] + j;
            let orig_row = sorted_rows[sorted_pos] as usize;
            let src_base = (i * n_max + j) * width;
            let dst_base = orig_row * width;
            scores_flat[dst_base..dst_base + width]
                .copy_from_slice(&scores_host[src_base..src_base + width]);
        }
    }

    ForwardOutput {
        picks,
        scores_flat,
        width,
    }
}

/// Forward pass + argmax, returning picks only. Thin wrapper over
/// `forward_scores` for callers that don't need the logits.
pub fn forward_and_pick(
    stack: &WeightStack,
    obs: &Tensor,
    acts: &Tensor,
    mask: &Tensor,
    genome_idx: &[usize],
    present: &[usize],
    n_max: usize,
) -> Vec<usize> {
    forward_scores(stack, obs, acts, mask, genome_idx, present, n_max).picks
}

/// Build a WeightStack from a single genome (for evaluation).
pub fn single_genome_weights(genome: &Genome, arch: &Arch, device: &Device) -> WeightStack {
    WeightStack::from_roster(&[genome], arch, device)
}
