//! Batched policy scoring via candle (pure Rust ML with CUDA via cudarc).
//!
//! Mirrors `training/python/canastra_train/policy.py` exactly: trunk layers
//! all tanh, head hidden layers tanh, final layer linear. The stacked forward
//! processes the whole genome roster in one batched pass.

use crate::genome::{Arch, Genome};
use candle_core::{DType, Device, Tensor};

/// Per-layer weight tensors, stacked on a leading G (genome) dimension.
/// Built once per generation, reused every ply.
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
fn stacked_bmm(x: &Tensor, w: &Tensor) -> Tensor {
    // x: [G, N, in], w: [G, out, in] → need [G, in, out] for bmm
    let w_t = w.transpose(1, 2).unwrap(); // [G, in, out]
    x.matmul(&w_t).unwrap()
}

/// Forward pass + argmax. Returns one pick per row.
///
/// obs: [N, OBS_DIM] on device
/// acts: [N, width, ACT_DIM] on device
/// mask: [N, width] on device
/// genome_idx: [N] which genome owns each row
/// present: distinct genome indices present
/// n_max: max rows per genome (padded to ≥4)
pub fn forward_and_pick(
    stack: &WeightStack,
    obs: &Tensor,  // [N, OBS_DIM]
    acts: &Tensor, // [N, width, ACT_DIM]
    mask: &Tensor, // [N, width] (bool)
    genome_idx: &[usize],
    present: &[usize],
    n_max: usize,
) -> Vec<usize> {
    let n_rows = obs.dims()[0];
    let obs_dim = obs.dims()[1];
    let width = acts.dims()[1];
    let act_dim = acts.dims()[2];
    let n_present = present.len();

    if n_rows == 0 {
        return Vec::new();
    }

    // Build the [G_present, n_max] padded index into global rows.
    let mut sorted: Vec<(usize, usize)> = genome_idx
        .iter()
        .enumerate()
        .map(|(row, &g)| (row, g))
        .collect();
    sorted.sort_by(|a, b| a.1.cmp(&b.1));

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
    // Split the first head layer's weight: [G, out, emb_dim + act_dim]
    // into emb_w [G, out, emb_dim] and act_w [G, out, act_dim].
    let emb_w = head_w0.narrow(2, 0, emb_dim).unwrap();
    let act_w = head_w0.narrow(2, emb_dim, act_dim).unwrap();

    // emb_in: [G, n_max, H] via bmm([G, n_max, emb_dim] × [G, emb_dim, H])
    let emb_in = stacked_bmm(&emb, &emb_w); // [G, n_max, H]
    let emb_in = emb_in.unsqueeze(2).unwrap(); // [G, n_max, 1, H]

    // act_in: [G, n_max, width, H] via reshaped bmm
    let g_dim = n_present as usize;
    let acts_flat = acts_g.reshape((g_dim, n_max * width, act_dim)).unwrap();
    let act_w_t = act_w.transpose(1, 2).unwrap(); // [G, act_dim, H]
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

    // argmax over width dimension (dim=2)
    let picks_sorted = masked.argmax(2).unwrap();

    // Extract picks and scatter back to original row order.
    let picks_host: Vec<u32> = picks_sorted
        .flatten_all()
        .unwrap()
        .to_vec1::<u32>()
        .unwrap();

    let mut picks = vec![0usize; n_rows];
    for (i, &cnt) in counts.iter().enumerate() {
        for j in 0..cnt {
            let sorted_pos = starts[i] + j;
            let orig_row = sorted_rows[sorted_pos] as usize;
            picks[orig_row] = picks_host[i * n_max + j] as usize;
        }
    }
    picks
}

/// Build a WeightStack from a single genome (for evaluation).
pub fn single_genome_weights(genome: &Genome, arch: &Arch, device: &Device) -> WeightStack {
    WeightStack::from_roster(&[genome], arch, device)
}
