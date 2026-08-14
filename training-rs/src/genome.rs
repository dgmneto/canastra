//! Flat parameter genomes ↔ weights JSON, and the pinned arch definition.
//!
//! Mirrors `training/python/canastra_train/genome.py` exactly: the same
//! `canastra-weights@1` format, the same layer ordering, the same random init.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

pub const FORMAT: &str = "canastra-weights@1";

/// The training architecture (must match Python `TRAINING_ARCH`).
pub const TRAINING_ARCH: Arch = Arch {
    obs: 2002,
    act: 101,
    trunk: &[512, 256],
    head: &[128],
};

/// Network architecture descriptor. Stored as const slices for zero-allocation
/// access, with a Serialize/Deserialize impl for JSON.
#[derive(Debug, Clone)]
pub struct Arch {
    pub obs: usize,
    pub act: usize,
    pub trunk: &'static [usize],
    pub head: &'static [usize],
}

/// (name, out, in) for every layer, in genome order.
pub fn layer_shapes(arch: &Arch) -> Vec<(&'static str, usize, usize)> {
    let mut result: Vec<(&'static str, usize, usize)> = Vec::new();
    let mut prev = arch.obs;
    for (i, &width) in arch.trunk.iter().enumerate() {
        result.push(("trunk", width, prev));
        prev = width;
    }
    prev += arch.act;
    for (i, &width) in arch.head.iter().enumerate() {
        result.push(("head", width, prev));
        prev = width;
    }
    result.push(("head.out", 1, prev));
    result
}

/// Genome size (total floats) for the given arch.
pub fn genome_size(arch: &Arch) -> usize {
    layer_shapes(arch)
        .iter()
        .map(|(_, out, inn)| out * inn + out)
        .sum()
}

/// A flat genome as owned f32 values.
pub type Genome = Vec<f32>;

/// Random genome initialized with N(0, 0.1).
pub fn random_genome(arch: &Arch, seed: u64) -> Genome {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let size = genome_size(arch);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut g = Genome::with_capacity(size);

    for i in (0..size).step_by(2) {
        let u1 = (rng.gen::<u32>() as f64 / u32::MAX as f64).max(1e-10);
        let u2 = rng.gen::<u32>() as f64 / u32::MAX as f64;
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        g.push((r * theta.cos() * 0.1) as f32);
        if i + 1 < size {
            g.push((r * theta.sin() * 0.1) as f32);
        }
    }
    g
}

/// Save genome as `canastra-weights@1` JSON.
pub fn save_json(path: &str, arch: &Arch, vec: &[f32]) -> anyhow::Result<()> {
    let shapes = layer_shapes(arch);
    let mut params = HashMap::new();
    let mut offset = 0usize;

    // Build named layers.
    let mut named: Vec<(String, usize, usize)> = Vec::new();
    let mut prev = arch.obs;
    for (i, &width) in arch.trunk.iter().enumerate() {
        named.push((format!("trunk.{}", i), width, prev));
        prev = width;
    }
    prev += arch.act;
    for (i, &width) in arch.head.iter().enumerate() {
        named.push((format!("head.{}", i), width, prev));
        prev = width;
    }
    named.push(("head.out".to_string(), 1, prev));

    for (name, out, inn) in &named {
        let w_size = out * inn;
        let w_data: Vec<f64> = vec[offset..offset + w_size]
            .iter()
            .map(|&v| ((v as f64) * 1e6).round() / 1e6)
            .collect();
        offset += w_size;
        let b_data: Vec<f64> = vec[offset..offset + out]
            .iter()
            .map(|&v| ((v as f64) * 1e6).round() / 1e6)
            .collect();
        offset += out;
        params.insert(
            format!("{}.weight", name),
            serde_json::json!({"shape": [out, inn], "data": w_data}),
        );
        params.insert(
            format!("{}.bias", name),
            serde_json::json!({"shape": [out], "data": b_data}),
        );
    }

    let payload = serde_json::json!({
        "format": FORMAT,
        "arch": {
            "obs": arch.obs,
            "act": arch.act,
            "trunk": arch.trunk,
            "head": arch.head,
            "activation": "tanh",
        },
        "params": params,
    });

    std::fs::write(path, serde_json::to_string_pretty(&payload)?)?;
    Ok(())
}
