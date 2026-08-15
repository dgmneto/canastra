//! Flat parameter genomes ↔ weights JSON, and the pinned arch definition.
//!
//! Mirrors `training/python/canastra_train/genome.py` exactly: the same
//! `canastra-weights@1` format, the same layer ordering, the same random init.

use serde_json::Map;

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
pub fn layer_shapes(arch: &Arch) -> Vec<(String, usize, usize)> {
    let mut result: Vec<(String, usize, usize)> = Vec::new();
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

/// Round to 6 decimals matching Python's `np.round(f32, 6).tolist()`.
///
/// Python: rounds in f64, casts result to f32, then `.tolist()` converts back
/// to f64 (giving the full f64 representation of the f32). We replicate: round
/// in f64, cast to f32, cast back to f64. This makes the JSON values
/// byte-for-byte the same as Python's output.
pub fn round6_f32(v: f32) -> f64 {
    let rounded_f64 = (v as f64 * 1e6).round_ties_even() / 1e6;
    rounded_f64 as f32 as f64
}

/// Round to 6 decimals using round-half-to-even (banker's rounding), matching
/// numpy's `np.round(x, 6)` semantics for f64 inputs.
pub fn round6(v: f64) -> f64 {
    (v * 1e6).round_ties_even() / 1e6
}

/// Save genome as `canastra-weights@1` JSON.
pub fn save_json(path: &str, arch: &Arch, vec: &[f32]) -> anyhow::Result<()> {
    let shapes = layer_shapes(arch);
    let mut params = Map::new();
    let mut offset = 0usize;

    for (name, out, inn) in &shapes {
        let w_size = out * inn;
        let w_data: Vec<f64> = vec[offset..offset + w_size]
            .iter()
            .map(|&v| round6_f32(v))
            .collect();
        offset += w_size;
        let b_data: Vec<f64> = vec[offset..offset + out]
            .iter()
            .map(|&v| round6_f32(v))
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

    // Match Python's `json.dump` formatting: compact with `, ` and `: ` separators.
    // serde_json::to_string gives `,` and `:` (no spaces). The string values in
    // the payload (format, activation, param keys) contain no `,` or `:`, so a
    // global replace is safe.
    let compact = serde_json::to_string(&payload)?;
    let python_formatted = compact.replace(',', ", ").replace(':', ": ");
    std::fs::write(path, python_formatted)?;
    Ok(())
}

/// Load a `canastra-weights@1` JSON file into a flat genome.
///
/// Mirrors Python `genome.load_json`: validates format/activation, checks
/// every shape and length, and fills the flat vector in genome (layer) order.
pub fn load_json(path: &str) -> anyhow::Result<(Arch, Genome)> {
    let payload: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    if payload.get("format").and_then(|v| v.as_str()) != Some(FORMAT) {
        anyhow::bail!("unsupported weights format: {:?}", payload.get("format"));
    }
    let arch = payload
        .get("arch")
        .ok_or_else(|| anyhow::anyhow!("missing arch"))?;
    if arch.get("activation").and_then(|v| v.as_str()) != Some("tanh") {
        anyhow::bail!("only tanh weights are supported");
    }
    let trunk: Vec<usize> = serde_json::from_value(arch.get("trunk").cloned().unwrap())?;
    let head: Vec<usize> = serde_json::from_value(arch.get("head").cloned().unwrap())?;
    let obs = arch.get("obs").and_then(|v| v.as_u64()).unwrap() as usize;
    let act = arch.get("act").and_then(|v| v.as_u64()).unwrap() as usize;
    // `Arch` holds static slices; clone the parsed vecs into leaked boxes so the
    // returned `Arch` owns them. Callers use the const `TRAINING_ARCH` in
    // practice; this path exists for arbitrary files (validation/evaluation).
    let trunk_box = trunk.leak();
    let head_box = head.leak();
    let arch = Arch {
        obs,
        act,
        trunk: trunk_box,
        head: head_box,
    };

    let size = genome_size(&arch);
    let mut vec = Genome::with_capacity(size);
    let mut offset = 0usize;
    for (name, out, inn) in &layer_shapes(&arch) {
        for (key, size_len) in [
            (format!("{}.weight", name), out * inn),
            (format!("{}.bias", name), *out),
        ] {
            let entry = payload["params"]
                .get(&key)
                .ok_or_else(|| anyhow::anyhow!("missing key: {}", key))?;
            let want_shape = if key.ends_with(".weight") {
                vec![*out as i64, *inn as i64]
            } else {
                vec![*out as i64]
            };
            let got_shape: Vec<i64> = serde_json::from_value(entry.get("shape").cloned().unwrap())?;
            if got_shape != want_shape {
                anyhow::bail!("{}: shape {:?} does not match the arch", key, got_shape);
            }
            let data: Vec<f32> = serde_json::from_value(entry.get("data").cloned().unwrap())?;
            if data.len() != size_len {
                anyhow::bail!("{}: {} values, expected {}", key, data.len(), size_len);
            }
            vec.extend_from_slice(&data);
            offset += size_len;
        }
    }
    assert_eq!(
        offset, size,
        "load_json consumed {offset} but genome_size is {size}"
    );
    Ok((arch, vec))
}
