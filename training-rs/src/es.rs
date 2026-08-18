//! Seed-based Evolution Strategies (ES) for gradient-free optimisation.
//!
//! Replaces GA's independent genomes with a single shared policy `θ` plus a
//! set of perturbation seeds. Each "genome" in the population is `θ ± σεᵢ`,
//! regenerated on demand from `(seed, sigma)` — the population collapses from
//! `G × genome_size` storage to `base_params + Vec<(seed, sigma)>`.
//!
//! **Two structural wins over GA:**
//!
//! 1. **Memory:** 2000 genomes × 1.2M × 4B = 9.6 GB → `base_params` (4.8 MB)
//!    + `Vec<(seed, sigma)>` (kilobytes). Population 10k+ becomes trivial.
//! 2. **Noise tolerance:** Mirrored (antithetic) sampling + rank-normalised
//!    fitness shaping averages the gradient estimate over the whole population.
//!    Noisy per-genome fitness is fine because nothing is selected — everything
//!    is weighted by rank.
//!
//! **The grouped-GEMM insight:** Under ES, `xᵢWᵢ = xᵢW + σ·(xᵢEᵢ)`. The first
//! term is one dense GEMM over all rows against a single shared weight matrix
//! (M = G·K = 64,000+, perfect efficiency, no grouping). Only the perturbation
//! term needs per-genome treatment. This is future work — the current
//! implementation materialises genomes and reuses the existing league.

use crate::genome::{genome_size, Arch, Genome};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// ES configuration.
pub struct ESConfig {
    /// Number of perturbation pairs (each pair = θ+σε and θ−σε).
    /// Population size for evaluation = 2 × n_perturbations.
    pub n_perturbations: usize,
    /// Perturbation noise scale (σ).
    pub sigma: f64,
    /// Sigma decay per generation.
    pub sigma_decay: f64,
    /// Minimum sigma (floor).
    pub sigma_floor: f64,
    /// Adam learning rate.
    pub lr: f64,
    /// Adam beta1 (momentum decay).
    pub beta1: f64,
    /// Adam beta2 (variance decay).
    pub beta2: f64,
    /// Weight decay (L2 regularisation).
    pub weight_decay: f64,
    /// Adam epsilon (numerical stability).
    pub eps: f64,
    /// HOF archival interval.
    pub hof_interval: u32,
}

impl Default for ESConfig {
    fn default() -> Self {
        Self {
            n_perturbations: 48, // 96 genomes = 48 mirrored pairs
            sigma: 0.02,
            sigma_decay: 0.995,
            sigma_floor: 0.002,
            lr: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.0,
            eps: 1e-8,
            hof_interval: 5,
        }
    }
}

/// The ES population state: a single base policy + Adam moments + perturbation
/// seeds. The actual genomes (θ ± σεᵢ) are materialised on demand.
pub struct ESState {
    /// Base policy parameters (the shared θ being optimised).
    pub base_params: Genome,
    /// Perturbation seeds: one per perturbation pair. Regenerating εᵢ from
    /// `seed_i` is deterministic, so no genome storage is needed.
    pub perturbation_seeds: Vec<u64>,
    /// Current sigma (decays over generations).
    pub sigma: f64,
    /// Adam first moment (momentum) — same shape as base_params.
    pub m: Vec<f32>,
    /// Adam second moment (variance) — same shape as base_params.
    pub v: Vec<f32>,
    /// Adam timestep.
    pub t: u32,
}

impl ESState {
    /// Initialise ES from a starting genome (e.g. random init or a loaded
    /// checkpoint). Perturbation seeds are derived from `run_seed`.
    pub fn new(base_params: Genome, cfg: &ESConfig, run_seed: u64) -> Self {
        let size = base_params.len();
        let mut rng = StdRng::seed_from_u64(run_seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let perturbation_seeds: Vec<u64> =
            (0..cfg.n_perturbations).map(|_| rng.gen::<u64>()).collect();
        ESState {
            base_params,
            perturbation_seeds,
            sigma: cfg.sigma,
            m: vec![0.0f32; size],
            v: vec![0.0f32; size],
            t: 0,
        }
    }

    /// Materialise the full evaluation population: 2 × n_perturbations genomes.
    /// Genome `2*i` = θ + σεᵢ, genome `2*i+1` = θ − σεᵢ (mirrored/antithetic).
    pub fn materialise_population(&self, arch: &Arch) -> Vec<Genome> {
        let size = genome_size(arch);
        let n = self.perturbation_seeds.len();
        let mut pop = Vec::with_capacity(2 * n);
        for &seed in &self.perturbation_seeds {
            let eps = generate_perturbation(seed, size);
            let plus = perturb(&self.base_params, &eps, self.sigma as f32);
            let minus = perturb(&self.base_params, &eps, -(self.sigma as f32));
            pop.push(plus);
            pop.push(minus);
        }
        pop
    }

    /// The number of genomes in the materialised population.
    pub fn pop_size(&self) -> usize {
        2 * self.perturbation_seeds.len()
    }

    /// Update the base policy using rank-normalised fitness from the mirrored
    /// evaluation. `fitness[i]` is the fitness of genome `i` (in materialise
    /// order: pair 0 plus, pair 0 minus, pair 1 plus, ...).
    ///
    /// The gradient estimate for perturbation pair `j` (seeds εⱼ) is:
    ///   grad ≈ Σ_j rank_normalized(f_{2j} - f_{2j+1}) · εⱼ / (n · σ)
    ///
    /// Mirrored sampling: the difference f⁺ − f⁻ cancels the bias of the
    /// base policy's own fitness, leaving only the directional derivative.
    ///
    /// That cancellation is only real if the twins were *measured under the
    /// same conditions* — anything that differs between their evaluations
    /// survives into the difference as noise. Schedule them with
    /// [`crate::league::schedule_pairings_mirrored`], which gives pair `j` a
    /// single opponent list shared by both twins; plain `schedule_pairings`
    /// draws opponents independently per genome and leaves the opponent draw
    /// dominating the signal at small σ.
    pub fn update(&mut self, fitness: &[f64], cfg: &ESConfig) {
        let n = self.perturbation_seeds.len();
        assert_eq!(
            fitness.len(),
            2 * n,
            "fitness must cover all mirrored pairs"
        );

        // Rank-normalise fitness: centred ranks in [−0.5, 0.5].
        let ranks = centred_ranks(fitness);

        // For each pair, compute the weighted perturbation direction.
        // grad += (rank_plus - rank_minus) * eps_j / (n * sigma)
        // This is the standard ES gradient estimate with antithetic sampling.
        let size = self.base_params.len();
        let mut grad = vec![0.0f64; size];
        let scale = 1.0 / (n as f64 * self.sigma);
        for j in 0..n {
            let rank_plus = ranks[2 * j];
            let rank_minus = ranks[2 * j + 1];
            let weight = (rank_plus - rank_minus) * scale;
            let eps = generate_perturbation(self.perturbation_seeds[j], size);
            for i in 0..size {
                grad[i] += weight * eps[i] as f64;
            }
        }

        // Adam update (with optional weight decay).
        self.t += 1;
        let lr_t = cfg.lr * (1.0 - cfg.beta2.powi(self.t as i32)).sqrt()
            / (1.0 - cfg.beta1.powi(self.t as i32));

        for (i, &g_raw) in grad.iter().enumerate() {
            let g = g_raw + cfg.weight_decay * self.base_params[i] as f64;
            let new_m = (cfg.beta1 * self.m[i] as f64 + (1.0 - cfg.beta1) * g) as f32;
            let new_v = (cfg.beta2 * self.v[i] as f64 + (1.0 - cfg.beta2) * g * g) as f32;
            self.m[i] = new_m;
            self.v[i] = new_v;
            self.base_params[i] +=
                (lr_t * new_m as f64 / (new_v as f64).sqrt().max(cfg.eps)) as f32;
        }

        // Decay sigma.
        self.sigma = (self.sigma * cfg.sigma_decay).max(cfg.sigma_floor);
    }

    /// Decay sigma for generation `gen` (for deterministic sigma scheduling
    /// independent of update count, matching the GA path).
    pub fn sigma_for_generation(&mut self, cfg: &ESConfig, gen: u32) {
        self.sigma = (cfg.sigma * cfg.sigma_decay.powf(gen as f64)).max(cfg.sigma_floor);
    }
}

/// Generate a deterministic Gaussian perturbation from a seed.
/// Uses Box-Muller (same as the GA's mutation sampler) for reproducibility.
pub fn generate_perturbation(seed: u64, size: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut eps = Vec::with_capacity(size);
    for i in (0..size).step_by(2) {
        let u1 = (rng.gen::<u32>() as f64 / u32::MAX as f64).max(1e-10);
        let u2 = rng.gen::<u32>() as f64 / u32::MAX as f64;
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        eps.push((r * theta.cos()) as f32);
        if i + 1 < size {
            eps.push((r * theta.sin()) as f32);
        }
    }
    eps
}

/// Apply a perturbation to base params: `base + scale * eps`.
fn perturb(base: &[f32], eps: &[f32], scale: f32) -> Genome {
    base.iter()
        .zip(eps.iter())
        .map(|(&b, &e)| b + scale * e)
        .collect()
}

/// Centred ranks in [−0.5, 0.5]: sort indices by fitness, assign ranks,
/// centre to [−0.5, 0.5]. This is rank-normalised fitness shaping — it
/// removes the scale of fitness differences, leaving only the ordering,
/// which makes the gradient estimate robust to outliers.
fn centred_ranks(fitness: &[f64]) -> Vec<f64> {
    let n = fitness.len();
    let mut indexed: Vec<(usize, f64)> = fitness.iter().enumerate().map(|(i, &f)| (i, f)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0f64; n];
    // Handle ties: average rank for equal fitness values.
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && indexed[j].1 == indexed[i].1 {
            j += 1;
        }
        let avg_rank = (i + j - 1) as f64 / 2.0; // 0-indexed average
        let centred = avg_rank / (n - 1) as f64 - 0.5; // [−0.5, 0.5]
        for k in i..j {
            ranks[indexed[k].0] = centred;
        }
        i = j;
    }
    ranks
}

/// Save an ES checkpoint (base params + Adam moments + seeds + sigma + t).
/// Atomic: writes to a `.tmp` file, then renames. Rotation: keeps the last
/// `keep_recent` checkpoints plus every `keep_every`-th (matching GA's
/// rotation). Without rotation, a 2000-gen run at `checkpoint_interval=5`
/// would leave 400 checkpoints (each growing with the HOF) and fill the disk.
pub fn save_es_checkpoint(
    dir: &std::path::Path,
    generation: u32,
    state: &ESState,
    elo: &crate::elo::EloTracker,
    hof: &crate::hof::HallOfFame,
    seeds: &[u64],
) -> anyhow::Result<()> {
    save_es_checkpoint_with_rotation(dir, generation, state, elo, hof, seeds, 10, 50)
}

#[allow(clippy::too_many_arguments)]
pub fn save_es_checkpoint_with_rotation(
    dir: &std::path::Path,
    generation: u32,
    state: &ESState,
    elo: &crate::elo::EloTracker,
    hof: &crate::hof::HallOfFame,
    seeds: &[u64],
    keep_recent: usize,
    keep_every: u32,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("gen-{:05}.es.bin", generation));
    let tmp = dir.join(format!("gen-{:05}.es.bin.tmp", generation));

    let size = state.base_params.len();
    let mut data = Vec::new();

    // Header
    data.extend_from_slice(&generation.to_le_bytes());
    data.extend_from_slice(&(1u64).to_le_bytes()); // 1 base genome
    data.extend_from_slice(&(size as u64).to_le_bytes());
    data.extend_from_slice(&(elo.ratings.len() as u64).to_le_bytes());
    data.extend_from_slice(&(seeds.len() as u64).to_le_bytes());
    data.extend_from_slice(&(hof.genomes.len() as u64).to_le_bytes());
    data.extend_from_slice(&(state.perturbation_seeds.len() as u64).to_le_bytes());
    data.extend_from_slice(&state.t.to_le_bytes());
    data.extend_from_slice(&state.sigma.to_le_bytes());

    // Base params
    for &v in &state.base_params {
        data.extend_from_slice(&v.to_le_bytes());
    }
    // Adam m
    for &v in &state.m {
        data.extend_from_slice(&v.to_le_bytes());
    }
    // Adam v
    for &v in &state.v {
        data.extend_from_slice(&v.to_le_bytes());
    }
    // Perturbation seeds
    for &s in &state.perturbation_seeds {
        data.extend_from_slice(&s.to_le_bytes());
    }
    // ELO
    for &r in &elo.ratings {
        data.extend_from_slice(&r.to_le_bytes());
    }
    // Gen seeds
    for &s in seeds {
        data.extend_from_slice(&s.to_le_bytes());
    }
    // HOF
    for g in &hof.genomes {
        data.extend_from_slice(&(g.len() as u64).to_le_bytes());
        for &v in g {
            data.extend_from_slice(&v.to_le_bytes());
        }
    }
    for &r in &hof.elo_ratings {
        data.extend_from_slice(&r.to_le_bytes());
    }
    for &gen in &hof.generations {
        data.extend_from_slice(&gen.to_le_bytes());
    }

    std::fs::write(&tmp, &data)?;
    std::fs::rename(&tmp, &path)?;

    // Prune: keep last `keep_recent` plus every `keep_every`-th.
    // Only prune ES checkpoints (".es.bin"), never GA checkpoints (".bin").
    let mut checkpoints: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.starts_with("gen-") && name.ends_with(".es.bin") && !name.ends_with(".tmp")
        })
        .collect();
    checkpoints.sort_by_key(|e| e.file_name());
    let total = checkpoints.len();
    if total > keep_recent {
        let mut to_remove = Vec::new();
        for (idx, entry) in checkpoints.iter().enumerate() {
            if idx >= total - keep_recent {
                continue;
            }
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(num_str) = name
                .strip_prefix("gen-")
                .and_then(|s| s.strip_suffix(".es.bin"))
            {
                if let Ok(gen) = num_str.parse::<u32>() {
                    if gen % keep_every == 0 {
                        continue;
                    }
                }
            }
            to_remove.push(entry.path());
        }
        for path in to_remove {
            let _ = std::fs::remove_file(path);
        }
    }

    Ok(())
}

/// Loaded ES checkpoint.
pub type ESCheckpoint = (u32, ESState, Vec<f64>, Vec<u64>, crate::hof::HallOfFame);

/// Load an ES checkpoint.
pub fn load_es_checkpoint(dir: &std::path::Path) -> anyhow::Result<ESCheckpoint> {
    let mut checkpoints: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.starts_with("gen-") && name.ends_with(".es.bin")
        })
        .collect();
    checkpoints.sort_by_key(|e| e.file_name());
    let path = checkpoints
        .last()
        .ok_or_else(|| anyhow::anyhow!("no ES checkpoint in {}", dir.display()))?
        .path();

    let data = std::fs::read(&path)?;
    let mut pos = 0;
    let read_u32 = |data: &[u8], pos: &mut usize| -> u32 {
        let v = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        v
    };
    let read_u64 = |data: &[u8], pos: &mut usize| -> u64 {
        let v = u64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
        *pos += 8;
        v
    };
    let read_f64 = |data: &[u8], pos: &mut usize| -> f64 {
        let v = f64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
        *pos += 8;
        v
    };
    let read_f32_slice = |data: &[u8], pos: &mut usize, n: usize| -> Vec<f32> {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(f32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap()));
            *pos += 4;
        }
        v
    };

    let generation = read_u32(&data, &mut pos);
    let _n_base = read_u64(&data, &mut pos) as usize;
    let size = read_u64(&data, &mut pos) as usize;
    let elo_size = read_u64(&data, &mut pos) as usize;
    let seeds_size = read_u64(&data, &mut pos) as usize;
    let hof_size = read_u64(&data, &mut pos) as usize;
    let n_perturb = read_u64(&data, &mut pos) as usize;
    let t = read_u32(&data, &mut pos);
    let sigma = read_f64(&data, &mut pos);

    let base_params = read_f32_slice(&data, &mut pos, size);
    let m = read_f32_slice(&data, &mut pos, size);
    let v = read_f32_slice(&data, &mut pos, size);
    let perturbation_seeds: Vec<u64> = (0..n_perturb).map(|_| read_u64(&data, &mut pos)).collect();

    let elo: Vec<f64> = (0..elo_size).map(|_| read_f64(&data, &mut pos)).collect();
    let gen_seeds: Vec<u64> = (0..seeds_size).map(|_| read_u64(&data, &mut pos)).collect();

    let mut hof = crate::hof::HallOfFame::new();
    for _ in 0..hof_size {
        let gsz = read_u64(&data, &mut pos) as usize;
        hof.genomes.push(read_f32_slice(&data, &mut pos, gsz));
    }
    for _ in 0..hof_size {
        hof.elo_ratings.push(read_f64(&data, &mut pos));
    }
    for _ in 0..hof_size {
        hof.generations.push(read_u32(&data, &mut pos));
    }

    let state = ESState {
        base_params,
        perturbation_seeds,
        sigma,
        m,
        v,
        t,
    };

    Ok((generation, state, elo, gen_seeds, hof))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perturbation_is_reproducible() {
        let size = 100;
        let eps1 = generate_perturbation(42, size);
        let eps2 = generate_perturbation(42, size);
        assert_eq!(eps1, eps2, "same seed must produce same perturbation");

        let eps3 = generate_perturbation(43, size);
        assert_ne!(
            eps1, eps3,
            "different seeds must produce different perturbations"
        );
    }

    #[test]
    fn mirrored_pairs_are_exact_negations() {
        let cfg = ESConfig::default();
        let base = vec![0.5f32; 50];
        let state = ESState::new(base, &cfg, 42);
        let pop = state.materialise_population(&crate::genome::TRAINING_ARCH);

        assert_eq!(pop.len(), 2 * cfg.n_perturbations, "population size");

        // Each pair: plus and minus must be symmetric around base.
        let genome_size = pop[0].len();
        for j in 0..cfg.n_perturbations {
            let plus = &pop[2 * j];
            let minus = &pop[2 * j + 1];
            for i in 0..genome_size {
                let mid = (plus[i] + minus[i]) / 2.0;
                assert!(
                    (mid - state.base_params[i]).abs() < 1e-5,
                    "pair {j} element {i}: midpoint {mid} != base {}",
                    state.base_params[i]
                );
            }
        }
    }

    #[test]
    fn centred_ranks_are_symmetric() {
        // Uniform fitness → all ranks should be ~0.
        let fitness = vec![1.0f64; 10];
        let ranks = centred_ranks(&fitness);
        for &r in &ranks {
            assert!(r.abs() < 1e-10, "uniform fitness should give zero ranks");
        }

        // Sorted fitness → ranks should span [−0.5, 0.5].
        let fitness: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ranks = centred_ranks(&fitness);
        assert!((ranks[0] + 0.5).abs() < 1e-10, "min fitness → rank -0.5");
        assert!((ranks[9] - 0.5).abs() < 1e-10, "max fitness → rank +0.5");
        // For even n, the median element is not exactly 0 — it's 1/(n-1) off.
        // Just check it's small.
        assert!(
            ranks[5].abs() < 0.1,
            "median fitness → rank near 0, got {}",
            ranks[5]
        );
    }

    #[test]
    fn es_update_moves_base_toward_better() {
        // If the +perturbation wins and the -perturbation loses, the base
        // should move toward the +perturbation direction.
        let cfg = ESConfig {
            n_perturbations: 4,
            lr: 0.1,
            ..Default::default()
        };
        let base = vec![0.0f32; 100];
        let mut state = ESState::new(base.clone(), &cfg, 42);
        let original_base = state.base_params.clone();

        // Fitness: all plus (even indices) win, all minus (odd) lose.
        let fitness: Vec<f64> = (0..8)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();

        state.update(&fitness, &cfg);

        // The base should have moved — not all zeros.
        let moved = original_base
            .iter()
            .zip(state.base_params.iter())
            .any(|(o, n)| (o - n).abs() > 1e-6);
        assert!(moved, "base params should have changed after update");
    }

    #[test]
    fn es_checkpoint_round_trips() {
        let cfg = ESConfig {
            n_perturbations: 4,
            ..Default::default()
        };
        let base = vec![0.5f32; 50];
        let state = ESState::new(base, &cfg, 42);
        let elo = crate::elo::EloTracker::new(8);
        let hof = crate::hof::HallOfFame::new();
        let seeds = vec![1u64, 2, 3];

        let tmp = std::env::temp_dir().join("canastra_es_checkpoint_test");
        let _ = std::fs::remove_dir_all(&tmp);
        save_es_checkpoint(&tmp, 5, &state, &elo, &hof, &seeds).unwrap();

        let (gen, loaded_state, loaded_elo, loaded_seeds, _loaded_hof) =
            load_es_checkpoint(&tmp).unwrap();
        let _ = std::fs::remove_dir_all(&tmp);

        assert_eq!(gen, 5);
        assert_eq!(loaded_state.base_params, state.base_params);
        assert_eq!(loaded_state.m, state.m);
        assert_eq!(loaded_state.v, state.v);
        assert_eq!(loaded_state.sigma, state.sigma);
        assert_eq!(loaded_state.t, state.t);
        assert_eq!(loaded_state.perturbation_seeds, state.perturbation_seeds);
        assert_eq!(loaded_elo, elo.ratings);
        assert_eq!(loaded_seeds, seeds);
    }
}
