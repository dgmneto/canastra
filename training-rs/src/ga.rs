//! GA core: selection, mutation, elitism, hall of fame, checkpoints.
//!
//! Ported from `training/python/canastra_train/ga.py`. Tabula rasa: random
//! init, self-play fitness, scalar reward is the match result.

use crate::elo::EloTracker;
use crate::genome::{genome_size, Arch, Genome};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::path::Path;

pub struct GAConfig {
    pub population: usize,
    pub elites: usize,
    pub tournament: usize,
    pub sigma: f64,
    pub sigma_decay: f64,
    pub sigma_floor: f64,
    pub hof_interval: u32,
}

impl Default for GAConfig {
    fn default() -> Self {
        Self {
            population: 96,
            elites: 8,
            tournament: 4,
            sigma: 0.02,
            sigma_decay: 0.995,
            sigma_floor: 0.002,
            hof_interval: 5,
        }
    }
}

pub fn initial_population(arch: &Arch, cfg: &GAConfig, run_seed: u64) -> Vec<Genome> {
    let size = genome_size(arch);
    let mut rng = StdRng::seed_from_u64(run_seed);
    (0..cfg.population)
        .map(|_| {
            let mut g = Genome::with_capacity(size);
            // Box-Muller normal(0, 0.1) — same as genome::random_genome.
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
        })
        .collect()
}

pub fn sigma_for(cfg: &GAConfig, generation: u32) -> f64 {
    (cfg.sigma * cfg.sigma_decay.powf(generation as f64)).max(cfg.sigma_floor)
}

fn tournament(elo: &[f64], k: usize, rng: &mut StdRng) -> usize {
    let mut best = 0usize;
    let mut best_elo = f64::NEG_INFINITY;
    for _ in 0..k {
        let contender = (rng.gen::<u32>() as usize) % elo.len();
        if elo[contender] > best_elo {
            best = contender;
            best_elo = elo[contender];
        }
    }
    best
}

/// Elites carried unchanged; the rest are mutated tournament winners.
/// Returns (next_pop, next_elo) where children inherit parent's ELO.
pub fn next_generation(
    pop: &[Genome],
    elo_ratings: &[f64],
    cfg: &GAConfig,
    generation: u32,
    rng: &mut StdRng,
) -> (Vec<Genome>, Vec<f64>) {
    // Sort by ELO descending to find elites.
    let mut order: Vec<usize> = (0..elo_ratings.len()).collect();
    order.sort_by(|&a, &b| elo_ratings[b].partial_cmp(&elo_ratings[a]).unwrap());

    let sigma = sigma_for(cfg, generation);
    let mut next_pop = Vec::with_capacity(cfg.population);
    let mut next_elo = Vec::with_capacity(cfg.population);

    // Elites
    for &i in &order[..cfg.elites.min(order.len())] {
        next_pop.push(pop[i].clone());
        next_elo.push(elo_ratings[i]);
    }

    // Children: tournament selection + Gaussian mutation
    while next_pop.len() < cfg.population {
        let parent = tournament(elo_ratings, cfg.tournament, rng);
        let mut child = pop[parent].clone();
        // Gaussian mutation
        for gene in child.iter_mut() {
            let u1 = (rng.gen::<u32>() as f64 / u32::MAX as f64).max(1e-10);
            let u2 = rng.gen::<u32>() as f64 / u32::MAX as f64;
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            let noise = r * theta.cos() * sigma;
            *gene += noise as f32;
        }
        next_elo.push(elo_ratings[parent]);
        next_pop.push(child);
    }

    (next_pop, next_elo)
}

pub struct HallOfFame {
    pub genomes: Vec<Genome>,
    pub elo_ratings: Vec<f64>,
    pub generations: Vec<u32>,
}

impl Default for HallOfFame {
    fn default() -> Self {
        Self::new()
    }
}

impl HallOfFame {
    pub fn new() -> Self {
        Self {
            genomes: Vec::new(),
            elo_ratings: Vec::new(),
            generations: Vec::new(),
        }
    }

    pub fn archive(&mut self, genome: &Genome, elo: f64, generation: u32) {
        self.genomes.push(genome.clone());
        self.elo_ratings.push(elo);
        self.generations.push(generation);
    }

    pub fn len(&self) -> usize {
        self.genomes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.genomes.is_empty()
    }

    pub fn sample(&self, rng: &mut StdRng) -> &Genome {
        let idx = (rng.gen::<u32>() as usize) % self.genomes.len();
        &self.genomes[idx]
    }
}

/// Save a checkpoint as a simple binary format (not .npz — Python compatibility
/// is not needed since this is a standalone Rust project).
///
/// Atomic: writes to a `.tmp` file, then renames — a crash mid-write does not
/// corrupt the latest checkpoint. Rotation: keeps the last `keep_recent`
/// checkpoints plus every `keep_every`-th (so both recent and milestone
/// checkpoints survive pruning).
pub fn save_checkpoint(
    dir: &Path,
    generation: u32,
    pop: &[Genome],
    elo: &EloTracker,
    hof: &HallOfFame,
    seeds: &[u64],
) -> anyhow::Result<()> {
    save_checkpoint_with_rotation(dir, generation, pop, elo, hof, seeds, 10, 50)
}

/// Save with configurable rotation: keep the last `keep_recent` plus every
/// `keep_every`-th checkpoint.
#[allow(clippy::too_many_arguments)]
pub fn save_checkpoint_with_rotation(
    dir: &Path,
    generation: u32,
    pop: &[Genome],
    elo: &EloTracker,
    hof: &HallOfFame,
    seeds: &[u64],
    keep_recent: usize,
    keep_every: u32,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("gen-{:05}.bin", generation));
    let tmp = dir.join(format!("gen-{:05}.bin.tmp", generation));

    // Simple binary format: generation, pop_size, genome_size, then flat data.
    let genome_size = if pop.is_empty() { 0 } else { pop[0].len() };
    let mut data = Vec::new();

    // Header
    data.extend_from_slice(&generation.to_le_bytes());
    data.extend_from_slice(&(pop.len() as u64).to_le_bytes());
    data.extend_from_slice(&(genome_size as u64).to_le_bytes());
    data.extend_from_slice(&(elo.ratings.len() as u64).to_le_bytes());
    data.extend_from_slice(&(seeds.len() as u64).to_le_bytes());
    data.extend_from_slice(&(hof.genomes.len() as u64).to_le_bytes());

    // Population
    for g in pop {
        for &v in g {
            data.extend_from_slice(&v.to_le_bytes());
        }
    }

    // ELO
    for &r in &elo.ratings {
        data.extend_from_slice(&r.to_le_bytes());
    }

    // Seeds
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

    // Atomic write: write to temp, then rename.
    std::fs::write(&tmp, &data)?;
    std::fs::rename(&tmp, &path)?;

    // Prune: keep last `keep_recent` plus every `keep_every`-th.
    let mut checkpoints: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.starts_with("gen-") && name.ends_with(".bin") && !name.ends_with(".tmp")
        })
        .collect();
    checkpoints.sort_by_key(|e| e.file_name());
    let total = checkpoints.len();
    if total > keep_recent {
        let mut to_remove = Vec::new();
        for (idx, entry) in checkpoints.iter().enumerate() {
            // Keep the last `keep_recent`.
            if idx >= total - keep_recent {
                continue;
            }
            // Keep every `keep_every`-th (milestone).
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(num_str) = name
                .strip_prefix("gen-")
                .and_then(|s| s.strip_suffix(".bin"))
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

/// Loaded checkpoint: (generation, population, elo ratings, seeds, hall of fame).
pub type Checkpoint = (u32, Vec<Genome>, Vec<f64>, Vec<u64>, HallOfFame);

pub fn load_checkpoint(dir: &Path) -> anyhow::Result<Checkpoint> {
    let mut checkpoints: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("gen-"))
        .collect();
    checkpoints.sort_by_key(|e| e.file_name());
    let path = checkpoints
        .last()
        .ok_or_else(|| anyhow::anyhow!("no checkpoint in {}", dir.display()))?
        .path();

    let data = std::fs::read(&path)?;
    let mut pos = 0;

    let generation = u32::from_le_bytes(data[pos..pos + 4].try_into()?);
    pos += 4;
    let pop_size = u64::from_le_bytes(data[pos..pos + 8].try_into()?) as usize;
    pos += 8;
    let genome_size = u64::from_le_bytes(data[pos..pos + 8].try_into()?) as usize;
    pos += 8;
    let elo_size = u64::from_le_bytes(data[pos..pos + 8].try_into()?) as usize;
    pos += 8;
    let seeds_size = u64::from_le_bytes(data[pos..pos + 8].try_into()?) as usize;
    pos += 8;
    let hof_size = u64::from_le_bytes(data[pos..pos + 8].try_into()?) as usize;
    pos += 8;

    let mut pop = Vec::with_capacity(pop_size);
    for _ in 0..pop_size {
        let mut g = Genome::with_capacity(genome_size);
        for _ in 0..genome_size {
            g.push(f32::from_le_bytes(data[pos..pos + 4].try_into()?));
            pos += 4;
        }
        pop.push(g);
    }

    let mut elo = Vec::with_capacity(elo_size);
    for _ in 0..elo_size {
        elo.push(f64::from_le_bytes(data[pos..pos + 8].try_into()?));
        pos += 8;
    }

    let mut seeds = Vec::with_capacity(seeds_size);
    for _ in 0..seeds_size {
        seeds.push(u64::from_le_bytes(data[pos..pos + 8].try_into()?));
        pos += 8;
    }

    let mut hof = HallOfFame::new();
    for _ in 0..hof_size {
        let gsz = u64::from_le_bytes(data[pos..pos + 8].try_into()?) as usize;
        pos += 8;
        let mut g = Genome::with_capacity(gsz);
        for _ in 0..gsz {
            g.push(f32::from_le_bytes(data[pos..pos + 4].try_into()?));
            pos += 4;
        }
        hof.genomes.push(g);
    }
    for _ in 0..hof_size {
        hof.elo_ratings
            .push(f64::from_le_bytes(data[pos..pos + 8].try_into()?));
        pos += 8;
    }
    for _ in 0..hof_size {
        hof.generations
            .push(u32::from_le_bytes(data[pos..pos + 4].try_into()?));
        pos += 4;
    }

    Ok((generation, pop, elo, seeds, hof))
}
