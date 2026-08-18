//! Duplicate-deal paired evaluation: play two genomes head to head over a set
//! of deals, each deal in both seatings, and report the mean point differential
//! with a 95% CI.
//!
//! **Currently unused.** It computes the same paired differential that
//! [`crate::fitness`] now uses as the ES selection signal — the idea was
//! already here in Rust while the training loop was ranking on ELO. The two are
//! kept separate because they have different shapes: this owns a rollout and is
//! an A/B tool for two specific genomes; `fitness` is a pure fold over a whole
//! generation's results. `mean_diff` here and `fitness` there are in the same
//! unit (mean points per game), so they can be read against each other.

use crate::genome::{Arch, Genome};
use crate::policy::{forward_picks, WeightStack};
use crate::pool::Pool;
use candle_core::{DType, Device, Tensor};

pub struct PairReport {
    pub pairs: usize,
    pub mean_diff: f64,
    pub ci95: f64,
    pub wins_a: usize,
    pub wins_b: usize,
    pub unfinished: usize,
}

pub fn evaluate_pair(
    vec_a: &Genome,
    vec_b: &Genome,
    arch: &Arch,
    seeds: &[u64],
    device: &Device,
) -> PairReport {
    let count = seeds.len();
    let pool_seeds: Vec<u64> = seeds.iter().chain(seeds.iter()).copied().collect();
    let mut pool = Pool::new(pool_seeds, Some(200_000), None);

    let dtype = match device {
        Device::Cuda(_) => DType::BF16,
        _ => DType::F32,
    };
    let stack = WeightStack::from_roster(&[vec_a, vec_b], arch, device, dtype);

    while pool.has_live() {
        let encoded = pool.encode();
        let n_rows = encoded.rows.len();
        if n_rows == 0 {
            break;
        }
        let genome_idx: Vec<usize> = encoded
            .rows
            .iter()
            .map(|&(game_idx, seat)| {
                let a_is_team_zero = game_idx < count;
                if (seat % 2 == 0) == a_is_team_zero {
                    0
                } else {
                    1
                }
            })
            .collect();

        let obs = Tensor::from_vec(encoded.obs, (n_rows, arch.obs), device)
            .unwrap_or_else(|e| panic!("obs: {e}"));
        let acts = Tensor::from_vec(encoded.acts, (n_rows, encoded.width, arch.act), device)
            .unwrap_or_else(|e| panic!("acts: {e}"));
        let mask = Tensor::from_vec(
            encoded
                .mask
                .iter()
                .map(|&v| if v { 1u32 } else { 0u32 })
                .collect::<Vec<_>>(),
            (n_rows, encoded.width),
            device,
        )
        .unwrap_or_else(|e| panic!("mask: {e}"));

        let picks = forward_picks(&stack, &obs, &acts, &mask, &genome_idx);
        let _ = pool.apply(&picks);
    }

    let results = pool.results();
    let mut diffs = Vec::new();
    let mut wins_a = 0;
    let mut wins_b = 0;
    let mut unfinished = 0;
    let mut by_seed: std::collections::HashMap<u64, Vec<f64>> = std::collections::HashMap::new();

    for (index, (_seed, scores, winner, _hands, is_unfinished)) in results.iter().enumerate() {
        let a_team_zero = index < count;
        let a_score = if a_team_zero { scores[0] } else { scores[1] };
        let b_score = if a_team_zero { scores[1] } else { scores[0] };
        by_seed
            .entry(*_seed)
            .or_default()
            .push((a_score - b_score) as f64);
        if *is_unfinished {
            unfinished += 1;
        } else if let Some(w) = winner {
            let won_a = (*w == 0) == a_team_zero;
            if won_a {
                wins_a += 1;
            } else {
                wins_b += 1;
            }
        }
    }

    for seed in seeds {
        if let Some(pair) = by_seed.get(seed) {
            if pair.len() == 2 {
                diffs.push((pair[0] + pair[1]) / 2.0);
            }
        }
    }

    let mean = diffs.iter().sum::<f64>() / diffs.len().max(1) as f64;
    let variance = if diffs.len() > 1 {
        diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (diffs.len() - 1) as f64
    } else {
        f64::INFINITY
    };
    let ci95 = 1.96 * variance.sqrt() / (count as f64).sqrt();

    PairReport {
        pairs: count,
        mean_diff: mean,
        ci95,
        wins_a,
        wins_b,
        unfinished,
    }
}
