//! Duplicate-deal paired evaluation.

use crate::genome::{Arch, Genome};
use crate::policy::{forward_and_pick, WeightStack};
use crate::pool::Pool;
use candle_core::{Device, Tensor};

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

    let stack_a = WeightStack::from_roster(&[vec_a], arch, device);
    let stack_b = WeightStack::from_roster(&[vec_b], arch, device);

    let mut diffs = Vec::new();
    let mut wins_a = 0;
    let mut wins_b = 0;
    let mut unfinished = 0;

    let mut by_seed: std::collections::HashMap<u64, Vec<f64>> = std::collections::HashMap::new();

    while pool.has_live() {
        let encoded = pool.encode();
        let n_rows = encoded.rows.len();
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

        // Forward with the right genome per row.
        // For simplicity, use the flat picker (no bucketing) since n_rows is small.
        let obs_dim = arch.obs;
        let act_dim = arch.act;
        let width = encoded.width;

        let device = &stack_a.device;
        let obs = Tensor::from_vec(encoded.obs, (n_rows, obs_dim), device)
            .unwrap_or_else(|e| panic!("obs: {e}"));
        let acts = Tensor::from_vec(encoded.acts, (n_rows, width, act_dim), device)
            .unwrap_or_else(|e| panic!("acts: {e}"));
        let mask = Tensor::from_vec(
            encoded
                .mask
                .iter()
                .map(|&v| if v { 1u32 } else { 0u32 })
                .collect::<Vec<_>>(),
            (n_rows, width),
            device,
        )
        .unwrap_or_else(|e| panic!("mask: {e}"));

        // Use stack_a for genome 0, stack_b for genome 1.
        // Since we only have 2 genomes, do two forwards.
        let mut picks = vec![0usize; n_rows];
        let a_rows: Vec<usize> = genome_idx
            .iter()
            .enumerate()
            .filter(|(_, &g)| g == 0)
            .map(|(i, _)| i)
            .collect();
        let b_rows: Vec<usize> = genome_idx
            .iter()
            .enumerate()
            .filter(|(_, &g)| g == 1)
            .map(|(i, _)| i)
            .collect();

        if !a_rows.is_empty() {
            let a_idx = Tensor::from_vec(
                a_rows.iter().map(|&r| r as u32).collect::<Vec<_>>(),
                a_rows.len(),
                &stack_a.device,
            )
            .unwrap();
            let a_obs = obs.index_select(&a_idx, 0).unwrap();
            let a_acts = acts.index_select(&a_idx, 0).unwrap();
            let a_mask = mask.index_select(&a_idx, 0).unwrap();
            let a_gidx: Vec<usize> = a_rows.iter().map(|_| 0).collect();
            let a_picks = forward_and_pick(
                &stack_a,
                &a_obs,
                &a_acts,
                &a_mask,
                &a_gidx,
                &[0],
                4.max(a_rows.len()),
            );
            for (i, &r) in a_rows.iter().enumerate() {
                picks[r] = a_picks[i];
            }
        }
        if !b_rows.is_empty() {
            let b_idx = Tensor::from_vec(
                b_rows.iter().map(|&r| r as u32).collect::<Vec<_>>(),
                b_rows.len(),
                &stack_b.device,
            )
            .unwrap();
            let b_obs = obs.index_select(&b_idx, 0).unwrap();
            let b_acts = acts.index_select(&b_idx, 0).unwrap();
            let b_mask = mask.index_select(&b_idx, 0).unwrap();
            let b_gidx: Vec<usize> = b_rows.iter().map(|_| 0).collect();
            let b_picks = forward_and_pick(
                &stack_b,
                &b_obs,
                &b_acts,
                &b_mask,
                &b_gidx,
                &[0],
                4.max(b_rows.len()),
            );
            for (i, &r) in b_rows.iter().enumerate() {
                picks[r] = b_picks[i];
            }
        }

        let _ = pool.apply(&picks);
    }

    let results = pool.results();
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
