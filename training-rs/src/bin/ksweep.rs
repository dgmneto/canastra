//! K-sweep benchmark: measure wall/gen, decisions/s, and rank correlation
//! at different games-per-genome (K) values.
//!
//! K = opponents × seeds × 2 (2 for seat swap). The current default is
//! K=64 (opponents=4, seeds=8). The sweep tests K ∈ {64, 128, 256, 512}.
//!
//! Rank correlation: run the same population twice with different deal seeds,
//! compute Spearman's ρ of the ELO rankings. High ρ means the fitness signal
//! is stable; low ρ means selection is mostly noise.
//!
//! Run with:
//!   target\release\canastra-ksweep.exe --population 96 --device cuda

use canastra_train::elo::EloTracker;
use canastra_train::ga::{self, GAConfig};
use canastra_train::genome::TRAINING_ARCH;
use canastra_train::league::{self, Rollout};
use canastra_train::seedstream;
use candle_core::Device;
use clap::Parser;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "canastra-ksweep")]
#[command(about = "K-sweep: wall/gen, decisions/s, rank correlation vs K")]
struct Args {
    /// Population size.
    #[arg(long, default_value = "96")]
    population: usize,

    /// Device: "cuda" or "cpu".
    #[arg(long, default_value = "cuda")]
    device: String,

    /// Rollout path: "auto", "lockstep", or "coalesced".
    #[arg(long, default_value = "auto")]
    rollout: String,

    /// Max legal actions per row (menu width cap). 0 = no cap.
    #[arg(long, default_value = "64")]
    max_width: usize,

    /// K values to sweep (games per genome). Comma-separated.
    #[arg(long, default_value = "64,128,256,512")]
    k_values: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let arch = &TRAINING_ARCH;
    let cfg = GAConfig {
        population: args.population,
        ..Default::default()
    };
    let pop = ga::initial_population(arch, &cfg, 7);
    let hof = ga::HallOfFame::new();

    let device = match args.device.as_str() {
        "cuda" => Device::new_cuda(0).unwrap_or(Device::Cpu),
        _ => Device::Cpu,
    };
    let device_label = if matches!(device, Device::Cuda(_)) {
        "cuda"
    } else {
        "cpu"
    };
    let rollout = match args.rollout.as_str() {
        "lockstep" => Rollout::Lockstep,
        "coalesced" => Rollout::Coalesced,
        _ => Rollout::default_for(args.population),
    };

    // Parse K values.
    let k_values: Vec<usize> = args
        .k_values
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    // For each K, decompose into opponents × seeds. Use opponents=4 fixed,
    // vary seeds. K = opponents * seeds * 2.
    let opponents = 4usize;

    eprintln!(
        "K-sweep: pop={} device={} rollout={} max_width={}",
        args.population,
        device_label,
        match rollout {
            Rollout::Lockstep => "lockstep",
            Rollout::Coalesced => "coalesced",
        },
        args.max_width
    );
    eprintln!(
        "{:>6} {:>8} {:>8} {:>10} {:>12} {:>10} {:>10}",
        "K", "games", "wall_s", "games/s", "dec/s", "rank_ρ", "rank_ρ2"
    );

    for &k in &k_values {
        let seeds = k / (opponents * 2);
        if seeds == 0 {
            continue;
        }
        let actual_k = opponents * seeds * 2;
        let games = args.population * actual_k;

        // Run 1: deal seeds from run_seed=7, generation=0.
        let mut rng1 = StdRng::seed_from_u64(seedstream::splitmix64(7));
        let pairings1 = league::schedule_pairings(args.population, opponents, &hof, &mut rng1);
        let gen_seeds1 = seedstream::generation_seeds(7, 0, seeds);
        let mut elo1 = EloTracker::new(args.population);

        let began = Instant::now();
        league::evaluate_generation(
            &league::EvalInputs {
                pop: &pop,
                hof: &hof,
                pairings: &pairings1,
                arch,
                seeds: &gen_seeds1,
                max_hands: Some(1),
                device: &device,
                n_workers: 8,
                rollout,
                max_width: if args.max_width == 0 {
                    usize::MAX
                } else {
                    args.max_width
                },
            },
            &mut elo1,
        );
        let wall1 = began.elapsed().as_secs_f64();
        let games_s = games as f64 / wall1;
        // Decisions/s: roughly games * plies_per_game / wall. We don't have
        // exact ply count, but ~200 plies/game is typical for 1-hand games.
        let dec_s = games as f64 * 200.0 / wall1;

        // Run 2: deal seeds from run_seed=99 (different deals, same population).
        let mut rng2 = StdRng::seed_from_u64(seedstream::splitmix64(99));
        let pairings2 = league::schedule_pairings(args.population, opponents, &hof, &mut rng2);
        let gen_seeds2 = seedstream::generation_seeds(99, 0, seeds);
        let mut elo2 = EloTracker::new(args.population);

        let began = Instant::now();
        league::evaluate_generation(
            &league::EvalInputs {
                pop: &pop,
                hof: &hof,
                pairings: &pairings2,
                arch,
                seeds: &gen_seeds2,
                max_hands: Some(1),
                device: &device,
                n_workers: 8,
                rollout,
                max_width: if args.max_width == 0 {
                    usize::MAX
                } else {
                    args.max_width
                },
            },
            &mut elo2,
        );
        let wall2 = began.elapsed().as_secs_f64();

        // Run 3: a third run with yet another seed for a 3-way correlation.
        let mut rng3 = StdRng::seed_from_u64(seedstream::splitmix64(777));
        let pairings3 = league::schedule_pairings(args.population, opponents, &hof, &mut rng3);
        let gen_seeds3 = seedstream::generation_seeds(777, 0, seeds);
        let mut elo3 = EloTracker::new(args.population);

        league::evaluate_generation(
            &league::EvalInputs {
                pop: &pop,
                hof: &hof,
                pairings: &pairings3,
                arch,
                seeds: &gen_seeds3,
                max_hands: Some(1),
                device: &device,
                n_workers: 8,
                rollout,
                max_width: if args.max_width == 0 {
                    usize::MAX
                } else {
                    args.max_width
                },
            },
            &mut elo3,
        );

        // Spearman rank correlation between run 1 and run 2, and 1 and 3.
        let ratings1 = &elo1.ratings[..args.population];
        let ratings2 = &elo2.ratings[..args.population];
        let ratings3 = &elo3.ratings[..args.population];
        let rho12 = spearman_rho(ratings1, ratings2);
        let rho13 = spearman_rho(ratings1, ratings3);

        let avg_wall = (wall1 + wall2) / 2.0;
        println!(
            "{:>6} {:>8} {:>8.1} {:>10.1} {:>12.0} {:>10.3} {:>10.3}",
            actual_k, games, avg_wall, games_s, dec_s, rho12, rho13
        );
    }

    Ok(())
}

/// Spearman's rank correlation coefficient.
fn spearman_rho(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let ranks_a = rank(a);
    let ranks_b = rank(b);

    // Pearson on ranks = Spearman.
    let mean_a = ranks_a.iter().sum::<f64>() / n as f64;
    let mean_b = ranks_b.iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut den_a = 0.0;
    let mut den_b = 0.0;
    for i in 0..n {
        let da = ranks_a[i] - mean_a;
        let db = ranks_b[i] - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }
    let den = (den_a * den_b).sqrt();
    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}

/// Rank values (1 = smallest). Ties get average rank.
fn rank(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && indexed[j].1 == indexed[i].1 {
            j += 1;
        }
        // Average rank for ties: (i+1 + j) / 2 (1-indexed).
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    ranks
}
