//! K-sweep benchmark: wall/gen, throughput, and rank correlation at different
//! games-per-genome (K) values.
//!
//! K = opponents × seeds × 2 (2 for seat swap). The default is K=64
//! (opponents=4, seeds=8). The sweep tests K ∈ {64, 128, 256, 512}.
//!
//! Rank correlation: run the same population three times with different deal
//! seeds, compute Spearman's ρ between runs. High ρ means the fitness signal is
//! stable; low ρ means selection is mostly noise.
//!
//! **Three correlations are reported so the ranking design is measured rather
//! than argued**, all from the *same* games:
//!
//! - `elo_ρ` — the old within-generation ELO.
//! - `diff_ρ` — the paired duplicate-deal differential that replaced it.
//! - `grad_ρ` — stability of the per-pair `f⁺ − f⁻`, which is what ES's
//!   gradient is actually built from. Population-wide ρ cannot see this: a
//!   ranking can look stable while every pairwise difference is noise.
//!
//! **Use `--sigma` to pick the regime.** `--sigma 0` ranks independent random
//! genomes, where the signal is strong and every metric looks fine — this is
//! what the original Task 4 sweep measured, and its conclusions did not
//! transfer. `--sigma 0.02` builds an ES-shaped population (one base policy,
//! mirrored θ±σε pairs), which is what training actually ranks and where the
//! metric choice bites. See `docs/decision-ranking-metric.md`.
//!
//! Run with:
//!   target\release\canastra-ksweep.exe --population 96 --device cuda --sigma 0.02

use canastra_train::elo::EloTracker;
use canastra_train::fitness;
use canastra_train::genome::{self, TRAINING_ARCH};
use canastra_train::hof::HallOfFame;
use canastra_train::league;
use canastra_train::seedstream;
use candle_core::Device;
use clap::Parser;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "canastra-ksweep")]
#[command(about = "K-sweep: wall/gen, games/s, rank correlation vs K")]
struct Args {
    /// Population size.
    #[arg(long, default_value = "96")]
    population: usize,

    /// Device: "cuda" or "cpu".
    #[arg(long, default_value = "cuda")]
    device: String,

    /// Max legal actions per row (menu width cap). 0 = no cap.
    #[arg(long, default_value = "64")]
    max_width: usize,

    /// K values to sweep (games per genome). Comma-separated.
    #[arg(long, default_value = "64,128,256,512")]
    k_values: String,

    /// Use mirrored-pair common random numbers when scheduling (matches what
    /// training does). Off reproduces the pre-fix independent draw.
    #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
    mirrored: bool,

    /// Perturbation scale for the sweep population. 0 = independent random
    /// genomes (the historical setting). Non-zero builds an ES-shaped
    /// population — one base policy plus mirrored θ±σε pairs — which is what
    /// training actually ranks, and where the metric choice bites hardest.
    #[arg(long, default_value = "0.0")]
    sigma: f64,
}

/// One run's two rankings, computed from the same games.
struct Ranking {
    elo: Vec<f64>,
    diff: Vec<f64>,
    /// Per mirrored pair, `f⁺ − f⁻` under the paired-differential metric.
    pair_delta: Vec<f64>,
    wall: f64,
    /// Diagnostics: how many games ended level, and the score range seen.
    draws: usize,
    total: usize,
    score_lo: i32,
    score_hi: i32,
    unfinished: usize,
}

#[allow(clippy::too_many_arguments)]
fn run_once(
    pop: &[Vec<f32>],
    hof: &HallOfFame,
    population: usize,
    opponents: usize,
    seeds: usize,
    run_seed: u64,
    device: &Device,
    max_width: usize,
    mirrored: bool,
) -> Ranking {
    let arch = &TRAINING_ARCH;
    let mut rng = StdRng::seed_from_u64(seedstream::splitmix64(run_seed));
    let pairings = if mirrored {
        league::schedule_pairings_mirrored(population, opponents, hof, &mut rng)
    } else {
        league::schedule_pairings(population, opponents, hof, &mut rng)
    };
    let gen_seeds = seedstream::generation_seeds(run_seed, 0, seeds);

    let began = Instant::now();
    let results = league::play_generation(&league::EvalInputs {
        pop,
        hof,
        pairings: &pairings,
        arch,
        seeds: &gen_seeds,
        max_hands: Some(1),
        device,
        max_width,
        dtype: None,
    });
    let wall = began.elapsed().as_secs_f64();

    // Same games, two ways of folding them into a per-genome number.
    let mut elo = EloTracker::new(population + hof.len());
    elo.batch_update(&league::elo_updates(&results, &pairings, gen_seeds.len()));
    let report =
        fitness::score_generation(&results, &pairings, gen_seeds.len(), population + hof.len());

    let draws = results.iter().filter(|(_, s, ..)| s[0] == s[1]).count();
    let unfinished = results.iter().filter(|&&(.., u)| u).count();
    let score_lo = results
        .iter()
        .flat_map(|(_, s, ..)| [s[0], s[1]])
        .min()
        .unwrap_or(0);
    let score_hi = results
        .iter()
        .flat_map(|(_, s, ..)| [s[0], s[1]])
        .max()
        .unwrap_or(0);

    // The quantity ES's gradient is actually built from: per mirrored pair,
    // f⁺ − f⁻. Population-wide ρ says nothing about this — it can look stable
    // while every pairwise difference is noise — so it gets its own column.
    let fit = &report.fitness;
    let pair_delta: Vec<f64> = (0..population / 2)
        .map(|j| fit[2 * j] - fit[2 * j + 1])
        .collect();

    Ranking {
        elo: elo.ratings[..population].to_vec(),
        diff: report.fitness[..population].to_vec(),
        pair_delta,
        wall,
        draws,
        total: results.len(),
        score_lo,
        score_hi,
        unfinished,
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let arch = &TRAINING_ARCH;
    let hof = HallOfFame::new();
    // Either independent random genomes (historical) or an ES-shaped
    // population: one base policy, perturbed into mirrored θ±σε pairs.
    let pop = if args.sigma > 0.0 {
        let cfg = canastra_train::es::ESConfig {
            n_perturbations: args.population / 2,
            sigma: args.sigma,
            ..Default::default()
        };
        let base = genome::random_genome(arch, 7);
        canastra_train::es::ESState::new(base, &cfg, 7).materialise_population(arch)
    } else {
        genome::random_population(arch, args.population, 7)
    };

    let device = match args.device.as_str() {
        "cuda" => Device::new_cuda(0).unwrap_or(Device::Cpu),
        _ => Device::Cpu,
    };
    let device_label = if matches!(device, Device::Cuda(_)) {
        "cuda"
    } else {
        "cpu"
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
    let max_width = if args.max_width == 0 {
        usize::MAX
    } else {
        args.max_width
    };

    eprintln!(
        "K-sweep: pop={} device={} max_width={} mirrored={} sigma={}",
        args.population, device_label, args.max_width, args.mirrored, args.sigma
    );
    eprintln!(
        "rank_ρ columns: elo = within-generation ELO, diff = paired duplicate-deal differential"
    );
    eprintln!("grad_ρ = stability of the per-pair f⁺−f⁻ that ES's gradient is built from");
    eprintln!(
        "{:>6} {:>8} {:>8} {:>10} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "K",
        "games",
        "wall_s",
        "games/s",
        "elo_ρ12",
        "elo_ρ13",
        "diff_ρ12",
        "diff_ρ13",
        "grad_ρ12",
        "grad_ρ13"
    );

    for &k in &k_values {
        let seeds = k / (opponents * 2);
        if seeds == 0 {
            continue;
        }
        let actual_k = opponents * seeds * 2;
        let games = args.population * actual_k;

        // Three runs of the same population over different deal seeds. Both
        // rankings come out of each run, so the comparison is apples to apples.
        let runs: Vec<_> = [7u64, 99, 777]
            .iter()
            .map(|&run_seed| {
                run_once(
                    &pop,
                    &hof,
                    args.population,
                    opponents,
                    seeds,
                    run_seed,
                    &device,
                    max_width,
                    args.mirrored,
                )
            })
            .collect();

        // A population whose games all end level produces a flat ranking under
        // *either* metric, and ρ is then meaningless rather than bad. Say so
        // loudly instead of printing a misleading 0.000.
        let r0 = &runs[0];
        if r0.draws == r0.total {
            eprintln!(
                "  !! all {} games ended level (scores {}..{}, {} unfinished) — \
                 no fitness signal exists to rank, ρ below is vacuous",
                r0.total, r0.score_lo, r0.score_hi, r0.unfinished
            );
        } else if std::env::var("KSWEEP_DEBUG").is_ok() {
            eprintln!(
                "  draws {}/{} scores {}..{} unfinished {}",
                r0.draws, r0.total, r0.score_lo, r0.score_hi, r0.unfinished
            );
        }

        let games_s = games as f64 / runs[0].wall;
        let avg_wall = (runs[0].wall + runs[1].wall) / 2.0;

        println!(
            "{:>6} {:>8} {:>8.1} {:>10.1} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3}",
            actual_k,
            games,
            avg_wall,
            games_s,
            spearman_rho(&runs[0].elo, &runs[1].elo),
            spearman_rho(&runs[0].elo, &runs[2].elo),
            spearman_rho(&runs[0].diff, &runs[1].diff),
            spearman_rho(&runs[0].diff, &runs[2].diff),
            spearman_rho(&runs[0].pair_delta, &runs[1].pair_delta),
            spearman_rho(&runs[0].pair_delta, &runs[2].pair_delta),
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
