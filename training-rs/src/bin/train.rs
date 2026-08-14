use canastra_train::elo::EloTracker;
use canastra_train::ga::{self, GAConfig, HallOfFame};
use canastra_train::genome::{self, Arch, TRAINING_ARCH};
use canastra_train::league;
use canastra_train::seedstream;
use candle_core::Device;
use clap::Parser;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "canastra-train")]
#[command(about = "GA training for Canastra policy networks (pure Rust)")]
struct Args {
    /// How many generations to run.
    #[arg(long)]
    generations: u32,

    /// Output directory.
    #[arg(long, default_value = "runs/auto")]
    run_dir: PathBuf,

    /// Population size.
    #[arg(long, default_value = "96")]
    population: usize,

    /// Number of elites carried unchanged.
    #[arg(long, default_value = "8")]
    elites: usize,

    /// Tournament size.
    #[arg(long, default_value = "4")]
    tournament: usize,

    /// Opponents per genome per generation.
    #[arg(long, default_value = "4")]
    opponents: usize,

    /// Seeds per opponent pairing.
    #[arg(long, default_value = "8")]
    seeds: usize,

    /// Hands per game (0 = full matches).
    #[arg(long, default_value = "1")]
    max_hands: u32,

    /// Run seed (derives every seed stream).
    #[arg(long, default_value = "7")]
    run_seed: u64,

    /// Gaussian mutation scale.
    #[arg(long, default_value = "0.02")]
    sigma: f64,

    /// HOF archival interval.
    #[arg(long, default_value = "5")]
    hof_interval: u32,

    /// Worker threads (coalesced GPU server count).
    #[arg(long, default_value = "4")]
    workers: usize,

    /// Device: "cuda" or "cpu".
    #[arg(long, default_value = "cuda")]
    device: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let arch = &TRAINING_ARCH;
    let cfg = GAConfig {
        population: args.population,
        elites: args.elites,
        tournament: args.tournament,
        sigma: args.sigma,
        sigma_decay: 0.995,
        sigma_floor: 0.002,
        hof_interval: args.hof_interval,
    };

    let device = match args.device.as_str() {
        "cuda" => Device::new_cuda(0)
            .unwrap_or_else(|e| panic!("CUDA device 0 not available: {e}. Use --device cpu.")),
        _ => Device::Cpu,
    };

    std::fs::create_dir_all(&args.run_dir)?;

    let mut pop = ga::initial_population(arch, &cfg, args.run_seed);
    let mut hof = HallOfFame::new();
    let mut elo = EloTracker::new(args.population);
    let mut best_ever = f64::NEG_INFINITY;

    for generation in 0..args.generations {
        let began = Instant::now();
        let mut gen_rng =
            StdRng::seed_from_u64(seedstream::splitmix64(args.run_seed + generation as u64));
        let gen_seeds = seedstream::generation_seeds(args.run_seed, generation, args.seeds);
        let pairings =
            league::schedule_pairings(args.population, args.opponents, &hof, &mut gen_rng);

        let max_hands = if args.max_hands > 0 {
            Some(args.max_hands)
        } else {
            None
        };
        league::evaluate_generation(
            &pop,
            &hof,
            &pairings,
            arch,
            &gen_seeds,
            max_hands,
            &device,
            &mut elo,
            args.workers,
        );

        let champion = elo.ratings[..args.population]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let elo_best = elo.ratings[champion];
        let elo_mean = elo.ratings[..args.population].iter().sum::<f64>() / args.population as f64;
        let sigma = ga::sigma_for(&cfg, generation);

        let improved = elo_best > best_ever;
        if improved {
            best_ever = elo_best;
        }

        let wall = began.elapsed().as_secs_f64();
        println!(
            "gen {}: best {:.1} mean {:.1} sigma {:.4} ({:.1}s){}",
            generation,
            elo_best,
            elo_mean,
            sigma,
            wall,
            if improved { "  new best-ever" } else { "" }
        );

        // Export champion.
        if improved || generation % cfg.hof_interval == 0 {
            let path = args
                .run_dir
                .join(format!("champion-gen{:05}.json", generation));
            let _ = genome::save_json(path.to_str().unwrap(), arch, &pop[champion]);
        }

        // HOF.
        if generation % cfg.hof_interval == 0 {
            hof.archive(&pop[champion], elo_best, generation);
            elo.grow(1);
            elo.ratings.push(elo_best);
        }

        // Evolve.
        let (next_pop, next_elo) = ga::next_generation(
            &pop,
            &elo.ratings[..args.population],
            &cfg,
            generation,
            &mut gen_rng,
        );
        pop = next_pop;
        elo.ratings[..args.population].copy_from_slice(&next_elo);

        // Checkpoint.
        if generation % 5 == 0 || generation == args.generations - 1 {
            let _ = ga::save_checkpoint(&args.run_dir, generation, &pop, &elo, &hof, &gen_seeds);
        }
    }

    // Final champion.
    let final_champ = elo.ratings[..args.population]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let path = args.run_dir.join("champion-final.json");
    let _ = genome::save_json(path.to_str().unwrap(), arch, &pop[final_champ]);

    println!("done: {}", args.run_dir.display());
    Ok(())
}
