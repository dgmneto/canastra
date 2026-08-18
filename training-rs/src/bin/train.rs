use canastra_train::anchors::AnchorSet;
use canastra_train::elo::EloTracker;
use canastra_train::es::{self, ESConfig, ESState};
use canastra_train::genome::{self, TRAINING_ARCH};
use canastra_train::hof::HallOfFame;
use canastra_train::league;
use canastra_train::seedstream;
use candle_core::Device;
use clap::Parser;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "canastra-train")]
#[command(about = "ES training for Canastra policy networks (pure Rust)")]
struct Args {
    /// How many generations to run.
    #[arg(long)]
    generations: u32,

    /// Output directory.
    #[arg(long, default_value = "runs/auto")]
    run_dir: PathBuf,

    /// ES: number of perturbation pairs (population = 2 × this).
    #[arg(long, default_value = "48")]
    n_perturbations: usize,

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

    /// Gaussian perturbation scale (σ).
    #[arg(long, default_value = "0.02")]
    sigma: f64,

    /// Sigma decay per generation.
    #[arg(long, default_value = "0.995")]
    sigma_decay: f64,

    /// Minimum sigma (floor).
    #[arg(long, default_value = "0.002")]
    sigma_floor: f64,

    /// HOF archival interval.
    #[arg(long, default_value = "5")]
    hof_interval: u32,

    /// Device: "cuda" or "cpu".
    #[arg(long, default_value = "cuda")]
    device: String,

    /// Max legal actions per row (menu width cap). 0 = no cap.
    #[arg(long, default_value = "64")]
    max_width: usize,

    /// Resume from a checkpoint in this directory. Loads the latest ES
    /// checkpoint and continues from the next generation.
    #[arg(long)]
    resume: Option<PathBuf>,

    /// Checkpoint every N generations.
    #[arg(long, default_value = "5")]
    checkpoint_interval: u32,

    /// Number of recent checkpoints to keep (rotation).
    #[arg(long, default_value = "10")]
    keep_recent: usize,

    /// Keep every Nth checkpoint as a milestone (rotation).
    #[arg(long, default_value = "50")]
    keep_every: u32,

    /// Anchor evaluation interval (generations). 0 = disabled.
    #[arg(long, default_value = "1")]
    anchor_interval: u32,

    /// Seeds for anchor evaluation games.
    #[arg(long, default_value = "8")]
    anchor_seeds: usize,

    /// Add the champion as a new anchor every N generations (frozen champions).
    #[arg(long, default_value = "50")]
    anchor_freeze_interval: u32,

    /// ES: Adam learning rate.
    #[arg(long, default_value = "0.01")]
    lr: f64,

    /// ES: weight decay (L2).
    #[arg(long, default_value = "0.0")]
    weight_decay: f64,
}

/// Write one JSON record per generation to `<run_dir>/generations.jsonl`.
fn log_generation(run_dir: &std::path::Path, record: &serde_json::Value) {
    let path = run_dir.join("generations.jsonl");
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let line = serde_json::to_string(record).unwrap_or_default();
        let _ = writeln!(f, "{line}");
    }
}

/// Compute fitness distribution stats from ELO ratings.
fn fitness_stats(ratings: &[f64]) -> serde_json::Value {
    let mut sorted: Vec<f64> = ratings.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let pct = |p: usize| sorted[(p * n / 100).min(n - 1)];
    serde_json::json!({
        "min": sorted[0],
        "p25": pct(25),
        "median": pct(50),
        "p75": pct(75),
        "p95": pct(95),
        "max": sorted[n - 1],
        "mean": sorted.iter().sum::<f64>() / n as f64,
    })
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let arch = &TRAINING_ARCH;
    let device = match args.device.as_str() {
        "cuda" => Device::new_cuda(0)
            .unwrap_or_else(|e| panic!("CUDA device 0 not available: {e}. Use --device cpu.")),
        _ => Device::Cpu,
    };

    std::fs::create_dir_all(&args.run_dir)?;

    let es_cfg = ESConfig {
        n_perturbations: args.n_perturbations,
        sigma: args.sigma,
        sigma_decay: args.sigma_decay,
        sigma_floor: args.sigma_floor,
        lr: args.lr,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: args.weight_decay,
        eps: 1e-8,
        hof_interval: args.hof_interval,
    };
    let pop_size = es_cfg.n_perturbations * 2;

    let max_width = if args.max_width == 0 {
        usize::MAX
    } else {
        args.max_width
    };
    let max_hands = if args.max_hands > 0 {
        Some(args.max_hands)
    } else {
        None
    };

    // Resume or fresh start.
    let (start_gen, mut es_state, mut hof, mut best_ever) =
        if let Some(ref resume_dir) = args.resume {
            let (gen, state, _elo, _seeds, loaded_hof) = es::load_es_checkpoint(resume_dir)?;
            let next_gen = gen + 1;
            eprintln!("ES: resumed from gen {gen}, continuing at gen {next_gen}");
            (next_gen, state, loaded_hof, f64::NEG_INFINITY)
        } else {
            let base = genome::random_genome(arch, args.run_seed);
            let state = ESState::new(base, &es_cfg, args.run_seed);
            (0u32, state, HallOfFame::new(), f64::NEG_INFINITY)
        };

    let mut anchors = AnchorSet::new(arch);
    let anchor_seeds: Vec<u64> = (1000..1000 + args.anchor_seeds as u64).collect();

    for generation in start_gen..(start_gen + args.generations) {
        let began = Instant::now();
        es_state.sigma_for_generation(&es_cfg, generation);

        // Materialise population: 2 × n_perturbations genomes (mirrored pairs).
        let pop = es_state.materialise_population(arch);

        // Self-play evaluation (lockstep rollout).
        let mut gen_rng =
            StdRng::seed_from_u64(seedstream::splitmix64(args.run_seed + generation as u64));
        let gen_seeds = seedstream::generation_seeds(args.run_seed, generation, args.seeds);
        let pairings = league::schedule_pairings(pop_size, args.opponents, &hof, &mut gen_rng);
        let mut elo = EloTracker::new(pop_size);
        // Grow ELO to include HOF entries (schedule_pairings may assign HOF
        // opponents at indices ≥ pop_size).
        if !hof.is_empty() {
            elo.grow(hof.len(), None);
        }

        league::evaluate_generation(
            &league::EvalInputs {
                pop: &pop,
                hof: &hof,
                pairings: &pairings,
                arch,
                seeds: &gen_seeds,
                max_hands,
                device: &device,
                max_width,
            },
            &mut elo,
        );

        // Extract fitness from ELO ratings.
        let fitness: Vec<f64> = elo.ratings[..pop_size].to_vec();

        // Find champion (highest ELO).
        let champion = fitness
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let elo_best = fitness[champion];
        let elo_mean = fitness.iter().sum::<f64>() / pop_size as f64;
        let sigma = es_state.sigma;

        let improved = elo_best > best_ever;
        if improved {
            best_ever = elo_best;
        }

        // ES update: rank-normalised fitness → gradient estimate → Adam.
        es_state.update(&fitness, &es_cfg);

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

        // Log to generations.jsonl.
        let games = pop_size * args.opponents * args.seeds * 2;

        // Anchored evaluation.
        let anchor_report = if args.anchor_interval > 0 && generation % args.anchor_interval == 0 {
            let report = anchors.evaluate(
                &pop[champion],
                arch,
                &anchor_seeds,
                &device,
                max_width,
                max_hands,
            );
            eprintln!("  anchor ELO: {:.1}", report.champion_rating);
            Some(report.to_json())
        } else {
            None
        };

        // Freeze the base policy as an anchor (NOT the perturbed champion).
        // For ES, the base policy θ is the actual policy being trained; the
        // "champion" (highest-ELO perturbation) is an ephemeral probe whose
        // advantage is mostly noise. Freezing it would pollute the anchor set
        // with random perturbations rather than tracking real progress.
        if args.anchor_freeze_interval > 0
            && generation > 0
            && generation % args.anchor_freeze_interval == 0
        {
            anchors.add_champion(&es_state.base_params, elo_best, generation);
        }

        let record = serde_json::json!({
            "gen": generation,
            "wall_s": wall,
            "games": games,
            "games_s": games as f64 / wall,
            "decisions_s": games as f64 * 200.0 / wall,
            "fitness": fitness_stats(&fitness),
            "elo_best": elo_best,
            "elo_mean": elo_mean,
            "sigma": sigma,
            "improved": improved,
            "best_ever": best_ever,
            "optimiser": "es",
            "anchor": anchor_report,
        });
        log_generation(&args.run_dir, &record);

        // Export champion (the base policy, not a perturbation).
        if improved || generation % es_cfg.hof_interval == 0 {
            let path = args
                .run_dir
                .join(format!("champion-gen{:05}.json", generation));
            let _ = genome::save_json(path.to_str().unwrap(), arch, &es_state.base_params);
        }

        // HOF.
        if generation % es_cfg.hof_interval == 0 {
            hof.archive(&es_state.base_params, elo_best, generation);
        }

        // Checkpoint (atomic, with rotation).
        if generation % args.checkpoint_interval == 0
            || generation == start_gen + args.generations - 1
        {
            let _ = es::save_es_checkpoint_with_rotation(
                &args.run_dir,
                generation,
                &es_state,
                &EloTracker::new(0),
                &hof,
                &gen_seeds,
                args.keep_recent,
                args.keep_every,
            );
        }
    }

    // Final champion = base params.
    let path = args.run_dir.join("champion-final.json");
    let _ = genome::save_json(path.to_str().unwrap(), arch, &es_state.base_params);
    println!("done: {}", args.run_dir.display());
    Ok(())
}
