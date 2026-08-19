use canastra_train::anchors::AnchorSet;
use canastra_train::elo::EloTracker;
use canastra_train::es::{self, ESConfig, ESState};
use canastra_train::fitness;
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

    /// Max archived HOF genomes. Every archived genome is re-uploaded to the
    /// GPU each generation and written into every checkpoint, so this is a
    /// hard cost, not a soft one. Over capacity, the archive is thinned to
    /// stay spread across training history.
    #[arg(long, default_value = "20")]
    hof_capacity: usize,

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
    let (start_gen, mut es_state, mut hof, mut best_ever, mut anchors) =
        if let Some(ref resume_dir) = args.resume {
            let ckpt = es::load_es_checkpoint(resume_dir)?;
            // The checkpoint carries whatever archive it was written with; the
            // flag governs from here on, so resuming with a smaller cap thins
            // an oversized archive rather than carrying it forever.
            let mut loaded_hof = ckpt.hof;
            loaded_hof.capacity = args.hof_capacity.max(1);
            let next_gen = ckpt.generation + 1;
            // Restore the progress metric and the anchor set, so a resumed run
            // continues from the same best_ever and the same fixed-reference
            // opponents (random bot + frozen champions). Older checkpoints that
            // predate anchor persistence have neither; fall back to the legacy
            // fresh start for those.
            let restored = ckpt.anchors.is_some();
            let (best_ever, anchors) = match (ckpt.best_ever, ckpt.anchors) {
                (Some(be), Some(snap)) => (be, AnchorSet::from_snapshot(snap)),
                _ => (f64::NEG_INFINITY, AnchorSet::new(arch)),
            };
            eprintln!(
                "ES: resumed from gen {}, continuing at gen {} (HOF {} entries, {})",
                ckpt.generation,
                next_gen,
                loaded_hof.len(),
                if restored {
                    "anchors restored"
                } else {
                    "fresh anchors"
                }
            );
            (next_gen, ckpt.state, loaded_hof, best_ever, anchors)
        } else {
            let base = genome::random_genome(arch, args.run_seed);
            let state = ESState::new(base, &es_cfg, args.run_seed);
            (
                0u32,
                state,
                HallOfFame::with_capacity(args.hof_capacity),
                f64::NEG_INFINITY,
                AnchorSet::new(arch),
            )
        };

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
        // Mirrored twins share an opponent list, so `f⁺ − f⁻` compares θ+σε and
        // θ−σε under identical conditions — the antithetic cancellation ES
        // assumes. See `league::schedule_pairings_mirrored`.
        let pairings =
            league::schedule_pairings_mirrored(pop_size, args.opponents, &hof, &mut gen_rng);

        let league_began = Instant::now();
        let results = league::play_generation(&league::EvalInputs {
            pop: &pop,
            hof: &hof,
            pairings: &pairings,
            arch,
            seeds: &gen_seeds,
            max_hands,
            device: &device,
            max_width,
            dtype: None,
        });

        // Fitness = mean duplicate-deal score differential. Antisymmetric, so
        // the population mean sits near zero by construction.
        let report =
            fitness::score_generation(&results, &pairings, gen_seeds.len(), pop_size + hof.len());
        let fitness: Vec<f64> = report.fitness[..pop_size].to_vec();

        // Throughput is the league's alone. `wall` additionally covers
        // materialising the population, the anchor evaluation (which plays its
        // own games, uncounted by `games`) and the Adam step — so dividing
        // `games` by it would understate games/s by ~40% at pop=1000.
        let league_wall = league_began.elapsed().as_secs_f64();

        let champion = fitness
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let fit_best = fitness[champion];
        let fit_mean = fitness.iter().sum::<f64>() / pop_size as f64;
        let win_best = report.win_rate[champion];
        let sigma = es_state.sigma;

        // Anchored evaluation — the *only* cross-generation progress metric.
        // Within-generation fitness cannot play that role: it is antisymmetric
        // (its population mean is zero by construction) and re-measured against
        // a fresh population each generation, so it tracks how much the twins
        // differ, not how strong they are. The old `elo_best` had exactly the
        // same defect.
        //
        // Rated on the base policy θ, not on `pop[champion]`. θ is the policy
        // actually being trained; the highest-fitness perturbation is an
        // ephemeral probe whose lead is mostly noise, and rating a different
        // random perturbation every generation would inject that noise straight
        // into the one clean signal. This is the same reasoning the freeze site
        // below already applied — the two now agree.
        //
        // Run *before* `es_state.update`, so "generation N's anchor rating" is
        // the θ that produced generation N's population.
        let anchor_report = if args.anchor_interval > 0 && generation % args.anchor_interval == 0 {
            let report = anchors.evaluate(
                &es_state.base_params,
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

        // "Improved" means the anchor rating reached a new high — a claim about
        // absolute strength. Generations without an anchor evaluation make no
        // claim either way.
        let improved = anchor_report.is_some() && anchors.champion_rating > best_ever;
        if improved {
            best_ever = anchors.champion_rating;
        }

        // ES update: rank-normalised fitness → gradient estimate → Adam.
        es_state.update(&fitness, &es_cfg);

        let wall = began.elapsed().as_secs_f64();
        println!(
            "gen {}: best {:+.1} pts (win {:.0}%) spread {:.1} sigma {:.4} ({:.1}s){}",
            generation,
            fit_best,
            win_best * 100.0,
            report.mean_abs_diff,
            sigma,
            wall,
            if improved { "  new best-ever" } else { "" }
        );

        // Log to generations.jsonl.
        let games = pop_size * args.opponents * args.seeds * 2;

        // Freeze the base policy as an anchor. A frozen anchor's rating must be
        // on the *anchor* scale, so later champions rated against it inherit a
        // comparable number; the old code froze it at the internal self-play
        // ELO, a scale that reset every generation.
        if args.anchor_freeze_interval > 0
            && generation > 0
            && generation % args.anchor_freeze_interval == 0
        {
            let frozen_at = anchors.champion_rating;
            anchors.add_champion(&es_state.base_params, frozen_at, generation);
        }

        let record = serde_json::json!({
            "gen": generation,
            "wall_s": wall,
            "games": games,
            // League games only, over the league's own wall time — `wall`
            // additionally covers the anchor evaluation and the Adam step.
            "league_wall_s": league_wall,
            "games_s": games as f64 / league_wall,
            // Not measured: a fixed 200 plies/game assumption, so this is
            // games_s × 200 and carries no information games_s doesn't. Kept
            // only because existing logs have the field.
            "decisions_s": games as f64 * 200.0 / league_wall,
            // Fitness is the mean duplicate-deal score differential, in points.
            "fitness": fitness::fitness_stats(&fitness),
            "fitness_best": fit_best,
            "fitness_mean": fit_mean,
            "win_rate_best": win_best,
            // Mean |paired differential| — the scale of the signal. Collapsing
            // toward zero means the population has stopped differentiating.
            "signal_spread": report.mean_abs_diff,
            "sigma": sigma,
            // `improved` / `best_ever` are on the anchor rating scale, the only
            // one comparable across generations.
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

        // HOF. Archived with the anchor rating, which is the only figure on a
        // scale that stays comparable across generations.
        if generation % es_cfg.hof_interval == 0 {
            hof.archive(&es_state.base_params, anchors.champion_rating, generation);
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
                best_ever,
                Some(&anchors.snapshot()),
            );
        }
    }

    // Final champion = base params.
    let path = args.run_dir.join("champion-final.json");
    let _ = genome::save_json(path.to_str().unwrap(), arch, &es_state.base_params);
    println!("done: {}", args.run_dir.display());
    Ok(())
}
