use canastra_train::elo::EloTracker;
use canastra_train::ga::{self, GAConfig};
use canastra_train::genome::TRAINING_ARCH;
use canastra_train::league::{self, Rollout};
use candle_core::Device;
use clap::Parser;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "canastra-bench")]
#[command(about = "Benchmark one GA generation")]
struct Args {
    /// Population size.
    #[arg(long, default_value = "96")]
    population: usize,

    /// Opponents per genome.
    #[arg(long, default_value = "4")]
    opponents: usize,

    /// Seeds per opponent pairing.
    #[arg(long, default_value = "8")]
    seeds: usize,

    /// Worker threads.
    #[arg(long, default_value = "8")]
    workers: usize,

    /// Device: "cuda" or "cpu".
    #[arg(long, default_value = "cuda")]
    device: String,

    /// Rollout path: "auto", "lockstep", or "coalesced".
    #[arg(long, default_value = "auto")]
    rollout: String,

    /// Max legal actions per row (menu width cap). 0 = no cap.
    #[arg(long, default_value = "64")]
    max_width: usize,
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
    let mut rng = StdRng::seed_from_u64(7);
    let pairings = league::schedule_pairings(args.population, args.opponents, &hof, &mut rng);
    let seeds: Vec<u64> = (11..11 + args.seeds).map(|i| i as u64).collect();
    let games = pairings.len() * 2 * seeds.len();

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
    let rollout_label = match rollout {
        Rollout::Lockstep => "lockstep",
        Rollout::Coalesced => "coalesced",
    };
    let mut elo = EloTracker::new(args.population);

    eprintln!(
        "pop={} opponents={} seeds={} games={} workers={} device={} rollout={}",
        args.population,
        args.opponents,
        args.seeds,
        games,
        args.workers,
        device_label,
        rollout_label
    );

    let began = Instant::now();
    league::evaluate_generation(
        &league::EvalInputs {
            pop: &pop,
            hof: &hof,
            pairings: &pairings,
            arch,
            seeds: &seeds,
            max_hands: Some(1),
            device: &device,
            n_workers: args.workers,
            rollout,
            max_width: if args.max_width == 0 {
                usize::MAX
            } else {
                args.max_width
            },
        },
        &mut elo,
    );
    let elapsed = began.elapsed().as_secs_f64();
    println!(
        "pop={} games={} device={} rollout={}: {:.1}s = {:.0} games/s",
        args.population,
        games,
        device_label,
        rollout_label,
        elapsed,
        games as f64 / elapsed
    );
    Ok(())
}
