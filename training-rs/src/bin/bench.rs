use canastra_train::es::{ESConfig, ESState};
use canastra_train::fitness;
use canastra_train::genome::{self, TRAINING_ARCH};
use canastra_train::hof::HallOfFame;
use canastra_train::league;
use candle_core::Device;
use clap::Parser;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "canastra-bench")]
#[command(about = "Benchmark one generation (lockstep)")]
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

    /// Device: "cuda" or "cpu".
    #[arg(long, default_value = "cuda")]
    device: String,

    /// Max legal actions per row (menu width cap). 0 = no cap.
    #[arg(long, default_value = "64")]
    max_width: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let arch = &TRAINING_ARCH;
    // Use an ES population (θ ± σε pairs) so the ES grouped-GEMM split is
    // available — the production training path always uses ES. The population
    // is 2 × n_perturbations (mirrored pairs), matching the production config.
    let es_cfg = ESConfig {
        n_perturbations: args.population / 2,
        ..Default::default()
    };
    let base = genome::random_genome(arch, 7);
    let es_state = ESState::new(base, &es_cfg, 7);
    let pop = es_state.materialise_population(arch);
    let hof = HallOfFame::new();
    let mut rng = StdRng::seed_from_u64(7);
    // Mirrored scheduling, matching the production training path.
    let pairings =
        league::schedule_pairings_mirrored(args.population, args.opponents, &hof, &mut rng);
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
    eprintln!(
        "pop={} opponents={} seeds={} games={} device={}",
        args.population, args.opponents, args.seeds, games, device_label
    );

    let began = Instant::now();
    let results = league::play_generation(&league::EvalInputs {
        pop: &pop,
        hof: &hof,
        pairings: &pairings,
        arch,
        seeds: &seeds,
        max_hands: Some(1),
        device: &device,
        max_width: if args.max_width == 0 {
            usize::MAX
        } else {
            args.max_width
        },
    });
    let elapsed = began.elapsed().as_secs_f64();

    // Report the signal alongside the throughput. A forward pass that has
    // silently degenerated — every policy taking the first legal action —
    // finishes games almost instantly and reads as a large speedup, which is
    // exactly how the f16 masking bug was recorded as a 3.9x win (see
    // docs/decision-ranking-metric.md). Throughput on games nobody played is
    // not a benchmark, so print the numbers that expose it.
    let report = fitness::score_generation(&results, &pairings, seeds.len(), args.population);
    let level = results.iter().filter(|(_, s, ..)| s[0] == s[1]).count();

    println!(
        "pop={} games={} device={}: {:.1}s = {:.0} games/s | level {:.1}% | mean |diff| {:.0} pts",
        args.population,
        games,
        device_label,
        elapsed,
        games as f64 / elapsed,
        100.0 * level as f64 / results.len() as f64,
        report.mean_abs_diff,
    );
    if level == results.len() {
        eprintln!(
            "!! every game ended level — the forward pass is degenerate, this \
             throughput figure is meaningless"
        );
    }
    Ok(())
}
