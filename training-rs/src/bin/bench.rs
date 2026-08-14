use canastra_train::elo::EloTracker;
use canastra_train::ga::{self, GAConfig};
use canastra_train::genome::TRAINING_ARCH;
use canastra_train::league;
use candle_core::Device;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let arch = &TRAINING_ARCH;
    let pop = ga::initial_population(arch, &GAConfig::default(), 7);
    let hof = ga::HallOfFame::new();
    let mut rng = StdRng::seed_from_u64(7);
    let pairings = league::schedule_pairings(96, 4, &hof, &mut rng);
    let seeds: Vec<u64> = (11..19).map(|i| i as u64).collect();

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let mut elo = EloTracker::new(96);

    let began = Instant::now();
    league::evaluate_generation(
        &pop,
        &hof,
        &pairings,
        arch,
        &seeds,
        Some(1),
        &device,
        &mut elo,
        4,
    );
    let elapsed = began.elapsed().as_secs_f64();
    let games = pairings.len() * 2 * seeds.len();
    println!(
        "1 generation in {:.1}s = {:.0} games/s",
        elapsed,
        games as f64 / elapsed
    );
    Ok(())
}
