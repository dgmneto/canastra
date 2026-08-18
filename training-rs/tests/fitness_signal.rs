//! The fitness signal must actually discriminate.
//!
//! Every ranking metric in this crate — ELO before, paired score differentials
//! now — folds match results into a per-genome number. If the *matches* come
//! back with both partnerships on identical scores, every metric produces a
//! flat ranking, ES's gradient estimate is exactly zero, and training silently
//! does nothing while the logs still look plausible. That failure is invisible
//! from `generations.jsonl`: a flat ranking and a genuinely tied population
//! print the same numbers.
//!
//! These tests pin the property directly.

use canastra_train::fitness;
use canastra_train::genome::{self, TRAINING_ARCH};
use canastra_train::hof::HallOfFame;
use canastra_train::league;
use canastra_train::seedstream;
use candle_core::Device;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// The device under test. CPU unless `FITNESS_SIGNAL_DEVICE=cuda` and the
/// crate was built with `--features cuda`.
fn test_device() -> Device {
    let dev = match std::env::var("FITNESS_SIGNAL_DEVICE").as_deref() {
        Ok("cuda") => Device::new_cuda(0).expect("CUDA requested but unavailable"),
        _ => Device::Cpu,
    };
    println!(
        "  device = {}",
        if matches!(dev, Device::Cuda(_)) {
            "cuda"
        } else {
            "cpu"
        }
    );
    dev
}

/// Play a small generation and return (results, pairings, seeds).
fn small_generation(
    population: usize,
    opponents: usize,
    n_seeds: usize,
    max_hands: Option<u32>,
    max_width: usize,
) -> (
    Vec<canastra_train::pool::MatchResult>,
    Vec<(usize, usize)>,
    usize,
) {
    let arch = &TRAINING_ARCH;
    let pop = genome::random_population(arch, population, 7);
    let hof = HallOfFame::new();
    let mut rng = StdRng::seed_from_u64(seedstream::splitmix64(7));
    let pairings = league::schedule_pairings_mirrored(population, opponents, &hof, &mut rng);
    let gen_seeds = seedstream::generation_seeds(7, 0, n_seeds);

    let results = league::play_generation(&league::EvalInputs {
        pop: &pop,
        hof: &hof,
        pairings: &pairings,
        arch,
        seeds: &gen_seeds,
        max_hands,
        device: &test_device(),
        max_width,
        dtype: None,
    });
    (results, pairings, gen_seeds.len())
}

fn describe(results: &[canastra_train::pool::MatchResult]) -> String {
    let draws = results.iter().filter(|(_, s, ..)| s[0] == s[1]).count();
    let unfinished = results.iter().filter(|&&(.., u)| u).count();
    let lo = results
        .iter()
        .flat_map(|(_, s, ..)| [s[0], s[1]])
        .min()
        .unwrap_or(0);
    let hi = results
        .iter()
        .flat_map(|(_, s, ..)| [s[0], s[1]])
        .max()
        .unwrap_or(0);
    let hands: Vec<u32> = results.iter().map(|&(.., h, _)| h).collect();
    let max_h = hands.iter().copied().max().unwrap_or(0);
    format!(
        "{} games, {draws} level, {unfinished} unfinished, scores {lo}..{hi}, max hands {max_h}",
        results.len()
    )
}

/// With `max_hands = 1` — the setting `ksweep`, `bench` and the smoke config
/// all use — do games produce *any* score separation?
#[test]
fn one_hand_games_produce_a_non_degenerate_signal() {
    let (results, pairings, n_seeds) = small_generation(8, 2, 2, Some(1), 64);
    println!("max_hands=1: {}", describe(&results));

    let report = fitness::score_generation(&results, &pairings, n_seeds, 8);
    println!("  fitness = {:?}", report.fitness);
    println!("  mean |diff| = {}", report.mean_abs_diff);

    let level = results.iter().filter(|(_, s, ..)| s[0] == s[1]).count();
    assert!(
        level < results.len(),
        "every game ended level ({}) — no metric can rank a population from \
         this, and ES's gradient is identically zero. Untrained policies that \
         never meet §6's opening minimum both score the flat −300 of §13.3, \
         which is a tie by construction.",
        describe(&results)
    );
    assert!(
        report.mean_abs_diff > 0.0,
        "paired differentials are all zero: {}",
        describe(&results)
    );
}

/// Full matches (no hand cap) must separate too — this is the configuration a
/// real run would fall back to if the 1-hand signal proves degenerate.
#[test]
fn full_matches_produce_a_non_degenerate_signal() {
    let (results, pairings, n_seeds) = small_generation(4, 2, 1, Some(8), 64);
    println!("max_hands=8: {}", describe(&results));

    let report = fitness::score_generation(&results, &pairings, n_seeds, 4);
    println!("  fitness = {:?}", report.fitness);
    println!("  mean |diff| = {}", report.mean_abs_diff);

    let level = results.iter().filter(|(_, s, ..)| s[0] == s[1]).count();
    assert!(
        level < results.len(),
        "every game ended level: {}",
        describe(&results)
    );
}

/// Diagnostic at the benchmark population (96) and the exact K=64 sweep config,
/// rather than a toy 8. This is the case that exposed the f16 masking bug — all
/// 6144 games level at −300 to −300 on CUDA, while CPU scored −1015..+2480 — so
/// point it at the GPU when touching the forward path or its dtypes:
///
/// ```powershell
/// $env:FITNESS_SIGNAL_DEVICE="cuda"
/// cargo test --release --features cuda --test fitness_signal -- --ignored --nocapture
/// ```
///
/// Ignored by default: 6144 games is ~4 minutes on CPU in release.
#[test]
#[ignore = "slow: 6144 games"]
fn signal_survives_at_benchmark_population() {
    // The exact ksweep K=64 config: opponents=4, seeds=8.
    let (results, pairings, n_seeds) = small_generation(96, 4, 8, Some(1), 64);
    println!("pop=96 max_hands=1: {}", describe(&results));

    let report = fitness::score_generation(&results, &pairings, n_seeds, 96);
    println!("  mean |diff| = {}", report.mean_abs_diff);
    println!("  fitness[..8] = {:?}", &report.fitness[..8]);

    let level = results.iter().filter(|(_, s, ..)| s[0] == s[1]).count();
    assert!(
        level < results.len(),
        "every game ended level at pop=96: {}",
        describe(&results)
    );
}

/// The paired differential must be antisymmetric on real games: summed over
/// the whole roster it is exactly zero, because every pair contributes `+d` to
/// one genome and `−d` to the other. A non-zero total means the game→pairing
/// attribution has slipped.
#[test]
fn paired_differentials_sum_to_zero_over_the_roster() {
    let (results, pairings, n_seeds) = small_generation(8, 2, 2, Some(1), 64);
    let report = fitness::score_generation(&results, &pairings, n_seeds, 8);

    let weighted: f64 = report
        .fitness
        .iter()
        .zip(report.games.iter())
        .map(|(&f, &g)| f * g as f64)
        .sum();
    assert!(
        weighted.abs() < 1e-9,
        "differentials must cancel across the roster, got {weighted}"
    );
}

/// Mirrored scheduling must give twin `2j` and twin `2j+1` the same opponents,
/// so that `f⁺ − f⁻` compares them under identical conditions.
#[test]
fn mirrored_twins_share_their_opponent_list() {
    let hof = HallOfFame::new();
    let mut rng = StdRng::seed_from_u64(42);
    let pairings = league::schedule_pairings_mirrored(8, 3, &hof, &mut rng);

    for j in 0..4 {
        let plus: Vec<usize> = pairings
            .iter()
            .filter(|&&(me, _)| me == 2 * j)
            .map(|&(_, opp)| opp)
            .collect();
        let minus: Vec<usize> = pairings
            .iter()
            .filter(|&&(me, _)| me == 2 * j + 1)
            .map(|&(_, opp)| opp)
            .collect();
        assert_eq!(plus, minus, "twins of pair {j} got different opponents");
        assert!(!plus.is_empty(), "pair {j} got no opponents");
        assert!(
            !plus.contains(&(2 * j)) && !plus.contains(&(2 * j + 1)),
            "pair {j} was scheduled against itself: {plus:?}"
        );
    }
}
