//! Dump observation vectors from real rollouts and measure sparsity.
//!
//! Runs a short lockstep rollout (pop=8, seeds=2, max_hands=1) and collects
//! ≥10,000 observation vectors. Reports non-zero fraction, binary fraction,
//! max non-zeros per row, and whether non-zeros are contiguous one-hot blocks
//! or scattered.
//!
//! Run with:
//!   target\release\canastra-sparsity.exe

use canastra_encode::{encode_observation, OBS_DIM};
use canastra_engine::{apply, enumerate, new_game, observe, Phase};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn main() {
    let n_games = 2000;
    let target_samples = 10_000;
    let mut all_obs: Vec<Vec<f32>> = Vec::with_capacity(target_samples);
    let mut rng = StdRng::seed_from_u64(42);

    // Play many single-hand games with random actions, collecting observations.
    for game_i in 0..n_games {
        if all_obs.len() >= target_samples {
            break;
        }
        let seed = (game_i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut state = new_game(seed);
        let mut actions_played = 0u64;

        loop {
            if matches!(state.phase, Phase::MatchOver) {
                break;
            }
            if actions_played > 500 {
                break; // safety
            }
            let seat = state.turn;
            let legal = enumerate(&state, seat);
            if legal.is_empty() {
                break;
            }
            // Collect observation at decision points.
            if matches!(
                state.phase,
                Phase::AwaitingDraw | Phase::AwaitingRefusalChoice | Phase::Melding
            ) {
                let view = observe(&state, seat);
                let mut obs = vec![0.0f32; OBS_DIM];
                encode_observation(&view, &mut obs);
                all_obs.push(obs);
                if all_obs.len() >= target_samples {
                    break;
                }
            }
            // Pick a random legal action.
            let pick = (rng.gen::<u32>() as usize) % legal.len();
            match apply(&state, seat, &legal[pick]) {
                Ok(next) => state = next,
                Err(_) => break,
            }
            actions_played += 1;
        }
    }

    let n = all_obs.len();
    println!("Collected {} observation vectors (OBS_DIM={})", n, OBS_DIM);

    // ── Sparsity stats ──
    let mut total_nonzeros: usize = 0;
    let mut total_binary_nonzeros: usize = 0;
    let mut max_nonzeros: usize = 0;
    let mut min_nonzeros: usize = usize::MAX;
    let mut nonzero_counts = Vec::with_capacity(n);

    for obs in &all_obs {
        let mut nz = 0usize;
        let mut binary_nz = 0usize;
        for &v in obs {
            if v != 0.0 {
                nz += 1;
                if v == 1.0 {
                    binary_nz += 1;
                }
            }
        }
        total_nonzeros += nz;
        total_binary_nonzeros += binary_nz;
        max_nonzeros = max_nonzeros.max(nz);
        min_nonzeros = min_nonzeros.min(nz);
        nonzero_counts.push(nz);
    }

    let mean_nz = total_nonzeros as f64 / n as f64;
    let density = total_nonzeros as f64 / (n as f64 * OBS_DIM as f64);
    let binary_pct = total_binary_nonzeros as f64 / total_nonzeros as f64 * 100.0;

    nonzero_counts.sort();
    let p25 = nonzero_counts[n / 4];
    let p50 = nonzero_counts[n / 2];
    let p75 = nonzero_counts[3 * n / 4];
    let p95 = nonzero_counts[(95 * n) / 100];
    let p99 = nonzero_counts[(99 * n) / 100];

    println!();
    println!("=== Sparsity ===");
    println!("OBS_DIM:              {}", OBS_DIM);
    println!("Total elements:       {}", n * OBS_DIM);
    println!("Total non-zeros:      {}", total_nonzeros);
    println!("Density (mean):       {:.4}%", density * 100.0);
    println!(
        "Binary non-zeros:     {} ({:.1}%)",
        total_binary_nonzeros, binary_pct
    );
    println!();
    println!("Non-zeros per row:");
    println!("  min:   {}", min_nonzeros);
    println!("  p25:   {}", p25);
    println!("  p50:   {}", p50);
    println!("  p75:   {}", p75);
    println!("  p95:   {}", p95);
    println!("  p99:   {}", p99);
    println!("  max:   {}", max_nonzeros);
    println!("  mean:  {:.1}", mean_nz);

    // ── Structure check: are non-zeros contiguous one-hot blocks? ──
    // Check if the non-zero values are always 0.0 or 1.0 (binary) and whether
    // they appear in contiguous runs (one-hot blocks).
    let mut max_run = 0usize;
    let mut total_runs = 0usize;
    for obs in &all_obs {
        let mut current_run = 0usize;
        let mut in_run = false;
        for &v in obs {
            if v != 0.0 {
                if !in_run {
                    in_run = true;
                    current_run = 1;
                    total_runs += 1;
                } else {
                    current_run += 1;
                }
            } else if in_run {
                in_run = false;
                max_run = max_run.max(current_run);
                current_run = 0;
            }
        }
        if in_run {
            max_run = max_run.max(current_run);
        }
    }
    let mean_run_len = total_nonzeros as f64 / total_runs.max(1) as f64;

    println!();
    println!("=== Structure ===");
    println!("Contiguous runs of non-zeros: {}", total_runs);
    println!("Mean run length:              {:.1}", mean_run_len);
    println!("Max run length:               {}", max_run);

    // ── Check for any non-binary values ──
    let mut non_binary_values = std::collections::HashSet::new();
    for obs in &all_obs {
        for &v in obs {
            if v != 0.0 && v != 1.0 {
                non_binary_values.insert(v.to_bits());
            }
        }
    }
    println!("Distinct non-binary values:   {}", non_binary_values.len());
    if !non_binary_values.is_empty() {
        for bits in non_binary_values.iter().take(10) {
            println!("  {}", f32::from_bits(*bits));
        }
    }

    // ── Segment-level breakdown (from obs.rs layout) ──
    println!();
    println!("=== Segment-level density (first 100 rows) ===");
    let segments: &[(&str, usize, usize)] = &[
        ("phase one-hot", 0, 5),
        ("laid_value therm", 5, 18),
        ("took_pile", 18, 19),
        ("refusal_available", 19, 20),
        ("pending card", 20, 73),
        ("hand_number", 73, 79),
        ("my hand census", 79, 183),
        ("my hand jokers", 183, 187),
        ("frozen census", 187, 295),
        ("my hand size", 295, 303),
        ("other hand counts", 303, 339),
        ("stock count", 339, 350),
        ("my score", 350, 370),
        ("their score", 370, 390),
        (">=2500 bits", 390, 392),
        ("opening min", 392, 395),
        ("opened bits", 395, 397),
        ("clean canastra", 397, 399),
        ("red threes", 399, 407),
        ("pile top", 407, 460),
        ("pile size", 460, 475),
        ("pile census", 475, 583),
        ("meld tokens", 583, 2002),
    ];
    for (name, start, end) in segments {
        let mut seg_nz = 0usize;
        let seg_len = end - start;
        for obs in all_obs.iter().take(100) {
            for &v in &obs[*start..*end] {
                if v != 0.0 {
                    seg_nz += 1;
                }
            }
        }
        let seg_density = seg_nz as f64 / (100 * seg_len) as f64;
        println!(
            "  {:>20} [{:>4}:{:>4}] ({:>4} wide): density {:>6.1}%  mean_nz {:>5.1}",
            name,
            start,
            end,
            seg_len,
            seg_density * 100.0,
            seg_nz as f64 / 100.0
        );
    }
}
