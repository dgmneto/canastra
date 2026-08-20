//! Correctness gate for the optimization harness.
//!
//! Independent of the benchmark. Runs ONE fixed-seed generation through the
//! production lockstep path (`league::play_generation`) on CPU (fp32, exact),
//! serializes the full per-game result vector to canonical JSON, and either
//! writes the golden fixture (`--gen`) or checks the live output is
//! byte-identical to the committed golden (`--check`).
//!
//! The generation is a pure function of the hardcoded seed 7 (same setup as
//! `bench.rs`: `random_genome(arch, 7)` + `ESState::new(base, cfg, 7)` +
//! `StdRng::seed_from_u64(7)` + mirrored pairings + seeds 11..). A change that
//! speeds the loop up by altering behaviour will flip at least one argmax pick
//! → a different score → a different result vector → FAIL. The gate does not
//! run on the benchmark's timing path and adds nothing to it.
//!
//! Build (CPU-only, no CUDA toolchain needed):
//!   cargo build --release --target-dir target/gate --bin gate
//! Generate golden from current HEAD:
//!   target/gate/release/gate --gen -o bench/golden/generation.json
//! Check (exit 0 = pass, 1 = fail):
//!   target/gate/release/gate --check -i bench/golden/generation.json
//!
//! This binary is harness infrastructure, not a hot path. It does not touch the
//! rollout, the forward pass, or any timing code.

use canastra_train::es::{ESConfig, ESState};
use canastra_train::genome::{self, TRAINING_ARCH};
use canastra_train::hof::HallOfFame;
use canastra_train::league;
use candle_core::Device;
use clap::Parser;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

/// Fixed gate config — CPU, small, deterministic. Mirrors bench.rs's setup but
/// tiny so the gate runs in seconds and never needs the GPU.
const POP: usize = 8;
const OPPONENTS: usize = 2;
const SEEDS: usize = 2;
const SEED: u64 = 7;

#[derive(Parser)]
#[command(
    name = "gate",
    about = "Byte-identical correctness gate for the optimisation harness"
)]
struct Args {
    /// Write the golden fixture from the live run.
    #[arg(long)]
    gen: bool,
    /// Check the live run against the committed golden.
    #[arg(long)]
    check: bool,
    /// Output path for --gen.
    #[arg(short = 'o', long)]
    out: Option<PathBuf>,
    /// Golden input path for --check.
    #[arg(short = 'i', long)]
    input: Option<PathBuf>,
}

/// Canonical JSON for a MatchResult = (seed, [s0,s1], winner, hands, unplayed).
/// Compact, deterministic ordering — byte-identical across runs and machines.
fn results_to_json(results: &[canastra_train::pool::MatchResult]) -> String {
    let arr: Vec<serde_json::Value> = results
        .iter()
        .map(|(seed, scores, winner, hands, unplayed)| {
            serde_json::json!([
                seed,
                [scores[0], scores[1]],
                winner.map(|w| w as u32),
                hands,
                unplayed,
            ])
        })
        .collect();
    // to_string is deterministic (no map key ordering involved — pure arrays).
    serde_json::to_string(&serde_json::Value::Array(arr)).expect("serialize results")
}

fn build_and_run() -> Vec<canastra_train::pool::MatchResult> {
    let arch = &TRAINING_ARCH;
    // Same setup as bench.rs: ES mirrored population from a fixed base genome.
    let es_cfg = ESConfig {
        n_perturbations: POP / 2,
        ..Default::default()
    };
    let base = genome::random_genome(arch, SEED);
    let es_state = ESState::new(base, &es_cfg, SEED);
    let pop = es_state.materialise_population(arch);
    let hof = HallOfFame::new();
    let mut rng = StdRng::seed_from_u64(SEED);
    let pairings = league::schedule_pairings_mirrored(POP, OPPONENTS, &hof, &mut rng);
    let seeds: Vec<u64> = (11..11 + SEEDS).map(|i| i as u64).collect();

    league::play_generation(&league::EvalInputs {
        pop: &pop,
        hof: &hof,
        pairings: &pairings,
        arch,
        seeds: &seeds,
        max_hands: Some(1),
        device: &Device::Cpu,
        max_width: 64,
        dtype: None,
    })
}

fn main() -> ExitCode {
    let args = Args::parse();
    if args.gen == args.check {
        eprintln!("gate: pass exactly one of --gen / --check");
        return ExitCode::from(2);
    }
    let results = build_and_run();
    let live = results_to_json(&results);

    if args.gen {
        let out = args
            .out
            .unwrap_or_else(|| PathBuf::from("bench/golden/generation.json"));
        if let Some(parent) = out.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(&out, &live).expect("write golden");
        eprintln!(
            "gate: wrote golden ({} bytes, {} games) -> {}",
            live.len(),
            results.len(),
            out.display()
        );
        ExitCode::SUCCESS
    } else {
        let input = args
            .input
            .unwrap_or_else(|| PathBuf::from("bench/golden/generation.json"));
        let golden = match fs::read_to_string(&input) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("gate: cannot read golden {}: {e}", input.display());
                return ExitCode::from(1);
            }
        };
        // Compare as parsed JSON (tolerant of trailing whitespace) then also
        // require byte-identity of the canonical form.
        let golden_canon = serde_json::to_string(
            &serde_json::from_str::<serde_json::Value>(&golden).expect("golden is valid JSON"),
        )
        .expect("reserialize golden");
        if golden_canon == live {
            eprintln!(
                "gate: PASS ({} bytes, {} games) byte-identical to {}",
                live.len(),
                results.len(),
                input.display()
            );
            ExitCode::SUCCESS
        } else {
            // Find first divergence for the detail string.
            let gv: Vec<serde_json::Value> =
                serde_json::from_str(&golden_canon).expect("golden array");
            let lv: Vec<serde_json::Value> = serde_json::from_str(&live).expect("live array");
            let mut detail = String::from("result vector diverged");
            if gv.len() != lv.len() {
                detail = format!(
                    "result count changed: golden={} live={}",
                    gv.len(),
                    lv.len()
                );
            } else {
                for (i, (g, l)) in gv.iter().zip(lv.iter()).enumerate() {
                    if g != l {
                        detail = format!("first divergence at game {i}: golden={g} live={l}");
                        break;
                    }
                }
            }
            eprintln!("gate: FAIL — {detail}");
            // Write the live output next to the golden for easy diffing.
            let diff_path = input.with_extension("json.live");
            let _ = fs::write(&diff_path, &live);
            eprintln!("gate: live output written to {}", diff_path.display());
            ExitCode::from(1)
        }
    }
}
