//! Small repeatable benchmark for the cloning and non-cloning legality paths.
//!
//! This is intentionally an engine-only measurement. It captures real positions
//! first, then times the checks against the same accepted action corpus without
//! including deal/setup work in either path.

use std::hint::black_box;
use std::time::Instant;

use canastra_engine::score::settle_hand;
use canastra_engine::{Action, GameState, Phase, Seat, apply, enumerate, new_game, validate};

struct Case {
    state: GameState,
    seat: Seat,
    action: Action,
}

const REPEATS: usize = 20;

fn mix(seed: u64, ply: u64) -> usize {
    let mut x = seed ^ ply.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (x ^ (x >> 31)) as usize
}

fn corpus(limit: usize) -> (Vec<(GameState, Seat)>, Vec<Case>) {
    let mut positions = Vec::new();
    let mut cases = Vec::new();

    for seed in 0..10u64 {
        let mut state = new_game(seed);
        let mut turn_start = state.clone();
        let mut safe = false;
        for ply in 0..20_000u64 {
            if state.phase == Phase::HandOver {
                state = settle_hand(&state).expect("settle");
                continue;
            }
            if state.phase == Phase::MatchOver {
                break;
            }
            if matches!(
                state.phase,
                Phase::AwaitingDraw | Phase::AwaitingRefusalChoice
            ) {
                turn_start = state.clone();
                safe = false;
            }

            let seat = state.turn;
            let actions = enumerate(&state, seat);
            if actions.is_empty() {
                assert!(!safe, "the safe retry dead-ended");
                state = turn_start.clone();
                safe = true;
                continue;
            }

            positions.push((state.clone(), seat));
            for action in &actions {
                cases.push(Case {
                    state: state.clone(),
                    seat,
                    action: action.clone(),
                });
                if cases.len() == limit {
                    return (positions, cases);
                }
            }

            let pick = if safe {
                0
            } else {
                mix(seed, ply) % actions.len()
            };
            let action = actions[pick].clone();
            state = apply(&state, seat, &action).expect("enumerated action applies");
        }
    }

    (positions, cases)
}

fn main() {
    let (positions, cases) = corpus(50_000);
    assert!(!positions.is_empty() && !cases.is_empty());
    let checks = cases.len() * REPEATS;
    let enumerate_calls = positions.len() * REPEATS;

    let began = Instant::now();
    let mut valid = 0usize;
    for _ in 0..REPEATS {
        for case in &cases {
            valid += usize::from(black_box(validate(&case.state, case.seat, &case.action)).is_ok());
        }
    }
    let validate_elapsed = began.elapsed();

    let began = Instant::now();
    let mut applied = 0usize;
    for _ in 0..REPEATS {
        for case in &cases {
            applied += usize::from(black_box(apply(&case.state, case.seat, &case.action)).is_ok());
        }
    }
    let apply_elapsed = began.elapsed();

    let began = Instant::now();
    let mut enumerated_actions = 0usize;
    for _ in 0..REPEATS {
        for (state, seat) in &positions {
            enumerated_actions += black_box(enumerate(state, *seat)).len();
        }
    }
    let enumerate_elapsed = began.elapsed();

    println!(
        "checks {}/{}; validate {:.3}s ({:.0}/s); apply {:.3}s ({:.0}/s); enumerate {} actions from {} calls in {:.3}s ({:.0} calls/s)",
        valid,
        applied,
        validate_elapsed.as_secs_f64(),
        checks as f64 / validate_elapsed.as_secs_f64(),
        apply_elapsed.as_secs_f64(),
        checks as f64 / apply_elapsed.as_secs_f64(),
        enumerated_actions,
        enumerate_calls,
        enumerate_elapsed.as_secs_f64(),
        enumerate_calls as f64 / enumerate_elapsed.as_secs_f64(),
    );
}
