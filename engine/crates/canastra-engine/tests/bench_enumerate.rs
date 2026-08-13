//! Microbenchmark for `enumerate`, measuring three representative positions.
//!
//! Run with: cargo test -p canastra-engine bench_enumerate -- --nocapture --ignored

use canastra_engine::enumerate::enumerate;
use canastra_engine::state::{GameState, Phase};
use canastra_engine::testkit::Rig;
use std::time::Instant;

fn bench(label: &str, state: &GameState, iterations: usize) {
    let seat = state.turn;
    // Warmup
    let _ = enumerate(state, seat);

    let start = Instant::now();
    let mut total_actions = 0usize;
    for _ in 0..iterations {
        let actions = enumerate(state, seat);
        total_actions += actions.len();
    }
    let elapsed = start.elapsed();

    let per_call = elapsed / iterations as u32;
    let avg_actions = total_actions / iterations;
    println!(
        "{label:30} {avg_actions:5} actions  {per_call:10?}/call  ({iterations} iters, {elapsed:?} total)"
    );
}

fn early_hand_awaiting_draw() -> GameState {
    Rig::new()
        .hand(1, "4H 5H 6H 7H 8H 9H TH JH QH KH AS AD AC 2H 2S")
        .stock("6D 7D 8D 9D TD JD QD KD 4C 5C 6C 7C 8C 9C TC JC QC KC")
        .discard("9S")
        .turn(1)
        .build()
}

fn mid_hand_melding() -> GameState {
    Rig::new()
        .hand(1, "4H 5H 6H 7H 8H 9H TH AS AD AC 2H 2S JOKER")
        .stock("6D 7D 8D")
        .meld(1, "6C 7C 8C")
        .meld(1, "9C TC JC")
        .meld(1, "QC KC AC")
        .opened(1)
        .phase(Phase::Melding)
        .turn(1)
        .build()
}

fn ace_rich_melding() -> GameState {
    Rig::new()
        .hand(1, "AH AH AD AD AC AS 2H 2S 2D 2C JOKER JOKER 4H 5H 6H")
        .stock("6D 7D")
        .meld(1, "6C 7C 8C")
        .opened(1)
        .phase(Phase::Melding)
        .turn(1)
        .build()
}

fn worst_case_melding() -> GameState {
    Rig::new()
        .hand(
            1,
            "AH AH AD AD AC AS 2H 2S 2D 2C JOKER JOKER 4H 5H 6H 7H 8H 9H TH JH QH KH",
        )
        .stock("6D 7D")
        .meld(1, "6C 7C 8C")
        .meld(1, "9C TC JC")
        .meld(1, "QC KC AC")
        .meld(1, "4D 5D 6D")
        .meld(1, "7D 8D 9D")
        .opened(1)
        .phase(Phase::Melding)
        .turn(1)
        .build()
}

#[test]
#[ignore = "microbenchmark — run with --ignored --nocapture"]
fn bench_enumerate() {
    let early = early_hand_awaiting_draw();
    let mid = mid_hand_melding();
    let ace_rich = ace_rich_melding();
    let worst = worst_case_melding();

    // Quick action count for each position
    println!("\n=== Action counts ===");
    println!(
        "early_hand:   {} actions",
        enumerate(&early, early.turn).len()
    );
    println!("mid_hand:     {} actions", enumerate(&mid, mid.turn).len());
    println!(
        "ace_rich:     {} actions",
        enumerate(&ace_rich, ace_rich.turn).len()
    );
    println!(
        "worst_case:   {} actions",
        enumerate(&worst, worst.turn).len()
    );

    println!("\n=== Benchmarks (release build) ===");
    bench("early_hand_awaiting_draw", &early, 10_000);
    bench("mid_hand_melding", &mid, 10_000);
    bench("ace_rich_melding", &ace_rich, 5_000);
    bench("worst_case_melding", &worst, 2_000);
}
