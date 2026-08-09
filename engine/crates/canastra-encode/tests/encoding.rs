//! The invariants the layout lives or dies by, pinned end to end.

use canastra_encode::{OBS_DIM, encode_observation};
use canastra_engine::state::Phase;
use canastra_engine::testkit::Rig;
use canastra_engine::{GameState, Seat, apply, new_game, observe};

fn seat(index: u8) -> Seat {
    Seat::new(index).unwrap()
}

fn encoded(state: &GameState, seat: Seat) -> Vec<f32> {
    let mut out = vec![f32::NAN; OBS_DIM];
    encode_observation(&observe(state, seat), &mut out);
    out
}

/// SplitMix64 — deterministic pseudo-randomness for the fuzz, no dependency.
fn mix(seed: u64, ply: u64) -> usize {
    let mut x = seed ^ ply.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (x ^ (x >> 31)) as usize
}

#[test]
fn the_length_is_fixed_across_real_play() {
    // Principle 1: a silently variable vector fails as garbage learning.
    // Encode at every ply of whole matches and demand a finite OBS_DIM vector.
    for seed in 0..3u64 {
        let mut state = new_game(seed);
        for ply in 0..50_000u64 {
            if state.phase == Phase::HandOver {
                state = canastra_engine::settle_hand(&state).unwrap();
                continue;
            }
            if state.phase == Phase::MatchOver {
                break;
            }
            let out = encoded(&state, state.turn);
            assert!(out.iter().all(|x| x.is_finite()), "seed {seed} ply {ply}");
            assert!(out.iter().all(|x| *x == 0.0 || *x == 1.0), "bits only");
            let legal = canastra_engine::enumerate(&state, state.turn);
            if legal.is_empty() {
                break;
            }
            let pick = legal[mix(seed, ply) % legal.len()].clone();
            state = apply(&state, state.turn, &pick).unwrap();
        }
    }
}

#[test]
fn the_encoding_is_relative_to_the_acting_seat() {
    let state = Rig::new()
        .hand(0, "4C 5C 6C")
        .hand(1, "4D")
        .hand(2, "4H 5H")
        .hand(3, "4S 5S 6S 7S")
        .scores(300, 1300)
        .build();
    // Seat 0's "right" block (303..315) is seat 1's count; seat 1's "right"
    // block is seat 2's count; seat 2's "partner" block (315..327) is seat 0's.
    let from0 = encoded(&state, seat(0));
    let from1 = encoded(&state, seat(1));
    let from2 = encoded(&state, seat(2));
    assert_eq!(from0[303..315].iter().sum::<f32>(), 0.0, "seat 1 holds 1");
    assert_eq!(from1[303..315].iter().sum::<f32>(), 1.0, "seat 2 holds 2");
    assert_eq!(from2[315..327].iter().sum::<f32>(), 1.0, "seat 0 holds 3");
    // Scores: 300 for team 0 reads as "mine" from seat 0, "theirs" from seat 1.
    assert_eq!(from0[350..370].iter().sum::<f32>(), 1.0);
    assert_eq!(from1[370..390].iter().sum::<f32>(), 1.0);
}

#[test]
fn hidden_information_never_reaches_the_vector() {
    // F6: two states that differ only in what the acting seat may not see
    // must encode identically — the stock's order, and the contents (not
    // counts) of the other hands.
    let base = Rig::new()
        .stock("8C 9D TH JS")
        .hand(0, "4C 5C 6C")
        .hand(1, "4D 5D")
        .hand(2, "4H 5H 6H")
        .hand(3, "4S 5S")
        .discard("9C")
        .build();
    let shuffled_hidden = Rig::new()
        .stock("JS TH 9D 8C") // same cards, different order — face down
        .hand(0, "QD QC JC") // different contents, same counts
        .hand(1, "4D 5D")
        .hand(2, "KD QD JD")
        .hand(3, "KS QS")
        .discard("9C")
        .build();
    assert_eq!(encoded(&base, seat(1)), encoded(&shuffled_hidden, seat(1)));
}
