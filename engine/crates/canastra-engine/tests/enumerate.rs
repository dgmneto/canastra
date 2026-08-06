//! `enumerate` is the answer to "what may I do right now?" (F7). These tests
//! pin coverage (every legal move is offered), soundness (nothing illegal
//! survives), and determinism (same position, same list).

use std::collections::HashSet;

#[allow(unused_imports)]
use canastra_engine::action::MeldTarget;
use canastra_engine::state::Phase;
use canastra_engine::testkit::Rig;
#[allow(unused_imports)]
use canastra_engine::{Action, GameState, Seat, apply, enumerate, new_game, settle_hand};

fn seat(index: u8) -> Seat {
    Seat::new(index).expect("0..=3")
}

/// The `LayMeld` payloads in a list, as sorted card strings for
/// order-free comparison.
#[allow(dead_code)]
fn lay_melds(actions: &[Action]) -> HashSet<Vec<String>> {
    actions
        .iter()
        .filter_map(|action| match action {
            Action::LayMeld { cards } => Some(cards.iter().map(|card| card.to_string()).collect()),
            _ => None,
        })
        .collect()
}

#[test]
fn awaiting_draw_with_no_takeable_pile_yields_exactly_draw() {
    let state = Rig::new()
        .stock("8C 9D")
        .discard("9C")
        .hand(1, "4D 5D")
        .build();
    assert_eq!(enumerate(&state, seat(1)), vec![Action::Draw]);
}

#[test]
fn refusal_choice_yields_keep_and_refuse() {
    // The stock must be non-empty: refusing means drawing a replacement.
    let state = Rig::new()
        .phase(Phase::AwaitingRefusalChoice)
        .pending_refusal("KH")
        .refusal_available()
        .stock("8C 9D")
        .hand(1, "4D 5D")
        .build();
    let actions = enumerate(&state, seat(1));
    assert!(actions.contains(&Action::KeepDrawnCard));
    assert!(actions.contains(&Action::RefuseDrawnCard));
    assert_eq!(actions.len(), 2);
}

#[test]
fn refusal_without_stock_leaves_only_keep() {
    // Refusing means drawing a replacement, so an empty stock takes the
    // option off the table even mid-decision.
    let state = Rig::new()
        .phase(Phase::AwaitingRefusalChoice)
        .pending_refusal("KH")
        .refusal_available()
        .hand(1, "4D 5D")
        .build();
    assert_eq!(enumerate(&state, seat(1)), vec![Action::KeepDrawnCard]);
}

#[test]
fn other_seats_and_terminal_phases_get_nothing() {
    let state = Rig::new()
        .stock("8C 9D")
        .discard("9C")
        .hand(1, "4D 5D")
        .build();
    assert!(enumerate(&state, seat(0)).is_empty(), "not seat 0's turn");

    for phase in [Phase::HandOver, Phase::MatchOver] {
        let state = Rig::new().phase(phase).build();
        assert!(
            enumerate(&state, seat(1)).is_empty(),
            "{phase:?} decides nothing"
        );
    }
}
