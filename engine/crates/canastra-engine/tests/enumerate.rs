//! `enumerate` is the answer to "what may I do right now?" (F7). These tests
//! pin coverage (every legal move is offered), soundness (nothing illegal
//! survives), and determinism (same position, same list).

use std::collections::HashSet;

use canastra_engine::action::MeldTarget;
use canastra_engine::state::Phase;
use canastra_engine::testkit::Rig;
use canastra_engine::{Action, GameState, Seat, apply, enumerate, new_game, settle_hand};

fn seat(index: u8) -> Seat {
    Seat::new(index).expect("0..=3")
}

/// The `LayMeld` payloads in a list, as sorted card strings for
/// order-free comparison.
fn lay_melds(actions: &[Action]) -> HashSet<Vec<String>> {
    actions
        .iter()
        .filter_map(|action| match action {
            Action::LayMeld { cards } => {
                Some(cards.iter().map(|card| card.to_string()).collect())
            }
            _ => None,
        })
        .collect()
}

#[test]
fn awaiting_draw_with_no_takeable_pile_yields_exactly_draw() {
    let state = Rig::new().stock("8C 9D").discard("9C").hand(1, "4D 5D").build();
    assert_eq!(enumerate(&state, seat(1)), vec![Action::Draw]);
}
