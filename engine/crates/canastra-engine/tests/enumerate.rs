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
fn lay_melds(actions: &[Action]) -> HashSet<Vec<String>> {
    actions
        .iter()
        .filter_map(|action| match action {
            Action::LayMeld { cards } => Some(cards.iter().map(|card| card.to_string()).collect()),
            _ => None,
        })
        .collect()
}

/// Two action lists are the same multiset, compared order-free via their
/// Debug strings (`Action` has no `Hash`, and test assertions should not
/// depend on the enumeration's sort order).
fn assert_same_actions(actual: &[Action], expected: &[Action]) {
    let mut left: Vec<String> = actual.iter().map(|a| format!("{a:?}")).collect();
    let mut right: Vec<String> = expected.iter().map(|a| format!("{a:?}")).collect();
    left.sort();
    right.sort();
    assert_eq!(left, right);
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

#[test]
fn two_aces_and_a_wild_make_an_aces_meld() {
    // The wild counts toward the three-card minimum (Meld::new), so a pair of
    // natural aces plus one wild is a legal lay — and must be offered.
    let state = Rig::new()
        .phase(Phase::Melding)
        .opened(1)
        .hand(1, "AH AD JOKER QS")
        .discard("9C")
        .build();
    let melds = lay_melds(&enumerate(&state, seat(1)));
    assert_eq!(
        melds,
        HashSet::from([vec![
            "AD".to_string(),
            "AH".to_string(),
            "JOKER".to_string()
        ]])
    );
}

#[test]
fn lay_meld_windows_from_a_four_card_run() {
    // §6 note: the partnership is marked open so the eager opening-minimum
    // check does not filter the small melds this test is about.
    let state = Rig::new()
        .phase(Phase::Melding)
        .opened(1)
        .hand(1, "4H 5H 6H 7H QS")
        .discard("9C")
        .build();
    let melds = lay_melds(&enumerate(&state, seat(1)));
    let expected: HashSet<Vec<String>> = [
        vec!["4H", "5H", "6H"],
        vec!["5H", "6H", "7H"],
        vec!["4H", "5H", "6H", "7H"],
    ]
    .into_iter()
    .map(|meld| meld.into_iter().map(String::from).collect())
    .collect();
    assert_eq!(melds, expected);
}

#[test]
fn lay_meld_with_a_joker_filling_the_gap() {
    // The spare QS keeps the lay from emptying the hand: going out needs a
    // clean canastra (§11.1), which this table does not have.
    let state = Rig::new()
        .phase(Phase::Melding)
        .opened(1)
        .hand(1, "6H 7H JOKER QS")
        .discard("9C")
        .build();
    let melds = lay_melds(&enumerate(&state, seat(1)));
    assert_eq!(
        melds,
        HashSet::from([vec!["6H".to_string(), "7H".to_string(), "JOKER".to_string()]])
    );
}

#[test]
fn lay_meld_plain_and_wild_capped_variants() {
    // Spare QS as above: a four-card lay must not empty the hand. Three legal
    // lays exist: the natural run, the Joker capping it, and the Joker
    // standing in for the 7 with just 5H 6H (a window with one gap).
    let state = Rig::new()
        .phase(Phase::Melding)
        .opened(1)
        .hand(1, "4H 5H 6H JOKER QS")
        .discard("9C")
        .build();
    let melds = lay_melds(&enumerate(&state, seat(1)));
    let expected: HashSet<Vec<String>> = [
        vec!["4H", "5H", "6H"],
        vec!["4H", "5H", "6H", "JOKER"],
        vec!["5H", "6H", "JOKER"],
    ]
    .into_iter()
    .map(|meld| meld.into_iter().map(String::from).collect())
    .collect();
    assert_eq!(melds, expected);
}

#[test]
fn each_usable_wild_yields_its_own_meld() {
    // §8: a 2 works only in its own suit. With both a Joker and the 2♥ in
    // hand, each one-gap window produces two distinct melds; the 2♦ is
    // unusable in hearts and irrelevant without diamonds naturals.
    let with_two_wilds = Rig::new()
        .phase(Phase::Melding)
        .opened(1)
        .hand(1, "4H 5H 6H JOKER 2H")
        .discard("9C")
        .build();
    let melds = lay_melds(&enumerate(&with_two_wilds, seat(1)));
    let expected: HashSet<Vec<String>> = [
        vec!["4H", "5H", "6H"],
        vec!["4H", "5H", "6H", "JOKER"],
        vec!["2H", "4H", "5H", "6H"],
        vec!["5H", "6H", "JOKER"],
        vec!["2H", "5H", "6H"],
    ]
    .into_iter()
    .map(|meld| meld.into_iter().map(String::from).collect())
    .collect();
    assert_eq!(melds, expected);

    let foreign_two = Rig::new()
        .phase(Phase::Melding)
        .opened(1)
        .hand(1, "4H 5H 6H 2D")
        .discard("9C")
        .build();
    let melds = lay_melds(&enumerate(&foreign_two, seat(1)));
    assert_eq!(
        melds,
        HashSet::from([vec!["4H".to_string(), "5H".to_string(), "6H".to_string()]])
    );
}
