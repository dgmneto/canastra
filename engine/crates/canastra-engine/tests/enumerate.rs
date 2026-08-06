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
        HashSet::from([vec![
            "6H".to_string(),
            "7H".to_string(),
            "JOKER".to_string()
        ]])
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

#[test]
fn ace_melds_are_sub_multisets_with_and_without_a_wild() {
    // §7.2 with duplicated aces: AH AH AD AS yields every sub-multiset of 3+
    // aces plain, and every sub-multiset of 2+ aces capped with the held
    // Joker (the wild counts toward the three-card minimum). The QS also runs
    // with AS and the Joker into a spade sequence, and keeps even the five-card
    // lay from emptying the hand (§11.1).
    let state = Rig::new()
        .phase(Phase::Melding)
        .opened(1)
        .hand(1, "AH AH AD AS JOKER QS")
        .discard("9C")
        .build();
    let melds = lay_melds(&enumerate(&state, seat(1)));
    let expected: HashSet<Vec<String>> = [
        vec!["AD", "AH", "AH"],
        vec!["AH", "AH", "AS"],
        vec!["AD", "AH", "AS"],
        vec!["AD", "AH", "AH", "AS"],
        vec!["AD", "AH", "JOKER"],
        vec!["AD", "AS", "JOKER"],
        vec!["AH", "AH", "JOKER"],
        vec!["AH", "AS", "JOKER"],
        vec!["AD", "AH", "AH", "JOKER"],
        vec!["AH", "AH", "AS", "JOKER"],
        vec!["AD", "AH", "AS", "JOKER"],
        vec!["AD", "AH", "AH", "AS", "JOKER"],
        vec!["AS", "JOKER", "QS"],
    ]
    .into_iter()
    .map(|meld| meld.into_iter().map(String::from).collect())
    .collect();
    assert_eq!(melds, expected);
}

#[test]
fn discards_dedup_identical_cards() {
    let state = Rig::new()
        .phase(Phase::Melding)
        .hand(1, "6D 6D KH")
        .meld(1, "4H 5H 6H")
        .discard("9C")
        .build();
    let actions = enumerate(&state, seat(1));
    let discards: Vec<Action> = actions
        .iter()
        .filter(|a| matches!(a, Action::Discard { .. }))
        .cloned()
        .collect();
    assert_same_actions(
        &discards,
        &[
            Action::Discard {
                card: "6D".parse().unwrap(),
            },
            Action::Discard {
                card: "KH".parse().unwrap(),
            },
        ],
    );
    // A legal discard exists, so the no-discard escape hatch stays shut.
    assert!(!actions.contains(&Action::EndTurnWithoutDiscard));
}

#[test]
fn frozen_cards_can_be_discarded_but_not_melded() {
    // §5 + clarification #5: cards swept from the pile are frozen this turn —
    // unmeldable, still discardable.
    let state = Rig::new()
        .phase(Phase::Melding)
        .hand(1, "7D 8D 9D 4H 5H 6H")
        .frozen("7D 8D 9D")
        .meld(1, "4D 5D 6D")
        .discard("9C")
        .build();
    let actions = enumerate(&state, seat(1));

    let frozen = ["7D", "8D", "9D"];
    let melding_with_frozen = actions.iter().any(|action| match action {
        Action::LayMeld { cards } => cards
            .iter()
            .any(|c| frozen.contains(&c.to_string().as_str())),
        Action::AddToMeld { cards, .. } => cards
            .iter()
            .any(|c| frozen.contains(&c.to_string().as_str())),
        _ => false,
    });
    assert!(
        !melding_with_frozen,
        "frozen cards must not be melded: {actions:?}"
    );

    // The natural extension that would be legal without the freeze.
    assert!(!actions.contains(&Action::AddToMeld {
        meld: 0,
        cards: vec!["7D".parse().unwrap()]
    }));
    // …while discarding a frozen card stays legal (clarification #5).
    assert!(actions.contains(&Action::Discard {
        card: "7D".parse().unwrap()
    }));
    // And the unfrozen run is still offered.
    assert!(lay_melds(&actions).contains(&vec![
        "4H".to_string(),
        "5H".to_string(),
        "6H".to_string()
    ]));
}

#[test]
fn cornered_player_may_only_end_without_discarding() {
    // CLAUDE.md clarification #6: one card in hand, no clean canastra — no
    // legal discard exists, so EndTurnWithoutDiscard is the only way out.
    let state = Rig::new()
        .phase(Phase::Melding)
        .hand(1, "KH")
        .meld(1, "4H 5H 6H")
        .discard("9C")
        .build();
    assert_eq!(
        enumerate(&state, seat(1)),
        vec![Action::EndTurnWithoutDiscard]
    );
}

#[test]
fn taking_the_pile_the_rules_spec_worked_example() {
    // §5: with the 6♦ on top, the legal cores are 4D 5D, 5D 7D and 7D 8D —
    // each into a new meld, and into whichever existing meld the three join.
    let state = Rig::new()
        .stock("8C 9D")
        .discard("9C 6D")
        .hand(1, "4D 5D 7D 8D KH")
        .meld(1, "7D 8D 9D")
        .meld(1, "8D 9D TD")
        .build();
    let takes: Vec<Action> = enumerate(&state, seat(1))
        .into_iter()
        .filter(|a| matches!(a, Action::TakeDiscardPile { .. }))
        .collect();
    let take = |a: &str, b: &str, target: MeldTarget| Action::TakeDiscardPile {
        core: [a.parse().unwrap(), b.parse().unwrap()],
        target,
    };
    assert_same_actions(
        &takes,
        &[
            take("4D", "5D", MeldTarget::NewMeld),
            take("4D", "5D", MeldTarget::Existing { meld: 0 }),
            take("5D", "7D", MeldTarget::NewMeld),
            take("5D", "7D", MeldTarget::Existing { meld: 1 }),
            take("7D", "8D", MeldTarget::NewMeld),
        ],
    );
}

#[test]
fn taking_the_pile_with_a_same_value_core() {
    // A pair of aces with an ace on top is a legal capture (aces meld).
    let state = Rig::new()
        .stock("8C 9D")
        .discard("9C AS")
        .hand(1, "AH AH KD QC")
        .opened(1)
        .build();
    let actions = enumerate(&state, seat(1));
    assert!(actions.contains(&Action::TakeDiscardPile {
        core: ["AH".parse().unwrap(), "AH".parse().unwrap()],
        target: MeldTarget::NewMeld,
    }));
    assert_eq!(
        actions.len(),
        2,
        "just Draw and the one capture: {actions:?}"
    );
}

#[test]
fn a_blocked_pile_offers_no_takes() {
    // §5: a black 3 on top puts the pile out of reach.
    let state = Rig::new()
        .stock("8C 9D")
        .discard("9C 3S")
        .hand(1, "4S 5S KD QC")
        .build();
    assert_eq!(enumerate(&state, seat(1)), vec![Action::Draw]);
}

#[test]
fn enumeration_is_deterministic() {
    let state = Rig::new()
        .stock("8C 9D")
        .discard("9C 6D")
        .hand(1, "4D 5D 7D 8D KH")
        .meld(1, "7D 8D 9D")
        .build();
    assert_eq!(enumerate(&state, seat(1)), enumerate(&state, seat(1)));
}

/// SplitMix64 finalizer — a deterministic pseudo-random pick without pulling
/// an rng dependency into the test suite.
fn mix(seed: u64, ply: u64) -> usize {
    let mut x = seed ^ ply.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (x ^ (x >> 31)) as usize
}

#[test]
fn everything_enumerated_is_legal_across_whole_matches() {
    for seed in 0..20u64 {
        let mut state: GameState = new_game(seed);
        let mut turn_start = state.clone();
        let mut safe = false;
        for ply in 0..200_000u64 {
            match state.phase {
                Phase::HandOver => {
                    state = settle_hand(&state).expect("settle");
                    continue;
                }
                Phase::MatchOver => break,
                _ => {}
            }
            // A turn can dead-end (§6's eager check is optimistic), so keep
            // the position it began from — the harness driver does the same.
            if matches!(
                state.phase,
                Phase::AwaitingDraw | Phase::AwaitingRefusalChoice
            ) {
                turn_start = state.clone();
                safe = false;
            }
            let turn = state.turn;
            let actions = enumerate(&state, turn);
            for action in &actions {
                assert!(
                    apply(&state, turn, action).is_ok(),
                    "offered an illegal {action:?}"
                );
            }
            if actions.is_empty() {
                // The residual self-strand: back out and finish the turn
                // plainly, which is exactly the driver's safeMode path.
                assert!(!safe, "even the plain retry dead-ended (seed {seed})");
                state = turn_start.clone();
                safe = true;
                continue;
            }
            let pick = if safe {
                0
            } else {
                mix(seed, ply) % actions.len()
            };
            state = apply(&state, turn, &actions[pick]).expect("enumerated action applies");
        }
    }
}
