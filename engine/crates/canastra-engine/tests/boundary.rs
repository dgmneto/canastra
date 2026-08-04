//! The contract other languages actually see.
//!
//! These are the shapes a TypeScript or Python caller writes by hand, so they
//! are pinned here as literals. Changing one of these means changing every
//! consumer, which is exactly what a test should make you notice.

use canastra_engine::action::MeldTarget;
use canastra_engine::state::Phase;
use canastra_engine::testkit::{Rig, card, cards};
use canastra_engine::{
    Action, GameState, PlayerView, RuleViolation, Seat, apply, new_game, observe,
};

fn seat(index: u8) -> Seat {
    Seat::new(index).expect("valid seat")
}

/// Cards travel as compact strings rather than nested objects.
#[test]
fn cards_are_plain_strings_on_the_wire() {
    assert_eq!(serde_json::to_string(&card("6D")).unwrap(), r#""6D""#);
    assert_eq!(serde_json::to_string(&card("JOKER")).unwrap(), r#""JOKER""#);
    assert_eq!(serde_json::to_string(&card("TH")).unwrap(), r#""TH""#);
    assert_eq!(
        serde_json::from_str::<canastra_engine::card::Card>(r#""AS""#).unwrap(),
        card("AS")
    );
}

#[test]
fn a_malformed_card_is_rejected_rather_than_guessed_at() {
    assert!(serde_json::from_str::<canastra_engine::card::Card>(r#""ZZ""#).is_err());
}

/// Every `Action` variant is internally tagged on `type`, which is what makes
/// the JS side a natural discriminated union.
#[test]
fn actions_arrive_as_tagged_objects() {
    let cases: [(&str, Action); 5] = [
        (r#"{"type":"Draw"}"#, Action::Draw),
        (r#"{"type":"KeepDrawnCard"}"#, Action::KeepDrawnCard),
        (
            r#"{"type":"Discard","card":"3S"}"#,
            Action::Discard { card: card("3S") },
        ),
        (
            r#"{"type":"LayMeld","cards":["6D","7D","8D"]}"#,
            Action::LayMeld {
                cards: cards("6D 7D 8D"),
            },
        ),
        (
            r#"{"type":"AddToMeld","meld":0,"cards":["9D"]}"#,
            Action::AddToMeld {
                meld: 0,
                cards: cards("9D"),
            },
        ),
    ];

    for (json, expected) in cases {
        let parsed: Action = serde_json::from_str(json).expect(json);
        assert_eq!(parsed, expected, "parsing {json}");
        let round_tripped: Action =
            serde_json::from_str(&serde_json::to_string(&expected).unwrap())
                .expect("re-parse our own output");
        assert_eq!(round_tripped, expected);
    }
}

#[test]
fn taking_the_pile_carries_its_target_as_a_tagged_object() {
    let parsed: Action = serde_json::from_str(
        r#"{"type":"TakeDiscardPile","core":["4D","5D"],"target":{"kind":"NewMeld"}}"#,
    )
    .expect("a shape TypeScript can write");
    assert_eq!(
        parsed,
        Action::TakeDiscardPile {
            core: [card("4D"), card("5D")],
            target: MeldTarget::NewMeld,
        }
    );

    let existing: Action = serde_json::from_str(
        r#"{"type":"TakeDiscardPile","core":["7D","8D"],"target":{"kind":"Existing","meld":0}}"#,
    )
    .expect("a shape TypeScript can write");
    assert!(matches!(
        existing,
        Action::TakeDiscardPile {
            target: MeldTarget::Existing { meld: 0 },
            ..
        }
    ));
}

#[test]
fn an_unknown_action_type_is_an_error_not_a_silent_default() {
    assert!(serde_json::from_str::<Action>(r#"{"type":"Teleport"}"#).is_err());
}

/// A rejected move comes back as structured data, so a client can branch on the
/// rule that fired rather than parsing a message.
#[test]
fn rule_violations_are_structured() {
    let violation = RuleViolation::OpeningMinimumNotMet {
        laid: 45,
        required: 75,
    };
    let json = serde_json::to_string(&violation).unwrap();
    assert_eq!(
        json,
        r#"{"error":"OpeningMinimumNotMet","laid":45,"required":75}"#
    );
    assert_eq!(
        serde_json::from_str::<RuleViolation>(&json).unwrap(),
        violation
    );
}

#[test]
fn a_whole_game_state_survives_a_round_trip() {
    let state = new_game(31);
    let json = serde_json::to_string(&state).unwrap();
    assert_eq!(serde_json::from_str::<GameState>(&json).unwrap(), state);
}

#[test]
fn a_player_view_survives_a_round_trip() {
    let state = new_game(31);
    let view = observe(&state, state.turn);
    let json = serde_json::to_string(&view).unwrap();
    assert_eq!(serde_json::from_str::<PlayerView>(&json).unwrap(), view);
}

/// A view is a fraction of the size of the state it came from, and — far more
/// importantly — none of the hidden cards appear anywhere in its bytes.
#[test]
fn a_serialized_view_does_not_contain_the_hidden_cards() {
    let state = new_game(31);
    let viewer = state.turn;
    let json = serde_json::to_string(&observe(&state, viewer)).unwrap();

    // The stock is face down; nothing in it should be nameable from the view.
    // Pick a card that is in the stock and in nobody's visible zone.
    let visible: Vec<_> = state.hand(viewer).iter().map(|c| c.to_string()).collect();
    let leaked = state
        .stock
        .iter()
        .map(|card| card.to_string())
        .filter(|code| !visible.contains(code))
        .filter(|code| json.contains(&format!("\"{code}\"")))
        .count();
    assert_eq!(leaked, 0, "a face-down card was nameable from the view");
}

#[test]
fn a_seat_outside_the_table_is_refused_at_the_boundary() {
    assert!(serde_json::from_str::<Seat>("4").is_err());
    assert_eq!(serde_json::from_str::<Seat>("3").unwrap(), seat(3));
}

/// The replay guarantee: a seed plus an ordered action log reproduces the game
/// exactly. This is what lets a bug report be a pair of numbers and a list.
#[test]
fn a_game_replays_from_its_seed_and_action_log() {
    let mut state = new_game(2024);
    let mut log: Vec<(Seat, Action)> = Vec::new();

    let mut record = |state: &mut GameState, action: Action| {
        let seat = state.turn;
        *state = apply(state, seat, &action).expect("scripted move should be legal");
        log.push((seat, action));
    };

    // Twelve plain turns: draw, then throw the first card back.
    for _ in 0..12 {
        assert_eq!(state.phase, Phase::AwaitingDraw);
        record(&mut state, Action::Draw);
        if state.phase == Phase::AwaitingRefusalChoice {
            record(&mut state, Action::KeepDrawnCard);
        }
        let throwaway = state.hand(state.turn)[0];
        record(&mut state, Action::Discard { card: throwaway });
    }

    let mut replayed = new_game(2024);
    for (seat, action) in &log {
        replayed = apply(&replayed, *seat, action).expect("the log replays cleanly");
    }
    assert_eq!(replayed, state);
}

/// The same seed deals the same match on every machine and every release.
///
/// This literal is the point of the test. If it ever needs regenerating, the
/// shuffle has changed and every previously recorded game — every replayed bug
/// report, every stored bot training run — no longer reproduces. That should be
/// a deliberate decision, not something noticed later.
#[test]
fn the_shuffle_is_pinned_so_recorded_games_keep_replaying() {
    let state = new_game(2024);
    let opening: Vec<String> = state
        .hand(state.turn)
        .iter()
        .map(|card| card.to_string())
        .collect();
    assert_eq!(
        opening.join(" "),
        "JS 7D QC 6H 8D AS 7D QC TH JC 9H 6C 9H 8C 4D"
    );
}

/// A rigged position serializes too, which is how a server persists a game in
/// progress and how a bug report travels.
#[test]
fn a_rigged_position_survives_persistence() {
    let state = Rig::new()
        .hand(1, "4D 5D JOKER")
        .discard("9C 6D")
        .meld(1, "5S 6S 7S 8S 9S TS JS")
        .red_threes(1, "3H")
        .scores(1200, 3400)
        .turn(1)
        .build();

    let restored: GameState =
        serde_json::from_str(&serde_json::to_string(&state).unwrap()).unwrap();
    assert_eq!(restored, state);

    // And it is still playable after the round trip.
    let taken = apply(
        &restored,
        seat(1),
        &Action::TakeDiscardPile {
            core: [card("4D"), card("5D")],
            target: MeldTarget::NewMeld,
        },
    )
    .expect("the restored position behaves identically");
    assert_eq!(taken.phase, Phase::Melding);
}

// ---- validated deserialization (adversarial review F2, F3) ----

use canastra_engine::meld::Meld;

/// F2: serde reconstructs a struct field by field, which walks straight past
/// every check the constructors make. These payloads all parsed before the fix,
/// and reading the resulting meld panicked.
#[test]
fn a_malformed_sequence_is_refused_at_the_boundary() {
    let bad = [
        // No slots at all, and a `low` far outside 4..=A. Sequence::low() used
        // to panic on this one.
        r#"{"kind":"Sequence","meld":{"suit":"Hearts","low":250,"slots":[]}}"#,
        // A natural card that is not the rank its position claims it is.
        r#"{"kind":"Sequence","meld":{"suit":"Hearts","low":0,"slots":[
            {"kind":"Natural","card":"KH"},{"kind":"Natural","card":"5H"},
            {"kind":"Natural","card":"6H"}]}}"#,
        // Two wild cards, which §8 forbids.
        r#"{"kind":"Sequence","meld":{"suit":"Hearts","low":0,"slots":[
            {"kind":"Wild","card":"JOKER"},{"kind":"Wild","card":"2H"},
            {"kind":"Natural","card":"6H"}]}}"#,
        // A natural of the wrong suit.
        r#"{"kind":"Sequence","meld":{"suit":"Hearts","low":0,"slots":[
            {"kind":"Natural","card":"4S"},{"kind":"Natural","card":"5H"},
            {"kind":"Natural","card":"6H"}]}}"#,
        // §8: a 2 standing in for a rank of another suit.
        r#"{"kind":"Sequence","meld":{"suit":"Hearts","low":0,"slots":[
            {"kind":"Natural","card":"4H"},{"kind":"Natural","card":"5H"},
            {"kind":"Wild","card":"2S"}]}}"#,
        // Shorter than a meld can be.
        r#"{"kind":"Sequence","meld":{"suit":"Hearts","low":0,"slots":[
            {"kind":"Natural","card":"4H"},{"kind":"Natural","card":"5H"}]}}"#,
        // Runs off the top of the Ace.
        r#"{"kind":"Sequence","meld":{"suit":"Hearts","low":9,"slots":[
            {"kind":"Natural","card":"KH"},{"kind":"Natural","card":"AH"},
            {"kind":"Wild","card":"JOKER"}]}}"#,
    ];
    for json in bad {
        assert!(
            serde_json::from_str::<Meld>(json).is_err(),
            "should have been refused: {json}"
        );
    }
}

#[test]
fn a_malformed_ace_meld_is_refused_at_the_boundary() {
    let bad = [
        r#"{"kind":"Aces","meld":{"aces":["KH","AD","AS"],"wild":null}}"#,
        r#"{"kind":"Aces","meld":{"aces":["AH","AD"],"wild":"7C"}}"#,
        r#"{"kind":"Aces","meld":{"aces":["AH"],"wild":null}}"#,
    ];
    for json in bad {
        assert!(
            serde_json::from_str::<Meld>(json).is_err(),
            "should have been refused: {json}"
        );
    }
}

/// The validation must not cost legitimate melds their round trip.
#[test]
fn every_kind_of_real_meld_still_round_trips() {
    for spec in [
        "6H 7H 8H",
        "6H 7H 2H",
        "JOKER QS KS AS",
        "4H 5H 6H 7H 8H 9H TH JH QH KH AH",
        "AH AD AS",
        "AH AD AS AC AH AD AS JOKER",
    ] {
        let meld = Meld::new(&cards(spec)).expect(spec);
        let json = serde_json::to_string(&meld).unwrap();
        assert_eq!(serde_json::from_str::<Meld>(&json).unwrap(), meld, "{spec}");
    }
}

/// F3: a whole state can be tampered with even when every meld inside it is
/// individually valid, so conservation has to be checked separately.
#[test]
fn a_dealt_game_satisfies_its_invariants() {
    for seed in 0..20 {
        new_game(seed)
            .check_invariants()
            .unwrap_or_else(|e| panic!("seed {seed}: {e}"));
    }
}

#[test]
fn a_state_that_invented_cards_is_caught() {
    let mut raw = serde_json::to_value(new_game(5)).unwrap();
    raw["hands"][1] = serde_json::json!(["6H", "6H", "6H", "6H", "6H", "6H", "6H", "6H"]);
    let tampered: GameState = serde_json::from_value(raw).expect("shape is still valid");
    assert!(
        tampered.check_invariants().is_err(),
        "eight copies of one card"
    );
}

/// §12: a red 3 goes to the table on sight, so it can never be in a hand or in
/// the pile. A state claiming otherwise did not come from play.
#[test]
fn a_red_three_where_it_cannot_be_is_caught() {
    let mut raw = serde_json::to_value(new_game(5)).unwrap();
    raw["discard"] = serde_json::json!(["3H"]);
    let tampered: GameState = serde_json::from_value(raw).expect("shape is still valid");
    assert!(tampered.check_invariants().is_err());
}

#[test]
fn a_frozen_card_the_player_does_not_hold_is_caught() {
    let mut raw = serde_json::to_value(new_game(5)).unwrap();
    raw["turn_context"]["frozen"] = serde_json::json!(["3H"]);
    let tampered: GameState = serde_json::from_value(raw).expect("shape is still valid");
    assert!(tampered.check_invariants().is_err());
}

/// Invariants have to survive real play, not just the deal.
#[test]
fn invariants_hold_all_the_way_through_a_played_hand() {
    let mut state = new_game(77);
    for _ in 0..30 {
        if state.phase == Phase::HandOver || state.phase == Phase::MatchOver {
            break;
        }
        let seat = state.turn;
        state = apply(&state, seat, &Action::Draw).unwrap();
        if state.phase == Phase::AwaitingRefusalChoice {
            state = apply(&state, seat, &Action::KeepDrawnCard).unwrap();
        }
        let throwaway = state.hand(seat)[0];
        state = apply(&state, seat, &Action::Discard { card: throwaway }).unwrap();
        state.check_invariants().expect("still a legal position");
    }
}

// ---- F1: the position with no legal discard ----

/// CLAUDE.md clarification #6. A player holding one card draws a red 3 as the
/// last card of the stock, so §12's replacement never arrives and the hand is
/// left at one card. With no clean canastra they may not empty their hand, and
/// they may not keep it either — before the fix there was no legal move at all.
#[test]
fn a_player_with_no_legal_discard_keeps_the_card_and_the_hand_ends() {
    let state = Rig::new()
        .hand(1, "4H")
        .meld(1, "5S 6S 7S 8S 9S TS 2S") // dirty canastra only
        .stock("3H")
        .turn(1)
        .build();
    let drawn = apply(&state, seat(1), &Action::Draw).unwrap();
    assert_eq!(drawn.hand(seat(1)).len(), 1, "the replacement never came");
    assert!(drawn.stock.is_empty());

    assert_eq!(
        apply(&drawn, seat(1), &Action::Discard { card: card("4H") }),
        Err(RuleViolation::NoCleanCanastra)
    );

    let done = apply(&drawn, seat(1), &Action::EndTurnWithoutDiscard).unwrap();
    assert_eq!(done.phase, Phase::HandOver);
    assert_eq!(
        done.went_out, None,
        "nobody went out, so nobody gets the bonus"
    );
    assert_eq!(
        done.hand(seat(1)),
        cards("4H"),
        "the card stays in hand and scores against them"
    );
}

#[test]
fn the_discard_cannot_be_skipped_while_the_stock_still_has_cards() {
    let state = Rig::new()
        .hand(1, "4H")
        .meld(1, "5S 6S 7S 8S 9S TS 2S")
        .stock("KC QC")
        .phase(Phase::Melding)
        .turn(1)
        .build();
    assert_eq!(
        apply(&state, seat(1), &Action::EndTurnWithoutDiscard),
        Err(RuleViolation::MustDiscard)
    );
}

/// A player who *can* legally discard has to.
#[test]
fn the_discard_cannot_be_skipped_when_a_legal_one_exists() {
    let with_spare = Rig::new()
        .hand(1, "4H 5H")
        .meld(1, "5S 6S 7S 8S 9S TS 2S")
        .phase(Phase::Melding)
        .turn(1)
        .build();
    assert_eq!(
        apply(&with_spare, seat(1), &Action::EndTurnWithoutDiscard),
        Err(RuleViolation::MustDiscard)
    );

    // One card, but a clean canastra behind it: discarding is going out, which
    // is legal and pays 100, so skipping is not on offer.
    let can_go_out = Rig::new()
        .hand(1, "4H")
        .meld(1, "5S 6S 7S 8S 9S TS JS")
        .phase(Phase::Melding)
        .turn(1)
        .build();
    assert_eq!(
        apply(&can_go_out, seat(1), &Action::EndTurnWithoutDiscard),
        Err(RuleViolation::MustDiscard)
    );
}
