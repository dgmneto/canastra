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
