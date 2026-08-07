//! What one player is allowed to know.

use crate::card::Card;
use crate::state::{GameState, Phase, Seat, TeamTable};
use serde::{Deserialize, Serialize};

/// The game as a single seat sees it.
///
/// This, not [`GameState`], is what belongs on the wire. `GameState` is
/// omniscient — it holds every hand and the order of the stock — so handing it
/// to a client would leak the whole game. The redaction is structural rather
/// than a matter of remembering to strip fields: there is simply nowhere here to
/// put another player's cards or the stock's order.
///
/// Bots train against this for the same reason: a policy that could see the
/// stock would learn nothing transferable to a real table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlayerView {
    /// Whose view this is.
    pub seat: Seat,
    pub hand: Vec<Card>,
    /// §5: which of the cards in hand came out of the pile this turn and so
    /// cannot be melded yet.
    pub frozen: Vec<Card>,
    /// Both partnerships' melds and red 3s — all of it is face up.
    pub tables: [TeamTable; 2],
    /// §5: the whole pile. Every card in it was discarded face up in front of
    /// all four players, and a player deciding whether to take it needs to know
    /// what they would be picking up.
    pub discard: Vec<Card>,
    /// §16: how many cards remain, which is public and worth counting. Their
    /// order is not.
    pub stock_count: usize,
    /// How many cards each seat holds, in seat order.
    pub hand_counts: [usize; 4],
    pub scores: [i32; 2],
    pub phase: Phase,
    pub turn: Seat,
    pub dealer: Seat,
    pub hand_number: u32,
    pub went_out: Option<Seat>,
    /// §6: what this player's partnership needs to open, already resolved
    /// against their score so clients do not each re-derive it.
    pub opening_minimum: u32,
    /// How much the current turn has laid so far, toward §6's minimum.
    pub laid_value: u32,
    /// §5: whether the current turn began by capturing the discard pile.
    pub took_pile: bool,
    /// §3: whether the current turn's player still holds the first-turn refusal.
    pub refusal_available: bool,
    /// §3: the card the lead player has been shown and not yet kept or refused.
    /// Only ever populated for the player making that choice.
    pub pending_refusal: Option<Card>,
}

/// Redact `state` down to what `seat` may legitimately see.
pub fn observe(state: &GameState, seat: Seat) -> PlayerView {
    let is_current = seat == state.turn;

    PlayerView {
        seat,
        hand: state.hand(seat).to_vec(),
        // The frozen set belongs to the turn in progress, so it means nothing
        // to anyone but the player taking that turn.
        frozen: if is_current {
            state.turn_context.frozen.clone()
        } else {
            Vec::new()
        },
        tables: state.tables.clone(),
        discard: state.discard.clone(),
        stock_count: state.stock.len(),
        hand_counts: std::array::from_fn(|index| state.hands[index].len()),
        scores: state.scores,
        phase: state.phase,
        turn: state.turn,
        dealer: state.dealer,
        hand_number: state.hand_number,
        went_out: state.went_out,
        opening_minimum: state.opening_minimum_for(seat.team()),
        // Unlike the frozen set and the pending card, these describe the turn
        // in progress, which is public at a real table: everyone watches the
        // acting player lay melds and take the pile. Populated for every seat.
        laid_value: state.turn_context.laid_value,
        took_pile: state.turn_context.took_pile,
        refusal_available: state.turn_context.refusal_available,
        pending_refusal: if is_current {
            state.turn_context.pending_refusal
        } else {
            None
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deal::new_game;
    use crate::testkit::{Rig, cards};

    fn seat(index: u8) -> Seat {
        Seat::new(index).expect("valid seat")
    }

    #[test]
    fn a_player_sees_their_own_hand() {
        let state = Rig::new().hand(1, "6H 7H 8H").build();
        assert_eq!(observe(&state, seat(1)).hand, cards("6H 7H 8H"));
    }

    /// The whole point of the view: it carries no field that could hold another
    /// player's cards, so a client cannot be handed them by accident.
    #[test]
    fn a_player_sees_only_how_many_cards_the_others_hold() {
        let state = Rig::new()
            .hand(0, "6H 7H")
            .hand(1, "8H")
            .hand(2, "9H TH JH")
            .hand(3, "QH QS QD QC")
            .build();
        let view = observe(&state, seat(1));
        assert_eq!(view.hand_counts, [2, 1, 3, 4]);
        assert_eq!(view.hand, cards("8H"));
    }

    /// §4.1: how many cards are left matters — §16 calls counting the stock a
    /// real skill — but their order is secret.
    #[test]
    fn the_stock_is_a_count_and_not_a_list() {
        let state = new_game(7);
        let view = observe(&state, state.turn);
        assert_eq!(view.stock_count, state.stock.len());
    }

    /// §5: every card in the pile was played face up, so the pile is public.
    /// A player deciding whether to take it needs to know what is in there.
    #[test]
    fn the_whole_discard_pile_is_public() {
        let state = Rig::new().discard("9C TC 6D").build();
        assert_eq!(observe(&state, seat(1)).discard, cards("9C TC 6D"));
    }

    #[test]
    fn both_tables_are_public() {
        let state = Rig::new()
            .meld(0, "6H 7H 8H")
            .meld(1, "9S TS JS")
            .red_threes(0, "3H")
            .build();
        let view = observe(&state, seat(1));
        assert_eq!(view.tables[0].melds.len(), 1);
        assert_eq!(view.tables[1].melds.len(), 1);
        assert_eq!(view.tables[0].red_threes, cards("3H"));
    }

    /// §3: the card the lead player is deciding on is theirs alone to see.
    #[test]
    fn the_card_on_offer_is_shown_only_to_the_player_deciding() {
        let state = new_game(7);
        let lead = state.turn;
        let drawn = crate::apply(&state, lead, &crate::Action::Draw).unwrap();
        assert_eq!(drawn.phase, Phase::AwaitingRefusalChoice);

        assert!(observe(&drawn, lead).pending_refusal.is_some());
        assert_eq!(observe(&drawn, lead.next()).pending_refusal, None);
    }

    /// §5: a player who took the pile needs to know which of their cards are
    /// frozen, since it changes what they can legally do this turn.
    #[test]
    fn frozen_cards_are_flagged_to_their_owner() {
        let state = Rig::new().hand(1, "6H 7H").frozen("7H").turn(1).build();
        assert_eq!(observe(&state, seat(1)).frozen, cards("7H"));
        assert!(observe(&state, seat(2)).frozen.is_empty());
    }

    /// §6: the bar depends on the viewer's own partnership score, so the view
    /// answers it directly rather than making every client re-derive it.
    #[test]
    fn the_view_carries_the_opening_minimum_for_that_player() {
        let state = Rig::new().scores(0, 2500).build();
        assert_eq!(observe(&state, seat(0)).opening_minimum, 75);
        assert_eq!(observe(&state, seat(1)).opening_minimum, 120);
    }

    /// The current turn's progress is public at a real table — everyone
    /// watches the acting player lay melds and take the pile — so the view
    /// publishes it to every seat, not just the one taking the turn.
    #[test]
    fn the_view_carries_the_current_turns_public_progress() {
        let state = Rig::new()
            .hand(1, "6H 7H 8H 9H")
            .frozen("9H")
            .laid_value(45)
            .refusal_available()
            .build();
        let view = observe(&state, seat(2));
        assert_eq!(view.laid_value, 45);
        assert!(view.took_pile);
        assert!(view.refusal_available);
    }

    #[test]
    fn the_view_carries_the_public_run_of_play() {
        let state = Rig::new().scores(300, 400).turn(2).dealer(1).build();
        let view = observe(&state, seat(0));
        assert_eq!(view.seat, seat(0));
        assert_eq!(view.turn, seat(2));
        assert_eq!(view.dealer, seat(1));
        assert_eq!(view.scores, [300, 400]);
        assert_eq!(view.phase, Phase::AwaitingDraw);
    }
}
