//! Shuffling and dealing (§2, §12).

use crate::card::{Card, deck};
use crate::state::{GameState, HAND_SIZE, Phase, Seat, TurnContext};
use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};

/// Shuffle deterministically from a seed.
///
/// Fisher-Yates is spelled out rather than taken from a helper crate on purpose.
/// `seed -> deal` is part of this engine's contract — a whole game replays from a
/// seed plus its action log — and no shuffling helper promises a stable algorithm
/// across releases, so depending on one would let a routine upgrade silently
/// invalidate every recorded game. ChaCha8 is a frozen specification, so this
/// pairing stays reproducible. (The modulo is very slightly biased; with 108
/// cards against a 64-bit draw the bias is far below anything observable.)
fn shuffled(seed: u64) -> Vec<Card> {
    let mut cards = deck();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for index in (1..cards.len()).rev() {
        let target = (rng.next_u64() % (index as u64 + 1)) as usize;
        cards.swap(index, target);
    }
    cards
}

/// Start a fresh match: seat 0 deals the first hand.
pub fn new_game(seed: u64) -> GameState {
    deal_hand(shuffled(seed), Seat::ALL[0], [0, 0], 1)
}

/// Deal one hand from `stock`.
///
/// §2: fifteen cards each, dealt round-robin starting to the dealer's right, and
/// the discard pile starts empty. §12: any red 3 among the opening fifteen goes
/// straight to its partnership's table and pulls a replacement.
pub fn deal_hand(
    mut stock: Vec<Card>,
    dealer: Seat,
    scores: [i32; 2],
    hand_number: u32,
) -> GameState {
    let mut hands: [Vec<Card>; 4] = Default::default();
    let mut seat = dealer.next();
    for _ in 0..HAND_SIZE {
        for _ in 0..Seat::ALL.len() {
            hands[seat.index()].push(stock.pop().expect("108 cards cover four hands"));
            seat = seat.next();
        }
    }

    let mut state = GameState {
        stock,
        discard: Vec::new(),
        hands,
        tables: Default::default(),
        scores,
        dealer,
        // §2: "começa o jogador à direita do carteador".
        turn: dealer.next(),
        phase: Phase::AwaitingDraw,
        turn_context: TurnContext {
            // §3: only the lead player of the hand gets the refusal.
            refusal_available: true,
            ..TurnContext::default()
        },
        hand_number,
    };

    for seat in Seat::ALL {
        resolve_red_threes(&mut state, seat);
    }
    state
}

/// §12: move every red 3 out of `seat`'s hand onto its partnership's table,
/// drawing a replacement for each.
///
/// This drains rather than passing once, because a replacement card can itself
/// be a red 3.
pub fn resolve_red_threes(state: &mut GameState, seat: Seat) {
    while let Some(position) = state.hands[seat.index()]
        .iter()
        .position(|card| card.is_red_three())
    {
        let three = state.hands[seat.index()].remove(position);
        state.tables[seat.team().index()].red_threes.push(three);
        let Some(replacement) = state.stock.pop() else {
            break;
        };
        state.hands[seat.index()].push(replacement);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every card in play, wherever it currently sits.
    fn all_cards(state: &GameState) -> Vec<Card> {
        let mut cards: Vec<Card> = state.stock.clone();
        cards.extend(state.discard.iter().copied());
        for hand in &state.hands {
            cards.extend(hand.iter().copied());
        }
        for table in &state.tables {
            cards.extend(table.red_threes.iter().copied());
            for meld in &table.melds {
                cards.extend(meld.cards());
            }
        }
        cards
    }

    /// §2: 15 cards each, the rest face down as the stock.
    #[test]
    fn every_player_starts_with_fifteen_cards() {
        let state = new_game(7);
        for hand in &state.hands {
            assert_eq!(hand.len(), HAND_SIZE);
        }
    }

    /// §2: the discard pile starts empty, so nobody can take it on turn one.
    #[test]
    fn the_discard_pile_starts_empty() {
        assert!(new_game(7).discard.is_empty());
    }

    #[test]
    fn dealing_conserves_the_whole_deck() {
        for seed in 0..25 {
            let state = new_game(seed);
            let mut dealt = all_cards(&state);
            let mut expected = crate::card::deck();
            dealt.sort_by_key(|card| card.to_string());
            expected.sort_by_key(|card| card.to_string());
            assert_eq!(dealt, expected, "seed {seed} lost or invented a card");
        }
    }

    /// §12: a red 3 never stays in a hand — not even one dealt in the opening
    /// fifteen — and its owner draws a replacement for it.
    #[test]
    fn red_threes_never_remain_in_a_hand_after_the_deal() {
        for seed in 0..50 {
            let state = new_game(seed);
            for (index, hand) in state.hands.iter().enumerate() {
                assert!(
                    !hand.iter().any(|card| card.is_red_three()),
                    "seed {seed}: seat {index} kept a red 3"
                );
                assert_eq!(hand.len(), HAND_SIZE, "seed {seed}: seat {index} short");
            }
        }
    }

    #[test]
    fn red_threes_dealt_out_land_on_their_partnership_table() {
        // Across many seeds at least one red 3 is certain to be dealt.
        let tabled: usize = (0..50)
            .map(|seed| {
                let state = new_game(seed);
                state
                    .tables
                    .iter()
                    .map(|table| table.red_threes.len())
                    .sum::<usize>()
            })
            .sum();
        assert!(tabled > 0, "no red 3 was ever dealt across 50 seeds");

        let state = new_game(0);
        for table in &state.tables {
            assert!(table.red_threes.iter().all(|card| card.is_red_three()));
        }
    }

    /// §12: a red 3 is not a meld, so it never opens the partnership.
    #[test]
    fn a_red_three_does_not_open_the_partnership() {
        let state = new_game(0);
        for table in &state.tables {
            assert!(!table.opened);
            assert!(table.melds.is_empty());
        }
    }

    /// §2: "começa o jogador à direita do carteador".
    #[test]
    fn the_player_to_the_dealers_right_leads() {
        let state = new_game(7);
        assert_eq!(state.turn, state.dealer.next());
        assert_eq!(state.phase, Phase::AwaitingDraw);
    }

    /// §3: the lead player, and only the lead player, may refuse their first card.
    #[test]
    fn the_lead_player_starts_with_the_refusal_available() {
        assert!(new_game(7).turn_context.refusal_available);
    }

    #[test]
    fn both_partnerships_start_at_zero() {
        assert_eq!(new_game(7).scores, [0, 0]);
    }

    #[test]
    fn the_same_seed_always_deals_the_same_game() {
        assert_eq!(new_game(42), new_game(42));
    }

    #[test]
    fn different_seeds_deal_different_games() {
        assert_ne!(new_game(1), new_game(2));
    }

    /// The stock holds whatever the deal did not: 108 less four hands, less any
    /// red 3 that went to a table and pulled a replacement out of the stock.
    #[test]
    fn the_stock_holds_the_remainder_of_the_deck() {
        let state = new_game(7);
        let tabled: usize = state
            .tables
            .iter()
            .map(|table| table.red_threes.len())
            .sum();
        assert_eq!(
            state.stock.len(),
            108 - Seat::ALL.len() * HAND_SIZE - tabled
        );
    }
}
