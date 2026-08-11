//! Legal-move enumeration (ADVERSARIAL-REVIEW.md F7).
//!
//! `enumerate` answers the question every bot has to ask — "what am I allowed
//! to do right now?" — so policies rank real moves instead of guessing and
//! checking `apply` errors. It generates a cheap superset of candidates and
//! lets the engine's non-cloning validator judge every one. The validator shares
//! the transition's rule predicates, while `apply` remains the final pure state
//! transition.

use crate::action::{Action, MeldTarget};
use crate::apply::validate;
use crate::card::{Card, Rank, Suit};
use crate::state::{GameState, Phase, Seat};

/// Every action `seat` may legally take right now, one ply, in deterministic
/// order. Call it with the player whose turn it is; any other seat, or a
/// finished hand or match, gets an empty list.
pub fn enumerate(state: &GameState, seat: Seat) -> Vec<Action> {
    if seat != state.turn {
        return Vec::new();
    }
    let candidates = candidate_actions(state, seat);

    let mut actions: Vec<Action> = candidates
        .into_iter()
        .filter(|action| validate(state, seat, action).is_ok())
        .collect();
    // Deterministic order matters: seed + action log must keep replaying, and
    // bots keyed on the list need a stable input. Candidates are already
    // emitted with their cards in canonical (string) order, so equivalent
    // candidates are equal values and dedup collapses them.
    actions.sort_by_key(|action| format!("{action:?}"));
    actions.dedup();
    actions
}

fn candidate_actions(state: &GameState, seat: Seat) -> Vec<Action> {
    match state.phase {
        Phase::AwaitingDraw => draw_candidates(state, seat),
        Phase::AwaitingRefusalChoice => refusal_candidates(),
        Phase::Melding => melding_candidates(state, seat),
        Phase::HandOver | Phase::MatchOver => Vec::new(),
    }
}

/// §3: the lead player's one-time choice about the card they were shown.
fn refusal_candidates() -> Vec<Action> {
    vec![Action::KeepDrawnCard, Action::RefuseDrawnCard]
}

/// §4.1 / §5: draw from the stock, or capture the pile with a natural core.
fn draw_candidates(state: &GameState, seat: Seat) -> Vec<Action> {
    let mut candidates = vec![Action::Draw];
    let meld_count = state.table(seat.team()).melds.len();
    for core in natural_pairs(state.hand(seat)) {
        candidates.push(Action::TakeDiscardPile {
            core,
            target: MeldTarget::NewMeld,
        });
        for meld in 0..meld_count {
            candidates.push(Action::TakeDiscardPile {
                core,
                target: MeldTarget::Existing { meld },
            });
        }
    }
    candidates
}

/// §5 cores: every unordered pair of natural cards held. Multiset-based — a
/// same-value pair (e.g. two A♥) is a candidate when two copies are held,
/// since a pair of aces with an ace on top is a legal capture. Wilds are
/// excluded statically; the validator filters blocked tops, frozen cores, §6
/// reachability, and invalid joins.
fn natural_pairs(hand: &[Card]) -> Vec<[Card; 2]> {
    let mut uniques: Vec<Card> = hand
        .iter()
        .copied()
        .filter(|card| card.is_natural())
        .collect();
    uniques.sort_by_key(|card| card.to_string());
    uniques.dedup();

    let mut pairs = Vec::new();
    for (i, &a) in uniques.iter().enumerate() {
        for &b in &uniques[i..] {
            if a == b && hand.iter().filter(|&&card| card == a).count() < 2 {
                continue;
            }
            pairs.push([a, b]);
        }
    }
    pairs
}

/// §4.2–§4.3: meld, lay off, discard, or (in exactly one corner) end the
/// turn holding a card that cannot legally be thrown.
fn melding_candidates(state: &GameState, seat: Seat) -> Vec<Action> {
    let hand = state.hand(seat);
    let mut candidates = lay_meld_candidates(hand);

    let mut distinct: Vec<Card> = hand.to_vec();
    distinct.sort_by_key(|card| card.to_string());
    distinct.dedup();

    for meld in 0..state.table(seat.team()).melds.len() {
        // Laying off several cards at once is recovered as successive
        // single-card adds, so single-card candidates are complete.
        for &card in &distinct {
            candidates.push(Action::AddToMeld {
                meld,
                cards: vec![card],
            });
        }
    }
    for &card in &distinct {
        candidates.push(Action::Discard { card });
    }
    candidates.push(Action::EndTurnWithoutDiscard);
    candidates
}

/// §7: every distinct meld this hand can lay — sequence windows plus ace
/// sub-multisets. Cards within each candidate are in canonical (string)
/// order so equivalent candidates dedup to one.
fn lay_meld_candidates(hand: &[Card]) -> Vec<Action> {
    let mut candidates = Vec::new();

    for suit in Suit::ALL {
        // One card per rank: a sequence covers each rank once.
        let mut held: Vec<Card> = hand
            .iter()
            .copied()
            .filter(|card| card.suit() == Some(suit))
            .filter(|card| card.rank().and_then(Rank::sequence_index).is_some())
            .collect();
        held.sort_by_key(|card| card.rank().and_then(Rank::sequence_index));
        held.dedup();

        // §8: wilds usable in this suit's sequences — the Joker anywhere, a
        // 2 only in its own suit. Each wild held yields its own candidate.
        let mut wilds: Vec<Card> = hand
            .iter()
            .copied()
            .filter(|card| {
                card.is_joker() || (card.suit() == Some(suit) && card.rank() == Some(Rank::Two))
            })
            .collect();
        wilds.sort_by_key(|card| card.to_string());
        wilds.dedup();

        // Every rank window of length ≥ 3 in 4..A (indices 0..=10).
        for start in 0..11u8 {
            for len in 3..=(11u8 - start) {
                let present: Vec<Card> = (start..start + len)
                    .filter_map(|index| {
                        held.iter()
                            .find(|card| card.rank().and_then(Rank::sequence_index) == Some(index))
                            .copied()
                    })
                    .collect();
                match len as usize - present.len() {
                    0 => candidates.push(lay(present)),
                    1 => {
                        for &wild in &wilds {
                            let mut cards = present.clone();
                            cards.push(wild);
                            candidates.push(lay(cards));
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // §7.2: every sub-multiset of natural aces held, each with and without one
    // held wild (an ace meld takes a Joker or any 2). The wild counts toward
    // the three-card minimum (`Meld::new`), so a pair of natural aces plus a
    // wild is a legal meld and must be offered. Sub-multiset, not subset: the
    // deck holds two copies of each ace and `AcesMeld` accepts duplicates, so
    // `AH AH AD` is a legal meld.
    let mut aces: Vec<Card> = hand
        .iter()
        .copied()
        .filter(|card| card.rank() == Some(Rank::Ace))
        .collect();
    aces.sort_by_key(|card| card.to_string());

    let mut wilds: Vec<Card> = hand.iter().copied().filter(|card| card.is_wild()).collect();
    wilds.sort_by_key(|card| card.to_string());
    wilds.dedup();

    for size in 2..=aces.len() {
        for combo in combinations(&aces, size) {
            // Three or more natural aces can stand alone; a backbone of two
            // aces needs a wild to clear the three-card minimum.
            if size >= 3 {
                candidates.push(lay(combo.clone()));
            }
            for &wild in &wilds {
                let mut cards = combo.clone();
                cards.push(wild);
                candidates.push(lay(cards));
            }
        }
    }

    candidates
}

/// Every `size`-card combination of `cards`, deduplicated by value (the hand
/// may hold two copies of the same card).
///
/// `cards` must arrive sorted and let indices ascend, so each emitted combination
/// is already in sorted order. Output order follows `next` ascending.
fn combinations(cards: &[Card], size: usize) -> Vec<Vec<Card>> {
    fn pick(
        cards: &[Card],
        size: usize,
        next: usize,
        stack: &mut Vec<usize>,
        seen: &mut HashSet<Vec<Card>>,
        out: &mut Vec<Vec<Card>>,
    ) {
        if stack.len() == size {
            let combo: Vec<Card> = stack.iter().map(|&i| cards[i]).collect();
            if seen.insert(combo.clone()) {
                out.push(combo);
            }
            return;
        }
        for i in next..cards.len() {
            stack.push(i);
            pick(cards, size, i + 1, stack, seen, out);
            stack.pop();
        }
    }

    use std::collections::HashSet;
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    pick(cards, size, 0, &mut Vec::new(), &mut seen, &mut out);
    out
}

/// A `LayMeld` candidate with its cards in canonical (string) order.
fn lay(mut cards: Vec<Card>) -> Action {
    cards.sort_by_key(|card| card.to_string());
    Action::LayMeld { cards }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::apply::{apply, validate};
    use crate::deal::new_game;
    use crate::score::settle_hand;

    fn mix(seed: u64, ply: u64) -> usize {
        let mut x = seed ^ ply.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (x ^ (x >> 31)) as usize
    }

    /// Compare the non-cloning path with the existing transition for every
    /// candidate the enumerator considers, including candidates that are
    /// deliberately only a cheap superset. The wider seeded run is available
    /// with `CANASTRA_VALIDATE_SEEDS=50`.
    #[test]
    fn validation_matches_apply_for_candidate_corpus() {
        let seeds: u64 = std::env::var("CANASTRA_VALIDATE_SEEDS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(2);

        for seed in 0..seeds {
            let mut state = new_game(seed);
            let mut turn_start = state.clone();
            let mut safe = false;
            for ply in 0..20_000u64 {
                if state.phase == Phase::HandOver {
                    state = settle_hand(&state).expect("settle");
                    continue;
                }
                if state.phase == Phase::MatchOver {
                    break;
                }
                if matches!(
                    state.phase,
                    Phase::AwaitingDraw | Phase::AwaitingRefusalChoice
                ) {
                    turn_start = state.clone();
                    safe = false;
                }

                let seat = state.turn;
                for action in candidate_actions(&state, seat) {
                    let expected = apply(&state, seat, &action).map(|_| ());
                    assert_eq!(
                        validate(&state, seat, &action),
                        expected,
                        "seed {seed}, ply {ply}, phase {:?}, action {action:?}",
                        state.phase,
                    );
                }

                let actions = enumerate(&state, seat);
                if actions.is_empty() {
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
                state = apply(&state, seat, &actions[pick]).expect("enumerated action applies");
            }
        }
    }
}
