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
use std::cmp::Ordering;

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
    // bots keyed on the list need a stable input. This key mirrors the old
    // derived-Debug lexicographic order without formatting an owned String.
    actions.sort_unstable_by(|left, right| action_sort_key(left).cmp(&action_sort_key(right)));
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
    uniques.sort_unstable_by_key(|&card| card_string_key(card));
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
    distinct.sort_unstable_by_key(|&card| card_string_key(card));
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
        wilds.sort_unstable_by_key(|&card| card_string_key(card));
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
    aces.sort_unstable_by_key(|&card| card_string_key(card));

    let mut wilds: Vec<Card> = hand.iter().copied().filter(|card| card.is_wild()).collect();
    wilds.sort_unstable_by_key(|&card| card_string_key(card));
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
    cards.sort_unstable_by_key(|&card| card_string_key(card));
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
/// The fixed-width key for the card's wire/display string (`"6D"`, `"JOKER"`).
/// Zeroes are the string terminator for the two-byte standard-card form, so the
/// array's ordinary lexicographic order is the same as `Card::to_string()`.
fn card_string_key(card: Card) -> [u8; 5] {
    match card {
        Card::Joker => *b"JOKER",
        Card::Standard { rank, suit } => [rank.code() as u8, suit_code(suit), 0, 0, 0],
    }
}

fn suit_code(suit: Suit) -> u8 {
    match suit {
        Suit::Clubs => b'C',
        Suit::Diamonds => b'D',
        Suit::Hearts => b'H',
        Suit::Spades => b'S',
    }
}

/// A borrowed action key that compares exactly like the old derived `Debug`
/// string, but does not allocate while sorting.
struct ActionSortKey<'a>(&'a Action);

fn action_sort_key(action: &Action) -> ActionSortKey<'_> {
    ActionSortKey(action)
}

impl Ord for ActionSortKey<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        let kind = action_debug_kind(self.0).cmp(&action_debug_kind(other.0));
        if kind != Ordering::Equal {
            return kind;
        }

        match (self.0, other.0) {
            (
                Action::TakeDiscardPile {
                    core: left_core,
                    target: left_target,
                },
                Action::TakeDiscardPile {
                    core: right_core,
                    target: right_target,
                },
            ) => cmp_debug_cards(left_core, right_core)
                .then_with(|| cmp_debug_targets(left_target, right_target)),
            (Action::LayMeld { cards: left }, Action::LayMeld { cards: right }) => {
                cmp_debug_cards(left, right)
            }
            (
                Action::AddToMeld {
                    meld: left_meld,
                    cards: left_cards,
                },
                Action::AddToMeld {
                    meld: right_meld,
                    cards: right_cards,
                },
            ) => cmp_decimal(*left_meld, *right_meld)
                .then_with(|| cmp_debug_cards(left_cards, right_cards)),
            (Action::Discard { card: left }, Action::Discard { card: right }) => {
                card_debug_key(*left).cmp(&card_debug_key(*right))
            }
            _ => Ordering::Equal,
        }
    }
}

impl PartialOrd for ActionSortKey<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ActionSortKey<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for ActionSortKey<'_> {}

/// Lexical order of the action variant names emitted by derived `Debug`.
fn action_debug_kind(action: &Action) -> u8 {
    match action {
        Action::AddToMeld { .. } => 0,       // AddToMeld
        Action::Discard { .. } => 1,         // Discard
        Action::Draw => 2,                   // Draw
        Action::EndTurnWithoutDiscard => 3,  // EndTurnWithoutDiscard
        Action::KeepDrawnCard => 4,          // KeepDrawnCard
        Action::LayMeld { .. } => 5,         // LayMeld
        Action::RefuseDrawnCard => 6,        // RefuseDrawnCard
        Action::TakeDiscardPile { .. } => 7, // TakeDiscardPile
    }
}

/// Derived `Debug` prints lists with `]` after a shared prefix and `,` when the
/// other list continues, so a longer list sorts before its exact prefix.
fn cmp_debug_cards(left: &[Card], right: &[Card]) -> Ordering {
    for (&left, &right) in left.iter().zip(right) {
        let ordering = card_debug_key(left).cmp(&card_debug_key(right));
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    right.len().cmp(&left.len())
}

/// Lexical order of a `usize` as rendered by derived `Debug` (no leading zero).
fn cmp_decimal(mut left: usize, mut right: usize) -> Ordering {
    let mut left_divisor = decimal_factor(left);
    let mut right_divisor = decimal_factor(right);
    loop {
        let ordering = (left / left_divisor).cmp(&(right / right_divisor));
        if ordering != Ordering::Equal {
            return ordering;
        }
        if left_divisor == 1 || right_divisor == 1 {
            // A shorter string sorts first when the shared digits are a prefix.
            return left_divisor.cmp(&right_divisor);
        }
        left %= left_divisor;
        right %= right_divisor;
        left_divisor /= 10;
        right_divisor /= 10;
    }
}

fn decimal_factor(mut value: usize) -> usize {
    let mut factor = 1;
    while value >= 10 {
        value /= 10;
        factor *= 10;
    }
    factor
}

fn cmp_debug_targets(left: &MeldTarget, right: &MeldTarget) -> Ordering {
    match (left, right) {
        (MeldTarget::Existing { meld: left }, MeldTarget::Existing { meld: right }) => {
            cmp_decimal(*left, *right)
        }
        (MeldTarget::Existing { .. }, MeldTarget::NewMeld) => Ordering::Less,
        (MeldTarget::NewMeld, MeldTarget::Existing { .. }) => Ordering::Greater,
        (MeldTarget::NewMeld, MeldTarget::NewMeld) => Ordering::Equal,
    }
}

/// Lexical order of the derived `Debug` output for `Card`, `Rank`, and `Suit`.
fn card_debug_key(card: Card) -> (u8, u8, u8) {
    match card {
        Card::Joker => (0, 0, 0), // Joker
        Card::Standard { rank, suit } => (1, rank_debug_key(rank), suit_debug_key(suit)),
    }
}

fn rank_debug_key(rank: Rank) -> u8 {
    match rank {
        Rank::Ace => 0,   // Ace
        Rank::Eight => 1, // Eight
        Rank::Five => 2,  // Five
        Rank::Four => 3,  // Four
        Rank::Jack => 4,  // Jack
        Rank::King => 5,  // King
        Rank::Nine => 6,  // Nine
        Rank::Queen => 7, // Queen
        Rank::Seven => 8, // Seven
        Rank::Six => 9,   // Six
        Rank::Ten => 10,  // Ten
        Rank::Three => 11,
        Rank::Two => 12,
    }
}

fn suit_debug_key(suit: Suit) -> u8 {
    match suit {
        Suit::Clubs => 0,
        Suit::Diamonds => 1,
        Suit::Hearts => 2,
        Suit::Spades => 3,
    }
}

#[cfg(test)]
mod sort_tests {
    use super::*;

    #[test]
    fn allocation_free_keys_match_the_legacy_sort_orders() {
        let mut cards: Vec<Card> = Suit::ALL
            .into_iter()
            .flat_map(|suit| {
                Rank::ALL
                    .into_iter()
                    .map(move |rank| Card::Standard { rank, suit })
            })
            .collect();
        cards.push(Card::Joker);

        let mut legacy_cards = cards.clone();
        legacy_cards.sort_by_key(|card| card.to_string());
        cards.sort_unstable_by_key(|&card| card_string_key(card));
        assert_eq!(cards, legacy_cards);

        let mut actions = vec![
            Action::Draw,
            Action::KeepDrawnCard,
            Action::RefuseDrawnCard,
            Action::EndTurnWithoutDiscard,
            Action::LayMeld { cards: Vec::new() },
        ];
        actions.extend(cards.iter().copied().map(|card| Action::Discard { card }));
        for &meld in &[0, 1, 9, 10, 11, 99] {
            actions.extend(cards.iter().copied().map(|card| Action::AddToMeld {
                meld,
                cards: vec![card],
            }));
        }
        actions.extend([
            Action::LayMeld {
                cards: vec![cards[0], cards[1]],
            },
            Action::LayMeld {
                cards: vec![cards[0], cards[1], cards[2]],
            },
            Action::TakeDiscardPile {
                core: [cards[0], cards[1]],
                target: MeldTarget::NewMeld,
            },
            Action::TakeDiscardPile {
                core: [cards[0], cards[1]],
                target: MeldTarget::Existing { meld: 10 },
            },
        ]);

        for left in &actions {
            for right in &actions {
                let expected = format!("{left:?}").cmp(&format!("{right:?}"));
                let actual = action_sort_key(left).cmp(&action_sort_key(right));
                assert_eq!(actual, expected, "{left:?} vs {right:?}");
            }
        }
    }
}
