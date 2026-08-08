//! Game state: seats, partnerships, the table, and the turn machine.

use crate::card::Card;
use crate::meld::Meld;
use serde::{Deserialize, Serialize};
use std::fmt;

/// §2: fifteen cards per player.
pub const HAND_SIZE: usize = 15;

/// §14: the first partnership to reach this wins the match.
pub const TARGET_SCORE: i32 = 5000;

/// §6: the cumulative score at which a partnership's opening minimum rises.
pub const OPENING_THRESHOLD: i32 = 2500;

/// §6: opening minimum below [`OPENING_THRESHOLD`].
pub const OPENING_MINIMUM_LOW: u32 = 75;

/// §6: opening minimum at or above [`OPENING_THRESHOLD`].
pub const OPENING_MINIMUM_HIGH: u32 = 120;

/// §6.1: the bar for a penalized partnership already on [`OPENING_MINIMUM_HIGH`].
pub const OPENING_MINIMUM_PENALIZED: u32 = 240;

/// §13.1: bonus for going out.
pub const GOING_OUT_BONUS: i32 = 100;

/// §12: what each red 3 is worth, positive or negative.
pub const RED_THREE_VALUE: i32 = 100;

/// §13.3: a flat penalty for a partnership that never opened in a hand.
pub const UNOPENED_PENALTY: i32 = 300;

/// §6: what a partnership on `score` must lay to open.
///
/// The rising bar is a catch-up mechanic — the trailing partnership opens on 75
/// while the leader needs 120.
pub fn opening_minimum(score: i32) -> u32 {
    if score >= OPENING_THRESHOLD {
        OPENING_MINIMUM_HIGH
    } else {
        OPENING_MINIMUM_LOW
    }
}

/// One of the four places at the table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(into = "u8", try_from = "u8")]
pub struct Seat(u8);

impl Seat {
    pub const ALL: [Seat; 4] = [Seat(0), Seat(1), Seat(2), Seat(3)];

    pub fn new(index: u8) -> Option<Seat> {
        (index < Seat::ALL.len() as u8).then_some(Seat(index))
    }

    pub fn index(self) -> usize {
        self.0 as usize
    }

    /// §2: "o jogo segue nessa mesma direção" — play passes to the right.
    pub fn next(self) -> Seat {
        Seat((self.0 + 1) % Seat::ALL.len() as u8)
    }

    /// §2: partners sit facing each other, so alternate seats share a table.
    /// This is also why the player after you is always an opponent.
    pub fn team(self) -> Team {
        Team(self.0 % Team::ALL.len() as u8)
    }
}

/// Seats cross the wire as plain numbers, but come back through [`Seat::new`],
/// so a payload naming seat 9 is rejected at the boundary rather than panicking
/// on an array index later.
impl From<Seat> for u8 {
    fn from(seat: Seat) -> u8 {
        seat.0
    }
}

impl TryFrom<u8> for Seat {
    type Error = &'static str;

    fn try_from(index: u8) -> Result<Seat, Self::Error> {
        Seat::new(index).ok_or("seat must be 0, 1, 2 or 3")
    }
}

/// One of the two partnerships.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(into = "u8", try_from = "u8")]
pub struct Team(u8);

impl Team {
    pub const ALL: [Team; 2] = [Team(0), Team(1)];

    pub fn new(index: u8) -> Option<Team> {
        (index < Team::ALL.len() as u8).then_some(Team(index))
    }

    pub fn index(self) -> usize {
        self.0 as usize
    }

    pub fn opponent(self) -> Team {
        Team(1 - self.0)
    }
}

impl From<Team> for u8 {
    fn from(team: Team) -> u8 {
        team.0
    }
}

impl TryFrom<u8> for Team {
    type Error = &'static str;

    fn try_from(index: u8) -> Result<Team, Self::Error> {
        Team::new(index).ok_or("team must be 0 or 1")
    }
}

/// Everything a partnership has in front of it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TeamTable {
    pub melds: Vec<Meld>,
    /// §12: red 3s sit in their own place, outside every meld. They are not a
    /// meld, do not open the partnership, and carry no card value — their only
    /// effect is the ±100 at the end of the hand.
    pub red_threes: Vec<Card>,
    /// §6: whether the partnership has already met its opening minimum.
    pub opened: bool,
}

/// Where the current turn stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    /// §4.1: draw from the stock, or take the whole discard pile.
    AwaitingDraw,
    /// §3: the lead player has seen their first card and may refuse it once.
    AwaitingRefusalChoice,
    /// §4.2–4.3: lay down, lay off, and eventually discard.
    Melding,
    HandOver,
    MatchOver,
}

/// State that lives for exactly one turn and resets at the next.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnContext {
    /// §6: card value of everything laid this turn. The opening minimum has to
    /// be met inside a single turn, so this never carries across turns.
    pub laid_value: u32,
    pub laid_anything: bool,
    /// §5: cards swept up from the discard pile, unusable until the next turn.
    /// Held as a multiset so a card the player already had stays playable.
    pub frozen: Vec<Card>,
    pub took_pile: bool,
    /// §5: the meld the three-card core went into. No wild may join *that* meld
    /// for the rest of this turn, though other melds are unrestricted.
    pub pile_core_meld: Option<usize>,
    /// §3: the card the lead player has drawn but not yet kept or refused.
    pub pending_refusal: Option<Card>,
    /// §3: whether the once-per-hand refusal is still available.
    pub refusal_available: bool,
}

/// The complete, omniscient game state.
///
/// Fields are public for serialization and for bots that need to read widely.
/// Mutating them outside [`crate::apply`] is outside the engine's contract —
/// nothing here re-checks the rules. Clients should be handed a redacted view
/// rather than this, since it contains every hand and the stock order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GameState {
    pub stock: Vec<Card>,
    pub discard: Vec<Card>,
    pub hands: [Vec<Card>; 4],
    pub tables: [TeamTable; 2],
    /// Cumulative match score, updated at the end of each hand.
    pub scores: [i32; 2],
    pub dealer: Seat,
    pub turn: Seat,
    pub phase: Phase,
    pub turn_context: TurnContext,
    pub hand_number: u32,
    /// §11.1: who went out, if anyone. `None` after a hand that ended because
    /// the stock ran dry (§11.2), where nobody collects the bonus.
    pub went_out: Option<Seat>,
    /// §6.1: whether each partnership has already paid the failed-opening
    /// penalty this hand. A latch, not a counter — a second failed opening in
    /// the same hand changes nothing, and a fresh deal clears it.
    #[serde(default)]
    pub opening_penalty: [bool; 2],
    /// The match seed. Every hand's shuffle is derived from it, so a whole
    /// match replays from this one number plus its action log.
    pub seed: u64,
}

impl GameState {
    pub fn hand(&self, seat: Seat) -> &[Card] {
        &self.hands[seat.index()]
    }

    pub fn table(&self, team: Team) -> &TeamTable {
        &self.tables[team.index()]
    }

    pub fn score(&self, team: Team) -> i32 {
        self.scores[team.index()]
    }

    /// §6: what this partnership must lay in one turn to open.
    pub fn opening_minimum_for(&self, team: Team) -> u32 {
        let base = opening_minimum(self.score(team));
        if !self.opening_penalty[team.index()] {
            return base;
        }
        // §6.1: a failed opening steps the bar up one tier — 75 becomes 120,
        // and 120 becomes 240.
        if base >= OPENING_MINIMUM_HIGH {
            OPENING_MINIMUM_PENALIZED
        } else {
            OPENING_MINIMUM_HIGH
        }
    }

    /// §6.1: abandoning the turn in progress right now counts as a failed
    /// opening — the partnership has not opened and has already laid cards it
    /// would be taking back.
    pub fn restart_penalizes_opening(&self, seat: Seat) -> bool {
        !self.table(seat.team()).opened && self.turn_context.laid_anything
    }

    /// §6.1: latch the one-time penalty. Returns whether the bar actually
    /// moved — `false` when the partnership had already been penalized.
    pub fn penalize_opening(&mut self, team: Team) -> bool {
        let already = self.opening_penalty[team.index()];
        self.opening_penalty[team.index()] = true;
        !already
    }

    /// §14: the partnership that won, once the match is over.
    ///
    /// Derived rather than stored, so it cannot disagree with the scores.
    pub fn winner(&self) -> Option<Team> {
        if self.phase != Phase::MatchOver {
            return None;
        }
        let [first, second] = self.scores;
        match first.cmp(&second) {
            std::cmp::Ordering::Greater => Some(Team::ALL[0]),
            std::cmp::Ordering::Less => Some(Team::ALL[1]),
            std::cmp::Ordering::Equal => None,
        }
    }

    /// §11.1 and §12: whether the partnership holds a clean canastra.
    ///
    /// This one predicate gates two separate things — whether the partnership
    /// may go out at all, and whether their red 3s score +100 or −100 — which is
    /// why closing a clean canastra is not optional in practice.
    pub fn has_clean_canastra(&self, team: Team) -> bool {
        self.table(team)
            .melds
            .iter()
            .any(|meld| meld.canastra().is_clean())
    }
}

/// Why a state could not have arisen from play.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "problem")]
pub enum StateError {
    /// The cards in play are not the 108-card deck: some were invented, lost,
    /// or duplicated beyond the two copies that exist.
    DeckNotConserved,
    /// §12: a red 3 goes to the table the moment it is seen, so it can never be
    /// in a hand, and having never been in a hand it can never reach the pile.
    RedThreeOutOfPlace,
    /// A partnership's red-3 pile holds something that is not a red 3.
    NotARedThree,
    /// §5: the frozen set is meant to be part of the current player's hand.
    FrozenCardNotHeld,
    /// §5: the meld that captured the pile does not exist.
    DanglingPileCoreMeld,
}

impl fmt::Display for StateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            StateError::DeckNotConserved => "the cards in play are not the 108-card deck",
            StateError::RedThreeOutOfPlace => "a red 3 is in a hand or in the discard pile",
            StateError::NotARedThree => "a partnership's red 3s include another card",
            StateError::FrozenCardNotHeld => "a frozen card is not in the player's hand",
            StateError::DanglingPileCoreMeld => "the pile's capturing meld does not exist",
        };
        f.write_str(message)
    }
}

impl std::error::Error for StateError {}

impl GameState {
    /// Check that this state could actually have arisen from play.
    ///
    /// serde rebuilds a `GameState` field by field, which walks past every check
    /// [`crate::apply`] makes. Individually valid melds are not enough: a payload
    /// can still invent cards, or park a red 3 somewhere §12 says it can never
    /// be. Call this on anything that came from outside the process — a stored
    /// snapshot, a client payload — before trusting it.
    ///
    /// This is a soundness check, not a rules check. It does not care whether
    /// the position is *reachable*, only that it is not impossible.
    pub fn check_invariants(&self) -> Result<(), StateError> {
        for hand in &self.hands {
            if hand.iter().any(|card| card.is_red_three()) {
                return Err(StateError::RedThreeOutOfPlace);
            }
        }
        // A red 3 never enters a hand, so it can never have been discarded
        // either. One still in the stock is fine — it just has not been drawn.
        if self.discard.iter().any(|card| card.is_red_three()) {
            return Err(StateError::RedThreeOutOfPlace);
        }

        for table in &self.tables {
            if !table.red_threes.iter().all(|card| card.is_red_three()) {
                return Err(StateError::NotARedThree);
            }
        }

        let held = &self.hands[self.turn.index()];
        for frozen in &self.turn_context.frozen {
            let in_hand = held.iter().filter(|card| *card == frozen).count();
            let frozen_copies = self
                .turn_context
                .frozen
                .iter()
                .filter(|card| *card == frozen)
                .count();
            if in_hand < frozen_copies {
                return Err(StateError::FrozenCardNotHeld);
            }
        }

        if let Some(index) = self.turn_context.pile_core_meld
            && index >= self.table(self.turn.team()).melds.len()
        {
            return Err(StateError::DanglingPileCoreMeld);
        }

        let mut in_play: Vec<Card> = self.stock.clone();
        in_play.extend(self.discard.iter().copied());
        for hand in &self.hands {
            in_play.extend(hand.iter().copied());
        }
        for table in &self.tables {
            in_play.extend(table.red_threes.iter().copied());
            for meld in &table.melds {
                in_play.extend(meld.cards());
            }
        }
        // §3: before the lead player resolves the refusal, the card on offer is
        // out of the stock, not yet in hand, and not on the table — it lives
        // only in `pending_refusal`, so it must be counted separately or a state
        // in `AwaitingRefusalChoice` reads as short one card.
        if let Some(offered) = self.turn_context.pending_refusal {
            in_play.push(offered);
        }
        let mut expected = crate::card::deck();
        in_play.sort_by_key(|card| card.to_string());
        expected.sort_by_key(|card| card.to_string());
        if in_play != expected {
            return Err(StateError::DeckNotConserved);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::Rig;

    fn seat(index: u8) -> Seat {
        Seat::new(index).expect("valid seat")
    }

    #[test]
    fn there_are_four_seats() {
        assert!(Seat::new(3).is_some());
        assert!(Seat::new(4).is_none());
    }

    /// §2: play passes to the right, and the table wraps around.
    #[test]
    fn play_passes_to_the_right_and_wraps() {
        assert_eq!(seat(0).next(), seat(1));
        assert_eq!(seat(3).next(), seat(0));
    }

    /// §2: partners sit facing each other, so they are two seats apart.
    #[test]
    fn partners_sit_opposite_each_other() {
        assert_eq!(seat(0).team(), seat(2).team());
        assert_eq!(seat(1).team(), seat(3).team());
        assert_ne!(seat(0).team(), seat(1).team());
    }

    /// §5: the player who follows you is always an opponent, which is why a
    /// blocking discard can never hurt your partner.
    #[test]
    fn the_next_player_is_always_an_opponent() {
        for index in 0..4 {
            assert_ne!(seat(index).team(), seat(index).next().team());
        }
    }

    #[test]
    fn seats_iterate_in_play_order() {
        let seats: Vec<u8> = Seat::ALL.iter().map(|seat| seat.index() as u8).collect();
        assert_eq!(seats, vec![0, 1, 2, 3]);
    }

    /// §6: the bar rises once a partnership is past 2500.
    #[test]
    fn the_opening_minimum_rises_past_twenty_five_hundred() {
        assert_eq!(opening_minimum(0), 75);
        assert_eq!(opening_minimum(2499), 75);
        assert_eq!(opening_minimum(2500), 120);
        assert_eq!(opening_minimum(4000), 120);
    }

    /// A partnership that has gone negative is still below the bar.
    #[test]
    fn a_negative_score_keeps_the_lower_opening_minimum() {
        assert_eq!(opening_minimum(-300), 75);
    }

    /// §6.1: a failed opening steps the bar up one tier — 75 to 120, 120 to
    /// 240 — and the latch means a second failure changes nothing.
    #[test]
    fn a_failed_opening_raises_the_bar_one_tier_only_once() {
        let mut state = Rig::new().build();
        let team = Team::ALL[0];

        assert!(state.penalize_opening(team));
        assert_eq!(state.opening_minimum_for(team), 120);
        // The latch is already down: a second mistake is free.
        assert!(!state.penalize_opening(team));
        assert_eq!(state.opening_minimum_for(team), 120);

        state.scores = [2500, 0];
        assert_eq!(state.opening_minimum_for(team), 240);
    }

    /// §6.1: the penalty is for taking back cards an unopened partnership had
    /// already laid — not for any restart, and not after opening.
    #[test]
    fn only_a_restart_with_cards_laid_before_opening_penalizes() {
        let seat = seat(1);
        let mut state = Rig::new().turn(1).build();
        // Nothing laid yet: backing out of a turn costs nothing.
        assert!(!state.restart_penalizes_opening(seat));

        state.turn_context.laid_anything = true;
        assert!(state.restart_penalizes_opening(seat));

        state.tables[seat.team().index()].opened = true;
        assert!(!state.restart_penalizes_opening(seat));
    }
}
