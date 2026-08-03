//! Game state: seats, partnerships, the table, and the turn machine.

use crate::card::Card;
use crate::meld::Meld;

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

/// §13.1: bonus for going out.
pub const GOING_OUT_BONUS: i32 = 100;

/// §12: what each red 3 is worth, positive or negative.
pub const RED_THREE_VALUE: i32 = 100;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

/// One of the two partnerships.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

/// Everything a partnership has in front of it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone, Default, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
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
        opening_minimum(self.score(team))
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
