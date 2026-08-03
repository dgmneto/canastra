//! What a player can do, and the ways it can be refused.

use crate::card::Card;
use crate::meld::MeldError;
use crate::state::{Phase, Seat};
use std::fmt;

/// A single move.
///
/// Actions are deliberately granular — one draw, one meld, one discard — rather
/// than one composite "here is my whole turn". That keeps the branching factor
/// small for a searching bot and lets a human UI show the table updating as
/// cards are placed.
///
/// The cost of granularity is that a player can reach a dead end: §6 requires
/// the opening minimum to be met across a single turn, and that is only checked
/// once they try to discard. Since [`crate::apply`] is pure, backing out is just
/// a matter of the caller keeping the state from the start of the turn.
///
/// No variant is a tuple variant. Serde's internally-tagged representation —
/// the one that gives other languages natural objects like
/// `{ "type": "Discard", "card": "3S" }` — cannot express tuple variants, and
/// the shape of this enum is a published interface once bindings are generated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// §4.1: take the top card of the stock.
    Draw,
    /// §3: the lead player accepts the card they were shown.
    KeepDrawnCard,
    /// §3: the lead player throws their first card away and takes another.
    RefuseDrawnCard,
    /// §5: take the whole discard pile instead of drawing.
    ///
    /// `core` is the two natural cards from hand that, together with the card on
    /// top of the pile, form the compulsory three. All three land in one meld —
    /// splitting them across melds is not allowed.
    TakeDiscardPile { core: [Card; 2], target: MeldTarget },
    /// §4.2: put a new meld on your partnership's table.
    LayMeld { cards: Vec<Card> },
    /// §4.2: add cards to a meld your partnership already has down.
    AddToMeld { meld: usize, cards: Vec<Card> },
    /// §4.3: put a card on the pile, ending the turn.
    Discard { card: Card },
}

/// Where the three cards captured from the discard pile are going.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeldTarget {
    /// Open a new meld with them.
    NewMeld,
    /// Fold them into a meld the partnership already has down.
    Existing { meld: usize },
}

/// Why a move was refused.
///
/// Rich enough to show a human what went wrong and to let a bot's tests assert
/// on the specific rule that fired.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuleViolation {
    /// Somebody else is to move.
    NotYourTurn { current: Seat },
    /// The action does not belong in the phase the turn is currently in.
    WrongPhase { phase: Phase },
    /// The hand, or the whole match, is already finished.
    HandIsOver,
    /// §11.2: there is nothing left to draw.
    StockEmpty,
    /// The player does not hold that card.
    CardNotInHand { card: Card },
    /// Your partnership has no meld at that position. Melds are addressed per
    /// partnership, so this also covers pointing at an opponent's meld.
    NoSuchMeld { meld: usize },
    /// The cards are not a legal meld, or will not join the one they were aimed at.
    InvalidMeld { reason: MeldError },
    /// §6: a partnership's first melds have to clear the bar inside one turn.
    OpeningMinimumNotMet { laid: u32, required: u32 },
    /// §5: there is no pile to take.
    DiscardPileEmpty,
    /// §5: a black 3 or a wild on top puts the pile out of reach. This is
    /// exactly what makes a black 3 worth holding.
    DiscardPileBlocked { card: Card },
    /// §5: "2s e coringas não podem ser usados" to capture the pile.
    WildInDiscardCore,
    /// §5: the meld that captured the pile takes no wild for the rest of the
    /// turn. Other melds are unaffected, and the restriction lifts next turn.
    WildInPileCoreMeld,
    /// §5: a card swept up from the pile cannot be melded until the next turn.
    /// It can still be discarded (CLAUDE.md clarification #5).
    CardFrozen { card: Card },
    /// §11.1: going out needs a clean canastra on the table. A dirty one does
    /// not qualify, so this move would have emptied the hand illegally.
    NoCleanCanastra,
}

impl fmt::Display for RuleViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuleViolation::NotYourTurn { current } => {
                write!(f, "it is seat {}'s turn", current.index())
            }
            RuleViolation::WrongPhase { phase } => write!(f, "not available during {phase:?}"),
            RuleViolation::HandIsOver => f.write_str("the hand is already over"),
            RuleViolation::StockEmpty => f.write_str("the stock is empty"),
            RuleViolation::CardNotInHand { card } => write!(f, "you do not hold the {card}"),
            RuleViolation::NoSuchMeld { meld } => {
                write!(f, "your partnership has no meld {meld}")
            }
            RuleViolation::InvalidMeld { reason } => write!(f, "{reason}"),
            RuleViolation::OpeningMinimumNotMet { laid, required } => {
                write!(f, "laid {laid} of the {required} needed to open")
            }
            RuleViolation::DiscardPileEmpty => f.write_str("the discard pile is empty"),
            RuleViolation::DiscardPileBlocked { card } => {
                write!(f, "the {card} on top blocks the pile")
            }
            RuleViolation::WildInDiscardCore => {
                f.write_str("the three that take the pile must all be natural")
            }
            RuleViolation::WildInPileCoreMeld => {
                f.write_str("no wild card may join that meld this turn")
            }
            RuleViolation::CardFrozen { card } => {
                write!(f, "the {card} came out of the pile and is frozen this turn")
            }
            RuleViolation::NoCleanCanastra => {
                f.write_str("going out needs a clean canastra on the table")
            }
        }
    }
}

impl std::error::Error for RuleViolation {}
