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
    /// §4.2: put a new meld on your partnership's table.
    LayMeld { cards: Vec<Card> },
    /// §4.2: add cards to a meld your partnership already has down.
    AddToMeld { meld: usize, cards: Vec<Card> },
    /// §4.3: put a card on the pile, ending the turn.
    Discard { card: Card },
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
        }
    }
}

impl std::error::Error for RuleViolation {}
