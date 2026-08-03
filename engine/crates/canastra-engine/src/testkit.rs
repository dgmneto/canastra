//! Helpers for building specific game positions.
//!
//! These live in the library rather than inside a `#[cfg(test)]` module so that
//! integration tests and downstream crates (bots, the server) can rig positions
//! too — reproducing a reported bug usually means describing a table, not a seed.

use crate::card::Card;
use crate::meld::Meld;
use crate::state::{GameState, Phase, Seat, TurnContext};

/// Parse one card, panicking on nonsense. For tests only.
pub fn card(spec: &str) -> Card {
    spec.parse()
        .unwrap_or_else(|_| panic!("not a card: {spec:?}"))
}

/// Parse a whitespace-separated run of cards: `cards("6D 7D JOKER")`.
pub fn cards(spec: &str) -> Vec<Card> {
    spec.split_whitespace().map(card).collect()
}

/// Builder for a hand-crafted [`GameState`].
///
/// Everything starts empty, with seat 1 to move in [`Phase::AwaitingDraw`] and
/// no refusal available, so a test only has to state what it actually cares about.
pub struct Rig {
    state: GameState,
}

impl Default for Rig {
    fn default() -> Self {
        Self::new()
    }
}

impl Rig {
    pub fn new() -> Rig {
        Rig {
            state: GameState {
                stock: Vec::new(),
                discard: Vec::new(),
                hands: Default::default(),
                tables: Default::default(),
                scores: [0, 0],
                dealer: Seat::ALL[0],
                turn: Seat::ALL[1],
                phase: Phase::AwaitingDraw,
                turn_context: TurnContext::default(),
                hand_number: 1,
            },
        }
    }

    pub fn hand(mut self, seat: usize, spec: &str) -> Rig {
        self.state.hands[seat] = cards(spec);
        self
    }

    /// Cards are drawn from the end, so the last card listed is drawn first.
    pub fn stock(mut self, spec: &str) -> Rig {
        self.state.stock = cards(spec);
        self
    }

    /// The last card listed is the top of the pile.
    pub fn discard(mut self, spec: &str) -> Rig {
        self.state.discard = cards(spec);
        self
    }

    pub fn scores(mut self, first: i32, second: i32) -> Rig {
        self.state.scores = [first, second];
        self
    }

    pub fn turn(mut self, seat: usize) -> Rig {
        self.state.turn = Seat::new(seat as u8).expect("valid seat");
        self
    }

    pub fn dealer(mut self, seat: usize) -> Rig {
        self.state.dealer = Seat::new(seat as u8).expect("valid seat");
        self
    }

    pub fn phase(mut self, phase: Phase) -> Rig {
        self.state.phase = phase;
        self
    }

    /// §3: hand the player to move the lead player's one-time refusal.
    pub fn refusal_available(mut self) -> Rig {
        self.state.turn_context.refusal_available = true;
        self
    }

    /// §6: mark the partnership as having already met its opening minimum.
    pub fn opened(mut self, team: usize) -> Rig {
        self.state.tables[team].opened = true;
        self
    }

    /// Put a meld on a partnership's table. Implies the partnership has opened.
    pub fn meld(mut self, team: usize, spec: &str) -> Rig {
        let meld = Meld::new(&cards(spec)).unwrap_or_else(|e| panic!("bad meld {spec:?}: {e}"));
        self.state.tables[team].melds.push(meld);
        self.state.tables[team].opened = true;
        self
    }

    /// §12: red 3s already sitting in front of a partnership.
    pub fn red_threes(mut self, team: usize, spec: &str) -> Rig {
        self.state.tables[team].red_threes = cards(spec);
        self
    }

    pub fn build(self) -> GameState {
        self.state
    }
}
