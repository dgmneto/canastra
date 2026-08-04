//! JavaScript bindings for the Canastra engine.
//!
//! Rust enums that carry data have no stable ABI, so nothing here hands a
//! `GameState` or an `Action` across the boundary as a Rust value. JavaScript
//! builds plain objects, serde converts and validates them on the way in, and a
//! malformed one comes back as an error rather than undefined behaviour.

use canastra_engine::{
    Action, GameState, Phase, RuleViolation, Seat, Team, apply, new_game, observe, score_hand,
    settle_hand,
};
use wasm_bindgen::prelude::*;

/// A game living in Rust memory, driven from JavaScript.
///
/// The full state deliberately never crosses the boundary. It holds all four
/// hands and the order of the stock, so a browser client has no business owning
/// one — what comes back is always a redacted view of a single seat. Keeping the
/// state on this side also avoids reserializing the whole table on every move.
#[wasm_bindgen]
pub struct Game {
    state: GameState,
    /// §6 and §11.1 can only be judged when a turn ends, so a player may find
    /// themselves unable to finish one — having laid too little to open, or
    /// down to a last card they cannot legally throw. Because the engine's
    /// `apply` is pure, offering a way out is just a matter of keeping the
    /// position the turn started from.
    turn_start: GameState,
}

#[wasm_bindgen]
impl Game {
    /// Start a match. The seed arrives as a BigInt: `new Game(7n)`.
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u64) -> Game {
        let state = new_game(seed);
        Game {
            turn_start: state.clone(),
            state,
        }
    }

    /// Play one move, given as a plain object such as
    /// `{ type: "Discard", card: "3S" }`.
    pub fn apply(&mut self, seat: u8, action: JsValue) -> Result<(), JsValue> {
        let seat = Seat::new(seat).ok_or_else(|| message("seat must be 0, 1, 2 or 3"))?;
        let action: Action = serde_wasm_bindgen::from_value(action)
            .map_err(|error| message(&format!("not an action: {error}")))?;

        if self.state.phase == Phase::AwaitingDraw {
            self.turn_start = self.state.clone();
        }
        self.state = apply(&self.state, seat, &action).map_err(violation)?;
        Ok(())
    }

    /// What one seat is allowed to see.
    pub fn view(&self, seat: u8) -> Result<JsValue, JsValue> {
        let seat = Seat::new(seat).ok_or_else(|| message("seat must be 0, 1, 2 or 3"))?;
        serde_wasm_bindgen::to_value(&observe(&self.state, seat))
            .map_err(|error| message(&error.to_string()))
    }

    /// Abandon the turn in progress and restore the position it began from.
    #[wasm_bindgen(js_name = rewindTurn)]
    pub fn rewind_turn(&mut self) {
        self.state = self.turn_start.clone();
    }

    /// §13: what this hand would bank for one partnership right now, itemised.
    ///
    /// Mid-hand this is a running total, not a result — `going_out_bonus` is
    /// still zero and `hand_cards` keeps falling as cards are laid down.
    ///
    /// **Not for a player's eyes in a real game.** `hand_cards` sums *both*
    /// partners' hands, and partners do not see each other's cards, so this
    /// answers a question no seat is entitled to ask. It belongs on the same
    /// side of the boundary as [`Game::snapshot`].
    #[wasm_bindgen(js_name = handScore)]
    pub fn hand_score(&self, team: u8) -> Result<JsValue, JsValue> {
        let team = Team::new(team).ok_or_else(|| message("team must be 0 or 1"))?;
        serde_wasm_bindgen::to_value(&score_hand(&self.state, team))
            .map_err(|error| message(&error.to_string()))
    }

    /// §13 and §14: bank a finished hand, then deal on or end the match.
    #[wasm_bindgen(js_name = settleHand)]
    pub fn settle_hand(&mut self) -> Result<(), JsValue> {
        self.state = settle_hand(&self.state).map_err(violation)?;
        self.turn_start = self.state.clone();
        Ok(())
    }

    /// The whole state as JSON, for a server that has to persist a game.
    pub fn snapshot(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.state).map_err(|error| message(&error.to_string()))
    }

    /// Rebuild a game from [`Game::snapshot`].
    ///
    /// Parsing is not enough on its own. serde rebuilds a state field by field,
    /// so a snapshot that is merely corrupt — let alone crafted — can describe a
    /// position no game could reach. The invariant check is what stands between
    /// that and the rest of the engine.
    pub fn restore(snapshot: &str) -> Result<Game, JsValue> {
        let state: GameState = serde_json::from_str(snapshot)
            .map_err(|error| message(&format!("not a game: {error}")))?;
        state
            .check_invariants()
            .map_err(|error| message(&format!("not a reachable position: {error}")))?;
        Ok(Game {
            turn_start: state.clone(),
            state,
        })
    }
}

/// Rejected moves reach JavaScript as structured objects — `{ error:
/// "OpeningMinimumNotMet", laid: 45, required: 75 }` — so a UI can react to the
/// specific rule rather than matching on a sentence.
fn violation(violation: RuleViolation) -> JsValue {
    serde_wasm_bindgen::to_value(&violation).unwrap_or_else(|_| message(&violation.to_string()))
}

fn message(text: &str) -> JsValue {
    JsValue::from_str(text)
}
