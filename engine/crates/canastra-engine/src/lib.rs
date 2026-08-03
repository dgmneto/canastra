//! Rules engine and state machine for Canastra, house rules.
//!
//! The authoritative rules live in `canastra-regras-da-casa.md` at the repository
//! root; section references throughout this crate (§5, §9, ...) point at it.

pub mod action;
pub mod apply;
pub mod card;
pub mod deal;
pub mod meld;
pub mod state;
pub mod testkit;

pub use action::{Action, RuleViolation};
pub use apply::{apply, validate};
pub use deal::new_game;
pub use state::{GameState, Phase, Seat, Team};
