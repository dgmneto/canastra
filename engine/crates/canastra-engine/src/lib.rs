//! Rules engine and state machine for Canastra, house rules.
//!
//! The authoritative rules live in `canastra-regras-da-casa.md` at the repository
//! root; section references throughout this crate (§5, §9, ...) point at it.

pub mod action;
pub mod apply;
pub mod card;
pub mod deal;
pub mod meld;
pub mod score;
pub mod state;
pub mod testkit;
pub mod view;

pub use action::{Action, RuleViolation};
pub use apply::{apply, validate};
pub use deal::new_game;
pub use score::{HandScore, score_hand, settle_hand};
pub use state::{GameState, Phase, Seat, StateError, Team};
pub use view::{PlayerView, observe};
