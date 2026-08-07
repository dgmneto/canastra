//! The single source of truth for what a policy network sees and what its
//! outputs mean.
//!
//! A neural network cannot eat a `PlayerView`: someone has to turn the game
//! into a fixed-length vector of floats, and turn each legal action into a
//! fixed-length feature row. That translation lives here — once. The Python
//! trainer and the TypeScript `JSONWeightsBot` both bind this crate, so the
//! layout can never drift between training and deployment.
//!
//! Layout principles (binding, from the design spec):
//! - Fixed length, asserted in `encode`. A silently variable vector fails as
//!   garbage learning, not as a crash.
//! - Thermometers over raw magnitudes: a network learns `>= 2` far more
//!   easily than `2.0`.
//! - Everything relative to the acting seat: "my", "right", "partner",
//!   "left". One network plays all four positions.
//! - Derivable facts (locked wild, is-canastra, extendable) get their own
//!   units; they are cheap to supply and expensive to infer.

pub mod actions;
pub mod cards;
pub mod obs;
pub mod tokens;

pub use actions::encode_actions;
pub use obs::encode_observation;

/// Observation vector length. Pinned by the segment table in `obs`.
pub const OBS_DIM: usize = 2002;

/// Per-action feature row length. Pinned by the block table in `actions`.
pub const ACT_DIM: usize = 101;

#[cfg(test)]
mod tests {
    use super::cards::card_index;
    use canastra_engine::testkit::card;

    #[test]
    fn card_identity_indexing_follows_the_layout() {
        // Suits in declaration order, ranks in game order (4..A, 2, 3), so
        // sequence-adjacent ranks are encoding-adjacent.
        assert_eq!(card_index(card("4C")), 0);
        assert_eq!(card_index(card("AC")), 10);
        assert_eq!(card_index(card("2C")), 11);
        assert_eq!(card_index(card("3C")), 12);
        assert_eq!(card_index(card("4D")), 13);
        assert_eq!(card_index(card("3S")), 51);
        assert_eq!(card_index(card("JOKER")), 52);
    }
}
