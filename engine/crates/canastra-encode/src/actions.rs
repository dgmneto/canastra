//! Placeholder — implemented in a later task.

use canastra_engine::{Action, PlayerView};

use crate::ACT_DIM;

pub fn encode_actions(_view: &PlayerView, legal: &[Action], out: &mut [f32]) {
    assert_eq!(out.len(), legal.len() * ACT_DIM);
}
