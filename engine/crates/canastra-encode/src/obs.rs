//! Placeholder — implemented in a later task.

use canastra_engine::PlayerView;

use crate::OBS_DIM;

pub fn encode_observation(_view: &PlayerView, out: &mut [f32]) {
    assert_eq!(out.len(), OBS_DIM);
}
