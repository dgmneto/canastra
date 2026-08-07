//! Placeholder — implemented in the meld-token task.

use canastra_engine::PlayerView;

use crate::obs::Writer;

pub(crate) fn write_tokens(_view: &PlayerView, w: &mut Writer) {
    for _ in 0..33 * 43 {
        w.bit(false);
    }
}
