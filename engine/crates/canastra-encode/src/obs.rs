//! The observation vector: what the network sees, always from the acting
//! seat's side of the table.
//!
//! Segment layout (offsets are the contract — the design spec pins them):
//!
//! | offset | width | segment |
//! |---|---|---|
//! | 0    | 5    | phase one-hot |
//! | 5    | 13   | laid_value thermometer >=10..>=130 step 10 |
//! | 18   | 1    | took_pile |
//! | 19   | 1    | refusal_available |
//! | 20   | 53   | pending card one-hot (zero when none) |
//! | 73   | 6    | hand_number thermometer >=2,4,...,12 |
//! | 79   | 104  | my hand census: per standard identity >=1, >=2 |
//! | 183  | 4    | my hand jokers >=1..>=4 |
//! | 187  | 108  | frozen census (same 104+4 shape) |
//! | 295  | 8    | my hand size thermometer >=16,18,...,30 |
//! | 303  | 36   | right/partner/left hand counts, each >=2,4,...,24 |
//! | 339  | 11   | stock count thermometer >=4,8,...,44 |
//! | 350  | 20   | my score thermometer >=250..>=5000 step 250 |
//! | 370  | 20   | their score thermometer |
//! | 390  | 2    | >=2500 bits (mine, theirs) |
//! | 392  | 3    | opening minimum one-hot {opened, 75, 120} |
//! | 395  | 2    | opened bits (mine, theirs) |
//! | 397  | 2    | clean canastra bits (mine, theirs) |
//! | 399  | 8    | red threes per team thermometer >=1..>=4 |
//! | 407  | 53   | pile top one-hot (zero when empty) |
//! | 460  | 15   | pile size thermometer >=2,4,...,30 |
//! | 475  | 108  | pile census |
//! | 583  | 1419 | 33 meld tokens x 43 features (see `tokens`) |

use canastra_engine::card::Card;
use canastra_engine::state::Phase;
use canastra_engine::{PlayerView, Seat};

use crate::OBS_DIM;
use crate::cards::{CARD_IDS, card_index};
use crate::tokens;

/// Write the acting seat's observation into `out` (length [`OBS_DIM`]).
///
/// # Panics
///
/// The final length is asserted, not debug-asserted: a silently variable
/// vector fails as garbage learning, not as a crash (spec principle 1).
pub fn encode_observation(view: &PlayerView, out: &mut [f32]) {
    assert_eq!(out.len(), OBS_DIM);
    out.fill(0.0);
    let mut w = Writer { out, at: 0 };

    w.one_hot(5, Some(phase_index(view.phase)));
    w.therm(
        view.laid_value,
        &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130],
    );
    w.bit(view.took_pile);
    w.bit(view.refusal_available);
    w.one_hot(CARD_IDS, view.pending_refusal.map(card_index));
    w.therm(view.hand_number, &[2, 4, 6, 8, 10, 12]);

    census(view.hand.iter().copied(), &mut w);
    census(view.frozen.iter().copied(), &mut w);
    w.therm(view.hand.len() as u32, &[16, 18, 20, 22, 24, 26, 28, 30]);

    for seat in relatives(view.seat) {
        w.therm(
            view.hand_counts[seat.index()] as u32,
            &[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
        );
    }

    w.therm(
        view.stock_count as u32,
        &[4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44],
    );

    let mine = view.seat.team().index();
    let theirs = 1 - mine;
    w.therm(clamped(view.scores[mine]), &score_thresholds());
    w.therm(clamped(view.scores[theirs]), &score_thresholds());
    w.bit(view.scores[mine] >= 2500);
    w.bit(view.scores[theirs] >= 2500);

    let my_table = &view.tables[mine];
    let their_table = &view.tables[theirs];
    let minimum = if my_table.opened {
        0
    } else if view.opening_minimum == 75 {
        1
    } else {
        2
    };
    w.one_hot(3, Some(minimum));
    w.bit(my_table.opened);
    w.bit(their_table.opened);
    w.bit(has_clean_canastra(my_table));
    w.bit(has_clean_canastra(their_table));
    w.therm(my_table.red_threes.len() as u32, &[1, 2, 3, 4]);
    w.therm(their_table.red_threes.len() as u32, &[1, 2, 3, 4]);

    w.one_hot(CARD_IDS, view.discard.last().copied().map(card_index));
    w.therm(
        view.discard.len() as u32,
        &[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
    );
    census(view.discard.iter().copied(), &mut w);

    tokens::write_tokens(view, &mut w);

    w.finish();
}

/// A cursor over the output slice. Every segment writes through it, so a
/// layout mistake fails loudly at `finish` instead of shifting every later
/// segment.
pub(crate) struct Writer<'a> {
    pub(crate) out: &'a mut [f32],
    pub(crate) at: usize,
}

impl Writer<'_> {
    pub(crate) fn bit(&mut self, on: bool) {
        self.out[self.at] = if on { 1.0 } else { 0.0 };
        self.at += 1;
    }

    pub(crate) fn one_hot(&mut self, width: usize, index: Option<usize>) {
        for i in 0..width {
            self.bit(Some(i) == index);
        }
    }

    pub(crate) fn therm(&mut self, value: u32, thresholds: &[u32]) {
        for &threshold in thresholds {
            self.bit(value >= threshold);
        }
    }

    pub(crate) fn finish(self) {
        assert_eq!(self.at, self.out.len(), "observation length drifted");
    }
}

/// A 108-wide copy census: per standard identity >=1 / >=2, then jokers
/// >=1..>=4. Used for the hand, the frozen set, and the discard pile.
fn census(cards: impl Iterator<Item = Card>, w: &mut Writer) {
    let mut counts = [0u8; CARD_IDS];
    for card in cards {
        counts[card_index(card)] += 1;
    }
    for &count in counts.iter().take(52) {
        w.bit(count >= 1);
        w.bit(count >= 2);
    }
    for n in 1..=4u8 {
        w.bit(counts[52] >= n);
    }
}

/// The three other seats in felt order: right opponent, partner, left opponent.
fn relatives(seat: Seat) -> [Seat; 3] {
    [seat.next(), seat.next().next(), seat.next().next().next()]
}

fn phase_index(phase: Phase) -> usize {
    match phase {
        Phase::AwaitingDraw => 0,
        Phase::AwaitingRefusalChoice => 1,
        Phase::Melding => 2,
        Phase::HandOver => 3,
        Phase::MatchOver => 4,
    }
}

fn score_thresholds() -> [u32; 20] {
    std::array::from_fn(|i| 250 * (i as u32 + 1))
}

fn clamped(score: i32) -> u32 {
    score.max(0) as u32
}

fn has_clean_canastra(table: &canastra_engine::state::TeamTable) -> bool {
    table.melds.iter().any(|meld| meld.canastra().is_clean())
}

#[cfg(test)]
mod tests {
    use super::*;
    use canastra_engine::state::Phase;
    use canastra_engine::testkit::Rig;
    use canastra_engine::{Seat, observe};

    fn view(state: &canastra_engine::GameState, seat: u8) -> canastra_engine::PlayerView {
        observe(state, Seat::new(seat).unwrap())
    }

    fn encoded(state: &canastra_engine::GameState, seat: u8) -> Vec<f32> {
        let mut out = vec![0.0; OBS_DIM];
        encode_observation(&view(state, seat), &mut out);
        out
    }

    #[test]
    fn phase_is_a_one_hot() {
        let state = Rig::new().stock("8C").phase(Phase::Melding).build();
        let out = encoded(&state, 1);
        assert_eq!(&out[0..5], &[0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn laid_value_is_a_thermometer() {
        let state = Rig::new().laid_value(45).build();
        let out = encoded(&state, 1);
        // thresholds 10..=130 step 10: 45 crosses 10,20,30,40 only.
        assert_eq!(
            &out[5..18],
            &[
                1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ]
        );
    }

    #[test]
    fn the_pending_card_is_a_card_one_hot() {
        let state = Rig::new()
            .phase(Phase::AwaitingRefusalChoice)
            .pending_refusal("KH")
            .refusal_available()
            .build();
        let out = encoded(&state, 1);
        // KH = suit Hearts (2) * 13 + rank King (9) = 35; segment at offset 20.
        assert_eq!(out[20 + 35], 1.0);
        assert_eq!(out[20..73].iter().sum::<f32>(), 1.0);
        assert_eq!(out[19], 1.0, "refusal_available bit");
    }

    #[test]
    fn the_hand_census_counts_copies() {
        let state = Rig::new().hand(1, "6D 6D KH JOKER").build();
        let out = encoded(&state, 1);
        // 6D = Diamonds (1) * 13 + Six (2) = 15; >=1 and >=2 units at 79+2*i.
        assert_eq!(out[79 + 30], 1.0);
        assert_eq!(out[79 + 31], 1.0);
        // KH = 35: >=1 only.
        assert_eq!(out[79 + 70], 1.0);
        assert_eq!(out[79 + 71], 0.0);
        // Joker thermometer at 183..187: exactly >=1.
        assert_eq!(&out[183..187], &[1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn other_seats_are_right_partner_left() {
        let state = Rig::new()
            .hand(0, "4C 5C 6C")
            .hand(1, "4D")
            .hand(2, "4H 5H")
            .hand(3, "4S 5S 6S 7S")
            .build();
        // Seat 0's right is seat 1 (1 card), partner seat 2 (2), left seat 3 (4).
        let out = encoded(&state, 0);
        // Each count is a thermometer >=2,4,...,24 (12 units), block at 303.
        let right = &out[303..315];
        let partner = &out[315..327];
        let left = &out[327..339];
        assert_eq!(right.iter().sum::<f32>(), 0.0, "1 card crosses no >=2");
        assert_eq!(partner[0], 1.0);
        assert_eq!(partner.iter().sum::<f32>(), 1.0, "2 cards cross >=2 only");
        assert_eq!(left[0], 1.0);
        assert_eq!(left[1], 1.0);
        assert_eq!(left.iter().sum::<f32>(), 2.0, "4 cards cross >=2 and >=4");
    }

    #[test]
    fn scores_are_relative_to_the_acting_team() {
        let state = Rig::new().scores(300, 1300).build();
        // Seat 0 is team 0: "my" score 300 crosses >=250 only; "their" 1300
        // crosses >=250..>=1250 (5 thresholds) but not >=1500.
        let out = encoded(&state, 0);
        assert_eq!(out[350..370].iter().sum::<f32>(), 1.0);
        assert_eq!(out[370..390].iter().sum::<f32>(), 5.0);
        // From seat 1 (team 1) the blocks swap.
        let theirs = encoded(&state, 1);
        assert_eq!(theirs[350..370].iter().sum::<f32>(), 5.0);
        assert_eq!(theirs[370..390].iter().sum::<f32>(), 1.0);
    }

    #[test]
    fn the_opening_minimum_is_a_three_way_one_hot() {
        let opened = Rig::new().opened(0).build();
        assert_eq!(&encoded(&opened, 0)[392..395], &[1.0, 0.0, 0.0]);
        let low = Rig::new().build();
        assert_eq!(&encoded(&low, 0)[392..395], &[0.0, 1.0, 0.0]);
        // The acting seat's own team at 2500 is what raises the bar to 120.
        let high = Rig::new().scores(2500, 0).build();
        assert_eq!(&encoded(&high, 0)[392..395], &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn the_pile_has_a_top_a_size_and_a_census() {
        let state = Rig::new().discard("9C 6D 6D").build();
        let out = encoded(&state, 1);
        // Top is 6D = 15, one-hot at 407.
        assert_eq!(out[407 + 15], 1.0);
        assert_eq!(out[407..460].iter().sum::<f32>(), 1.0);
        // Size 3 crosses >=2 only of the >=2,4,...,30 thermometer at 460.
        assert_eq!(out[460..475].iter().sum::<f32>(), 1.0);
        // Census at 475: 9C once (Clubs 0 * 13 + Nine 5 = 5; >=1 at 475+10,
        // >=2 at 475+11), 6D twice (15; >=1 at 475+30, >=2 at 475+31).
        assert_eq!(out[475 + 10], 1.0, "9C >=1");
        assert_eq!(out[475 + 11], 0.0, "9C >=2");
        assert_eq!(out[475 + 30], 1.0, "6D >=1");
        assert_eq!(out[475 + 31], 1.0, "6D >=2");
    }
}
