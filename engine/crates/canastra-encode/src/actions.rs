//! One feature row per legal action.
//!
//! Block layout (offsets are the contract):
//!
//! | offset | width | block |
//! |---|---|---|
//! | 0  | 8  | kind one-hot: Draw, KeepDrawnCard, RefuseDrawnCard, TakeDiscardPile, LayMeld, AddToMeld, Discard, EndTurnWithoutDiscard |
//! | 8  | 18 | primary card: rank one-hot 13 (game order) + suit one-hot 4 + is-joker |
//! | 26 | 28 | meld descriptor: suit 4 + low-rank 11 + length therm 10 + wild-present + wild-is-joker + is-aces |
//! | 54 | 34 | target: pool-token one-hot 33 + new-meld bit |
//! | 88 | 10 | points thermometer >=5..>=50 step 5 |
//! | 98 | 3  | opening context: reaches-minimum, exceeds-by->=25, exceeds-by->=50 |
//!
//! The target block's token index uses the same canonical pool ordering as
//! the observation's table segment, so the policy can attend to "the meld
//! this action touches" by position. Distinct legal actions may rarely
//! collide to identical rows; either choice is then fine.

use canastra_engine::action::MeldTarget;
use canastra_engine::card::{Card, Rank};
use canastra_engine::{Action, PlayerView};

use crate::ACT_DIM;
use crate::cards::{rank_index, suit_index};
use crate::obs::Writer;
use crate::tokens;

/// Write one [`ACT_DIM`] row per action in `legal`.
///
/// # Panics
///
/// If `out` is not exactly `legal.len() * ACT_DIM` — see principle 1.
pub fn encode_actions(view: &PlayerView, legal: &[Action], out: &mut [f32]) {
    assert_eq!(out.len(), legal.len() * ACT_DIM);
    out.fill(0.0);
    for (row, action) in legal.iter().enumerate() {
        let mut w = Writer {
            out: &mut out[row * ACT_DIM..(row + 1) * ACT_DIM],
            at: 0,
        };
        encode_action(view, action, &mut w);
        w.finish();
    }
}

fn encode_action(view: &PlayerView, action: &Action, w: &mut Writer) {
    w.one_hot(8, Some(kind_index(action)));

    match action {
        Action::Discard { card } => write_card(*card, w),
        Action::AddToMeld { cards, .. } => write_card(cards[0], w),
        Action::KeepDrawnCard | Action::RefuseDrawnCard => match view.pending_refusal {
            Some(card) => write_card(card, w),
            None => write_no_card(w),
        },
        _ => write_no_card(w),
    }

    match action {
        Action::LayMeld { cards } => describe(cards, w),
        Action::TakeDiscardPile { core, .. } => {
            let mut shape = core.to_vec();
            if let Some(&top) = view.discard.last() {
                shape.push(top);
            }
            describe(&shape, w);
        }
        _ => describe(&[], w),
    }

    let token = match action {
        Action::AddToMeld { meld, .. } => Some(tokens::target_index(view, *meld)),
        Action::TakeDiscardPile {
            target: MeldTarget::Existing { meld },
            ..
        } => Some(tokens::target_index(view, *meld)),
        _ => None,
    };
    w.one_hot(33, token);
    w.bit(matches!(
        action,
        Action::TakeDiscardPile {
            target: MeldTarget::NewMeld,
            ..
        }
    ));

    let points = involved_points(view, action);
    w.therm(points, &[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]);

    let my_table = &view.tables[view.seat.team().index()];
    let opening = !my_table.opened && view.laid_value < view.opening_minimum;
    w.bit(opening && view.laid_value + points >= view.opening_minimum);
    w.bit(opening && view.laid_value + points >= view.opening_minimum + 25);
    w.bit(opening && view.laid_value + points >= view.opening_minimum + 50);
}

fn kind_index(action: &Action) -> usize {
    match action {
        Action::Draw => 0,
        Action::KeepDrawnCard => 1,
        Action::RefuseDrawnCard => 2,
        Action::TakeDiscardPile { .. } => 3,
        Action::LayMeld { .. } => 4,
        Action::AddToMeld { .. } => 5,
        Action::Discard { .. } => 6,
        Action::EndTurnWithoutDiscard => 7,
    }
}

/// The primary-card block: rank one-hot 13 (game order), suit one-hot 4,
/// is-joker. A Joker has no rank or suit — only the flag.
fn write_card(card: Card, w: &mut Writer) {
    w.one_hot(13, card.rank().map(rank_index));
    w.one_hot(4, card.suit().map(suit_index));
    w.bit(card.is_joker());
}

fn write_no_card(w: &mut Writer) {
    w.one_hot(13, None);
    w.one_hot(4, None);
    w.bit(false);
}

/// The meld-descriptor block for a card set (a LayMeld's cards, or a pile
/// capture's core plus the top card). Shape facts only — the engine has
/// already judged legality.
fn describe(cards: &[Card], w: &mut Writer) {
    let naturals: Vec<Card> = cards.iter().copied().filter(|c| c.is_natural()).collect();
    let is_aces = !naturals.is_empty() && naturals.iter().all(|c| c.rank() == Some(Rank::Ace));
    let suit = naturals.first().and_then(|c| c.suit());
    let low = naturals
        .iter()
        .filter_map(|c| c.rank().and_then(|r| r.sequence_index()))
        .min();
    w.one_hot(4, suit.map(suit_index));
    w.one_hot(11, low.map(|i| i as usize));
    w.therm(cards.len() as u32, &[3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    let wild = cards.iter().copied().find(|c| c.is_wild());
    w.bit(wild.is_some());
    w.bit(wild.is_some_and(|c| c.is_joker()));
    w.bit(is_aces);
}

/// Face-value points of the cards the action moves.
fn involved_points(view: &PlayerView, action: &Action) -> u32 {
    match action {
        Action::Discard { card } => card.points(),
        Action::AddToMeld { cards, .. } | Action::LayMeld { cards } => {
            cards.iter().map(|c| c.points()).sum()
        }
        Action::TakeDiscardPile { core, .. } => {
            core.iter().map(|c| c.points()).sum::<u32>()
                + view.discard.last().map_or(0, |c| c.points())
        }
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use canastra_engine::testkit::{Rig, card, cards};
    use canastra_engine::{Seat, enumerate, observe};

    fn features(view: &PlayerView, action: &Action) -> Vec<f32> {
        let mut out = vec![0.0; ACT_DIM];
        encode_actions(view, std::slice::from_ref(action), &mut out);
        out
    }

    #[test]
    fn the_kind_is_a_one_hot_in_declaration_order() {
        let view = observe(&Rig::new().build(), Seat::new(1).unwrap());
        assert_eq!(
            features(&view, &Action::Draw)[0..8],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            features(&view, &Action::Discard { card: card("3S") })[0..8],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        );
        assert_eq!(
            features(&view, &Action::EndTurnWithoutDiscard)[0..8],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn a_discard_carries_its_card_and_points() {
        let view = observe(&Rig::new().build(), Seat::new(1).unwrap());
        let out = features(&view, &Action::Discard { card: card("KH") });
        // KH: rank King = 9 (block at 8..21), suit Hearts = 2 (21..25), not joker.
        assert_eq!(out[8 + 9], 1.0);
        assert_eq!(out[21 + 2], 1.0);
        assert_eq!(out[25], 0.0);
        // Points 10: thermometer >=5..>=50 step 5 at 88..98 crosses 5 and 10.
        assert_eq!(&out[88..90], &[1.0, 1.0]);
        assert_eq!(out[90..98].iter().sum::<f32>(), 0.0);
    }

    #[test]
    fn a_lay_meld_carries_its_shape() {
        let view = observe(&Rig::new().build(), Seat::new(1).unwrap());
        let out = features(
            &view,
            &Action::LayMeld {
                cards: cards("4H 5H JOKER"),
            },
        );
        // meld descriptor at 26: suit hearts (26+2), low 4 (30+0), length
        // therm at 41..51 (len 3 crosses >=3 only), wild present (51), wild
        // is joker (52), not aces (53).
        assert_eq!(out[26 + 2], 1.0);
        assert_eq!(out[30], 1.0);
        assert_eq!(&out[41..43], &[1.0, 0.0]);
        assert_eq!(out[51], 1.0);
        assert_eq!(out[52], 1.0);
        assert_eq!(out[53], 0.0);
    }

    #[test]
    fn an_add_to_meld_points_at_its_pool_token() {
        let state = Rig::new()
            .meld(0, "AH AD AS")
            .meld(1, "9S TS JS")
            .meld(1, "4H 5H 6H")
            .build();
        let view = observe(&state, Seat::new(1).unwrap());
        // Team-local meld 0 is the 9S run -> pool token 1 (mine sorted: 4H first).
        let out = features(
            &view,
            &Action::AddToMeld {
                meld: 0,
                cards: cards("QC"),
            },
        );
        assert_eq!(out[54 + 1], 1.0, "token one-hot at 54..87");
        assert_eq!(out[54..88].iter().sum::<f32>(), 1.0);
        assert_eq!(out[87], 0.0, "not new");
    }

    #[test]
    fn the_opening_context_marks_reaching_the_minimum() {
        let state = Rig::new().laid_value(60).build(); // minimum 75
        let view = observe(&state, Seat::new(1).unwrap());
        let out = features(
            &view,
            &Action::LayMeld {
                cards: cards("AH AD AS"),
            },
        ); // 45 points
        // reaches (60+45 >= 75), exceeds-by-25 (>= 100), not exceeds-by-50.
        assert_eq!(&out[98..101], &[1.0, 1.0, 0.0]);
        let small = features(&view, &Action::Discard { card: card("4H") }); // 5 points
        assert_eq!(&small[98..101], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn featurizing_the_real_legal_list_never_panics() {
        let state = Rig::new()
            .stock("8C 9D")
            .discard("9C 6D")
            .hand(1, "4D 5D 7D 8D KH")
            .meld(1, "7D 8D 9D")
            .build();
        let view = observe(&state, Seat::new(1).unwrap());
        let legal = enumerate(&state, Seat::new(1).unwrap());
        let mut out = vec![0.0; legal.len() * ACT_DIM];
        encode_actions(&view, &legal, &mut out);
        assert!(out.iter().all(|x| *x == 0.0 || *x == 1.0));
    }
}
