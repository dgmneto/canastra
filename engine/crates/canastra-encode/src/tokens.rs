//! The table as a shared pool of 33 meld tokens.
//!
//! 33 is the exact bound: the 8 threes can never appear in a meld, leaving
//! 100 meldable cards, and every meld holds at least 3. One pool with an
//! ownership bit, rather than per-team allocations — the network reads both
//! sides' tables through the same feature shape.
//!
//! Tokens are a concatenated list, so slot order matters. The canonical sort
//! (my team first, then theirs; each side by kind, suit, low rank, length) is
//! the mitigation for order sensitivity. Overflow is unreachable in a legal
//! game; the tail is dropped deterministically and the fuzz asserts it never
//! happens.

use canastra_engine::PlayerView;
use canastra_engine::meld::{CanastraKind, MAX_SEQUENCE_LEN, Meld};

use crate::cards::suit_index;
use crate::obs::Writer;

pub(crate) const MAX_TOKENS: usize = 33;
pub(crate) const TOKEN_DIM: usize = 43;

/// Where a pooled meld came from, so action targets can be mapped back.
pub(crate) struct TokenSource {
    pub mine: bool,
    pub meld: usize,
}

/// Both tables' melds in canonical pool order.
pub(crate) fn sorted_tokens(view: &PlayerView) -> Vec<(&Meld, TokenSource)> {
    let mine = view.seat.team().index();
    let mut tokens: Vec<(&Meld, TokenSource)> = Vec::new();
    for (side, &table_index) in [mine, 1 - mine].iter().enumerate() {
        let table = &view.tables[table_index];
        let mut owned: Vec<(&Meld, TokenSource)> = table
            .melds
            .iter()
            .enumerate()
            .map(|(meld, m)| {
                (
                    m,
                    TokenSource {
                        mine: side == 0,
                        meld,
                    },
                )
            })
            .collect();
        owned.sort_by_key(|(meld, _)| sort_key(meld));
        tokens.extend(owned);
    }
    tokens
}

/// The canonical order within one partnership: sequences (by suit, then low
/// rank, then length) before ace melds.
fn sort_key(meld: &Meld) -> (u8, u8, u8, u8) {
    match meld {
        Meld::Sequence(seq) => (
            0,
            suit_index(seq.suit()) as u8,
            seq.low().sequence_index().unwrap_or(0),
            seq.len() as u8,
        ),
        Meld::Aces(aces) => (1, 0, 0, aces.len() as u8),
    }
}

/// The pool position of a meld addressed the engine's way — per-partnership
/// index on the acting team's table. Test-only: the hot path uses
/// [`target_index_in`] over a pool built once per ply.
#[cfg(test)]
pub(crate) fn target_index(view: &PlayerView, meld: usize) -> usize {
    target_index_in(&sorted_tokens(view), meld)
}

/// As `target_index`, but over a pool the caller already built — the hot
/// path builds it once per ply, not once per targeted action.
pub(crate) fn target_index_in(tokens: &[(&Meld, TokenSource)], meld: usize) -> usize {
    tokens
        .iter()
        .position(|(_, source)| source.mine && source.meld == meld)
        .expect("every meld on the acting team's table is pooled")
}

/// Write the 33 x 43 token block.
pub(crate) fn write_tokens(view: &PlayerView, w: &mut Writer) {
    let tokens = sorted_tokens(view);
    debug_assert!(
        tokens.len() <= MAX_TOKENS,
        "more melds than the 33-token pool: unreachable in a legal game"
    );
    for slot in 0..MAX_TOKENS {
        match tokens.get(slot) {
            None => {
                for _ in 0..TOKEN_DIM {
                    w.bit(false);
                }
            }
            Some((meld, source)) => write_token(meld, source.mine, w),
        }
    }
}

/// Per-token features (43): present, my-team, kind (2), suit (4), low rank
/// over the 11 sequence ranks, length thermometer >=3..>=12 (10), wild
/// present, wild locked, is-canastra, tier one-hot (4), extendable low/high,
/// points thermometer >=25/50/75/100/150 (5).
fn write_token(meld: &Meld, mine: bool, w: &mut Writer) {
    w.bit(true);
    w.bit(mine);
    let is_sequence = matches!(meld, Meld::Sequence(_));
    w.bit(is_sequence);
    w.bit(!is_sequence);

    match meld {
        Meld::Sequence(seq) => {
            w.one_hot(4, Some(suit_index(seq.suit())));
            w.one_hot(11, seq.low().sequence_index().map(|i| i as usize));
        }
        Meld::Aces(_) => {
            w.one_hot(4, None);
            w.one_hot(11, None);
        }
    }

    w.therm(meld.len() as u32, &[3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

    let (wild_present, wild_locked) = match meld {
        Meld::Sequence(seq) => (seq.wild_index().is_some(), seq.wild_is_locked()),
        // §9's locking is a sequence rule; an ace meld's wild never locks.
        Meld::Aces(aces) => (aces.len() > aces.natural_aces(), false),
    };
    w.bit(wild_present);
    w.bit(wild_locked);

    let tier = meld.canastra();
    w.bit(tier != CanastraKind::None);
    w.bit(tier == CanastraKind::None);
    w.bit(tier == CanastraKind::Dirty);
    w.bit(tier == CanastraKind::Clean);
    w.bit(tier == CanastraKind::CleanAces);

    let (extendable_low, extendable_high) = match meld {
        Meld::Sequence(seq) => (
            seq.low().sequence_index() > Some(0) && seq.len() < MAX_SEQUENCE_LEN,
            seq.high().sequence_index() < Some(10) && seq.len() < MAX_SEQUENCE_LEN,
        ),
        // §7.2: aces extend until 8 naturals plus the one wild; nothing low.
        Meld::Aces(aces) => (false, aces.len() < 9),
    };
    w.bit(extendable_low);
    w.bit(extendable_high);

    w.therm(meld.card_points(), &[25, 50, 75, 100, 150]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use canastra_engine::testkit::Rig;
    use canastra_engine::{Seat, observe};

    fn encoded(view: &PlayerView) -> Vec<f32> {
        let mut out = vec![0.0; 33 * TOKEN_DIM];
        let mut w = Writer {
            out: &mut out,
            at: 0,
        };
        write_tokens(view, &mut w);
        out
    }

    fn token(out: &[f32], index: usize) -> &[f32] {
        &out[index * TOKEN_DIM..(index + 1) * TOKEN_DIM]
    }

    #[test]
    fn my_melds_come_first_then_theirs_each_sorted() {
        let state = Rig::new()
            .meld(1, "9S TS JS") // mine (seat 1 -> team 1)
            .meld(1, "4H 5H 6H")
            .meld(0, "AH AD AS") // theirs
            .build();
        let view = observe(&state, Seat::new(1).unwrap());
        let out = encoded(&view);
        // Mine sorted by (kind, suit, low): the hearts run before the spades run.
        let first = token(&out, 0);
        assert_eq!(first[0], 1.0, "present");
        assert_eq!(first[1], 1.0, "mine");
        assert_eq!(first[2], 1.0, "sequence");
        assert_eq!(first[4 + 2], 1.0, "hearts");
        assert_eq!(first[8], 1.0, "low = 4 (sequence index 0)");
        let second = token(&out, 1);
        assert_eq!(second[4 + 3], 1.0, "spades");
        assert_eq!(second[8 + 5], 1.0, "low = 9 (sequence index 5)");
        // Their ace meld third: kind = aces, suit and low rank zero.
        let third = token(&out, 2);
        assert_eq!(third[1], 0.0, "not mine");
        assert_eq!(third[3], 1.0, "aces kind");
        assert_eq!(third[4..8].iter().sum::<f32>(), 0.0, "no suit");
        assert_eq!(third[8..19].iter().sum::<f32>(), 0.0, "no low rank");
        // Everything after is absent.
        assert_eq!(token(&out, 3).iter().sum::<f32>(), 0.0);
    }

    #[test]
    fn a_locked_wild_is_marked() {
        // 6H 7H 2H 9H: the 2 stands in for the 8, naturals on both sides (§9).
        let state = Rig::new().meld(1, "6H 7H 2H 9H").build();
        let view = observe(&state, Seat::new(1).unwrap());
        let out = encoded(&view);
        let first = token(&out, 0);
        assert_eq!(first[29], 1.0, "wild present");
        assert_eq!(first[30], 1.0, "wild locked");
        assert_eq!(
            first[19..29].iter().sum::<f32>(),
            2.0,
            "len 4 crosses >=3 and >=4"
        );
    }

    #[test]
    fn canastra_tiers_are_a_one_hot() {
        let state = Rig::new()
            .meld(1, "4H 5H 6H 7H 8H 9H TH") // clean
            .meld(1, "4S 5S 6S 7S 8S 9S 2S") // dirty
            .meld(0, "AH AD AS AC AH AD AS") // clean aces
            .build();
        let view = observe(&state, Seat::new(1).unwrap());
        let out = encoded(&view);
        // My two sequences sort by suit index: hearts (2) before spades (3).
        assert_eq!(&token(&out, 0)[32..36], &[0.0, 0.0, 1.0, 0.0], "clean");
        assert_eq!(&token(&out, 1)[32..36], &[0.0, 1.0, 0.0, 0.0], "dirty");
        assert_eq!(&token(&out, 2)[32..36], &[0.0, 0.0, 0.0, 1.0], "clean aces");
        for index in 0..3 {
            assert_eq!(token(&out, index)[31], 1.0, "is-canastra");
        }
    }

    #[test]
    fn extendability_is_supplied_not_inferred() {
        let state = Rig::new()
            .meld(1, "4H 5H 6H") // low end blocked (4 is the floor)
            .meld(1, "JH QH KH AH") // high end blocked (A is the cap)
            .meld(1, "7D 8D 9D") // both ends open
            .build();
        let view = observe(&state, Seat::new(1).unwrap());
        let out = encoded(&view);
        // Sorted by suit: diamonds (1) first, then hearts (2) by low rank.
        assert_eq!(
            &token(&out, 0)[36..38],
            &[1.0, 1.0],
            "7D 8D 9D extends both ways"
        );
        assert_eq!(
            &token(&out, 1)[36..38],
            &[0.0, 1.0],
            "4H 5H 6H extends high only"
        );
        assert_eq!(
            &token(&out, 2)[36..38],
            &[1.0, 0.0],
            "JH QH KH AH extends low only"
        );
    }

    #[test]
    fn target_index_maps_a_team_local_meld_into_the_pool() {
        let state = Rig::new()
            .meld(0, "AH AD AS")
            .meld(1, "9S TS JS")
            .meld(1, "4H 5H 6H")
            .build();
        let view = observe(&state, Seat::new(1).unwrap());
        // Rig pushes in call order, so tables[1].melds == [9S run, 4H run].
        // The canonical pool orders mine as [4H run, 9S run].
        assert_eq!(target_index(&view, 0), 1, "team-local 0 (9S) -> pool 1");
        assert_eq!(target_index(&view, 1), 0, "team-local 1 (4H) -> pool 0");
    }
}
