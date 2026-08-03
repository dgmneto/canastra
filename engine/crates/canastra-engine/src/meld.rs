//! Melds: sequences, ace melds, and the wild-card placement rules of §7–§10.

use crate::card::{Card, Rank, Suit};
use std::collections::BTreeMap;
use std::fmt;

/// §7.1: the longest possible sequence runs 4 through Ace.
pub const MAX_SEQUENCE_LEN: usize = 11;

/// Bonus threshold — §10 defines a canastra as a meld of seven cards or more.
pub const CANASTRA_LEN: usize = 7;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeldError {
    /// §7: every meld is at least three cards.
    TooFewCards,
    /// §8: "cada jogo pode conter no máximo um curinga".
    TooManyWilds,
    /// §7.1: a sequence is single-suited.
    MixedSuits,
    /// The cards do not form a contiguous run, or include a rank (2, 3) that
    /// has no place in a sequence.
    NotASequence,
    /// §7.1: "não pode haver carta repetida na mesma sequência".
    DuplicateRank,
    /// The run would stretch past the eleven ranks between 4 and Ace.
    TooLong,
    /// §9: the wild has naturals on both sides and can never move again, so the
    /// card being added has nowhere to go.
    WildLocked,
    /// §8: "apenas o 2♥ pode entrar numa sequência de copas".
    WrongSuitForTwo,
    /// §7.2: an ace meld collects aces, and takes nothing else but a wild.
    NotAnAce,
}

impl fmt::Display for MeldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            MeldError::TooFewCards => "a meld needs at least three cards",
            MeldError::TooManyWilds => "a meld may hold at most one wild card",
            MeldError::MixedSuits => "a sequence is single-suited",
            MeldError::NotASequence => "these cards do not form a run",
            MeldError::DuplicateRank => "that rank is already in the sequence",
            MeldError::TooLong => "a sequence cannot stretch past 4 through Ace",
            MeldError::WildLocked => "the wild card is locked and cannot move",
            MeldError::WrongSuitForTwo => "a 2 only joins a sequence of its own suit",
            MeldError::NotAnAce => "only aces and wild cards join an ace meld",
        };
        f.write_str(message)
    }
}

impl std::error::Error for MeldError {}

/// One position in a sequence. Every position is filled, so a sequence is
/// contiguous by construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slot {
    Natural(Card),
    Wild(Card),
}

impl Slot {
    pub fn card(self) -> Card {
        match self {
            Slot::Natural(card) | Slot::Wild(card) => card,
        }
    }

    pub fn is_wild(self) -> bool {
        matches!(self, Slot::Wild(_))
    }
}

/// §10 bonus tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanastraKind {
    /// Fewer than seven cards — scores its cards, but no bonus.
    None,
    /// Seven or more cards including a 2.
    Dirty,
    /// Seven or more cards with no 2. A Joker is fine.
    Clean,
    /// Seven or more aces of which at least seven are real.
    CleanAces,
}

impl CanastraKind {
    pub fn bonus(self) -> u32 {
        match self {
            CanastraKind::None => 0,
            CanastraKind::Dirty => 200,
            CanastraKind::Clean => 500,
            CanastraKind::CleanAces => 1000,
        }
    }

    pub fn is_clean(self) -> bool {
        matches!(self, CanastraKind::Clean | CanastraKind::CleanAces)
    }
}

/// §7.1: three or more cards in sequence, same suit, between 4 and Ace.
///
/// Stored as a contiguous, fully-populated slot array where `slots[i]` covers
/// rank `low + i`. That representation is what makes §9's locking rule trivial:
/// since a sequence has no holes and holds at most one wild, a wild has naturals
/// on both sides exactly when it sits at an interior index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sequence {
    suit: Suit,
    low: u8,
    slots: Vec<Slot>,
}

impl Sequence {
    pub fn suit(&self) -> Suit {
        self.suit
    }

    pub fn low(&self) -> Rank {
        Rank::from_sequence_index(self.low).expect("a sequence starts on a real rank")
    }

    pub fn high(&self) -> Rank {
        Rank::from_sequence_index(self.low + self.slots.len() as u8 - 1)
            .expect("a sequence ends on a real rank")
    }

    #[allow(clippy::len_without_is_empty, reason = "a meld always holds 3+ cards")]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn slots(&self) -> &[Slot] {
        &self.slots
    }

    pub fn cards(&self) -> impl Iterator<Item = Card> + '_ {
        self.slots.iter().map(|slot| slot.card())
    }

    pub fn wild_index(&self) -> Option<usize> {
        self.slots.iter().position(|slot| slot.is_wild())
    }

    /// The rank the wild is currently standing in for — display and adjacency
    /// only. A wild always scores its own face value (§13.2).
    pub fn wild_represents(&self) -> Option<Rank> {
        self.wild_index()
            .map(|index| self.low + index as u8)
            .and_then(Rank::from_sequence_index)
    }

    /// §9: "Um curinga está trancado quando tem cartas naturais dos dois lados."
    ///
    /// Slots are contiguous and only one wild is allowed, so both neighbours are
    /// necessarily naturals whenever the wild is not on an end.
    pub fn wild_is_locked(&self) -> bool {
        match self.wild_index() {
            Some(index) => index > 0 && index < self.slots.len() - 1,
            None => false,
        }
    }

    pub fn card_points(&self) -> u32 {
        self.cards().map(|card| card.points()).sum()
    }

    pub fn canastra(&self) -> CanastraKind {
        if self.len() < CANASTRA_LEN {
            return CanastraKind::None;
        }
        if self.cards().any(|card| card.rank() == Some(Rank::Two)) {
            CanastraKind::Dirty
        } else {
            CanastraKind::Clean
        }
    }

    fn build(naturals: &[Card], wild: Option<Card>) -> Result<Sequence, MeldError> {
        let suit = naturals[0].suit().expect("naturals are standard cards");
        if naturals.iter().any(|card| card.suit() != Some(suit)) {
            return Err(MeldError::MixedSuits);
        }
        check_two_suit(wild, suit)?;

        let mut positions = BTreeMap::new();
        for &card in naturals {
            let index = card
                .rank()
                .and_then(Rank::sequence_index)
                .ok_or(MeldError::NotASequence)?;
            if positions.insert(index, card).is_some() {
                return Err(MeldError::DuplicateRank);
            }
        }
        assemble(suit, positions, wild, None)
    }

    /// Lay one more card onto this sequence, sliding a free wild if that is what
    /// it takes. Leaves the sequence untouched when the card cannot be placed.
    pub fn add_card(&mut self, card: Card) -> Result<(), MeldError> {
        let mut positions: BTreeMap<u8, Card> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(offset, slot)| match *slot {
                Slot::Natural(natural) => Some((self.low + offset as u8, natural)),
                Slot::Wild(_) => None,
            })
            .collect();

        let held_wild = self.wild_index().map(|index| self.slots[index].card());

        let (wild, pinned) = if card.is_wild() {
            if held_wild.is_some() {
                return Err(MeldError::TooManyWilds);
            }
            check_two_suit(Some(card), self.suit)?;
            (Some(card), None)
        } else {
            if card.suit() != Some(self.suit) {
                return Err(MeldError::MixedSuits);
            }
            let index = card
                .rank()
                .and_then(Rank::sequence_index)
                .ok_or(MeldError::NotASequence)?;
            if positions.insert(index, card).is_some() {
                return Err(MeldError::DuplicateRank);
            }
            // A locked wild is frozen at the rank it already represents.
            let pinned = self
                .wild_is_locked()
                .then(|| self.low + self.wild_index().expect("locked implies present") as u8);
            (held_wild, pinned)
        };

        let rebuilt = assemble(self.suit, positions, wild, pinned)?;
        *self = rebuilt;
        Ok(())
    }
}

/// §8: a 2 only stands in within its own suit. The Joker has no suit and goes
/// anywhere.
fn check_two_suit(wild: Option<Card>, suit: Suit) -> Result<(), MeldError> {
    match wild {
        Some(card) if card.rank() == Some(Rank::Two) && card.suit() != Some(suit) => {
            Err(MeldError::WrongSuitForTwo)
        }
        _ => Ok(()),
    }
}

/// Place the naturals and the wild into a contiguous run.
///
/// `pinned` names the rank index a locked wild is nailed to; a free wild is
/// positioned by this function, which is exactly the sliding of §9.
fn assemble(
    suit: Suit,
    positions: BTreeMap<u8, Card>,
    wild: Option<Card>,
    pinned: Option<u8>,
) -> Result<Sequence, MeldError> {
    let lowest = *positions.keys().next().ok_or(MeldError::TooFewCards)?;
    let highest = *positions.keys().next_back().expect("just proved non-empty");

    let (low, wild_at) = match (wild, pinned) {
        // Locked: the wild cannot move, so the run has to close around it.
        (Some(_), Some(pin)) => {
            if positions.contains_key(&pin) {
                return Err(MeldError::WildLocked);
            }
            let low = lowest.min(pin);
            let high = highest.max(pin);
            if (low..=high).any(|index| index != pin && !positions.contains_key(&index)) {
                return Err(MeldError::NotASequence);
            }
            (low, Some(pin))
        }
        // Free: the wild fills the single gap, or caps an end.
        (Some(_), None) => {
            let gaps: Vec<u8> = (lowest..=highest)
                .filter(|index| !positions.contains_key(index))
                .collect();
            match gaps.as_slice() {
                [] => {
                    // Both ends accept the same future cards while the wild stays
                    // free, so store it high and only fall back when the Ace is
                    // already taken (§9: `J-Q-K-A-Coringa` is not a thing).
                    if (highest as usize) + 1 < MAX_SEQUENCE_LEN {
                        (lowest, Some(highest + 1))
                    } else if lowest >= 1 {
                        (lowest - 1, Some(lowest - 1))
                    } else {
                        return Err(MeldError::TooLong);
                    }
                }
                [gap] => (lowest, Some(*gap)),
                _ => return Err(MeldError::NotASequence),
            }
        }
        (None, _) => {
            if (lowest..=highest).any(|index| !positions.contains_key(&index)) {
                return Err(MeldError::NotASequence);
            }
            (lowest, None)
        }
    };

    let high = wild_at.map_or(highest, |index| highest.max(index));
    if usize::from(high - low + 1) > MAX_SEQUENCE_LEN {
        return Err(MeldError::TooLong);
    }

    let slots = (low..=high)
        .map(|index| match wild_at {
            Some(at) if at == index => Slot::Wild(wild.expect("wild_at implies a wild")),
            _ => Slot::Natural(positions[&index]),
        })
        .collect();

    Ok(Sequence { suit, low, slots })
}

/// §7.2: a set of aces, any suits. Only eight aces exist across the two decks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcesMeld {
    aces: Vec<Card>,
    wild: Option<Card>,
}

impl AcesMeld {
    #[allow(clippy::len_without_is_empty, reason = "a meld always holds 3+ cards")]
    pub fn len(&self) -> usize {
        self.aces.len() + usize::from(self.wild.is_some())
    }

    /// How many of the cards are real aces, which is what the 1000 turns on.
    pub fn natural_aces(&self) -> usize {
        self.aces.len()
    }

    pub fn cards(&self) -> impl Iterator<Item = Card> + '_ {
        self.aces.iter().copied().chain(self.wild)
    }

    pub fn card_points(&self) -> u32 {
        self.cards().map(|card| card.points()).sum()
    }

    /// Lay one more card onto this ace meld.
    ///
    /// §8: unlike a sequence, an ace meld accepts a 2 of any suit — the
    /// own-suit restriction only exists because a 2 in a sequence stands in for
    /// a specific card, and here it stands in for nothing.
    pub fn add_card(&mut self, card: Card) -> Result<(), MeldError> {
        if card.is_wild() {
            if self.wild.is_some() {
                return Err(MeldError::TooManyWilds);
            }
            self.wild = Some(card);
            return Ok(());
        }
        if card.rank() != Some(Rank::Ace) {
            return Err(MeldError::NotAnAce);
        }
        self.aces.push(card);
        Ok(())
    }

    /// §10, plus CLAUDE.md clarification #1: a 2 caps the meld at the dirty tier
    /// no matter how many real aces sit beside it. The 1000 needs seven of them;
    /// a Joker rides along as an extra card but never substitutes for an ace.
    pub fn canastra(&self) -> CanastraKind {
        if self.len() < CANASTRA_LEN {
            return CanastraKind::None;
        }
        if self.wild.and_then(Card::rank) == Some(Rank::Two) {
            return CanastraKind::Dirty;
        }
        if self.natural_aces() >= CANASTRA_LEN {
            CanastraKind::CleanAces
        } else {
            CanastraKind::Clean
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Meld {
    Sequence(Sequence),
    Aces(AcesMeld),
}

impl Meld {
    /// Build a meld from the cards laid down together.
    ///
    /// A meld whose naturals are all aces is an aces meld (§7.2); anything else
    /// is read as a sequence, so an ace closing a run — `Q♠ K♠ A♠` — stays a
    /// sequence rather than being mistaken for the start of an ace collection.
    pub fn new(cards: &[Card]) -> Result<Meld, MeldError> {
        if cards.len() < 3 {
            return Err(MeldError::TooFewCards);
        }

        let mut wilds = cards.iter().copied().filter(|card| card.is_wild());
        let wild = wilds.next();
        if wilds.next().is_some() {
            return Err(MeldError::TooManyWilds);
        }

        let naturals: Vec<Card> = cards
            .iter()
            .copied()
            .filter(|card| card.is_natural())
            .collect();
        if naturals.iter().all(|card| card.rank() == Some(Rank::Ace)) {
            return Ok(Meld::Aces(AcesMeld {
                aces: naturals,
                wild,
            }));
        }
        Sequence::build(&naturals, wild).map(Meld::Sequence)
    }

    #[allow(clippy::len_without_is_empty, reason = "a meld always holds 3+ cards")]
    pub fn len(&self) -> usize {
        match self {
            Meld::Sequence(sequence) => sequence.len(),
            Meld::Aces(aces) => aces.len(),
        }
    }

    pub fn cards(&self) -> Vec<Card> {
        match self {
            Meld::Sequence(sequence) => sequence.cards().collect(),
            Meld::Aces(aces) => aces.cards().collect(),
        }
    }

    /// §13.1: the cards of a meld score their own face values on top of any bonus.
    pub fn card_points(&self) -> u32 {
        match self {
            Meld::Sequence(sequence) => sequence.card_points(),
            Meld::Aces(aces) => aces.card_points(),
        }
    }

    pub fn canastra(&self) -> CanastraKind {
        match self {
            Meld::Sequence(sequence) => sequence.canastra(),
            Meld::Aces(aces) => aces.canastra(),
        }
    }

    /// Lay one more card onto this meld. The meld is left untouched on failure.
    pub fn add_card(&mut self, card: Card) -> Result<(), MeldError> {
        match self {
            Meld::Sequence(sequence) => sequence.add_card(card),
            Meld::Aces(aces) => aces.add_card(card),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::{Card, Rank};

    fn cards(spec: &str) -> Vec<Card> {
        spec.split_whitespace()
            .map(|c| c.parse().expect("test card literal"))
            .collect()
    }

    fn meld(spec: &str) -> Meld {
        Meld::new(&cards(spec)).expect("test meld literal")
    }

    fn sequence(spec: &str) -> Sequence {
        match meld(spec) {
            Meld::Sequence(sequence) => sequence,
            Meld::Aces(_) => panic!("{spec} built an aces meld, not a sequence"),
        }
    }

    // ---- construction (§7) ----

    #[test]
    fn a_meld_needs_at_least_three_cards() {
        assert_eq!(Meld::new(&cards("6H 7H")), Err(MeldError::TooFewCards));
    }

    #[test]
    fn sequences_are_single_suited() {
        assert_eq!(Meld::new(&cards("6H 7S 8H")), Err(MeldError::MixedSuits));
    }

    #[test]
    fn sequences_must_be_contiguous() {
        assert_eq!(Meld::new(&cards("6H 7H 9H")), Err(MeldError::NotASequence));
    }

    #[test]
    fn sequences_reject_repeated_ranks() {
        assert_eq!(Meld::new(&cards("6H 6H 7H")), Err(MeldError::DuplicateRank));
    }

    /// §7.1: "2s e 3s nunca entram em sequências".
    #[test]
    fn threes_never_enter_sequences() {
        assert_eq!(Meld::new(&cards("3H 4H 5H")), Err(MeldError::NotASequence));
    }

    /// §8: at most one wild per meld, Joker or 2, never two.
    #[test]
    fn a_meld_holds_at_most_one_wild() {
        assert_eq!(
            Meld::new(&cards("6H 7H JOKER JOKER")),
            Err(MeldError::TooManyWilds)
        );
        assert_eq!(
            Meld::new(&cards("6H 7H JOKER 2H")),
            Err(MeldError::TooManyWilds)
        );
    }

    /// §8: "apenas o 2♥ pode entrar numa sequência de copas".
    #[test]
    fn a_two_only_joins_a_sequence_of_its_own_suit() {
        assert!(Meld::new(&cards("6H 7H 2H")).is_ok());
        assert_eq!(
            Meld::new(&cards("6H 7H 2S")),
            Err(MeldError::WrongSuitForTwo)
        );
    }

    /// §7.2: aces group by rank, not by suit.
    #[test]
    fn aces_form_their_own_meld_regardless_of_suit() {
        assert!(matches!(meld("AH AD AS"), Meld::Aces(_)));
        assert!(matches!(meld("AH AD JOKER"), Meld::Aces(_)));
    }

    /// An ace inside a run is part of that run, not an aces meld.
    #[test]
    fn an_ace_extending_a_run_stays_a_sequence() {
        assert!(matches!(meld("QS KS AS"), Meld::Sequence(_)));
    }

    /// §7.1: eleven ranks from 4 to Ace, and nothing longer exists.
    #[test]
    fn a_sequence_tops_out_at_eleven_cards() {
        let full = sequence("4H 5H 6H 7H 8H 9H TH JH QH KH AH");
        assert_eq!(full.len(), 11);
        assert_eq!(
            Meld::new(&cards("4H 5H 6H 7H 8H 9H TH JH QH KH AH JOKER")),
            Err(MeldError::TooLong)
        );
    }

    // ---- wild placement (§9) ----

    /// A wild laid on a contiguous run sits at an end, and an end wild is free.
    #[test]
    fn a_fresh_wild_lands_on_an_end_and_stays_free() {
        let seq = sequence("6H 7H 2H");
        assert_eq!(seq.wild_represents(), Some(Rank::Eight));
        assert!(!seq.wild_is_locked());
    }

    /// §9 limit: a wild always stands in for a real card, so `Coringa-4♥-5♥-6♥`
    /// is not a thing — the wild caps the run at the top instead. A free wild at
    /// either end accepts exactly the same future cards, so the engine stores it
    /// canonically high and only drops it low when the Ace is already taken.
    #[test]
    fn a_free_wild_is_stored_at_the_high_end() {
        let seq = sequence("4H 5H 6H JOKER");
        assert_eq!(seq.wild_represents(), Some(Rank::Seven));
        assert!(!seq.wild_is_locked());
    }

    /// §9 limit: nothing exists above the Ace, so a wild capping a run that
    /// already reaches the Ace has to drop to the bottom end.
    #[test]
    fn a_wild_never_represents_a_rank_above_ace() {
        let seq = sequence("JS QS KS AS JOKER");
        assert_eq!(seq.wild_represents(), Some(Rank::Ten));
        assert!(!seq.wild_is_locked());
    }

    /// §9, first worked table: starting from `6-7-2`.
    #[test]
    fn spec_table_wild_movement_from_six_seven_wild() {
        // (added card, rank the wild ends up representing, locked?)
        let table = [
            ("8H", Rank::Nine, false),
            ("9H", Rank::Eight, true),
            ("4H", Rank::Five, true),
        ];
        for (added, represents, locked) in table {
            let mut seq = sequence("6H 7H 2H");
            seq.add_card(added.parse().unwrap())
                .unwrap_or_else(|e| panic!("adding {added} to 6-7-2 should work, got {e:?}"));
            assert_eq!(
                seq.wild_represents(),
                Some(represents),
                "after adding {added}"
            );
            assert_eq!(seq.wild_is_locked(), locked, "after adding {added}");
        }
    }

    /// §9, second worked table: starting from `J♠-Q♠-K♠-Coringa`. The wild sits
    /// at the boundary yet stays free — being on an end is not what locks it.
    #[test]
    fn spec_table_wild_movement_from_jack_queen_king_wild() {
        let table = [
            ("AS", Rank::Ten, false),
            ("TS", Rank::Ace, false),
            ("9S", Rank::Ten, true),
        ];
        for (added, represents, locked) in table {
            let mut seq = sequence("JS QS KS JOKER");
            seq.add_card(added.parse().unwrap())
                .unwrap_or_else(|e| panic!("adding {added} to J-Q-K-W should work, got {e:?}"));
            assert_eq!(
                seq.wild_represents(),
                Some(represents),
                "after adding {added}"
            );
            assert_eq!(seq.wild_is_locked(), locked, "after adding {added}");
        }
    }

    /// §9: "Trancado, ele nunca mais se move." Once naturals flank the wild, a
    /// card that would need its slot has nowhere to go.
    #[test]
    fn a_locked_wild_never_moves_again() {
        let mut seq = sequence("6H 7H 2H");
        seq.add_card("9H".parse().unwrap()).unwrap();
        assert!(seq.wild_is_locked());
        assert_eq!(seq.wild_represents(), Some(Rank::Eight));

        // The 2♥ is standing on the 8; a natural 8♥ cannot displace it.
        assert_eq!(
            seq.add_card("8H".parse().unwrap()),
            Err(MeldError::WildLocked)
        );
    }

    #[test]
    fn a_second_wild_can_never_be_added() {
        let mut seq = sequence("6H 7H 2H");
        assert_eq!(
            seq.add_card("JOKER".parse().unwrap()),
            Err(MeldError::TooManyWilds)
        );
    }

    #[test]
    fn cards_of_another_suit_cannot_join_a_sequence() {
        let mut seq = sequence("6H 7H 8H");
        assert_eq!(
            seq.add_card("9S".parse().unwrap()),
            Err(MeldError::MixedSuits)
        );
    }

    #[test]
    fn a_card_already_present_cannot_join_a_sequence() {
        let mut seq = sequence("6H 7H 8H");
        assert_eq!(
            seq.add_card("7H".parse().unwrap()),
            Err(MeldError::DuplicateRank)
        );
    }

    #[test]
    fn a_disconnected_card_cannot_join_a_sequence() {
        let mut seq = sequence("6H 7H 8H");
        assert_eq!(
            seq.add_card("JH".parse().unwrap()),
            Err(MeldError::NotASequence)
        );
    }

    // ---- laying off onto ace melds (§7.2, §8) ----

    #[test]
    fn an_ace_joins_an_ace_meld() {
        let mut meld = meld("AH AD AS");
        meld.add_card("AC".parse().unwrap()).unwrap();
        assert_eq!(meld.len(), 4);
    }

    /// §8: a 2 may enter an ace meld whatever its suit, unlike in a sequence.
    #[test]
    fn a_two_of_any_suit_joins_an_ace_meld() {
        let mut meld = meld("AH AD AS");
        meld.add_card("2C".parse().unwrap()).unwrap();
        assert_eq!(meld.canastra(), CanastraKind::None); // only four cards so far
        assert_eq!(meld.len(), 4);
    }

    #[test]
    fn an_ace_meld_holds_at_most_one_wild() {
        let mut meld = meld("AH AD JOKER");
        assert_eq!(
            meld.add_card("2C".parse().unwrap()),
            Err(MeldError::TooManyWilds)
        );
    }

    #[test]
    fn only_aces_and_wilds_join_an_ace_meld() {
        let mut meld = meld("AH AD AS");
        assert_eq!(
            meld.add_card("KH".parse().unwrap()),
            Err(MeldError::NotAnAce)
        );
    }

    /// Laying off is how a meld actually reaches canastra length in play.
    #[test]
    fn laying_off_grows_a_meld_into_a_canastra() {
        let mut meld = meld("5H 6H 7H");
        for card in ["8H", "9H", "TH"] {
            meld.add_card(card.parse().unwrap()).unwrap();
        }
        assert_eq!(meld.canastra(), CanastraKind::None);
        meld.add_card("JH".parse().unwrap()).unwrap();
        assert_eq!(meld.canastra(), CanastraKind::Clean);
        assert_eq!(meld.card_points(), 55);
    }

    // ---- canastra classification (§10) ----

    #[test]
    fn a_meld_under_seven_cards_is_no_canastra() {
        assert_eq!(meld("5H 6H 7H 8H 9H TH").canastra(), CanastraKind::None);
        assert_eq!(CanastraKind::None.bonus(), 0);
    }

    #[test]
    fn seven_cards_without_a_two_is_a_clean_canastra() {
        assert_eq!(meld("5H 6H 7H 8H 9H TH JH").canastra(), CanastraKind::Clean);
        assert_eq!(CanastraKind::Clean.bonus(), 500);
    }

    /// §10: a Joker does not dirty a canastra; a 2 does.
    #[test]
    fn a_joker_keeps_a_canastra_clean_but_a_two_dirties_it() {
        assert_eq!(
            meld("5H 6H 7H 8H 9H TH JOKER").canastra(),
            CanastraKind::Clean
        );
        assert_eq!(meld("5H 6H 7H 8H 9H TH 2H").canastra(), CanastraKind::Dirty);
        assert_eq!(CanastraKind::Dirty.bonus(), 200);
    }

    /// §10, the ace-canastra table verbatim, plus CLAUDE.md clarification #1:
    /// a 2 caps an ace meld at the dirty tier however many natural aces it holds.
    #[test]
    fn spec_table_ace_canastra_bonuses() {
        let table = [
            ("AH AD AS AC AH AD AS JOKER", CanastraKind::CleanAces), // 7 aces + Joker
            ("AH AD AS AC AH AD AS AC", CanastraKind::CleanAces),    // 8 aces
            ("AH AD AS AC AH AD AS AC JOKER", CanastraKind::CleanAces), // 8 aces + Joker
            ("AH AD AS AC AH AD JOKER", CanastraKind::Clean),        // 6 aces + Joker
            ("AH AD AS AC AH AD 2C", CanastraKind::Dirty),           // 6 aces + 2
            ("AH AD AS AC AH AD AS 2C", CanastraKind::Dirty),        // 7 aces + 2
        ];
        for (spec, expected) in table {
            assert_eq!(meld(spec).canastra(), expected, "for {spec}");
        }
        assert_eq!(CanastraKind::CleanAces.bonus(), 1000);
    }

    /// §10: the 1000 needs seven *real* aces, so a short ace meld padded out by
    /// a Joker scores as an ordinary clean canastra.
    #[test]
    fn a_joker_does_not_substitute_for_the_seventh_ace() {
        let six_aces_and_a_joker = meld("AH AD AS AC AH AD JOKER");
        assert_eq!(six_aces_and_a_joker.len(), 7);
        assert_eq!(six_aces_and_a_joker.canastra(), CanastraKind::Clean);
    }

    // ---- card values (§13) ----

    /// §13 worked example: the clean canastra 5♥..J♥ is worth 55 in card value,
    /// on top of its 500 bonus.
    #[test]
    fn meld_card_points_sum_the_individual_cards() {
        let canastra = meld("5H 6H 7H 8H 9H TH JH");
        assert_eq!(canastra.card_points(), 55);
        assert_eq!(canastra.canastra().bonus(), 500);
    }

    #[test]
    fn a_wild_scores_its_own_face_value_not_the_rank_it_stands_for() {
        // The Joker stands in for the 7♥ (worth 5) but still scores 50.
        assert_eq!(meld("JOKER 5H 6H").card_points(), 50 + 5 + 5);
        // The 2♥ stands in for the 8♥ (worth 10) but still scores 20.
        assert_eq!(meld("6H 7H 2H").card_points(), 5 + 5 + 20);
    }
}
