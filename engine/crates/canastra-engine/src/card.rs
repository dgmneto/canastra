//! Cards, the 108-card deck, and the classification predicates the rules lean on.

use std::fmt;
use std::str::FromStr;

/// How a Joker is spelled in the string codec.
pub const JOKER_CODE: &str = "JOKER";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Suit {
    Clubs,
    Diamonds,
    Hearts,
    Spades,
}

impl Suit {
    pub const ALL: [Suit; 4] = [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades];

    pub fn is_red(self) -> bool {
        matches!(self, Suit::Diamonds | Suit::Hearts)
    }

    fn code(self) -> char {
        match self {
            Suit::Clubs => 'C',
            Suit::Diamonds => 'D',
            Suit::Hearts => 'H',
            Suit::Spades => 'S',
        }
    }

    fn from_code(code: char) -> Option<Suit> {
        Some(match code {
            'C' => Suit::Clubs,
            'D' => Suit::Diamonds,
            'H' => Suit::Hearts,
            'S' => Suit::Spades,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Rank {
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
    Jack,
    Queen,
    King,
    Ace,
}

impl Rank {
    pub const ALL: [Rank; 13] = [
        Rank::Two,
        Rank::Three,
        Rank::Four,
        Rank::Five,
        Rank::Six,
        Rank::Seven,
        Rank::Eight,
        Rank::Nine,
        Rank::Ten,
        Rank::Jack,
        Rank::Queen,
        Rank::King,
        Rank::Ace,
    ];

    /// Position within a sequence meld, where `Four` is 0 and `Ace` is 10.
    ///
    /// §7.1: sequences run 4 through Ace, so 2s and 3s have no position at all.
    /// Returning `None` for them is what keeps them out of sequences everywhere.
    pub fn sequence_index(self) -> Option<u8> {
        Some(match self {
            Rank::Four => 0,
            Rank::Five => 1,
            Rank::Six => 2,
            Rank::Seven => 3,
            Rank::Eight => 4,
            Rank::Nine => 5,
            Rank::Ten => 6,
            Rank::Jack => 7,
            Rank::Queen => 8,
            Rank::King => 9,
            Rank::Ace => 10,
            Rank::Two | Rank::Three => return None,
        })
    }

    /// Inverse of [`Rank::sequence_index`].
    pub fn from_sequence_index(index: u8) -> Option<Rank> {
        Some(match index {
            0 => Rank::Four,
            1 => Rank::Five,
            2 => Rank::Six,
            3 => Rank::Seven,
            4 => Rank::Eight,
            5 => Rank::Nine,
            6 => Rank::Ten,
            7 => Rank::Jack,
            8 => Rank::Queen,
            9 => Rank::King,
            10 => Rank::Ace,
            _ => return None,
        })
    }

    /// §13.2 card values.
    fn points(self) -> u32 {
        match self {
            Rank::Three => 0,
            Rank::Four | Rank::Five | Rank::Six | Rank::Seven => 5,
            Rank::Eight | Rank::Nine | Rank::Ten | Rank::Jack | Rank::Queen | Rank::King => 10,
            Rank::Ace => 15,
            Rank::Two => 20,
        }
    }

    fn code(self) -> char {
        match self {
            Rank::Two => '2',
            Rank::Three => '3',
            Rank::Four => '4',
            Rank::Five => '5',
            Rank::Six => '6',
            Rank::Seven => '7',
            Rank::Eight => '8',
            Rank::Nine => '9',
            Rank::Ten => 'T',
            Rank::Jack => 'J',
            Rank::Queen => 'Q',
            Rank::King => 'K',
            Rank::Ace => 'A',
        }
    }

    fn from_code(code: char) -> Option<Rank> {
        Some(match code {
            '2' => Rank::Two,
            '3' => Rank::Three,
            '4' => Rank::Four,
            '5' => Rank::Five,
            '6' => Rank::Six,
            '7' => Rank::Seven,
            '8' => Rank::Eight,
            '9' => Rank::Nine,
            'T' => Rank::Ten,
            'J' => Rank::Jack,
            'Q' => Rank::Queen,
            'K' => Rank::King,
            'A' => Rank::Ace,
            _ => return None,
        })
    }
}

/// A single card.
///
/// Deliberately a plain value with no per-card identity: the deck holds two
/// copies of every standard card, and no rule in the spec can tell them apart.
/// A client that wants stable identities for render animations assigns its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Card {
    Joker,
    Standard { rank: Rank, suit: Suit },
}

impl Card {
    pub fn is_joker(self) -> bool {
        matches!(self, Card::Joker)
    }

    pub fn rank(self) -> Option<Rank> {
        match self {
            Card::Joker => None,
            Card::Standard { rank, .. } => Some(rank),
        }
    }

    pub fn suit(self) -> Option<Suit> {
        match self {
            Card::Joker => None,
            Card::Standard { suit, .. } => Some(suit),
        }
    }

    /// §8: the Joker and every 2 are wild. They differ in what they do to a
    /// canastra, not in whether they are wild.
    pub fn is_wild(self) -> bool {
        matches!(
            self,
            Card::Joker
                | Card::Standard {
                    rank: Rank::Two,
                    ..
                }
        )
    }

    pub fn is_natural(self) -> bool {
        !self.is_wild()
    }

    /// §12: goes straight to the table when drawn, and scores ±100 at hand end.
    pub fn is_red_three(self) -> bool {
        matches!(self, Card::Standard { rank: Rank::Three, suit } if suit.is_red())
    }

    /// §12: worth nothing anywhere, never reaches the table, blocks the pile.
    pub fn is_black_three(self) -> bool {
        matches!(self, Card::Standard { rank: Rank::Three, suit } if !suit.is_red())
    }

    /// §13.2 card values.
    pub fn points(self) -> u32 {
        match self {
            Card::Joker => 50,
            Card::Standard { rank, .. } => rank.points(),
        }
    }
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Card::Joker => f.write_str(JOKER_CODE),
            Card::Standard { rank, suit } => write!(f, "{}{}", rank.code(), suit.code()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseCardError(pub String);

impl fmt::Display for ParseCardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "not a card: {:?}", self.0)
    }
}

impl std::error::Error for ParseCardError {}

impl FromStr for Card {
    type Err = ParseCardError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == JOKER_CODE {
            return Ok(Card::Joker);
        }
        let invalid = || ParseCardError(s.to_owned());
        let mut chars = s.chars();
        let (Some(rank), Some(suit), None) = (chars.next(), chars.next(), chars.next()) else {
            return Err(invalid());
        };
        Ok(Card::Standard {
            rank: Rank::from_code(rank).ok_or_else(invalid)?,
            suit: Suit::from_code(suit).ok_or_else(invalid)?,
        })
    }
}

/// §2: two full decks plus four Jokers — 108 cards.
pub fn deck() -> Vec<Card> {
    let mut cards = Vec::with_capacity(108);
    for _ in 0..2 {
        for suit in Suit::ALL {
            for rank in Rank::ALL {
                cards.push(Card::Standard { rank, suit });
            }
        }
    }
    cards.extend([Card::Joker; 4]);
    cards
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Terse card literal for tests: `c("6D")`, `c("JOKER")`.
    fn c(s: &str) -> Card {
        s.parse().unwrap()
    }

    #[test]
    fn deck_has_108_cards() {
        assert_eq!(deck().len(), 108);
    }

    #[test]
    fn deck_has_four_jokers() {
        assert_eq!(deck().iter().filter(|c| c.is_joker()).count(), 4);
    }

    #[test]
    fn deck_has_two_copies_of_every_standard_card() {
        let deck = deck();
        for suit in Suit::ALL {
            for rank in Rank::ALL {
                let card = Card::Standard { rank, suit };
                let copies = deck.iter().filter(|&&other| other == card).count();
                assert_eq!(copies, 2, "expected exactly two copies of {card}");
            }
        }
    }

    /// §13.2 card value table.
    #[test]
    fn card_points_follow_the_scoring_table() {
        assert_eq!(c("3S").points(), 0);
        assert_eq!(c("3H").points(), 0);
        assert_eq!(c("4D").points(), 5);
        assert_eq!(c("7C").points(), 5);
        assert_eq!(c("8H").points(), 10);
        assert_eq!(c("TD").points(), 10);
        assert_eq!(c("KS").points(), 10);
        assert_eq!(c("AD").points(), 15);
        assert_eq!(c("2C").points(), 20);
        assert_eq!(c("JOKER").points(), 50);
    }

    /// §8: both the Joker and any 2 act as wild cards.
    #[test]
    fn wild_cards_are_jokers_and_twos() {
        assert!(c("JOKER").is_wild());
        assert!(c("2H").is_wild());
        assert!(c("2S").is_wild());
        assert!(!c("3H").is_wild());
        assert!(!c("AS").is_wild());
    }

    /// §12: only hearts and diamonds carry the red-3 bonus.
    #[test]
    fn red_threes_are_hearts_and_diamonds_only() {
        assert!(c("3H").is_red_three());
        assert!(c("3D").is_red_three());
        assert!(!c("3S").is_red_three());
        assert!(!c("3C").is_red_three());
        assert!(!c("4H").is_red_three());
    }

    #[test]
    fn black_threes_are_spades_and_clubs_only() {
        assert!(c("3S").is_black_three());
        assert!(c("3C").is_black_three());
        assert!(!c("3H").is_black_three());
        assert!(!c("3D").is_black_three());
    }

    /// §7.1: sequences run 4 through Ace, so 2s and 3s have no sequence position.
    #[test]
    fn sequence_ranks_span_four_through_ace() {
        assert_eq!(Rank::Four.sequence_index(), Some(0));
        assert_eq!(Rank::Ace.sequence_index(), Some(10));
        assert_eq!(Rank::Two.sequence_index(), None);
        assert_eq!(Rank::Three.sequence_index(), None);
    }

    #[test]
    fn sequence_index_round_trips() {
        for index in 0..=10 {
            let rank = Rank::from_sequence_index(index).expect("4..=A is 11 ranks");
            assert_eq!(rank.sequence_index(), Some(index));
        }
        assert_eq!(Rank::from_sequence_index(11), None);
    }

    #[test]
    fn cards_round_trip_through_their_string_codec() {
        for card in deck() {
            assert_eq!(card.to_string().parse::<Card>(), Ok(card));
        }
    }

    #[test]
    fn card_codec_rejects_malformed_input() {
        assert!("".parse::<Card>().is_err());
        assert!("XD".parse::<Card>().is_err(), "unknown rank");
        assert!("6X".parse::<Card>().is_err(), "unknown suit");
        assert!("10D".parse::<Card>().is_err(), "ten is spelled T");
        assert!("6DD".parse::<Card>().is_err(), "trailing junk");
        assert!("6".parse::<Card>().is_err(), "missing suit");
    }
}
