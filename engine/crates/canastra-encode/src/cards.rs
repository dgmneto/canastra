//! The fixed card-identity space every segment indexes into.

use canastra_engine::card::{Card, Rank, Suit};

/// Width of a card-identity one-hot: 52 standard cards plus the Joker.
pub const CARD_IDS: usize = 53;

/// Where a card lives in the identity space. Suits in declaration order
/// (Clubs..Spades), ranks in *game* order (4,5,6,7,8,9,T,J,Q,K,A,2,3) so
/// sequence-adjacent ranks are encoding-adjacent. The Joker is 52.
pub fn card_index(card: Card) -> usize {
    match card {
        Card::Joker => 52,
        Card::Standard { rank, suit } => suit_index(suit) * 13 + rank_index(rank),
    }
}

/// Game-order rank position: 4 is 0, Ace is 10, 2 is 11, 3 is 12.
pub fn rank_index(rank: Rank) -> usize {
    match rank {
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
        Rank::Two => 11,
        Rank::Three => 12,
    }
}

/// Suit position in declaration order.
pub fn suit_index(suit: Suit) -> usize {
    match suit {
        Suit::Clubs => 0,
        Suit::Diamonds => 1,
        Suit::Hearts => 2,
        Suit::Spades => 3,
    }
}
