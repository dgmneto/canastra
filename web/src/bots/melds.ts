/**
 * Meld-finding, shared by every bot.
 *
 * Not engine rules — the engine is still the only judge of a legal meld. This
 * is a *search*: which combinations are worth proposing. It lives outside any
 * one bot because "what could I lay from this hand" is a question every bot
 * asks, and it is the expensive part to get right.
 */

import type { Card, Meld } from "../types";
import { cardValue } from "../types";

/** §7.1: sequences run 4 up to A. 2s and 3s never sit in one as naturals. */
const SEQUENCE_RANKS = "456789TJQKA";

/**
 * A set of melds this hand can lay, none of them sharing a card.
 *
 * Overlap matters more than it looks: callers total these to decide whether
 * §6's opening minimum is within reach, and counting the same 8♥ toward two
 * different runs would talk a bot into a lay it cannot finish. Candidates are
 * taken most-valuable-first and each one consumes the cards it uses.
 */
export function findMelds(hand: Card[]): Card[][] {
  const pool = new Map<Card, number>();
  for (const card of hand) pool.set(card, (pool.get(card) ?? 0) + 1);

  const chosen: Card[][] = [];
  for (const meld of enumerateMelds(hand).sort((a, b) => meldValue(b) - meldValue(a))) {
    if (!meld.every((card, at) => (pool.get(card) ?? 0) > meld.slice(0, at).filter((c) => c === card).length)) {
      continue;
    }
    for (const card of meld) pool.set(card, pool.get(card)! - 1);
    chosen.push(meld);
  }
  return chosen;
}

/** Every meld shape worth trying, overlaps included. */
export function enumerateMelds(hand: Card[]): Card[][] {
  const melds: Card[][] = [];

  // §7.2: three or more aces collect rather than run.
  const aces = hand.filter((card) => card.length === 2 && card[0] === "A");
  if (aces.length >= 3) melds.push(aces.slice(0, 7));

  for (const suit of "CDHS") {
    // One card per rank — a sequence covers each rank once, so a second 6♥ is
    // no use in the same meld.
    const byRank = new Map<number, Card>();
    for (const card of hand) {
      if (card.length !== 2 || card[1] !== suit) continue;
      const rank = SEQUENCE_RANKS.indexOf(card[0]);
      if (rank >= 0 && !byRank.has(rank)) byRank.set(rank, card);
    }

    // §8: a Joker goes anywhere, a 2 only into its own suit. One per meld.
    const wild = hand.find((card) => card === "JOKER" || (card.length === 2 && card[0] === "2" && card[1] === suit));

    for (let length = 7; length >= 3; length -= 1) {
      for (let start = 0; start + length <= SEQUENCE_RANKS.length; start += 1) {
        const held: Card[] = [];
        for (let at = start; at < start + length; at += 1) {
          const card = byRank.get(at);
          if (card) held.push(card);
        }
        const missing = length - held.length;
        if (missing === 0) melds.push(held);
        else if (missing === 1 && wild) melds.push([...held, wild]);
      }
    }
  }

  return melds;
}

export function meldValue(meld: Card[]): number {
  return meld.reduce((total, card) => total + cardValue(card), 0);
}

/** The cards in a laid meld, whichever shape it is. */
export function meldCards(meld: Meld): Card[] {
  if (meld.kind === "Aces") return meld.wild ? [...meld.aces, meld.wild] : meld.aces;
  return meld.cards.map((slot) => slot.card);
}
