/**
 * Random Plus — Random with four opinions.
 *
 * All four come from the same observation: §13 pays for *depth*, not breadth.
 * Random builds six shallow melds, scores 420 in table cards and still only
 * banks 90, because it earns no canastra bonus and its red 3s turn negative.
 * The bonuses are the game.
 *
 *  1. **Hoard 2s.** §8 lets a 2 into a meld; §10 then caps that meld at the
 *     dirty tier forever — 200 instead of 500. Worse, §12 pays red 3s ±100
 *     each on whether a *clean* canastra exists, so one careless 2 can cost
 *     300 in bonus and flip up to 400 more in red 3s. A Joker does none of
 *     that, so Jokers are spent freely and 2s are held back. The exception is
 *     opening: §6's minimum has to be met in one turn, and a partnership that
 *     never opens scores nothing at all.
 *
 *  2. **Deepen before widening.** A seventh card on a six-card meld is worth
 *     500; a fresh three-card meld is worth about 15. Lay-offs are proposed
 *     before new melds, longest meld first.
 *
 *  3. **Discard what has no future.** Random throws its cheapest card, which
 *     is exactly the 4–7 that runs are built from. This one scores every card
 *     by whether it extends a meld or has same-suit neighbours in hand, then
 *     throws the least useful — breaking ties toward *high* cards, since
 *     §13.2 charges for whatever is still in hand at the end.
 *
 *  4. **Use black 3s as blockers.** §5 says a black 3 on top puts the pile out
 *     of reach. They are worth nothing wherever they sit, so holding one is
 *     free and throwing one onto a fat pile denies it to the next player.
 */

import type { Action, Card, Meld, PlayerView } from "./types";
import { cardValue } from "./types";
import type { Bot, BotContext } from "./bot";
import { findMelds, meldCards, meldValue } from "./melds";

const SEQUENCE_RANKS = "456789TJQKA";
const SUIT_CODE: Record<string, string> = {
  Clubs: "C",
  Diamonds: "D",
  Hearts: "H",
  Spades: "S",
};

/** A pile worth denying. Below this, a black 3 is worth keeping for later. */
const PILE_WORTH_BLOCKING = 5;

export const randomPlusBot: Bot = {
  id: "random-plus",
  name: "Random Plus",
  blurb: "Hoards 2s for clean canastras, deepens melds, discards dead cards.",

  candidates(view, context: BotContext): Action[] {
    switch (view.phase) {
      case "AwaitingRefusalChoice": {
        // §3: one refusal per hand. Spend it on a card with no future.
        const card = view.pending_refusal;
        const dead = card !== null && !isWild(card) && usefulness(card, view) < 0;
        return dead
          ? [{ type: "RefuseDrawnCard" }, { type: "KeepDrawnCard" }]
          : [{ type: "KeepDrawnCard" }];
      }

      case "AwaitingDraw":
        return [{ type: "Draw" }];

      case "Melding": {
        const moves: Action[] = [];
        const table = view.tables[view.seat % 2];
        const playable = view.hand.filter((card) => !view.frozen.includes(card));

        if (!context.safeMode) {
          if (table.opened) {
            moves.push(...layOffs(playable, table.melds));
            // Clean melds only once open: there is no deadline any more, so a
            // 2 spent here buys nothing that waiting would not.
            for (const cards of findMelds(playable.filter((card) => !isTwo(card)))) {
              moves.push({ type: "LayMeld", cards });
            }
          } else {
            moves.push(...opening(view, playable));
          }
        }

        moves.push(...discards(view, context));
        moves.push({ type: "EndTurnWithoutDiscard" });
        return moves;
      }

      default:
        return [];
    }
  },
};

/**
 * §6: the opening lay.
 *
 * Clean first — if the minimum is reachable without touching a 2, that is
 * strictly better. Only when it is not does it spend them, because a
 * partnership that never opens scores nothing, and that dwarfs the tier.
 *
 * The reachability test mirrors Random's: the engine's eager check is
 * optimistic (it counts every remaining card at face value), so a bot that
 * trusts it can lay 45 toward 75 and then be unable to discard.
 */
function opening(view: PlayerView, playable: Card[]): Action[] {
  const minimum = view.opening_minimum;
  // Melds already down while `opened` is false were laid earlier in this same
  // turn — the only way to be in that position — so they are progress.
  const table = view.tables[view.seat % 2];
  const laid = table.melds.reduce((sum, meld) => sum + meldValue(meldCards(meld)), 0);

  const clean = findMelds(playable.filter((card) => !isTwo(card)));
  if (laid + total(clean) >= minimum) {
    return [...layOffs(playable, table.melds), ...clean.map(toLay)];
  }

  const withTwos = findMelds(playable);
  if (laid + total(withTwos) >= minimum) {
    return [...layOffs(playable, table.melds), ...withTwos.map(toLay)];
  }

  // Out of reach this turn either way. Laying anything now would only strand
  // the turn, so hold everything and discard.
  return [];
}

/**
 * §4.2: single-card lay-offs, longest meld first.
 *
 * Longest first is the whole point — the seventh card is worth 200 or 500, and
 * every card after that is worth its face value on a meld that already paid.
 * Cards that provably extend a meld are proposed before speculative ones, but
 * the speculative ones stay in the list because the engine is the real judge.
 */
function layOffs(playable: Card[], melds: Meld[]): Action[] {
  const order = melds
    .map((meld, index) => ({ meld, index }))
    .sort((a, b) => meldCards(b.meld).length - meldCards(a.meld).length);

  const moves: Action[] = [];
  for (const { meld, index } of order) {
    const dirty = meldCards(meld).some(isTwo);
    // A 2 is only ever added to a meld §10 has already spoiled.
    const usable = playable.filter((card) => !isTwo(card) || dirty);
    const fits = usable.filter((card) => extendsMeld(card, meld));
    const rest = usable.filter((card) => !extendsMeld(card, meld));
    for (const card of [...fits, ...rest]) moves.push({ type: "AddToMeld", meld: index, cards: [card] });
  }
  return moves;
}

/**
 * §4.3: the compulsory discard, least useful card first.
 *
 * Every card is listed — the engine can refuse a discard (§11.1 stranding),
 * so the tail of this list is what keeps the turn finishable.
 */
function discards(view: PlayerView, context: BotContext): Action[] {
  const blocking = view.discard.length >= PILE_WORTH_BLOCKING;

  const ranked = [...view.hand].sort((a, b) => {
    // §5: a black 3 on top freezes the pile. Free to hold, so it is thrown
    // when there is something worth denying and kept when there is not.
    const scoreA = isBlackThree(a) ? (blocking ? -100 : 40) : usefulness(a, view);
    const scoreB = isBlackThree(b) ? (blocking ? -100 : 40) : usefulness(b, view);
    if (scoreA !== scoreB) return scoreA - scoreB;
    // §13.2 charges for whatever is still in hand, so dump the dearer card.
    return cardValue(b) - cardValue(a);
  });

  // A little noise so two Random Plus seats do not play in lockstep.
  if (ranked.length > 2 && context.rng() < 0.15) {
    [ranked[0], ranked[1]] = [ranked[1], ranked[0]];
  }
  return ranked.map((card) => ({ type: "Discard", card }) as Action);
}

/**
 * How much this card is worth keeping. Higher means hold.
 *
 * Wilds are never voluntarily discarded: a Joker is a free slot in any future
 * canastra, and a 2 is being saved deliberately.
 */
function usefulness(card: Card, view: PlayerView): number {
  if (isWild(card)) return 1000;

  const ours = view.tables[view.seat % 2].melds;
  if (ours.some((meld) => extendsMeld(card, meld))) return 500;

  const rank = SEQUENCE_RANKS.indexOf(card[0]);
  if (rank < 0) return 0;

  // Same-suit cards within two ranks are the makings of a future run.
  let neighbours = 0;
  for (const other of view.hand) {
    if (other === card || other.length !== 2 || other[1] !== card[1]) continue;
    const distance = Math.abs(SEQUENCE_RANKS.indexOf(other[0]) - rank);
    if (distance >= 1 && distance <= 2) neighbours += 1;
  }

  // With no neighbours the card is dead weight, and the dearer it is the more
  // it costs to keep holding it.
  return neighbours * 30 - cardValue(card);
}

/** Whether this card would sit on either end of that meld. */
function extendsMeld(card: Card, meld: Meld): boolean {
  if (meld.kind === "Aces") return card.length === 2 && card[0] === "A";
  if (card.length !== 2 || card[1] !== SUIT_CODE[meld.suit]) return false;

  // Slots run low to high, and a wild reports the rank it stands in for, so
  // the run's ends are the first and last effective ranks.
  const effective = meld.cards.map((slot) => SEQUENCE_RANKS.indexOf(slot.standingInRank ?? slot.card[0]));
  const low = Math.min(...effective);
  const high = Math.max(...effective);
  const rank = SEQUENCE_RANKS.indexOf(card[0]);
  return rank === low - 1 || rank === high + 1;
}

function toLay(cards: Card[]): Action {
  return { type: "LayMeld", cards };
}

function total(melds: Card[][]): number {
  return melds.reduce((sum, meld) => sum + meldValue(meld), 0);
}

function isTwo(card: Card): boolean {
  return card.length === 2 && card[0] === "2";
}

function isWild(card: Card): boolean {
  return card === "JOKER" || isTwo(card);
}

function isBlackThree(card: Card): boolean {
  return card.length === 2 && card[0] === "3" && (card[1] === "S" || card[1] === "C");
}
