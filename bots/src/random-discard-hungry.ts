/**
 * Random Discard Hungry — Random, but it always reaches for the pile.
 *
 * Deliberately **identical to Random everywhere except the draw**: it delegates
 * the whole melding and discarding phase to `randomBot`. That is what makes the
 * comparison mean something — any difference in results is §5 pile-taking and
 * nothing else.
 *
 * §5 makes this an expensive habit, which is the interesting part:
 *
 *  - The three cards that capture the pile must be the top card plus **two
 *    natural cards from hand of the same suit**, forming a contiguous run. No
 *    wild may stand in, and all three land in one meld.
 *  - Everything else in the pile goes to hand and is **frozen** — unmeldable
 *    until the next turn (CLAUDE.md clarification #5 reads "usada" as *melded*,
 *    so a frozen card may still be discarded).
 *  - The capturing meld takes no wild for the rest of the turn.
 *
 * So a big pile is a big hand of dead cards, and §13.2 charges for every card
 * still held at the end. Hunger is not obviously good, which is why it is worth
 * measuring rather than assuming.
 */

import type { Action, Card, PlayerView } from "./types";
import type { Bot, BotContext } from "./bot";
import { randomBot } from "./random";

/** §7.1: a capturing run lives in 4..A, so 2s and 3s cannot anchor one. */
const SEQUENCE_RANKS = "456789TJQKA";

export const randomDiscardHungryBot: Bot = {
  id: "random-hungry",
  name: "Random Discard Hungry",
  blurb: "Random, but grabs the discard pile whenever §5 allows it.",

  candidates(view, context: BotContext): Action[] {
    if (view.phase === "AwaitingDraw") {
      // §5 taking is exactly what dead-ends a turn: the captured pile arrives
      // frozen, so a partnership that has not opened often cannot reach §6's
      // minimum and cannot then discard. `safeMode` says that already happened
      // — reaching for the pile again would reproduce it, forever, because the
      // deal is deterministic.
      if (context.safeMode) return [{ type: "Draw" }];
      // Every way of taking the pile, then the ordinary draw as the fallback.
      return [...pileTakes(view), { type: "Draw" }];
    }
    return randomBot.candidates(view, context);
  },
};

/**
 * §5: every legal-looking way to capture the pile with the hand as it stands.
 *
 * Only the shapes are worked out here — whether the pile is actually takeable
 * (a black 3 or a wild on top blocks it, §6's minimum may not be met) is left
 * to the engine, which refuses without disturbing the position.
 */
function pileTakes(view: PlayerView): Action[] {
  const top = view.discard[view.discard.length - 1];
  if (top === undefined || top.length !== 2) return [];

  const anchor = SEQUENCE_RANKS.indexOf(top[0]);
  // A 2, a 3 or a Joker on top: no run can be built on it. §5 blocks the first
  // and third outright, and a 3 has no place in a sequence.
  if (anchor < 0) return [];

  const suit = top[1];
  const byRank = new Map<number, Card>();
  for (const card of view.hand) {
    if (card.length !== 2 || card[1] !== suit) continue;
    const rank = SEQUENCE_RANKS.indexOf(card[0]);
    // Frozen cards are already unmeldable, and the core is melded immediately.
    if (rank >= 0 && !byRank.has(rank) && !view.frozen.includes(card)) byRank.set(rank, card);
  }

  const moves: Action[] = [];
  const ourMelds = view.tables[view.seat % 2].melds.length;

  // The three windows that contain the top card: it sits high, middle, or low.
  for (const offset of [-2, -1, 0]) {
    const run = [anchor + offset, anchor + offset + 1, anchor + offset + 2];
    if (run[0] < 0 || run[2] >= SEQUENCE_RANKS.length) continue;

    const fromHand = run.filter((rank) => rank !== anchor).map((rank) => byRank.get(rank));
    if (fromHand.some((card) => card === undefined)) continue;
    const core = fromHand as [Card, Card];

    // A new meld first: folding into an existing one is only possible when the
    // run actually joins it, and the engine is the judge of that.
    moves.push({ type: "TakeDiscardPile", core, target: { kind: "NewMeld" } });
    for (let meld = 0; meld < ourMelds; meld += 1) {
      moves.push({ type: "TakeDiscardPile", core, target: { kind: "Existing", meld } });
    }
  }

  return moves;
}
