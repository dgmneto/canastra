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

import type { Action } from "./types";
import type { Bot, BotContext } from "./bot";
import { ofType } from "./bot";
import { randomBot } from "./random";

export const randomDiscardHungryBot: Bot = {
  id: "random-hungry",
  name: "Random Discard Hungry",
  blurb: "Random, but grabs the discard pile whenever §5 allows it.",

  candidates(view, legal, context: BotContext): Action[] {
    if (view.phase === "AwaitingDraw") {
      // §5 taking is exactly what dead-ends a turn: the captured pile arrives
      // frozen, so a partnership that has not opened often cannot reach §6's
      // minimum and cannot then discard. `safeMode` says that already happened
      // — reaching for the pile again would reproduce it, forever, because the
      // deal is deterministic.
      if (context.safeMode) return ofType(legal, "Draw");
      // Every legal capture first, the ordinary draw last.
      return [...ofType(legal, "TakeDiscardPile"), ...ofType(legal, "Draw")];
    }
    return randomBot.candidates(view, legal, context);
  },
};
