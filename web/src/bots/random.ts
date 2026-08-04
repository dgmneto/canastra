/**
 * Random — the baseline.
 *
 * Lays whatever it can find, adds single cards to its own melds, and throws
 * away its cheapest card. It never takes the discard pile (§5), never holds a
 * black 3 as a blocker, and never plays toward a canastra — so a partnership of
 * these two builds wide and shallow, and is punished by §12's red 3s for it.
 *
 * That is the point: it is the floor a real bot has to beat.
 */

import type { Action } from "../types";
import { cardValue } from "../types";
import type { Bot, BotContext } from "./types";
import { findMelds, meldCards, meldValue } from "./melds";

export const randomBot: Bot = {
  id: "random",
  name: "Random",
  blurb: "Lays what it finds, discards its cheapest card. The floor.",

  candidates(view, context: BotContext): Action[] {
    switch (view.phase) {
      case "AwaitingRefusalChoice":
        // §3: the once-per-hand refusal. Cheap cards are worth throwing back.
        return view.pending_refusal && cardValue(view.pending_refusal) <= 5 && context.rng() < 0.5
          ? [{ type: "RefuseDrawnCard" }, { type: "KeepDrawnCard" }]
          : [{ type: "KeepDrawnCard" }];

      case "AwaitingDraw":
        return [{ type: "Draw" }];

      case "Melding": {
        const moves: Action[] = [];
        const playable = view.hand.filter((card) => !view.frozen.includes(card));

        if (!context.safeMode) {
          // §6: the opening minimum has to be met inside one turn. The engine's
          // eager check is optimistic — it counts every remaining card at face
          // value — so it will happily allow a 45-point lay that this hand can
          // never grow to 75, leaving a turn that cannot be discarded out of.
          // Only lay at all if hand plus table actually clears the bar.
          //
          // What is already down counts. A partnership that is not open yet but
          // has melds on the table laid them earlier in *this* turn — that is
          // the only way to be in that position — so their value is this turn's
          // progress, and `PlayerView` carries no `laid_value` to read instead.
          const table = view.tables[view.seat % 2];
          const found = findMelds(playable);
          const inHand = found.reduce((sum, meld) => sum + meldValue(meld), 0);
          const alreadyLaid = table.opened
            ? 0
            : table.melds.reduce((sum, meld) => sum + meldValue(meldCards(meld)), 0);
          const layable = table.opened || alreadyLaid + inHand >= view.opening_minimum ? found : [];

          if (context.rng() < 0.85) {
            for (const cards of layable) moves.push({ type: "LayMeld", cards });
          }
          // §4.2: lay-offs, one card at a time. Which cards fit is exactly the
          // combinatorial question the engine does not answer yet, so ask it.
          if (context.rng() < 0.7) {
            for (let meld = 0; meld < table.melds.length; meld += 1) {
              for (const card of playable) moves.push({ type: "AddToMeld", meld, cards: [card] });
            }
          }
        }

        // §4.3: the turn ends with a discard. Cheapest first, so it keeps the
        // cards worth points, and every card is tried before giving up.
        const discards = [...view.hand].sort((a, b) => cardValue(a) - cardValue(b));
        for (const card of discards) moves.push({ type: "Discard", card });
        moves.push({ type: "EndTurnWithoutDiscard" });
        return moves;
      }

      default:
        return [];
    }
  },
};
