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

import type { Action } from "./types";
import { cardValue } from "./types";
import type { Bot, BotContext } from "./bot";
import { ofType } from "./bot";
import { findMelds, meldCards, meldValue } from "./melds";

export const randomBot: Bot = {
  id: "random",
  name: "Random",
  blurb: "Lays what it finds, discards its cheapest card. The floor.",

  candidates(view, legal, context: BotContext): Action[] {
    switch (view.phase) {
      case "AwaitingRefusalChoice": {
        // §3: the once-per-hand refusal. Cheap cards are worth throwing back.
        const refuse = ofType(legal, "RefuseDrawnCard");
        const keep = ofType(legal, "KeepDrawnCard");
        return view.pending_refusal && cardValue(view.pending_refusal) <= 5 && context.rng() < 0.5
          ? [...refuse, ...keep]
          : [...keep, ...refuse];
      }

      case "AwaitingDraw":
        // Never reaches for the pile, but the tail stays ranked so the list
        // is always complete.
        return [...ofType(legal, "Draw"), ...legal.filter((a) => a.type !== "Draw")];

      case "Melding": {
        const moves: Action[] = [];
        const table = view.tables[view.seat % 2];

        if (!context.safeMode) {
          // §6: the opening minimum has to be met inside one turn. The engine's
          // eager check is optimistic — it counts every remaining card at face
          // value — so it will happily allow a 45-point lay that this hand can
          // never grow to 75, leaving a turn that cannot be discarded out of.
          // Only rank lays at all if hand plus table actually clears the bar.
          //
          // What is already down counts. A partnership that is not open yet but
          // has melds on the table laid them earlier in *this* turn — that is
          // the only way to be in that position — so their value is this turn's
          // progress, and `PlayerView` carries no `laid_value` to read instead.
          const playable = view.hand.filter((card) => !view.frozen.includes(card));
          const inHand = findMelds(playable).reduce((sum, meld) => sum + meldValue(meld), 0);
          const alreadyLaid = table.opened
            ? 0
            : table.melds.reduce((sum, meld) => sum + meldValue(meldCards(meld)), 0);
          const layable = table.opened || alreadyLaid + inHand >= view.opening_minimum;

          if (layable && context.rng() < 0.85) moves.push(...ofType(legal, "LayMeld"));
          if (context.rng() < 0.7) moves.push(...ofType(legal, "AddToMeld"));
        }

        // §4.3: the turn ends with a discard. Cheapest first, so it keeps the
        // cards worth points.
        moves.push(...ofType(legal, "Discard").sort((a, b) => cardValue(a.card) - cardValue(b.card)));
        moves.push(...ofType(legal, "EndTurnWithoutDiscard"));

        // Ranking is complete: anything not yet listed trails the discards.
        for (const action of legal) if (!moves.includes(action)) moves.push(action);
        return moves;
      }

      default:
        return [...legal];
    }
  },
};
