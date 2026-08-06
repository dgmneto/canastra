/**
 * The driver that runs a bot against the engine.
 *
 * Every bot goes through this, unchanged, so a match between two bots is
 * decided by their policies and nothing else. Anything that is not policy —
 * settling a finished hand, restarting a dead-ended turn, deciding a proposal
 * was refused — belongs here rather than in any bot.
 */

import type { Action, PlayerView, Seat } from "@canastra/bots";
import type { Bot, BotContext } from "@canastra/bots";
import type { Match } from "./match";

export interface StepResult {
  action: Action | "restartTurn" | "settleHand";
  /** The rules that refused the candidates tried first, in order. */
  refusals: string[];
}

/**
 * Advance the seat whose turn it is by one action.
 *
 * The refusals are worth surfacing: they are the record of which rule did the
 * work, which is usually what you want to know when a bot behaves oddly.
 */
export function step(match: Match, view: PlayerView, bot: Bot, context: BotContext): StepResult | null {
  if (view.phase === "MatchOver") return null;

  if (view.phase === "HandOver") {
    const refused = match.settleHand();
    return { action: "settleHand", refusals: refused ? [refused.error] : [] };
  }

  const seat = view.turn;
  const refusals: string[] = [];

  for (const action of bot.candidates(view, context)) {
    const refused = match.apply(seat, action);
    if (!refused) return { action, refusals };
    refusals.push(describe(action, refused.error));
  }

  // The bot ran out of ideas. §6's eager check makes this rare but not
  // impossible — that check is optimistic on purpose, so a turn can still
  // dead-end — and a bot may simply have proposed nothing legal.
  match.restartTurn(seat);
  return { action: "restartTurn", refusals };
}

function describe(action: Action, error: string): string {
  switch (action.type) {
    case "LayMeld":
      return `LayMeld ${action.cards.join(" ")} → ${error}`;
    case "AddToMeld":
      return `AddToMeld #${action.meld} ${action.cards.join(" ")} → ${error}`;
    case "Discard":
      return `Discard ${action.card} → ${error}`;
    default:
      return `${action.type} → ${error}`;
  }
}

/** A short human label for the move that was actually played. */
export function label(action: Action | "restartTurn" | "settleHand", seat: Seat, who: string): string {
  if (action === "restartTurn") return `${who} (seat ${seat}) restarted the turn`;
  if (action === "settleHand") return "hand settled";
  switch (action.type) {
    case "LayMeld":
      return `${who} laid ${action.cards.join(" ")}`;
    case "AddToMeld":
      return `${who} added ${action.cards.join(" ")} to meld #${action.meld}`;
    case "Discard":
      return `${who} discarded ${action.card}`;
    case "TakeDiscardPile":
      return `${who} took the pile with ${action.core.join(" ")}`;
    default:
      return `${who}: ${action.type}`;
  }
}
