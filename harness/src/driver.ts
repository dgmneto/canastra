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
  /** §6.1: set when a restartTurn was a failed opening and moved the bar. */
  penalized?: boolean;
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
  const penalized = match.restartTurn(seat);
  return { action: "restartTurn", refusals, penalized };
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

/** A card the way it appears at the table: `"TS"` reads as `10♠`. */
function fmt(card: string): string {
  if (card === "JOKER") return "Coringa";
  const suit: Record<string, string> = { C: "♣", D: "♦", H: "♥", S: "♠" };
  const rank = card[0] === "T" ? "10" : card[0];
  return `${rank}${suit[card[1]] ?? card[1]}`;
}

function fmtAll(cards: string[]): string {
  return cards.map(fmt).join(" ");
}

/** §6.1: the feed note for a failed opening, naming the partnership's new bar. */
export function penaltyLabel(view: PlayerView): string {
  return ` — abertura mal-sucedida: o mínimo da dupla sobe para ${view.opening_minimum}`;
}

/** Um rótulo curto, em português, para a jogada que de fato aconteceu. */
export function label(action: Action | "restartTurn" | "settleHand", seat: Seat, who: string): string {
  if (action === "restartTurn") return `${who} (lugar ${seat}) recomeçou o turno`;
  if (action === "settleHand") return "mão encerrada";
  switch (action.type) {
    case "LayMeld":
      return `${who} baixou ${fmtAll(action.cards)}`;
    case "AddToMeld":
      return `${who} adicionou ${fmtAll(action.cards)} ao jogo #${action.meld}`;
    case "Discard":
      return `${who} descartou ${fmt(action.card)}`;
    case "TakeDiscardPile":
      return `${who} pegou o lixo com ${fmtAll(action.core)}`;
    case "Draw":
      return `${who} comprou do monte`;
    case "KeepDrawnCard":
      return `${who} ficou com a carta oferecida`;
    case "RefuseDrawnCard":
      return `${who} recusou a carta oferecida`;
    case "EndTurnWithoutDiscard":
      return `${who} encerrou sem descartar`;
  }
}
