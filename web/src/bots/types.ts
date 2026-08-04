/**
 * What a bot is.
 *
 * A bot is a *policy* and nothing else: given a position, name the moves worth
 * trying, best first. It does not touch the engine, does not know whether a
 * move was accepted, and cannot end a turn on its own — `driver.ts` does all of
 * that, identically for every bot, so two bots are always judged on the same
 * terms.
 *
 * The shape is "propose a list" rather than "return the move" because the
 * engine has no `legal_actions` yet (ADVERSARIAL-REVIEW.md F7). A bot cannot
 * know which of its ideas are legal, so it offers several and lets `apply` be
 * the judge — refusal leaves the position untouched, which makes guessing free.
 * If `legal_actions` ever lands this interface should be revisited; until then
 * it is the honest shape.
 */

import type { Action, PlayerView } from "../types";
import type { Rng } from "../rng";

export interface BotContext {
  /** Seeded, so a match with bots in it still replays. */
  rng: Rng;
  /**
   * The previous attempt at this turn dead-ended and was restarted, so the
   * bot must not walk into the same wall — draw and discard, nothing clever.
   *
   * The deal is deterministic: an unmodified retry reproduces the position
   * exactly, so a bot that ignores this will loop forever.
   */
  safeMode: boolean;
}

export interface Bot {
  /** Stable across versions — it goes in the replay log. */
  readonly id: string;
  readonly name: string;
  /** One line, shown in the UI. */
  readonly blurb: string;

  /**
   * Moves worth trying in this position, best first.
   *
   * The driver plays the first one the engine accepts, so ordering *is* the
   * policy. Returning an empty list concedes the turn: the driver will restart
   * it, which is a real cost. The list should always end in something that can
   * end a turn.
   */
  candidates(view: PlayerView, context: BotContext): Action[];
}
