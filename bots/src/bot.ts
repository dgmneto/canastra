/**
 * What a bot is.
 *
 * A bot is a *policy* and nothing else: given a position and the complete
 * list of moves the engine allows in it, rank those moves best first. It does
 * not touch the engine, does not know whether a move was accepted, and cannot
 * end a turn on its own — the harness's driver does all of that, identically
 * for every bot, so two bots are always judged on the same terms.
 *
 * `legal` is the engine's own enumeration (ADVERSARIAL-REVIEW.md F7, closed).
 * The interface used to be "propose a list and let `apply` judge" because the
 * engine could not say what was legal; now that it can, a bot's body is pure
 * preference. The engine remains the referee: anything returned outside
 * `legal` is refused, and running out of ideas concedes the turn (the driver
 * restarts it), so a well-behaved bot returns every legal move, ranked.
 */

import type { Action, PlayerView } from "./types";
import type { Rng } from "./rng";

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
   * The legal moves in this position, best first.
   *
   * Ordering *is* the policy: the driver plays the first one. Returning an
   * empty list concedes the turn — the driver will restart it, which is a
   * real cost — so the list should always contain every legal move, however
   * low the tail ranks.
   */
  candidates(view: PlayerView, legal: Action[], context: BotContext): Action[];
}

/** The legal actions of one variant, narrowed for the caller. */
export function ofType<T extends Action["type"]>(
  legal: Action[],
  type: T,
): Extract<Action, { type: T }>[] {
  return legal.filter((a): a is Extract<Action, { type: T }> => a.type === type);
}
