/**
 * The engine, wrapped for a driver.
 *
 * This is the only module that touches the wasm bindings. It adds three things
 * the raw `Game` handle does not give us: a per-seat set of views, an
 * append-only action log, and a turn checkpoint.
 *
 * It is deliberately environment-agnostic. It imports the `Game` class from the
 * committed bindings, but who *initialises* the wasm is left to the caller —
 * the web page initialises it the browser way (fetch), the harness CLI the Node
 * way (a `WebAssembly.Module`). Neither is this module's concern.
 */

import { Game } from "../../web/src/engine/canastra.js";
import type { Action, HandScore, PlayerView, RuleViolation, Seat } from "@canastra/bots";
import { isRuleViolation } from "@canastra/bots";

/**
 * One line of the replay log.
 *
 * The header carries the seed, which is all the deal depends on — the engine
 * fixes the stock at deal time and never reshuffles, so seed plus this list
 * reproduces a whole match. `restartTurn` is recorded rather than truncating
 * the log, so an abandoned turn stays visible and a replayer honours it with
 * the same checkpoint logic used here.
 */
export type LogLine =
  | { seed: string; startedAt: string; bots: string[] }
  | { seat: Seat; action: Action }
  | { seat: Seat; restartTurn: true }
  | { settleHand: true };

export class Match {
  seed: bigint;
  log: LogLine[];
  private game: Game;
  /**
   * The position the current turn began from.
   *
   * `Game` ships a `rewindTurn`, but it refreshes its checkpoint only when the
   * phase *before* an action is `AwaitingDraw`, so calling it as the first move
   * of a turn reverts a whole turn too far (ADVERSARIAL-REVIEW.md F5). Holding
   * the snapshot here instead — refreshed whenever the resulting phase is
   * `AwaitingDraw` — is the checkpoint rule that finding asks for, and it costs
   * five lines.
   */
  private turnStart: string;

  /** `bots` is the bot id sitting in each seat, in seat order. */
  constructor(seed: bigint, bots: string[]) {
    this.seed = seed;
    this.game = new Game(seed);
    this.turnStart = this.game.snapshot();
    // The bots go in the header because the log is a record of a *match*, and
    // which policy sat where is the thing you want to know when comparing two.
    this.log = [{ seed: seed.toString(), startedAt: new Date().toISOString(), bots }];
  }

  /** Every seat's view of the position. The driver is deliberately omniscient. */
  views(): [PlayerView, PlayerView, PlayerView, PlayerView] {
    return [0, 1, 2, 3].map((seat) => this.game.view(seat) as PlayerView) as [
      PlayerView,
      PlayerView,
      PlayerView,
      PlayerView,
    ];
  }

  /**
   * §13: what the hand in progress would bank for each partnership, itemised.
   *
   * Straight from the engine rather than totalled here — canastra tiers, the
   * red-3 sign flip and black 3s scoring nothing are all §13, and a second copy
   * of those rules in the client would be a copy that drifts.
   */
  handScores(): [HandScore, HandScore] {
    return [0, 1].map((team) => this.game.handScore(team) as HandScore) as [HandScore, HandScore];
  }

  /**
   * Play a move. Returns `null` when it was accepted, or the rule that refused
   * it — the engine leaves the position untouched on refusal, so a caller may
   * guess and check freely.
   */
  apply(seat: Seat, action: Action): RuleViolation | null {
    try {
      this.game.apply(seat, action);
    } catch (thrown) {
      return normalize(thrown);
    }
    this.log.push({ seat, action });
    this.checkpoint();
    return null;
  }

  /** §13–14: bank a finished hand, then deal on or end the match. */
  settleHand(): RuleViolation | null {
    try {
      this.game.settleHand();
    } catch (thrown) {
      return normalize(thrown);
    }
    this.log.push({ settleHand: true });
    this.checkpoint();
    return null;
  }

  /**
   * Abandon the turn in progress and restore the position it began from.
   *
   * §6's opening minimum is now checked eagerly, so this is a backstop rather
   * than an everyday escape: the eager check is deliberately optimistic and a
   * player can still, rarely, reach a turn they cannot finish.
   */
  restartTurn(seat: Seat): void {
    const restored = Game.restore(this.turnStart);
    this.game.free();
    this.game = restored;
    this.log.push({ seat, restartTurn: true });
  }

  /**
   * The whole state as JSON. This is F6's sharp end — all four hands, the
   * stock order, the seed — so it is for the server's persistence only and
   * must never reach a browser.
   */
  snapshot(): string {
    return this.game.snapshot();
  }

  /**
   * Rebuild a match from a snapshot, for server persistence and replay
   * resume. `meta` carries what the snapshot does not: which bot (or human
   * name) sat where and when the match started, so the log header stays a
   * truthful record across a restart.
   *
   * The turn checkpoint becomes the restored position itself. After a
   * *mid-turn* restore that is not the true turn start: the pre-restore part of
   * the turn is already lost, so a `restartTurn` replays only the stretch after
   * the restore point rather than the whole turn. This is the conservative
   * direction — it can only ever make a restart undo *less*, never a whole turn
   * too far — and the spec accepted it (Risks / open points).
   */
  static restore(
    snapshot: string,
    meta: { seed: bigint; bots: string[]; startedAt: string; log?: LogLine[] },
  ): Match {
    // `Object.create` bypasses the constructor; the field assignments below
    // stay legal because these are ordinary TS-`private` runtime properties.
    // Converting them to `#private` ECMAScript fields would break this at
    // runtime, since `restore` builds the instance externally.
    const match = Object.create(Match.prototype) as Match;
    match.seed = meta.seed;
    match.game = Game.restore(snapshot);
    match.turnStart = snapshot;
    match.log = meta.log ?? [
      { seed: meta.seed.toString(), startedAt: meta.startedAt, bots: meta.bots },
    ];
    return match;
  }

  private checkpoint(): void {
    const view = this.game.view(0) as PlayerView;
    if (view.phase === "AwaitingDraw") {
      this.turnStart = this.game.snapshot();
    }
  }
}

/** The log as JSON Lines — one entry per line, the header first. */
export function logToText(log: LogLine[]): string {
  return log.map((line) => JSON.stringify(line)).join("\n") + "\n";
}

/**
 * wasm-bindgen rejects with whatever the Rust side produced: a structured
 * `RuleViolation` for a refused move, a plain string for a malformed call.
 */
function normalize(thrown: unknown): RuleViolation {
  if (isRuleViolation(thrown)) return thrown;
  return { error: String(thrown) };
}
