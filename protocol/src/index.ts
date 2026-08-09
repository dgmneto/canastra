/**
 * The client↔server wire protocol for the multiplayer table.
 *
 * One WebSocket at `/ws`, JSON messages discriminated by `type` — the same
 * tagging style the engine uses for `Action`. This package is the shared
 * language of `web/` (the game client) and `server/`; it holds no game logic and
 * no runtime dependencies (the wire types themselves live in `@canastra/bots`,
 * the leaf everything else depends on).
 *
 * The information rules that matter are encoded in what the messages can
 * carry: `view` is the engine's per-seat `observe`, `table` is public
 * information only, and there is no message that could put a `GameState`, a
 * snapshot, or another seat's hand on the wire (ADVERSARIAL-REVIEW.md F6).
 */

import type {
  Action,
  HandScore,
  Phase,
  PlayerView,
  RuleViolation,
  Seat,
} from "@canastra/bots";

/** Who plays a seat, as broadcast. A disconnected human stays a human, marked away. */
export type SeatOccupant =
  | { kind: "human"; name: string; connected: boolean }
  | { kind: "bot"; botId: string }
  | { kind: "empty" };

/**
 * Public table state. Everything here is information any observer at a
 * physical table would have: who sits where, whose turn, the match scores.
 */
export interface TableState {
  seats: [SeatOccupant, SeatOccupant, SeatOccupant, SeatOccupant];
  /** "lobby" before the first match; afterwards the engine's phase. */
  phase: "lobby" | Phase;
  turn: Seat | null;
  scores: [number, number] | null;
  handNumber: number | null;
}

export type ClientMessage =
  /** Identify. `token` (from localStorage) reclaims a seat after a drop. */
  | { type: "hello"; name: string; token?: string }
  /**
   * Claim an empty/bot/away seat. Mid-match this takes over the seat's hand.
   * The server answers with a fresh `welcome` (token now bound) plus a `view`.
   */
  | { type: "sit"; seat: Seat }
  /** Leave the seat; a bot takes over. Clears the token→seat binding. */
  | { type: "stand" }
  /** Begin a match (any seated human); empty seats become bots. */
  | { type: "start" }
  /**
   * Play an engine `Action`. There is deliberately no seat field — the server
   * passes the connection's bound seat to `Match.apply`, so a client cannot
   * claim to be someone else.
   */
  | { type: "action"; action: Action }
  /** Escape hatch for a dead-ended turn; `Match.restartTurn` semantics. */
  | { type: "restartTurn" }
  /** At HandOver: bank the hand now instead of waiting out the pause. Seated humans only. */
  | { type: "settle" };

export type ServerMessage =
  /** Answers `hello`. `seat` non-null means a reclaim happened. Store the token. */
  | { type: "welcome"; token: string; seat: Seat | null; table: TableState }
  /** Broadcast after any change to the public state. */
  | { type: "table"; table: TableState }
  /**
   * Private, per seated player: the engine's `observe(state, seat)`. Pushed
   * whenever it may have changed — after every accepted action, on deal, on
   * `sit`/takeover, and on reclaim.
   *
   * `legal` is the same seat's engine `legal_actions`, so a scripted human can
   * rank them the way the harness driver does. It names only the player's own
   * cards and the public discard pile — no other seat's cards — so it holds to
   * F6.
   */
  | { type: "view"; view: PlayerView; legal: Action[] }
  /** One move-log line, broadcast. */
  | { type: "event"; text: string }
  /** The rule that rejected your action (or why a lobby command was refused), to you only. */
  | { type: "refused"; violation: RuleViolation }
  /** §13 itemised settlement, broadcast at hand end — never mid-hand (it sums both partners' hands). */
  | { type: "handOver"; scores: [HandScore, HandScore] };

/**
 * Parse an incoming frame. Anything malformed is dropped (null) rather than
 * throwing — a garbage frame should cost nothing.
 */
export function parseClientMessage(text: string): ClientMessage | null {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return null;
  }
  if (typeof value !== "object" || value === null) {
    return null;
  }
  const msg = value as { type?: unknown; name?: unknown; seat?: unknown; action?: unknown };
  switch (msg.type) {
    case "hello":
      if (typeof msg.name !== "string") return null;
      break;
    case "sit":
      if (typeof msg.seat !== "number") return null;
      break;
    case "action":
      if (
        typeof msg.action !== "object" ||
        msg.action === null ||
        typeof (msg.action as { type?: unknown }).type !== "string"
      ) {
        return null;
      }
      break;
    case "stand":
    case "start":
    case "restartTurn":
    case "settle":
      break;
    default:
      return null;
  }
  return msg as ClientMessage;
}
