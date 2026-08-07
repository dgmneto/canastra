/**
 * The engine's wire shapes, hand-written.
 *
 * These live in `@canastra/bots` because every consumer (the bots themselves,
 * the harness, and the web sandbox) needs them, and the bots are the leaf that
 * everything else depends on. The literals they describe are pinned on the Rust
 * side by `engine/crates/canastra-engine/tests/boundary.rs` — that file is the
 * authority, and if it changes these have to follow.
 */

/** A card in the engine's compact codec: `"6H"`, `"TS"`, `"2D"`, `"JOKER"`. */
export type Card = string;

/** A bare rank as a single character: `"4"`…`"9"`, `"T"`, `"J"`, `"Q"`, `"K"`, `"A"`. */
export type Rank = string;

export type Suit = "Clubs" | "Diamonds" | "Hearts" | "Spades";

export type Phase =
  | "AwaitingDraw"
  | "AwaitingRefusalChoice"
  | "Melding"
  | "HandOver"
  | "MatchOver";

/**
 * One position in a laid sequence.
 *
 * A natural card carries nothing but itself — its rank is in the code. A wild
 * carries the two facts a client cannot cheaply compute: which rank it is
 * standing in for, and whether §9 has locked it in place.
 */
export interface MeldSlot {
  card: Card;
  standingInRank?: Rank;
  locked?: boolean;
}

export type Meld =
  | { kind: "Sequence"; suit: Suit; cards: MeldSlot[] }
  | { kind: "Aces"; aces: Card[]; wild: Card | null };

export interface TeamTable {
  melds: Meld[];
  red_threes: Card[];
  opened: boolean;
}

/** Seats cross the wire as plain numbers 0–3. */
export type Seat = number;

export interface PlayerView {
  seat: Seat;
  hand: Card[];
  frozen: Card[];
  tables: [TeamTable, TeamTable];
  discard: Card[];
  stock_count: number;
  hand_counts: [number, number, number, number];
  scores: [number, number];
  phase: Phase;
  turn: Seat;
  dealer: Seat;
  hand_number: number;
  went_out: Seat | null;
  opening_minimum: number;
  laid_value: number;
  took_pile: boolean;
  refusal_available: boolean;
  pending_refusal: Card | null;
}

export type MeldTarget = { kind: "NewMeld" } | { kind: "Existing"; meld: number };

export type Action =
  | { type: "Draw" }
  | { type: "KeepDrawnCard" }
  | { type: "RefuseDrawnCard" }
  | { type: "TakeDiscardPile"; core: [Card, Card]; target: MeldTarget }
  | { type: "LayMeld"; cards: Card[] }
  | { type: "AddToMeld"; meld: number; cards: Card[] }
  | { type: "Discard"; card: Card }
  | { type: "EndTurnWithoutDiscard" };

/**
 * Refusals arrive as structured objects, not sentences, so the UI can react to
 * the specific rule. Only the discriminant is modelled precisely; the payload
 * fields vary by variant and are read loosely for display.
 */
export interface RuleViolation {
  error: string;
  [field: string]: unknown;
}

export function isRuleViolation(value: unknown): value is RuleViolation {
  return typeof value === "object" && value !== null && typeof (value as RuleViolation).error === "string";
}

/**
 * §13: one partnership's score for one hand, itemised.
 *
 * Read mid-hand it is a running total rather than a result: `going_out_bonus`
 * is still zero, and `hand_cards` keeps falling as cards go down.
 */
export interface HandScore {
  canastra_bonus: number;
  going_out_bonus: number;
  red_three_bonus: number;
  table_cards: number;
  hand_cards: number;
  /**
   * §13.3: a partnership that never opened takes a flat −300 and nothing else
   * counts — no hand negatives, no red 3s. Zero for a team that opened.
   */
  unopened_penalty: number;
}

export function handScoreTotal(score: HandScore): number {
  return (
    score.canastra_bonus +
    score.going_out_bonus +
    score.red_three_bonus +
    score.table_cards -
    score.hand_cards +
    score.unopened_penalty
  );
}

/**
 * §13 card values, mirroring `Card::points` in the engine.
 *
 * Only the bots use this, to guess whether §6's opening minimum is within
 * reach. Anything that has to be *right* — a hand's score — comes back from
 * `Game::handScore` instead, so §13 is not reimplemented here.
 */
export function cardValue(card: Card): number {
  if (card === "JOKER") return 50;
  const rank = card[0];
  if (rank === "3") return 0;
  if (rank === "2") return 20;
  if (rank === "A") return 15;
  if ("89TJQK".includes(rank)) return 10;
  return 5;
}
