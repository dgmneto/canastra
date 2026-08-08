/**
 * Public API for `@canastra/harness`.
 *
 * The engine driver: a `Match` wrapper plus the `step`/`series` functions that
 * run bots against it. The web sandbox imports from here; the CLI is built on
 * top of the same pieces.
 */

export { Match, logToText } from "./match.js";
export type { LogLine } from "./match.js";
export { step, label, penaltyLabel } from "./driver.js";
export type { StepResult } from "./driver.js";
export { runMatch, series, headToHead } from "./series.js";
export type { MatchResult, SeriesReport } from "./series.js";
