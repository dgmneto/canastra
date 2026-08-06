/**
 * Headless bot-vs-bot runs, shared with the harness CLI.
 *
 * Exposed on `globalThis.lab` — it drives the engine directly and never touches
 * React, so a few hundred matches run in seconds.
 */

export { runMatch, series, headToHead } from "@canastra/harness";
export type { MatchResult, SeriesReport } from "@canastra/harness";
