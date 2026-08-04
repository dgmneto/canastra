/**
 * Headless bot-vs-bot runs, for telling whether a new bot is actually better.
 *
 * Watching one match proves nothing: a single deal swings hundreds of points on
 * whether three red 3s landed on the right side. The only honest answer comes
 * from many matches on many seeds, which is what this is for.
 *
 * Exposed on `globalThis.lab` — it drives the engine directly and never touches
 * React, so a few hundred matches run in seconds.
 */

import { Match } from "./match";
import { step } from "./driver";
import { botById } from "./bots";
import { makeRng } from "./rng";
import type { PlayerView } from "./types";

export interface MatchResult {
  seed: bigint;
  scores: [number, number];
  hands: number;
  /** Team index, or null if the cap was hit before anyone reached 5000. */
  winner: 0 | 1 | null;
  restarts: number;
}

/**
 * Play one match to §14's finish.
 *
 * `maxActions` is a guard, not a rule: a bot that proposes nothing legal gets
 * its turn restarted, and a bot that does so every turn would otherwise spin
 * forever. Hitting it is a bug in a bot, and the result reports it as no winner.
 */
export function runMatch(seed: bigint, botIds: string[], maxActions = 200_000): MatchResult {
  const match = new Match(seed, botIds);
  const rng = makeRng(Number(seed % 2147483647n) || 1);
  let safeMode = false;
  let restarts = 0;

  for (let actions = 0; actions < maxActions; actions += 1) {
    const view = match.views()[0] as PlayerView;
    if (view.phase === "MatchOver") {
      return finish(seed, view, restarts, view.scores[0] > view.scores[1] ? 0 : 1);
    }

    const acting = view.turn;
    const result = step(match, match.views()[acting], botById(botIds[acting]), { rng, safeMode });
    if (!result) break;

    if (result.action === "restartTurn") {
      safeMode = true;
      restarts += 1;
    } else if (
      result.action !== "settleHand" &&
      (result.action.type === "Discard" || result.action.type === "EndTurnWithoutDiscard")
    ) {
      safeMode = false;
    }
  }

  const view = match.views()[0] as PlayerView;
  return finish(seed, view, restarts, null);
}

function finish(seed: bigint, view: PlayerView, restarts: number, winner: 0 | 1 | null): MatchResult {
  return { seed, scores: view.scores, hands: view.hand_number, winner, restarts };
}

export interface SeriesReport {
  matches: number;
  /** Wins for the partnership in seats 0 and 2. */
  winsNos: number;
  winsEles: number;
  unfinished: number;
  meanScoreNos: number;
  meanScoreEles: number;
  meanHands: number;
  restarts: number;
}

/**
 * Run the same lineup over a range of seeds.
 *
 * Seats 0 and 2 are one partnership and 1 and 3 the other (§2), so a fair
 * head-to-head is `["a", "b", "a", "b"]`.
 */
export function series(botIds: string[], count = 100, firstSeed = 1n): SeriesReport {
  const results: MatchResult[] = [];
  for (let at = 0n; at < BigInt(count); at += 1n) results.push(runMatch(firstSeed + at, botIds));

  const sum = (pick: (result: MatchResult) => number) =>
    results.reduce((total, result) => total + pick(result), 0);

  return {
    matches: results.length,
    winsNos: results.filter((result) => result.winner === 0).length,
    winsEles: results.filter((result) => result.winner === 1).length,
    unfinished: results.filter((result) => result.winner === null).length,
    meanScoreNos: Math.round(sum((result) => result.scores[0]) / results.length),
    meanScoreEles: Math.round(sum((result) => result.scores[1]) / results.length),
    meanHands: Math.round(sum((result) => result.hands) / results.length),
    restarts: sum((result) => result.restarts),
  };
}

/**
 * Both seatings of a head-to-head, so the result cannot be an artefact of who
 * deals first or which seat the lucky deal landed in.
 */
export function headToHead(a: string, b: string, count = 100) {
  return {
    [`${a} in seats 0,2`]: series([a, b, a, b], count),
    [`${b} in seats 0,2`]: series([b, a, b, a], count),
  };
}
