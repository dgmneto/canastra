/**
 * The roster.
 *
 * To add a bot: write `src/<name>.ts` exporting a `Bot`, then add it to
 * `BOTS`. Nothing else needs to change — the harness driver, the web seat
 * pickers and the replay log all read from this list.
 */

import type { Bot } from "./bot";
import { randomBot } from "./random";
import { randomPlusBot } from "./random-plus";
import { randomDiscardHungryBot } from "./random-discard-hungry";
import { makeJsonWeightsBot } from "./json-weights";
import type { WeightsJson } from "./forward";
import randomInit from "./fixtures/random-init.json";

export const BOTS: Bot[] = [randomBot, randomPlusBot, randomDiscardHungryBot];

/**
 * Register a bot that cannot be a static constant — e.g. one built from a
 * weights file. Idempotent by id. Registered bots join the harness CLI, the
 * sandbox pickers, and anything else that reads `BOTS`.
 */
export function registerBot(bot: Bot): void {
  if (!BOTS.some((existing) => existing.id === bot.id)) BOTS.push(bot);
}

/** The committed random-init fixture, so a network can play without training. */
export const nnRandomBot = makeJsonWeightsBot(randomInit as WeightsJson, "nn-random");
registerBot(nnRandomBot);

export const DEFAULT_BOT = randomBot;

export function botById(id: string): Bot {
  return BOTS.find((bot) => bot.id === id) ?? DEFAULT_BOT;
}

export * from "./bot";
export * from "./types";
export * from "./forward";
export { makeJsonWeightsBot } from "./json-weights";
export { makeRng, type Rng } from "./rng";