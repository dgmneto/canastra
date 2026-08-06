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

export const BOTS: Bot[] = [randomBot, randomPlusBot, randomDiscardHungryBot];

export const DEFAULT_BOT = randomBot;

export function botById(id: string): Bot {
  return BOTS.find((bot) => bot.id === id) ?? DEFAULT_BOT;
}

export * from "./bot";
export * from "./types";
export { makeRng, type Rng } from "./rng";
