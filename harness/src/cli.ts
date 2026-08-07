#!/usr/bin/env node
/**
 * The harness executable.
 *
 * Takes a seed and a list of bot names, plays a whole match to §14's finish,
 * and prints the outcome as JSON Lines on stdout — the final score first, then
 * the complete move log. Deterministic: the same seed and lineup replay to the
 * same bytes.
 *
 * Usage:
 *   canastra-harness --seed 7 random random-plus random random-plus
 *
 * One name fills all four seats; two names are treated as partners, seating
 * [a, b, a, b]; four names are seated one per seat. `--seed` defaults to 7.
 */

import { BOTS, botById, makeRng } from "@canastra/bots";
import type { PlayerView, Seat } from "@canastra/bots";
import { Match, logToText } from "./match.js";
import { step } from "./driver.js";
import { loadEngine } from "./load-node.js";

const HELP = `canastra-harness [--seed N] <bot> [<bot> [<bot> <bot>]]

Plays a match to the 5000-point finish and writes JSON Lines to stdout:
the outcome line first, then one line per move.

  --seed N    the match seed (default 7)
  -h, --help  show this help

Bot names (one fills all seats, two seat as partners [a, b, a, b]):
  ${BOTS.map((bot) => `${bot.id} — ${bot.name}`).join("\n  ")}`;

const MAX_ACTIONS = 200_000;

interface Options {
  seed: bigint;
  lineup: string[];
}

function parseArgs(argv: string[]): Options {
  let seed = 7n;
  const names: string[] = [];

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--seed") {
      const next = argv[++i];
      if (next === undefined) {
        console.error("--seed needs a value");
        process.exit(2);
      }
      seed = BigInt(next);
    } else if (arg === "--help" || arg === "-h") {
      console.log(HELP);
      process.exit(0);
    } else if (arg.startsWith("--")) {
      console.error(`unknown option: ${arg}`);
      process.exit(2);
    } else {
      names.push(arg);
    }
  }

  for (const name of names) {
    if (!BOTS.some((bot) => bot.id === name)) {
      console.error(`unknown bot '${name}'. Known: ${BOTS.map((bot) => bot.id).join(", ")}`);
      process.exit(2);
    }
  }

  let lineup: string[];
  if (names.length === 0) {
    lineup = ["random", "random", "random", "random"];
  } else if (names.length === 1) {
    lineup = [0, 1, 2, 3].map(() => names[0]);
  } else if (names.length === 2) {
    // §2: partners sit facing each other, so [a, b, a, b].
    lineup = [names[0], names[1], names[0], names[1]];
  } else if (names.length === 4) {
    lineup = names;
  } else {
    console.error("pass 1, 2 or 4 bot names");
    process.exit(2);
  }

  return { seed, lineup };
}

async function main(): Promise<void> {
  const { seed, lineup } = parseArgs(process.argv.slice(2));
  await loadEngine();

  const match = new Match(seed, lineup);
  const rng = makeRng(Number(seed % 2147483647n) || 1);
  let safeMode = false;
  let restarts = 0;

  for (let actions = 0; actions < MAX_ACTIONS; actions += 1) {
    const view = match.views()[0] as PlayerView;
    if (view.phase === "MatchOver") break;

    const acting = view.turn as Seat;
    const result = step(match, match.views()[acting], botById(lineup[acting]), {
      rng,
      safeMode,
      encode: () => match.encodeState(acting),
    });
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

  const final = match.views()[0] as PlayerView;
  const winner: 0 | 1 | null =
    final.scores[0] > final.scores[1] ? 0 : final.scores[0] < final.scores[1] ? 1 : null;

  // The outcome first, then the move log — every line is JSON.
  console.log(
    JSON.stringify({
      type: "result",
      seed: seed.toString(),
      lineup,
      bot_names: lineup.map((id) => botById(id).name),
      scores: final.scores,
      winner,
      hands: final.hand_number,
      restarts,
      unfinished: final.phase !== "MatchOver",
    }),
  );
  process.stdout.write(logToText(match.log));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
