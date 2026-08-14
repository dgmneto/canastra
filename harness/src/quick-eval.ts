import { readFileSync } from "node:fs";
import { makeJsonWeightsBot, registerBot, makeRng, botById } from "@canastra/bots";
import type { Bot } from "@canastra/bots";
import { loadEngine } from "./load-node";
import { Match } from "./match";
import { step } from "./driver";

async function main(): Promise<void> {
  await loadEngine();
  const weights = JSON.parse(
    readFileSync(process.argv[2], "utf8"),
  ) as Parameters<typeof makeJsonWeightsBot>[0];
  registerBot(makeJsonWeightsBot(weights, "nn"));

  const opponent = process.argv[3] ?? "random";
  const maxActions = Number(process.argv[4] ?? "5000");

  const match = new Match(7n, ["nn", opponent, "nn", opponent]);
  const rng = makeRng(7);
  let safe = false;
  let restarts = 0;

  for (let actions = 0; actions < maxActions; actions += 1) {
    const view = match.views()[0];
    if (view.phase === "MatchOver") {
      console.log(`match over at ${actions} actions, ${restarts} restarts`);
      console.log(`scores: ${view.scores}`);
      return;
    }
    const acting = view.turn;
    const bot: Bot = botById(acting === 0 || acting === 2 ? "nn" : opponent);
    const result = step(match, match.views()[acting], bot, {
      rng,
      safeMode: safe,
      encode: () => match.encodeState(acting),
    });
    if (!result) break;
    if (result.action === "restartTurn") {
      safe = true;
      restarts += 1;
      if (restarts <= 5) console.log(`restart at action ${actions} (total ${restarts})`);
    } else if (
      result.action !== "settleHand" &&
      (result.action.type === "Discard" || result.action.type === "EndTurnWithoutDiscard")
    ) {
      safe = false;
    }
  }

  const view = match.views()[0];
  console.log(`stopped at ${maxActions} actions, ${restarts} restarts`);
  console.log(`scores: ${view.scores}, hand: ${view.hand_number}, phase: ${view.phase}`);
}

main().catch(console.error);
