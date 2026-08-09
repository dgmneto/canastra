/**
 * Play trained weights against a registered bot, both seatings, and print
 * the head-to-head report.
 *
 * Usage: npx tsx harness/src/eval-nn.ts <weights.json> <opponent-id> [count]
 *
 * This is the spec's external validation path: the Python evaluator measures
 * genomes against genomes; this one measures a weights file against the
 * heuristic bots on the TS side, through the same harness everyone else uses.
 */

import { readFileSync } from "node:fs";
import { makeJsonWeightsBot, registerBot, type WeightsJson } from "@canastra/bots";
import { loadEngine } from "./load-node";
import { headToHead } from "./series";

const [weightsPath, opponent, count = "40"] = process.argv.slice(2);
if (!weightsPath || !opponent) {
  console.error("usage: npx tsx harness/src/eval-nn.ts <weights.json> <opponent-id> [count]");
  process.exit(2);
}

async function main(): Promise<void> {
  await loadEngine();
  const weights = JSON.parse(readFileSync(weightsPath, "utf8")) as WeightsJson;
  registerBot(makeJsonWeightsBot(weights, "nn"));
  const report = headToHead("nn", opponent, Number(count));
  console.log(JSON.stringify(report, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});