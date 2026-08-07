/**
 * A bot that plays trained weights.
 *
 * The network sees exactly what training saw — the engine's own
 * `encodeState` — and ranks the legal list by score. Deterministic: no rng,
 * so a weights file always plays the same game from the same seed.
 */

import type { Action, PlayerView } from "./types";
import type { Bot, BotContext } from "./bot";
import { embed, scoreAction, validateWeights, type WeightsJson } from "./forward";

export function makeJsonWeightsBot(weights: WeightsJson, id: string): Bot {
  validateWeights(weights);
  return {
    id,
    name: `NN ${id}`,
    blurb: "Policy network loaded from JSON weights.",

    candidates(_view: PlayerView, legal: Action[], context: BotContext): Action[] {
      const encoded = context.encode?.();
      if (!encoded) {
        throw new Error(`${id}: neural bots need context.encode — the caller must wire encodeState`);
      }
      if (encoded.actions.length !== legal.length) {
        throw new Error(`${id}: encoded rows (${encoded.actions.length}) != legal moves (${legal.length})`);
      }
      if (encoded.obs.length !== weights.arch.obs) {
        throw new Error(`${id}: observation width ${encoded.obs.length} != ${weights.arch.obs}`);
      }
      const emb = embed(weights, encoded.obs);
      const scored = legal.map((action, index) => ({
        action,
        score: scoreAction(weights, emb, encoded.actions[index]),
      }));
      scored.sort((a, b) => b.score - a.score);
      return scored.map((entry) => entry.action);
    },
  };
}