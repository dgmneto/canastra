/**
 * The weights-JSON forward pass.
 *
 * Generic over the arch: layer widths come from the file, so architecture
 * changes never touch this code — only the format string does. Mirrors
 * `training/python/canastra_train/policy.py` exactly: trunk layers all tanh,
 * head hidden layers tanh, the final `head.out` layer linear.
 *
 * Format contract (pinned, spec Section E):
 *   { "format": "canastra-weights@1",
 *     "arch": { "obs", "act", "trunk": [...], "head": [...], "activation": "tanh" },
 *     "params": { "<name>.weight": { "shape": [out, in], "data": [...] }, ... } }
 * Layer names: trunk.{i}, head.{i}, head.out.
 */

export interface WeightsArch {
  obs: number;
  act: number;
  trunk: number[];
  head: number[];
  activation: "tanh";
}

export interface WeightsJson {
  format: string;
  arch: WeightsArch;
  params: Record<string, { shape: number[]; data: number[] }>;
}

export const WEIGHTS_FORMAT = "canastra-weights@1";

interface Layer {
  weight: number[]; // row-major, shape [out][in]
  bias: number[];
  out: number;
  inn: number;
}

export function validateWeights(weights: WeightsJson): void {
  if (weights.format !== WEIGHTS_FORMAT) {
    throw new Error(`unsupported weights format: ${weights.format}`);
  }
  if (weights.arch.activation !== "tanh") {
    throw new Error("only tanh weights are supported");
  }
  const names = layerNames(weights.arch);
  for (const name of names) {
    for (const part of ["weight", "bias"]) {
      const key = `${name}.${part}`;
      if (!(key in weights.params)) throw new Error(`missing params: ${key}`);
    }
  }
}

function layerNames(arch: WeightsArch): string[] {
  const names: string[] = [];
  for (let i = 0; i < arch.trunk.length; i += 1) names.push(`trunk.${i}`);
  for (let i = 0; i < arch.head.length; i += 1) names.push(`head.${i}`);
  names.push("head.out");
  return names;
}

function layer(weights: WeightsJson, name: string, expectedIn: number): Layer {
  const weight = weights.params[`${name}.weight`];
  const bias = weights.params[`${name}.bias`];
  const [out, inn] = weight.shape;
  if (inn !== expectedIn) {
    throw new Error(`${name}: weight expects input ${inn}, got ${expectedIn}`);
  }
  if (weight.data.length !== out * inn) throw new Error(`${name}: weight data length`);
  if (bias.data.length !== out) throw new Error(`${name}: bias data length`);
  return { weight: weight.data, bias: bias.data, out, inn };
}

function apply(layer: Layer, input: number[]): number[] {
  const out = new Array<number>(layer.out);
  for (let o = 0; o < layer.out; o += 1) {
    let acc = layer.bias[o];
    const base = o * layer.inn;
    for (let i = 0; i < layer.inn; i += 1) acc += layer.weight[base + i] * input[i];
    out[o] = acc;
  }
  return out;
}

const tanh = (xs: number[]) => xs.map(Math.tanh);

/** The observation embedding (trunk, all-tanh). */
export function embed(weights: WeightsJson, obs: number[]): number[] {
  if (obs.length !== weights.arch.obs) {
    throw new Error(`observation is ${obs.length} wide, weights expect ${weights.arch.obs}`);
  }
  let x = obs;
  let inn = weights.arch.obs;
  for (let i = 0; i < weights.arch.trunk.length; i += 1) {
    const l = layer(weights, `trunk.${i}`, inn);
    x = tanh(apply(l, x));
    inn = l.out;
  }
  return x;
}

/** One action's score: head over [embedding; features], final layer linear. */
export function scoreAction(weights: WeightsJson, emb: number[], feats: number[]): number {
  if (feats.length !== weights.arch.act) {
    throw new Error(`action row is ${feats.length} wide, weights expect ${weights.arch.act}`);
  }
  let x = [...emb, ...feats];
  let inn = emb.length + weights.arch.act;
  for (let i = 0; i < weights.arch.head.length; i += 1) {
    const l = layer(weights, `head.${i}`, inn);
    x = tanh(apply(l, x));
    inn = l.out;
  }
  const out = layer(weights, "head.out", inn);
  return apply(out, x)[0];
}