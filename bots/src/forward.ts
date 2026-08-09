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
  activation: string;
}

export interface WeightsJson {
  format: string;
  arch: WeightsArch;
  params: Record<string, { shape: number[]; data: number[] }>;
}

export const WEIGHTS_FORMAT = "canastra-weights@1";

export interface Layer {
  weight: number[]; // row-major, shape [out][in]
  bias: number[];
  out: number;
  inn: number;
}

/**
 * A weights file compiled once into its layer arrays, so the per-call shape
 * validation in `validateWeights` runs a single time instead of on every
 * action scored (~41 calls per ply per bot).
 */
export interface CompiledWeights {
  arch: WeightsArch;
  trunk: Layer[];
  head: Layer[];
}

export function validateWeights(weights: WeightsJson): void {
  if (weights.format !== WEIGHTS_FORMAT) {
    throw new Error(`unsupported weights format: ${weights.format}`);
  }
  if (weights.arch.activation !== "tanh") {
    throw new Error("only tanh weights are supported");
  }
  for (const name of layerNames(weights.arch)) {
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

/** Validate and precompile a weights file. Call once, then reuse the result. */
export function compileWeights(weights: WeightsJson): CompiledWeights {
  validateWeights(weights);
  const { arch } = weights;
  const trunk: Layer[] = [];
  let inn = arch.obs;
  for (const name of layerNames(arch).filter((n) => n.startsWith("trunk."))) {
    trunk.push(layer(weights, name, inn));
    inn = trunk[trunk.length - 1].out;
  }
  const head: Layer[] = [];
  inn = trunk[trunk.length - 1].out + arch.act;
  const headNames = layerNames(arch).filter((n) => n.startsWith("head."));
  for (let i = 0; i < headNames.length; i += 1) {
    head.push(layer(weights, headNames[i], inn));
    inn = head[head.length - 1].out;
  }
  return { arch, trunk, head };
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
export function embed(cw: CompiledWeights, obs: number[]): number[] {
  if (obs.length !== cw.arch.obs) {
    throw new Error(`observation is ${obs.length} wide, weights expect ${cw.arch.obs}`);
  }
  let x = obs;
  for (const l of cw.trunk) x = tanh(apply(l, x));
  return x;
}

/** One action's score: head over [embedding; features], final layer linear. */
export function scoreAction(cw: CompiledWeights, emb: number[], feats: number[]): number {
  if (feats.length !== cw.arch.act) {
    throw new Error(`action row is ${feats.length} wide, weights expect ${cw.arch.act}`);
  }
  let x = [...emb, ...feats];
  for (let i = 0; i < cw.head.length - 1; i += 1) x = tanh(apply(cw.head[i], x));
  return apply(cw.head[cw.head.length - 1], x)[0];
}