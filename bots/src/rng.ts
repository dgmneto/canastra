/**
 * Deterministic seeded randomness for bots.
 *
 * A bot that rolls differently on a replay would make the action log
 * unreproducible, so bots never touch `Math.random`.
 */

export interface Rng {
  (): number;
}

/**
 * mulberry32 — small and seedable.
 */
export function makeRng(seed: number): Rng {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
