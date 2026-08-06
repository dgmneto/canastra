/**
 * Loads the engine wasm the browser way.
 *
 * The committed bindings default to `fetch` over a URL, which is exactly right
 * in a browser — and exactly wrong for the Node harness, which hands over a
 * `WebAssembly.Module` instead (see `@canastra/harness`'s node loader). Each
 * environment brings its own loader; this one serves the sandbox page.
 */

import initWasm from "./engine/canastra.js";

let wasmReady: Promise<unknown> | null = null;

/** Load the wasm module once, whoever asks first. */
export function loadEngine(): Promise<unknown> {
  wasmReady ??= initWasm();
  return wasmReady;
}
