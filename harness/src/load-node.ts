/**
 * Loads the engine wasm the Node way.
 *
 * The committed bindings (in `web/src/engine`) default to the browser path —
 * `fetch` over a URL. Node has no URL to fetch, but the generated `init` accepts
 * a `WebAssembly.Module`, so we compile the bytes ourselves and hand it over.
 * The `Game` class is the same one the web page uses.
 *
 * This module is Node-only (`node:fs`); deliver it to the browser by other
 * means. It exists so the harness CLI can drive the real engine rather than a
 * JavaScript reimplementation.
 */

import { readFileSync } from "node:fs";
import initWasm from "../../web/src/engine/canastra.js";

export { Game } from "../../web/src/engine/canastra.js";

let ready: Promise<unknown> | null = null;

/** Compile and instantiate the committed wasm once, whoever asks first. */
export function loadEngine(): Promise<unknown> {
  ready ??= (async () => {
    const bytes = readFileSync(new URL("../../web/src/engine/canastra_bg.wasm", import.meta.url));
    await initWasm({ module_or_path: new WebAssembly.Module(bytes) });
  })();
  return ready;
}
