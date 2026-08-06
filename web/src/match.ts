/**
 * The engine, wrapped for the sandbox.
 *
 * The `Match` wrapper now lives in `@canastra/harness`, shared with the CLI.
 * This file re-exports it and adds a browser `loadEngine`, so the sandbox's
 * import graph stays put while the logic lives in one place.
 */

export { Match, logToText } from "@canastra/harness";
export type { LogLine } from "@canastra/harness";
export { loadEngine } from "./loadEngine";
