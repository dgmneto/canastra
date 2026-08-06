/**
 * Snapshot persistence: the table as one JSON file, rewritten on every
 * change. The file is small (a `GameState` serializes to tens of KB) and the
 * game is slow (one action per human heartbeat), so a synchronous write per
 * action is fine — and it means a server restart never costs a match.
 */

import { existsSync, mkdirSync, readFileSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

/** A seat as it survives a restart. A human seat keeps its token so the player can reclaim. */
export type SaveSeat =
  | { kind: "empty" }
  | { kind: "human"; name: string; token: string }
  | { kind: "bot"; botId: string };

export interface SaveGame {
  seats: SaveSeat[];
  /** Null while the table is in the lobby (no match to lose). */
  match: {
    seed: string;
    startedAt: string;
    /** The per-seat lineup recorded in the log header: bot ids and human names. */
    lineup: string[];
    snapshot: string;
    log: unknown[];
  } | null;
}

/** Write tmp-then-rename, so a crash mid-write never leaves half a file. */
export function saveGame(file: string, save: SaveGame): void {
  mkdirSync(dirname(file), { recursive: true });
  const tmp = `${file}.tmp`;
  writeFileSync(tmp, JSON.stringify(save));
  renameSync(tmp, file);
}

/** A missing or corrupt save costs the match in progress, not the server. */
export function loadGame(file: string): SaveGame | null {
  if (!existsSync(file)) return null;
  try {
    return JSON.parse(readFileSync(file, "utf8")) as SaveGame;
  } catch {
    return null;
  }
}

export function clearGame(file: string): void {
  try {
    unlinkSync(file);
  } catch {
    // already gone
  }
}
