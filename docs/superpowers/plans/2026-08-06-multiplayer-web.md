# Multiplayer Web App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `web/` into a multiplayer Canastra app: an authoritative Node WebSocket server holding the real engine, browsers as thin clients.

**Architecture:** Per `docs/superpowers/specs/2026-08-06-multiplayer-web-design.md` — one global table, bots fill empty seats, pick-a-seat lobby, token-based reclaim, snapshot persistence. Two new workspace packages (`protocol/`, `server/`), two small harness additions, and a second Vite page for the game client. The sandbox moves to `/sandbox.html` untouched.

**Tech Stack:** Node + `ws` + `tsx` (server), React 19 + Vite MPA (client), the existing Rust→wasm engine driven through `@canastra/harness`'s `Match`.

**Spec:** `docs/superpowers/specs/2026-08-06-multiplayer-web-design.md` (approved). Read it first.

**Conventions to hold:**
- All work happens in the worktree `/Users/dgmneto/web-game`, branch `web-game`.
- JS verification is `npm run typecheck` (no JS test framework exists — deliberately) plus the server smoke script built in Task 5. Rust gates (`cargo test --workspace`, `clippy -D warnings`, `fmt --check` from `engine/`) must stay green; the engine is not modified.
- Commit style: short imperative capitalized summary (see `git log`).
- Two deliberate deviations from the spec's wording, both in the spec's spirit:
  1. `@canastra/protocol` has a **type-only** dependency on `@canastra/bots` (the wire types live there; duplicating them would drift). "Zero dependencies" becomes "zero runtime dependencies".
  2. `loadEngine` is exposed as a **subpath export** `@canastra/harness/node`, not from the index — the index is imported by the browser sandbox, and `load-node.ts` imports `node:fs`.

---

### Task 1: `protocol/` package + root wiring

**Files:**
- Create: `protocol/package.json`
- Create: `protocol/tsconfig.json`
- Create: `protocol/src/index.ts`
- Modify: `package.json` (root: workspaces + typecheck chain)

- [ ] **Step 1: Create `protocol/package.json`**

```json
{
  "name": "@canastra/protocol",
  "private": true,
  "type": "module",
  "exports": {
    ".": "./src/index.ts"
  },
  "scripts": {
    "typecheck": "tsc --noEmit"
  },
  "dependencies": {
    "@canastra/bots": "*"
  },
  "devDependencies": {
    "typescript": "^5.9.0"
  }
}
```

- [ ] **Step 2: Create `protocol/tsconfig.json`** (mirrors `bots/tsconfig.json` exactly)

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "noEmit": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "isolatedModules": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src"]
}
```

- [ ] **Step 3: Create `protocol/src/index.ts`**

```ts
/**
 * The client↔server wire protocol for the multiplayer table.
 *
 * One WebSocket at `/ws`, JSON messages discriminated by `type` — the same
 * tagging style the engine uses for `Action`. This package is the shared
 * language of `web/` (the game client) and `server/`; it holds no logic and
 * no runtime dependencies (the wire types themselves live in `@canastra/bots`,
 * the leaf everything else depends on).
 *
 * The information rules that matter are encoded in what the messages can
 * carry: `view` is the engine's per-seat `observe`, `table` is public
 * information only, and there is no message that could put a `GameState`, a
 * snapshot, or another seat's hand on the wire (ADVERSARIAL-REVIEW.md F6).
 */

import type {
  Action,
  HandScore,
  Phase,
  PlayerView,
  RuleViolation,
  Seat,
} from "@canastra/bots";

/** Who plays a seat, as broadcast. A disconnected human stays a human, marked away. */
export type SeatOccupant =
  | { kind: "human"; name: string; connected: boolean }
  | { kind: "bot"; botId: string }
  | { kind: "empty" };

/**
 * Public table state. Everything here is information any observer at a
 * physical table would have: who sits where, whose turn, the match scores.
 */
export interface TableState {
  seats: [SeatOccupant, SeatOccupant, SeatOccupant, SeatOccupant];
  /** "lobby" before the first match; afterwards the engine's phase. */
  phase: "lobby" | Phase;
  turn: Seat | null;
  scores: [number, number] | null;
  handNumber: number | null;
}

export type ClientMessage =
  /** Identify. `token` (from localStorage) reclaims a seat after a drop. */
  | { type: "hello"; name: string; token?: string }
  /**
   * Claim an empty/bot/away seat. Mid-match this takes over the seat's hand.
   * The server answers with a fresh `welcome` (token now bound) plus a `view`.
   */
  | { type: "sit"; seat: Seat }
  /** Leave the seat; a bot takes over. Clears the token→seat binding. */
  | { type: "stand" }
  /** Begin a match (any seated human); empty seats become bots. */
  | { type: "start" }
  /**
   * Play an engine `Action`. There is deliberately no seat field — the server
   * passes the connection's bound seat to `Match.apply`, so a client cannot
   * claim to be someone else.
   */
  | { type: "action"; action: Action }
  /** Escape hatch for a dead-ended turn; `Match.restartTurn` semantics. */
  | { type: "restartTurn" }
  /** At HandOver: bank the hand now instead of waiting out the pause. Seated humans only. */
  | { type: "settle" };

export type ServerMessage =
  /** Answers `hello`. `seat` non-null means a reclaim happened. Store the token. */
  | { type: "welcome"; token: string; seat: Seat | null; table: TableState }
  /** Broadcast after any change to the public state. */
  | { type: "table"; table: TableState }
  /**
   * Private, per seated player: the engine's `observe(state, seat)`. Pushed
   * whenever it may have changed — after every accepted action, on deal, on
   * `sit`/takeover, and on reclaim.
   */
  | { type: "view"; view: PlayerView }
  /** One move-log line, broadcast. */
  | { type: "event"; text: string }
  /** The rule that rejected your action (or why a lobby command was refused), to you only. */
  | { type: "refused"; violation: RuleViolation }
  /** §13 itemised settlement, broadcast at hand end — never mid-hand (it sums both partners' hands). */
  | { type: "handOver"; scores: [HandScore, HandScore] };

/**
 * Parse an incoming frame. Anything malformed is dropped (null) rather than
 * throwing — a garbage frame should cost nothing.
 */
export function parseClientMessage(text: string): ClientMessage | null {
  try {
    const value: unknown = JSON.parse(text);
    if (typeof value === "object" && value !== null && typeof (value as { type?: unknown }).type === "string") {
      return value as ClientMessage;
    }
  } catch {
    // fall through
  }
  return null;
}
```

- [ ] **Step 4: Wire the root `package.json`**

Add `"protocol"` and `"server"` to `workspaces` (server's package is created in Task 3; adding it now is fine since npm tolerates a missing dir until `npm install` — create the `server/package.json` skeleton in Task 3 before running install, or add workspaces entries in the tasks that create them. **Do this now:**

```json
"workspaces": ["bots", "harness", "protocol", "web"],
"scripts": {
  ...,
  "typecheck:protocol": "npm run typecheck --prefix protocol",
  "typecheck": "npm run typecheck:bots && npm run typecheck:protocol && npm run typecheck:harness && npm run typecheck:web",
  ...
}
```

(Leave the existing `build:engine`, `typecheck:bots`, `typecheck:harness`, `typecheck:web`, `harness` scripts untouched; `"server"` is added to workspaces in Task 3.)

- [ ] **Step 5: Verify**

Run: `npm install && npm run typecheck`
Expected: installs cleanly, all four typechecks pass with no output/errors.

- [ ] **Step 6: Commit**

```bash
git add protocol/ package.json package-lock.json
git commit -m "Add @canastra/protocol with the client-server wire messages"
```

---

### Task 2: Harness additions — `Match.restore`, `Match.snapshot`, `./node` export

**Files:**
- Modify: `harness/src/match.ts` (add `snapshot()`, `restore()`; drop `readonly` on `seed`)
- Modify: `harness/package.json` (add `./node` subpath export)

Context: the server persists a match via `Game.snapshot()` and resumes it via `Game.restore()`. `Match` wraps `Game` with the log and the turn checkpoint, so persistence needs to go through it. The Node wasm loader stays off the index (the browser imports the index; `load-node.ts` imports `node:fs`) and gets a subpath export instead.

- [ ] **Step 1: Modify `harness/src/match.ts`**

In `Match`, change `readonly seed: bigint;` to `seed: bigint;` and `readonly log: LogLine[];` to `log: LogLine[];` (both are assigned in `restore`; still treated as read-only everywhere else), and add two members after `settleHand()`:

```ts
  /**
   * The whole state as JSON. This is F6's sharp end — all four hands, the
   * stock order, the seed — so it is for the server's persistence only and
   * must never reach a browser.
   */
  snapshot(): string {
    return this.game.snapshot();
  }

  /**
   * Rebuild a match from a snapshot, for server persistence and replay
   * resume. `meta` carries what the snapshot does not: which bot (or human
   * name) sat where and when the match started, so the log header stays a
   * truthful record across a restart.
   *
   * The turn checkpoint becomes the restored position itself: a `restartTurn`
   * shortly after a restore replays at most the turn in progress, which is
   * the same position the restart would have produced anyway.
   */
  static restore(
    snapshot: string,
    meta: { seed: bigint; bots: string[]; startedAt: string; log?: LogLine[] },
  ): Match {
    const match = Object.create(Match.prototype) as Match;
    match.seed = meta.seed;
    match.game = Game.restore(snapshot);
    match.turnStart = snapshot;
    match.log = meta.log ?? [
      { seed: meta.seed.toString(), startedAt: meta.startedAt, bots: meta.bots },
    ];
    return match;
  }
```

(`Object.create` bypasses the constructor; the field assignments are legal because `restore` is inside the class. `Game` is already imported.)

- [ ] **Step 2: Modify `harness/package.json`**

```json
"exports": {
  ".": "./src/index.ts",
  "./node": "./src/load-node.ts"
},
```

- [ ] **Step 3: Verify restore round-trips a real position**

Run from the repo root:

```bash
npx tsx -e "
import { loadEngine } from '@canastra/harness/node';
import { Match } from '@canastra/harness';
await loadEngine();
const a = new Match(7n, ['random','random','random','random']);
a.apply(1, { type: 'Draw' });
const snap = a.snapshot();
const b = Match.restore(snap, { seed: 7n, bots: ['random','random','random','random'], startedAt: 'x', log: [...a.log] });
const va = JSON.stringify(a.views()); const vb = JSON.stringify(b.views());
if (va !== vb) { console.error('views differ'); process.exit(1); }
if (b.log.length !== a.log.length) { console.error('log lost'); process.exit(1); }
console.log('restore round-trips');
"
```

Expected output: `restore round-trips`

- [ ] **Step 4: Verify**

Run: `npm run typecheck`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add harness/
git commit -m "Let a Match snapshot and restore itself, for server persistence"
```

---

### Task 3: `server/` package — scaffold, persistence, HTTP+WS server, lobby

**Files:**
- Create: `server/package.json`
- Create: `server/tsconfig.json`
- Create: `server/src/persistence.ts`
- Create: `server/src/table.ts` (lobby only this task: connect/disconnect/hello/sit/stand, `tableState`, broadcasting; match methods arrive in Task 4)
- Create: `server/src/server.ts` (HTTP static + WS + keepalive; entry-agnostic so the smoke script can boot it in-process)
- Create: `server/src/main.ts` (thin entry)
- Modify: `package.json` (root: add `"server"` to workspaces, `typecheck:server`)

- [ ] **Step 1: Create `server/package.json`**

```json
{
  "name": "@canastra/server",
  "private": true,
  "type": "module",
  "scripts": {
    "start": "tsx src/main.ts",
    "dev": "tsx watch src/main.ts",
    "smoke": "tsx src/smoke.ts",
    "typecheck": "tsc --noEmit"
  },
  "dependencies": {
    "@canastra/bots": "*",
    "@canastra/harness": "*",
    "@canastra/protocol": "*",
    "ws": "^8.18.0"
  },
  "devDependencies": {
    "@types/node": "^22.10.0",
    "@types/ws": "^8.5.10",
    "tsx": "^4.19.2",
    "typescript": "^5.9.0"
  }
}
```

- [ ] **Step 2: Create `server/tsconfig.json`** (mirrors `harness/tsconfig.json` exactly — `types: ["node"]`, DOM libs included)

Copy `harness/tsconfig.json` verbatim.

- [ ] **Step 3: Create `server/src/persistence.ts`**

```ts
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
```

- [ ] **Step 4: Create `server/src/table.ts`** (lobby complete; `start`/`action`/etc. refuse with `"NotRunning"` stubs replaced in Task 5 — structure it so Task 4 only adds methods)

```ts
/**
 * The one global table.
 *
 * Everything session-like lives here: who is connected, who sits where, the
 * match in progress, and the bots covering empty or abandoned seats. The
 * engine stays behind `Match` — this class never adjudicates a rule, it only
 * routes messages and drives bots.
 *
 * One process, one table: players find the game by connecting, not by picking
 * a room (the "single global table" decision in the spec).
 */

import { randomUUID } from "node:crypto";
import type { WebSocket } from "ws";
import { Match } from "@canastra/harness";
import type { LogLine } from "@canastra/harness";
import { makeRng } from "@canastra/bots";
import type { Action, Rng, Seat } from "@canastra/bots";
import type { ClientMessage, SeatOccupant, ServerMessage, TableState } from "@canastra/protocol";
import type { SaveGame, SaveSeat } from "./persistence.js";

/** The policy that covers a seat with no human deciding for it. */
const COVER_BOT = "random-plus";

/**
 * A human seat has a live socket; a covered seat is a human's seat being
 * played by a bot until they reclaim it with their token (the name and token
 * survive; the socket does not).
 */
type SeatState =
  | { kind: "empty" }
  | { kind: "human"; name: string; token: string; ws: WebSocket }
  | { kind: "covered"; name: string; token: string }
  | { kind: "bot"; botId: string };

interface Client {
  ws: WebSocket;
  name: string;
  token: string;
  seat: Seat | null;
}

export interface TableOptions {
  /** Pacing between bot actions, so humans can follow. 0 in the smoke run. */
  botDelayMs?: number;
  /** The pause at HandOver before the hand is banked. */
  settleDelayMs?: number;
  /** Persistence hook, called after every change worth saving. */
  onChange?: (save: SaveGame | null) => void;
}

export class Table {
  private clients = new Map<WebSocket, Client>();
  private seats: SeatState[] = [
    { kind: "empty" },
    { kind: "empty" },
    { kind: "empty" },
    { kind: "empty" },
  ];
  private match: Match | null = null;
  private rng: Rng = makeRng(1);
  /** §6 backstop bookkeeping — the same rule the harness CLI follows. */
  private safeMode = false;
  private botTimer: ReturnType<typeof setTimeout> | null = null;
  private settleTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(private options: TableOptions = {}) {}

  /** Rebuild a table from a save. Human seats come back covered: their owners reclaim by token. */
  static restore(save: SaveGame, options: TableOptions = {}): Table {
    const table = new Table(options);
    table.seats = save.seats.map((seat): SeatState => {
      if (seat.kind === "human") return { kind: "covered", name: seat.name, token: seat.token };
      if (seat.kind === "bot") return { kind: "bot", botId: seat.botId };
      return { kind: "empty" };
    });
    if (save.match) {
      table.match = Match.restore(save.match.snapshot, {
        seed: BigInt(save.match.seed),
        bots: save.match.lineup,
        startedAt: save.match.startedAt,
        log: save.match.log as LogLine[],
      });
      table.rng = makeRng(Number(BigInt(save.match.seed) % 2147483647n) || 1);
    }
    return table;
  }

  /** What gets written to disk after every change. Null match → the lobby. */
  serialize(): SaveGame {
    const seats: SaveSeat[] = this.seats.map((seat): SaveSeat => {
      if (seat.kind === "human" || seat.kind === "covered")
        return { kind: "human", name: seat.name, token: seat.token };
      if (seat.kind === "bot") return { kind: "bot", botId: seat.botId };
      return { kind: "empty" };
    });
    if (!this.match) return { seats, match: null };
    const header = this.match.log[0] as { startedAt: string; bots: string[] };
    return {
      seats,
      match: {
        seed: this.match.seed.toString(),
        startedAt: header.startedAt,
        lineup: header.bots,
        snapshot: this.match.snapshot(),
        log: this.match.log,
      },
    };
  }

  // --- connection management (called by server.ts) ---

  connect(ws: WebSocket): void {
    this.clients.set(ws, { ws, name: "", token: "", seat: null });
  }

  disconnect(ws: WebSocket): void {
    const client = this.clients.get(ws);
    this.clients.delete(ws);
    if (client?.seat != null) {
      this.coverSeat(client.seat);
      this.afterChange();
    }
  }

  handle(ws: WebSocket, message: ClientMessage): void {
    const client = this.clients.get(ws);
    if (!client) return;
    switch (message.type) {
      case "hello":
        return this.hello(client, message.name, message.token);
      case "sit":
        return this.sit(client, message.seat);
      case "stand":
        return this.stand(client);
      case "start":
        return this.start(client);
      case "action":
        return this.act(client, message.action);
      case "restartTurn":
        return this.restart(client);
      case "settle":
        return this.settleMessage(client);
    }
  }

  // --- lobby ---

  private hello(client: Client, name: string, token?: string): void {
    client.name = name.trim() || "Jogador";
    // Reclaim: a seat remembering this token? A token that matches nothing is
    // stale (the seat was taken by someone else) and this is a new arrival.
    const at = token
      ? this.seats.findIndex(
          (seat) =>
            (seat.kind === "human" || seat.kind === "covered") && seat.token === token,
        )
      : -1;
    if (at >= 0) {
      const seat = this.seats[at] as { name: string; token: string };
      // Two live tabs with the same token: the newest connection wins the seat.
      for (const other of this.clients.values()) {
        if (other !== client && other.seat === at) other.seat = null;
      }
      this.seats[at] = { kind: "human", name: seat.name, token: seat.token, ws: client.ws };
      client.name = seat.name;
      client.token = seat.token;
      client.seat = at;
    } else {
      client.token = token || randomUUID();
      client.seat = null;
    }
    this.send(client.ws, {
      type: "welcome",
      token: client.token,
      seat: client.seat,
      table: this.tableState(),
    });
    if (client.seat !== null) this.sendView(client.seat);
    this.broadcastTable();
  }

  private sit(client: Client, seat: Seat): void {
    if (seat < 0 || seat > 3) return;
    const target = this.seats[seat];
    if (target.kind === "human") {
      return this.refuse(client, "SeatTaken", `seat ${seat} is taken`);
    }
    if (client.seat !== null) this.vacate(client); // stand from wherever we were
    this.seats[seat] = { kind: "human", name: client.name, token: client.token, ws: client.ws };
    client.seat = seat;
    // The token is now bound to the seat — say so, and show the hand if there is one.
    this.send(client.ws, {
      type: "welcome",
      token: client.token,
      seat,
      table: this.tableState(),
    });
    if (this.match) this.sendView(seat);
    this.afterChange();
  }

  private stand(client: Client): void {
    if (client.seat === null) return;
    this.vacate(client);
    this.afterChange();
  }

  /** Leave a seat: a bot covers mid-match; the seat empties in the lobby. Token binding cleared. */
  private vacate(client: Client): void {
    const seat = client.seat;
    if (seat === null) return;
    this.seats[seat] = this.match
      ? { kind: "bot", botId: COVER_BOT }
      : { kind: "empty" };
    client.seat = null;
  }

  /** A dropped connection: the seat stays the human's, played by a bot, reclaimable by token. */
  private coverSeat(seat: Seat): void {
    const current = this.seats[seat];
    if (current.kind === "human") {
      this.seats[seat] = { kind: "covered", name: current.name, token: current.token };
    }
  }

  // --- match (implemented in Task 4) ---

  private start(client: Client): void {
    this.refuse(client, "NotRunning", "match flow lands in Task 4");
  }

  private act(client: Client, action: Action): void {
    void action;
    this.refuse(client, "NotRunning", "no match in progress");
  }

  private restart(client: Client): void {
    this.refuse(client, "NotRunning", "no match in progress");
  }

  private settleMessage(client: Client): void {
    this.refuse(client, "NotRunning", "no hand to settle");
  }

  // --- plumbing ---

  private refuse(client: Client, error: string, detail: string): void {
    this.send(client.ws, { type: "refused", violation: { error, detail } });
  }

  private send(ws: WebSocket, message: ServerMessage): void {
    if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(message));
  }

  private broadcast(message: ServerMessage): void {
    for (const client of this.clients.keys()) this.send(client, message);
  }

  private broadcastTable(): void {
    this.broadcast({ type: "table", table: this.tableState() });
  }

  private sendView(seat: Seat): void {
    const occupant = this.seats[seat];
    if (!this.match || occupant.kind !== "human") return;
    this.send(occupant.ws, { type: "view", view: this.match.views()[seat] });
  }

  private occupantOf(seat: SeatState): SeatOccupant {
    switch (seat.kind) {
      case "human":
        return { kind: "human", name: seat.name, connected: true };
      case "covered":
        return { kind: "human", name: seat.name, connected: false };
      case "bot":
        return { kind: "bot", botId: seat.botId };
      case "empty":
        return { kind: "empty" };
    }
  }

  private tableState(): TableState {
    const view = this.match?.views()[0] ?? null;
    return {
      seats: this.seats.map((seat) => this.occupantOf(seat)) as TableState["seats"],
      phase: view ? view.phase : "lobby",
      turn: view && view.phase !== "MatchOver" ? view.turn : null,
      scores: view ? view.scores : null,
      handNumber: view ? view.hand_number : null,
    };
  }

  /** Every state change routes through here: tell everyone, save, then let bots play. */
  private afterChange(): void {
    this.broadcastTable();
    for (let seat = 0; seat < 4; seat++) this.sendView(seat as Seat);
    this.options.onChange?.(this.serialize());
  }
}
```

Note: `afterChange` deliberately does not call `pump()` yet (bots arrive in Task 4), and `sendView` no-ops while `match` is null — safe for lobby-only use. The import list above is exactly what Task 3 uses; Task 4 adds `label`, `step` (from `@canastra/harness`) and `botById` (from `@canastra/bots`) when it wires up bot driving.

- [ ] **Step 5: Create `server/src/server.ts`**

```ts
/**
 * HTTP + WebSocket serving, kept apart from the table so the smoke run can
 * boot one in-process on an ephemeral port.
 *
 * One port carries everything: static files from web/dist when built, and the
 * game WebSocket at /ws. In development the Vite dev server proxies /ws here.
 */

import { createServer, type Server as HttpServer } from "node:http";
import type { AddressInfo } from "node:net";
import { existsSync, readFileSync, statSync } from "node:fs";
import { extname, join, normalize, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { WebSocketServer, type WebSocket } from "ws";
import { loadEngine } from "@canastra/harness/node";
import { parseClientMessage } from "@canastra/protocol";
import { Table, type TableOptions } from "./table.js";
import { clearGame, loadGame, saveGame } from "./persistence.js";

const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".map": "application/json",
  ".wasm": "application/wasm",
  ".svg": "image/svg+xml",
  ".png": "image/png",
};

export interface ServerOptions extends TableOptions {
  port: number;
  /** Where the save file lives; null keeps the table in memory (the smoke run). */
  saveFile: string | null;
  /** The built web client; defaults to web/dist relative to this package. */
  distDir?: string;
}

export interface RunningServer {
  port(): number;
  close(): Promise<void>;
}

export async function startServer(options: ServerOptions): Promise<RunningServer> {
  await loadEngine();

  const distDir =
    options.distDir ?? fileURLToPath(new URL("../../web/dist", import.meta.url));

  const persist = (save: import("./persistence.js").SaveGame | null): void => {
    if (!options.saveFile) return;
    if (save) saveGame(options.saveFile, save);
    else clearGame(options.saveFile);
  };

  // A save that no longer parses or fails the engine's invariant check costs
  // the match in progress, not the server.
  let table: Table;
  const restored = options.saveFile ? loadGame(options.saveFile) : null;
  try {
    table = restored
      ? Table.restore(restored, { ...options, onChange: persist })
      : new Table({ ...options, onChange: persist });
  } catch {
    table = new Table({ ...options, onChange: persist });
  }

  const http: HttpServer = createServer((req, res) => {
    if (!existsSync(distDir)) {
      res.writeHead(200, { "content-type": "text/plain; charset=utf-8" });
      res.end(
        "canastra server is up. The web client is not built — run `npm run build --prefix web`, or use the Vite dev server on :5173.\n",
      );
      return;
    }
    const pathname = (req.url ?? "/").split("?")[0];
    const file = normalize(join(distDir, pathname === "/" ? "index.html" : pathname));
    if (file !== distDir && !file.startsWith(distDir + sep)) {
      res.writeHead(403);
      res.end();
      return;
    }
    if (!existsSync(file) || !statSync(file).isFile()) {
      res.writeHead(404);
      res.end("not found");
      return;
    }
    res.writeHead(200, { "content-type": MIME[extname(file)] ?? "application/octet-stream" });
    res.end(readFileSync(file));
  });

  const wss = new WebSocketServer({ server: http, path: "/ws" });
  /** Dead-connection detection: miss two pongs (~20 s) and the close path runs. */
  const alive = new Map<WebSocket, boolean>();

  wss.on("connection", (ws) => {
    alive.set(ws, true);
    table.connect(ws);
    ws.on("pong", () => alive.set(ws, true));
    ws.on("message", (data) => {
      alive.set(ws, true);
      const message = parseClientMessage(String(data));
      if (message) table.handle(ws, message);
    });
    ws.on("close", () => {
      alive.delete(ws);
      table.disconnect(ws);
    });
  });

  const keepalive = setInterval(() => {
    for (const ws of wss.clients) {
      if (alive.get(ws) === false) {
        ws.terminate(); // 'close' fires, the seat gets covered
        continue;
      }
      alive.set(ws, false);
      ws.ping();
    }
  }, 10_000);

  await new Promise<void>((resolve) => http.listen(options.port, resolve));

  return {
    port: () => (http.address() as AddressInfo).port,
    close: () =>
      new Promise((resolve) => {
        clearInterval(keepalive);
        for (const ws of wss.clients) ws.terminate();
        wss.close();
        http.close(() => resolve());
      }),
  };
}
```

- [ ] **Step 6: Create `server/src/main.ts`**

```ts
/** The production entry: one table, one port, a save file under server/data/. */

import { fileURLToPath } from "node:url";
import { startServer } from "./server.js";

const port = Number(process.env.PORT ?? 3001);
const saveFile = fileURLToPath(new URL("../data/game.json", import.meta.url));

startServer({ port, saveFile }).then((server) => {
  console.log(`canastra server on http://localhost:${server.port()}`);
});
```

- [ ] **Step 7: Root `package.json` — add server to workspaces and typecheck**

```json
"workspaces": ["bots", "harness", "protocol", "server", "web"],
"scripts": {
  ...,
  "typecheck:server": "npm run typecheck --prefix server",
  "typecheck": "npm run typecheck:bots && npm run typecheck:protocol && npm run typecheck:harness && npm run typecheck:server && npm run typecheck:web",
  ...
}
```

Also add `"server/data/"` to the root `.gitignore` (the save file is runtime state).

- [ ] **Step 8: Verify**

Run: `npm install && npm run typecheck`
Expected: all five typechecks pass.

Then boot the server and exercise the lobby with a throwaway client:

```bash
npm run start --prefix server &
sleep 2
npx tsx -e "
import WebSocket from 'ws';
const ws = new WebSocket('ws://localhost:3001/ws');
ws.on('open', () => ws.send(JSON.stringify({ type: 'hello', name: 'Ana' })));
ws.on('message', (data) => {
  const msg = JSON.parse(String(data));
  console.log('got', msg.type);
  if (msg.type === 'welcome') { ws.send(JSON.stringify({ type: 'sit', seat: 0 })); }
  if (msg.type === 'table' && msg.table.seats[0].kind === 'human') { console.log('seated:', msg.table.seats[0].name); process.exit(0); }
});
setTimeout(() => { console.error('timeout'); process.exit(1); }, 5000);
"
kill %1
```

Expected: `got welcome`, `got table`, `seated: Ana`.

- [ ] **Step 9: Commit**

```bash
git add server/ package.json package-lock.json .gitignore
git commit -m "Scaffold @canastra/server: one global table, lobby, WS+static serving"
```

---

### Task 4: Table match lifecycle — start, actions, bot driving, settle flow

**Files:**
- Modify: `server/src/table.ts` (replace the four Task-3 stubs; add `pump`, `beginHandOver`, `settleNow`, `endMatch`, `name`, `botIdFor`; wire `pump()` into `afterChange`; add the `label`, `step`, `botById` imports)
- Modify: `server/src/main.ts` (add a `BOT_DELAY_MS` env override, used by Task 4's verification)

- [ ] **Step 1: Replace the match section of `server/src/table.ts`**

Replace the four stubs with:

```ts
  // --- match ---

  private start(client: Client): void {
    if (client.seat === null) return this.refuse(client, "NotSeated", "sit first");
    if (this.match && this.match.views()[0].phase !== "MatchOver") {
      return this.refuse(client, "MatchRunning", "a match is already in progress");
    }
    if (this.settleTimer) {
      clearTimeout(this.settleTimer);
      this.settleTimer = null;
    }
    // Empty seats become bots; human and covered seats stay as they are.
    this.seats = this.seats.map((seat) =>
      seat.kind === "empty" ? { kind: "bot", botId: COVER_BOT } : seat,
    );
    // The header records who sat where — bot ids and human names alike,
    // because the log is a record of a match.
    const lineup = this.seats.map((seat) =>
      seat.kind === "bot" ? seat.botId : (seat as { name: string }).name,
    );
    const seed = BigInt(Math.floor(Math.random() * 2 ** 31));
    this.match = new Match(seed, lineup);
    this.rng = makeRng(Number(seed % 2147483647n) || 1);
    this.safeMode = false;
    this.broadcast({ type: "event", text: `match started (seed ${seed})` });
    this.afterChange();
  }

  private act(client: Client, action: Action): void {
    if (client.seat === null || !this.match) {
      return this.refuse(client, "NotSeated", "you are not at the table");
    }
    const refused = this.match.apply(client.seat, action);
    if (refused) return this.refuse(client, refused.error, JSON.stringify(refused));
    this.broadcast({
      type: "event",
      text: label(action, client.seat, this.name(client.seat)),
    });
    this.afterChange();
  }

  private restart(client: Client): void {
    if (client.seat === null || !this.match) {
      return this.refuse(client, "NotSeated", "you are not at the table");
    }
    this.match.restartTurn(client.seat);
    this.broadcast({
      type: "event",
      text: label("restartTurn", client.seat, this.name(client.seat)),
    });
    this.afterChange();
  }

  private settleMessage(client: Client): void {
    if (client.seat === null) return this.refuse(client, "NotSeated", "spectators cannot settle");
    this.settleNow();
  }

  private settleNow(): void {
    if (!this.match) return;
    if (this.match.views()[0].phase !== "HandOver") return;
    if (this.settleTimer) {
      clearTimeout(this.settleTimer);
      this.settleTimer = null;
    }
    const refused = this.match.settleHand();
    if (refused) return;
    this.broadcast({ type: "event", text: "hand settled" });
    this.afterChange();
  }

  /**
   * Drive the bots until the game needs a human, a hand ends, or the match
   * ends. Each bot action is paced by a timer so people can follow; anything
   * that changes the state re-enters through `afterChange`.
   */
  private pump(): void {
    if (!this.match || this.botTimer) return;
    const view = this.match.views()[0];
    if (view.phase === "HandOver") return this.beginHandOver();
    if (view.phase === "MatchOver") return this.endMatch();
    if (this.seats[view.turn].kind === "human") return; // a person decides

    this.botTimer = setTimeout(() => {
      this.botTimer = null;
      if (!this.match) return;
      const now = this.match.views()[0];
      // The state may have moved while the timer was pending (a stand, a reclaim).
      if (now.phase === "HandOver" || now.phase === "MatchOver") return this.pump();
      const acting = now.turn as Seat;
      if (this.seats[acting].kind === "human") return;
      const bot = botById(this.botIdFor(acting));
      const result = step(this.match, this.match.views()[acting], bot, {
        rng: this.rng,
        safeMode: this.safeMode,
      });
      if (!result) return;
      // The same safeMode rule as the harness CLI: a restarted turn retries
      // with draw-and-discard only, cleared by the next completed turn.
      if (result.action === "restartTurn") this.safeMode = true;
      else if (
        result.action !== "settleHand" &&
        (result.action.type === "Discard" || result.action.type === "EndTurnWithoutDiscard")
      )
        this.safeMode = false;
      this.broadcast({
        type: "event",
        text: label(result.action, acting, this.name(acting)),
      });
      this.afterChange();
    }, this.options.botDelayMs ?? 500);
  }

  private beginHandOver(): void {
    if (this.settleTimer || !this.match) return;
    // §13 is only safe to show now: the hand is over, and partners would be
    // counting together at a physical table. Mid-hand it would leak the sum
    // of both partners' hands.
    this.broadcast({ type: "handOver", scores: this.match.handScores() });
    this.settleTimer = setTimeout(() => {
      this.settleTimer = null;
      this.settleNow();
    }, this.options.settleDelayMs ?? 10_000);
  }

  private endMatch(): void {
    const scores = this.match!.views()[0].scores;
    this.broadcast({ type: "event", text: `match over — ${scores[0]} vs ${scores[1]}` });
    // The finished match stays on the table until someone presses start again.
  }

  private name(seat: Seat): string {
    const occupant = this.seats[seat];
    if (occupant.kind === "human" || occupant.kind === "covered") return occupant.name;
    if (occupant.kind === "bot") return botById(occupant.botId).name;
    return `seat ${seat}`;
  }

  private botIdFor(seat: Seat): string {
    const occupant = this.seats[seat];
    return occupant.kind === "bot" ? occupant.botId : COVER_BOT;
  }
```

And in `afterChange()`, append `this.pump();` as the last statement. Add `"settle"` handling note: `settleMessage` already routes. Update imports: add `label`, `step` from `@canastra/harness`, `botById` from `@canastra/bots`.

- [ ] **Step 2: Verify**

Run: `npm run typecheck`
Expected: pass.

Then a scripted check — one human seat + three bots should reach HandOver on their own:

```bash
npm run start --prefix server &
sleep 2
npx tsx -e "
import WebSocket from 'ws';
const ws = new WebSocket('ws://localhost:3001/ws');
let table = null; let view = null;
ws.on('open', () => ws.send(JSON.stringify({ type: 'hello', name: 'Ana' })));
ws.on('message', (data) => {
  const msg = JSON.parse(String(data));
  if (msg.type === 'table') table = msg.table;
  if (msg.type === 'view') {
    view = msg.view;
    if (view.turn === view.seat) {
      // safe human: draw then discard the first card that goes through
      if (view.phase === 'AwaitingDraw') ws.send(JSON.stringify({ type: 'action', action: { type: 'Draw' } }));
      else if (view.phase === 'Melding') tryDiscard(0);
      else if (view.phase === 'AwaitingRefusalChoice') ws.send(JSON.stringify({ type: 'action', action: { type: 'KeepDrawnCard' } }));
    }
    if (view.phase === 'HandOver') { console.log('hand over; stock:', view.stock_count); process.exit(0); }
  }
  if (msg.type === 'refused' && view && view.phase === 'Melding') tryDiscard(last + 1);
});
let last = 0;
function tryDiscard(at) {
  last = at;
  if (at >= view.hand.length) { ws.send(JSON.stringify({ type: 'restartTurn' })); last = -1; return; }
  ws.send(JSON.stringify({ type: 'action', action: { type: 'Discard', card: view.hand[at] } }));
}
ws.on('open', () => setTimeout(() => { ws.send(JSON.stringify({ type: 'sit', seat: 0 })); setTimeout(() => ws.send(JSON.stringify({ type: 'start' })), 200); }, 100));
setTimeout(() => { console.error('timeout — no HandOver in 120s'); process.exit(1); }, 120000);
"
kill %1
```

Expected: `hand over; stock: <n>` within the timeout (bot actions are 500 ms apart, so a full hand takes a minute or two — the 120 s budget may be tight; if it times out, re-run with `botDelayMs` overridden via a `BOT_DELAY_MS` env var you add to `main.ts`: `botDelayMs: Number(process.env.BOT_DELAY_MS ?? 500)`. Add that line as part of this task.)

Also delete `server/data/game.json` after this manual run so it doesn't linger: `rm -f server/data/game.json`.

- [ ] **Step 3: Commit**

```bash
git add server/
git commit -m "Run matches on the table: bot driving, settle pacing, match flow"
```

---

### Task 5: The smoke run — full match, reclaim, persistence

**Files:**
- Create: `server/src/smoke.ts`
- Modify: `package.json` (root: add `"smoke": "npm run smoke --prefix server"`)

This is the spec's scripted end-to-end check: fake WS clients against a real in-process server. One seat is played by a scripted "human" that mirrors the harness driver's candidate loop.

- [ ] **Step 1: Create `server/src/smoke.ts`**

```ts
/**
 * The end-to-end smoke run — on demand, not a test framework:
 *
 *   npm run smoke
 *
 * Boots a real server in-process (ephemeral port, temp save file), connects
 * fake players over real WebSockets, and checks the promises the spec makes:
 * the lobby works, a match with bots runs to §14, a dropped player is covered
 * and can reclaim, and a restarted server resumes the table.
 *
 * The scripted humans reuse a real bot's candidate list against their own
 * `view`, exactly the way the harness driver does — so the check plays legal
 * Canastra without re-implementing any rules.
 */

import { mkdtempSync, rmSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import WebSocket from "ws";
import { botById, makeRng } from "@canastra/bots";
import type { Action, PlayerView, Seat } from "@canastra/bots";
import type { ClientMessage, ServerMessage, TableState } from "@canastra/protocol";
import { startServer, type RunningServer } from "./server.js";

let failures = 0;

function ok(condition: boolean, what: string): void {
  console.log(`${condition ? "ok" : "NOT OK"} - ${what}`);
  if (!condition) failures++;
}

async function waitFor(
  condition: () => boolean,
  what: string,
  timeoutMs = 15_000,
): Promise<boolean> {
  const started = Date.now();
  while (!condition()) {
    if (Date.now() - started > timeoutMs) {
      ok(false, `${what} (timed out)`);
      return false;
    }
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  ok(true, what);
  return true;
}

/** A scripted player: connects, and whenever its view says it must act, plays like the bot. */
class Player {
  view: PlayerView | null = null;
  table: TableState | null = null;
  seat: Seat | null = null;
  token: string | null = null;
  handOvers = 0;

  private ws: WebSocket | null = null;
  private rng = makeRng(42);
  private safeMode = false;
  private driving = false;
  private pending: ((accepted: boolean) => void) | null = null;
  private settledHand = -1;

  constructor(
    private name: string,
    private port: number,
    token?: string,
  ) {
    this.token = token ?? null;
  }

  connect(): Promise<void> {
    return new Promise((resolve) => {
      const ws = new WebSocket(`ws://localhost:${this.port}/ws`);
      this.ws = ws;
      ws.on("open", () => {
        this.send({ type: "hello", name: this.name, token: this.token ?? undefined });
        resolve();
      });
      ws.on("message", (data) => this.receive(JSON.parse(String(data)) as ServerMessage));
    });
  }

  terminate(): void {
    this.ws?.terminate(); // an unclean drop, like a dead network
  }

  send(message: ClientMessage): void {
    this.ws?.send(JSON.stringify(message));
  }

  sit(seat: Seat): void {
    this.send({ type: "sit", seat });
  }

  start(): void {
    this.send({ type: "start" });
  }

  private receive(message: ServerMessage): void {
    switch (message.type) {
      case "welcome":
        this.token = message.token;
        this.seat = message.seat;
        this.table = message.table;
        break;
      case "table":
        this.table = message.table;
        break;
      case "view": {
        this.view = message.view;
        this.pending?.(true);
        this.pending = null;
        void this.drive();
        break;
      }
      case "refused": {
        this.pending?.(false);
        this.pending = null;
        break;
      }
      case "handOver":
        this.handOvers++;
        break;
      case "event":
        break;
    }
  }

  /** Play our turn by proposing the bot's candidates until one is accepted. */
  private async drive(): Promise<void> {
    if (this.driving || !this.view || this.seat === null) return;
    this.driving = true;
    try {
      while (
        this.view &&
        this.view.turn === this.seat &&
        (this.view.phase === "AwaitingDraw" ||
          this.view.phase === "Melding" ||
          this.view.phase === "AwaitingRefusalChoice")
      ) {
        const bot = botById("random-plus");
        const candidates = bot.candidates(this.view, { rng: this.rng, safeMode: this.safeMode });
        let accepted: Action | null = null;
        for (const candidate of candidates) {
          if (await this.act(candidate)) {
            accepted = candidate;
            break;
          }
        }
        if (!accepted) {
          // Out of ideas — the same dead-end the driver's restart exists for.
          this.safeMode = true;
          this.send({ type: "restartTurn" });
          await new Promise((resolve) => setTimeout(resolve, 50));
        } else if (accepted.type === "Discard" || accepted.type === "EndTurnWithoutDiscard") {
          this.safeMode = false;
        }
      }
      if (this.view && this.view.phase === "HandOver" && this.settledHand !== this.view.hand_number) {
        this.settledHand = this.view.hand_number;
        this.send({ type: "settle" });
      }
    } finally {
      this.driving = false;
    }
  }

  /** The next message addressed to us — a view or a refusal — settles the attempt. */
  private act(action: Action): Promise<boolean> {
    return new Promise((resolve) => {
      this.pending = resolve;
      this.send({ type: "action", action });
    });
  }
}

async function main(): Promise<void> {
  const dir = mkdtempSync(join(tmpdir(), "canastra-smoke-"));
  const saveFile = join(dir, "game.json");
  let server: RunningServer | null = await startServer({
    port: 0,
    saveFile,
    botDelayMs: 0,
    settleDelayMs: 50,
  });

  try {
    const port = server.port();

    // --- lobby ---
    const ana = new Player("Ana", port);
    await ana.connect();
    ana.sit(0);
    await waitFor(() => ana.seat === 0 && ana.table?.seats[0].kind === "human", "Ana sits at seat 0");

    const bruno = new Player("Bruno", port);
    await bruno.connect();
    bruno.sit(1);
    await waitFor(() => bruno.seat === 1, "Bruno sits at seat 1");

    // --- the match runs ---
    ana.start();
    await waitFor(() => ana.view !== null && ana.view.hand.length === 15, "the deal reaches a seated player");

    // --- a dropped player is covered and reclaims ---
    await waitFor(() => ana.table !== null && ana.table.phase !== "lobby", "the match is running");
    bruno.terminate();
    await waitFor(
      () =>
        ana.table?.seats[1].kind === "human" &&
        ana.table.seats[1].connected === false,
      "a dropped seat is covered by a bot",
    );
    const brunoReturns = new Player("Bruno", port, bruno.token ?? undefined);
    await brunoReturns.connect();
    const reclaimed = await waitFor(
      () => brunoReturns.seat === 1 && brunoReturns.view !== null,
      "the token reclaims the seat mid-match",
    );
    let activeBruno: Player = reclaimed ? brunoReturns : bruno;
    void activeBruno;

    // --- a restarted server resumes the table ---
    await waitFor(() => existsSync(saveFile), "the save file exists");
    const scoresBefore = ana.table?.scores ?? null;
    await server.close();
    server = await startServer({ port: 0, saveFile, botDelayMs: 0, settleDelayMs: 50 });
    const port2 = server.port();
    const carla = new Player("Carla", port2);
    await carla.connect();
    await waitFor(
      () => carla.table !== null && carla.table.phase !== "lobby" && carla.table.phase !== undefined,
      "a restarted server resumes the match",
    );
    ok(
      JSON.stringify(carla.table?.scores) === JSON.stringify(scoresBefore),
      "the resumed table kept the scores",
    );
    const anaReturns = new Player("Ana", port2, ana.token ?? undefined);
    await anaReturns.connect();
    await waitFor(() => anaReturns.seat === 0 && anaReturns.view !== null, "Ana reclaims after the restart");

    // --- the match runs to §14 ---
    await waitFor(
      () => anaReturns.table?.phase === "MatchOver" || carla.table?.phase === "MatchOver",
      "the match runs to MatchOver",
      300_000,
    );
    const scores = anaReturns.table?.scores ?? carla.table?.scores;
    console.log(`final score: ${scores?.[0]} vs ${scores?.[1]}`);
    ok((anaReturns.handOvers + carla.handOvers) > 0, "at least one hand settlement was broadcast");
  } finally {
    await server?.close();
    rmSync(dir, { recursive: true, force: true });
  }

  if (failures) {
    console.error(`${failures} check(s) failed`);
    process.exit(1);
  }
  console.log("smoke: all good");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
```

- [ ] **Step 2: Run the smoke**

Run: `npm run smoke --prefix server`
Expected: every line `ok - ...`, ending with `smoke: all good`. The MatchOver wait has a 5-minute budget; locally it should take well under a minute at `botDelayMs: 0`.

Debug if it fails: the most likely first-run problems are (a) `view.seat` vs `turn` mixups in the driver loop, (b) the human player deadlocking on `AwaitingRefusalChoice` (the bot's candidates handle it — make sure the loop includes that phase), (c) restore throwing on a save written mid-turn (check `server.ts`'s try/catch actually falls back).

- [ ] **Step 3: Commit**

```bash
git add server/src/smoke.ts package.json
git commit -m "Smoke-run the server: match to MatchOver, reclaim, restart resume"
```

---

### Task 6: Web MPA split — sandbox to `/sandbox.html`, game client entry at `/`

**Files:**
- Create: `web/sandbox.html`
- Modify: `web/index.html` (point at `/src/game/main.tsx`, retitle)
- Modify: `web/vite.config.ts` (MPA inputs + `/ws` proxy)
- Create: `web/src/game/main.tsx` + `web/src/game/App.tsx` (placeholder rendering "connecting…" until Task 7/8 fill it in)

- [ ] **Step 1: Create `web/sandbox.html`** (the current `index.html` verbatim, script tag unchanged)

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Canastra — engine sandbox</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 2: Rewrite `web/index.html`**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Canastra</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/game/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 3: Rewrite `web/vite.config.ts`**

```ts
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // The game client talks to @canastra/server; the sandbox page needs nothing.
    proxy: { "/ws": { target: "ws://localhost:3001", ws: true } },
  },
  build: {
    rollupOptions: {
      input: {
        main: fileURLToPath(new URL("index.html", import.meta.url)),
        sandbox: fileURLToPath(new URL("sandbox.html", import.meta.url)),
      },
    },
  },
});
```

- [ ] **Step 4: Create `web/src/game/main.tsx` and a placeholder `App.tsx`**

`main.tsx`:

```tsx
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "../styles.css";

// No StrictMode, matching the sandbox: double-invoked effects would open two
// WebSockets per mount.
createRoot(document.getElementById("root")!).render(<App />);
```

`App.tsx` (placeholder — Task 8 replaces it):

```tsx
export function App() {
  return <div className="loading">canastra — client lands in task 8</div>;
}
```

- [ ] **Step 5: Verify**

Run: `npm run typecheck && npm run build --prefix web`
Expected: typecheck passes; the build emits `dist/index.html` and `dist/sandbox.html`.

Run: `npm run dev --prefix web` briefly and load both `/` (placeholder) and `/sandbox.html` (the full sandbox — it must still work: this is the regression check for the MPA split).

- [ ] **Step 6: Commit**

```bash
git add web/
git commit -m "Split web into game client (/) and sandbox (/sandbox.html)"
```

---

### Task 7: Game client — connection layer + lobby

**Files:**
- Create: `web/src/game/client.ts`
- Modify: `web/src/game/App.tsx` (real version: connect, route lobby/table)
- Create: `web/src/game/Lobby.tsx`
- Modify: `web/src/styles.css` (add game styles — lobby seats, buttons)

- [ ] **Step 1: Create `web/src/game/client.ts`**

```ts
/**
 * The game client's connection: one WebSocket, one immutable state blob for
 * React, reconnect with backoff. It holds no rules — it renders what the
 * server sends and sends what the player clicks.
 *
 * The token in localStorage is what makes reclaim work: close the tab
 * mid-match, reopen, and the server hands the seat back.
 */

import type { ClientMessage, ServerMessage, TableState } from "@canastra/protocol";
import type { HandScore, PlayerView, RuleViolation, Seat } from "@canastra/bots";

const TOKEN_KEY = "canastra:token";
const NAME_KEY = "canastra:name";

export interface ClientState {
  connected: boolean;
  seat: Seat | null;
  table: TableState | null;
  view: PlayerView | null;
  /** Newest first, capped — the move feed. */
  events: string[];
  /** The last refusal addressed to us, until the next accepted action clears it. */
  refusal: RuleViolation | null;
  /** §13 settlement, between HandOver and the settle. */
  handOver: [HandScore, HandScore] | null;
}

const INITIAL: ClientState = {
  connected: false,
  seat: null,
  table: null,
  view: null,
  events: [],
  refusal: null,
  handOver: null,
};

export class GameClient {
  private state: ClientState = INITIAL;
  private listeners = new Set<() => void>();
  private ws: WebSocket | null = null;
  private token: string | undefined = localStorage.getItem(TOKEN_KEY) ?? undefined;
  private retryMs = 1_000;
  name: string = localStorage.getItem(NAME_KEY) ?? "";

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getState = (): ClientState => this.state;

  private emit(patch: Partial<ClientState>): void {
    this.state = { ...this.state, ...patch };
    for (const listener of this.listeners) listener();
  }

  connect(): void {
    const url = new URL("/ws", window.location.href);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(url);
    this.ws = ws;
    ws.onopen = () => {
      this.retryMs = 1_000;
      this.send({ type: "hello", name: this.name || "Jogador", token: this.token });
    };
    ws.onmessage = (event) => this.receive(JSON.parse(String(event.data)) as ServerMessage);
    ws.onclose = () => {
      this.emit({ connected: false });
      setTimeout(() => this.connect(), this.retryMs);
      this.retryMs = Math.min(this.retryMs * 2, 10_000);
    };
  }

  setName(name: string): void {
    this.name = name;
    localStorage.setItem(NAME_KEY, name);
    this.send({ type: "hello", name: name || "Jogador", token: this.token });
  }

  send(message: ClientMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(message));
  }

  private receive(message: ServerMessage): void {
    switch (message.type) {
      case "welcome":
        this.token = message.token;
        localStorage.setItem(TOKEN_KEY, message.token);
        this.emit({ connected: true, seat: message.seat, table: message.table });
        break;
      case "table":
        // A settle (or a new match) closes the settlement panel.
        this.emit({
          table: message.table,
          handOver: message.table.phase === "HandOver" ? this.state.handOver : null,
        });
        break;
      case "view":
        this.emit({ view: message.view, refusal: null });
        break;
      case "event":
        this.emit({ events: [message.text, ...this.state.events].slice(0, 200) });
        break;
      case "refused":
        this.emit({ refusal: message.violation });
        break;
      case "handOver":
        this.emit({ handOver: message.scores });
        break;
    }
  }
}
```

- [ ] **Step 2: Rewrite `web/src/game/App.tsx`**

```tsx
import { useEffect, useSyncExternalStore } from "react";
import { GameClient } from "./client";
import { Lobby } from "./Lobby";
import { Table } from "./Table";

const client = new GameClient();

export function App() {
  useEffect(() => client.connect(), []);
  const state = useSyncExternalStore(client.subscribe, client.getState);

  if (!state.connected || !state.table) {
    return <div className="loading">connecting…</div>;
  }
  if (state.table.phase === "lobby") {
    return <Lobby client={client} state={state} />;
  }
  return <Table client={client} state={state} />;
}
```

(`Table` arrives in Task 8; for this task's typecheck, create a minimal `web/src/game/Table.tsx` placeholder exporting `export function Table() { return null; }` — Task 8 replaces it. Or land Task 8 before typechecking. Keep the placeholder and let Task 8 overwrite.)

- [ ] **Step 3: Create `web/src/game/Lobby.tsx`**

```tsx
import type { GameClient, ClientState } from "./client";
import type { SeatOccupant } from "@canastra/protocol";

export const SEAT_NAMES = ["Sul", "Oeste", "Norte", "Leste"];

function OccupantLabel({ occupant }: { occupant: SeatOccupant }) {
  if (occupant.kind === "human")
    return (
      <span>
        {occupant.name}
        {!occupant.connected && <em> (away — bot playing)</em>}
      </span>
    );
  if (occupant.kind === "bot") return <span>bot ({occupant.botId})</span>;
  return <span className="dim">empty</span>;
}

export function Lobby({ client, state }: { client: GameClient; state: ClientState }) {
  const { table, seat } = state;
  if (!table) return null;
  const seated = seat !== null;

  return (
    <div className="app">
      <header>
        <h1>Canastra</h1>
        <span className="sub">mesa única — pick a seat, partnerships are 0+2 vs 1+3</span>
      </header>

      <section className="lobby">
        <label>
          your name
          <input
            defaultValue={client.name}
            size={12}
            onBlur={(event) => client.setName(event.target.value)}
          />
        </label>

        <div className="lobby-seats">
          {table.seats.map((occupant, at) => (
            <div key={at} className={`lobby-seat${seat === at ? " mine" : ""}`}>
              <strong>{SEAT_NAMES[at]}</strong>
              <span className="team">{at % 2 === 0 ? "nós (0+2)" : "eles (1+3)"}</span>
              <OccupantLabel occupant={occupant} />
              {seat === at ? (
                <button onClick={() => client.send({ type: "stand" })}>stand</button>
              ) : (
                occupant.kind !== "human" && (
                  <button onClick={() => client.send({ type: "sit", seat: at })}>sit</button>
                )
              )}
            </div>
          ))}
        </div>

        <button disabled={!seated} onClick={() => client.send({ type: "start" })}>
          start the match
        </button>
        <p className="dim">Empty seats are played by bots. Anyone seated can start.</p>
      </section>
    </div>
  );
}
```

- [ ] **Step 4: Add lobby styles to `web/src/styles.css`**

Append:

```css
/* --- multiplayer game client --- */
.lobby { padding: 1rem; display: grid; gap: 1rem; justify-items: start; }
.lobby-seats { display: flex; gap: 1rem; }
.lobby-seat { border: 1px solid #444; border-radius: 8px; padding: 0.75rem 1rem; display: grid; gap: 0.35rem; min-width: 10rem; }
.lobby-seat.mine { border-color: #7c7; }
.dim { opacity: 0.55; }
```

- [ ] **Step 5: Verify**

Run: `npm run typecheck`
Expected: pass.

Manual: `npm run dev --prefix server` + `npm run dev --prefix web`, open `http://localhost:5173/` in two windows, set names, sit in seats 0 and 1 — both windows show both occupants live. (No start yet — the Table placeholder renders null.)

- [ ] **Step 6: Commit**

```bash
git add web/
git commit -m "Game client: connection layer with reclaim token, and the lobby"
```

---

### Task 8: Game client — the table screen

**Files:**
- Modify: `web/src/game/Table.tsx` (replace placeholder with the real screen)
- Modify: `web/src/ui/Cards.tsx` (export `compareCards` for the game client's hand)
- Modify: `web/src/styles.css` (table layout, feed, panels)

This task is rendering only — interactions land in Task 9. The screen shows: status bar, both partnerships' melds (reusing `MeldView`/`CardChip`), the discard pile, the move feed, your hand, the HandOver settlement panel, the MatchOver panel, and a stand button. Spectators (connected, not seated) see the same screen without a hand, plus sit buttons on bot seats.

- [ ] **Step 1: Export the comparator from `web/src/ui/Cards.tsx`**

Change `function compareCards` to `export function compareCards`.

- [ ] **Step 2: Write `web/src/game/Table.tsx`**

```tsx
import type { GameClient, ClientState } from "./client";
import type { Card, HandScore } from "@canastra/bots";
import { handScoreTotal } from "@canastra/bots";
import { CardChip, MeldView } from "../ui/Cards";
import { SEAT_NAMES } from "./Lobby";
import { TurnControls } from "./TurnControls";

/**
 * The game screen. Everything rendered here comes from the player's own
 * `view` (their hand, the tables, the pile) plus the public `table` state —
 * no other seat's cards ever reach the browser.
 *
 * Team labels follow the viewer: your partnership is "nós", the other "eles"
 * (seats 0+2 vs 1+3). Spectators get neutral labels.
 */
export function Table({ client, state }: { client: GameClient; state: ClientState }) {
  const { table, view, seat, events } = state;
  if (!table) return null;

  const myTeam = seat !== null ? seat % 2 : null;
  const teamName = (team: number) =>
    myTeam === null ? (team === 0 ? "Sul·Norte" : "Oeste·Leste") : team === myTeam ? "nós" : "eles";

  const over = table.phase === "MatchOver";
  const handOver = table.phase === "HandOver";

  return (
    <div className="app">
      <header>
        <h1>Canastra</h1>
        <span className="sub">
          {seat !== null ? `${SEAT_NAMES[seat]} — you` : "spectating"}
        </span>
        {seat !== null && <button onClick={() => client.send({ type: "stand" })}>stand</button>}
        <a href="/sandbox.html" className="dim">sandbox</a>
      </header>

      <section className="status">
        <Stat label="hand" value={String(table.handNumber ?? "—")} />
        <Stat label="phase" value={table.phase} />
        <Stat label="turn" value={table.turn !== null ? SEAT_NAMES[table.turn] : "—"} />
        <Stat label="stock" value={view ? String(view.stock_count) : "—"} />
        {table.scores && <Stat label={teamName(0)} value={String(table.scores[0])} />}
        {table.scores && <Stat label={teamName(1)} value={String(table.scores[1])} />}
        {view?.went_out != null && <Stat label="bateu" value={SEAT_NAMES[view.went_out]} />}
      </section>

      {state.refusal && <div className="refusal-banner">{describeRefusal(state.refusal)}</div>}

      {handOver && state.handOver && <HandOverPanel scores={state.handOver} teamName={teamName} client={client} seated={seat !== null} />}

      {over && table.scores && (
        <div className="panel">
          <h2>match over</h2>
          <p>
            {teamName(0)} {table.scores[0]} — {table.scores[1]} {teamName(1)}
          </p>
          {seat !== null && (
            <button onClick={() => client.send({ type: "start" })}>new match</button>
          )}
        </div>
      )}

      <main>
        <div className="middle">
          <div className="pile">
            <h2>discard ({view?.discard.length ?? 0})</h2>
            <div className="hand">
              {(view?.discard ?? []).slice(-14).map((card, index) => (
                <CardChip key={index} card={card} dim={index !== (view?.discard.length ?? 0) - 1} />
              ))}
            </div>
          </div>

          {[0, 1].map((team) => (
            <div key={team} className="team-table">
              <h2>
                {teamName(team)}
                {view && !view.tables[team].opened && ` — not open (needs ${view.opening_minimum})`}
              </h2>
              {view && view.tables[team].red_threes.length > 0 && (
                <div className="reds">
                  red 3s: {view.tables[team].red_threes.map((card, at) => <CardChip key={at} card={card} />)}
                </div>
              )}
              {view?.tables[team].melds.map((meld, index) => (
                <MeldView key={index} meld={meld} index={index} />
              ))}
            </div>
          ))}
        </div>

        <aside className="log">
          <h2>moves</h2>
          {events.map((text, index) => (
            <div key={index} className="event">{text}</div>
          ))}
        </aside>
      </main>

      <footer>
        {seat === null ? (
          <div className="spectate">
            {table.seats.map((occupant, at) =>
              occupant.kind === "human" ? null : (
                <button key={at} onClick={() => client.send({ type: "sit", seat: at })}>
                  take {SEAT_NAMES[at]}'s seat ({occupant.kind})
                </button>
              ),
            )}
          </div>
        ) : (
          view && <TurnControls client={client} view={view} seat={seat} />
        )}
      </footer>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

/** A refusal, spelled out with the numbers the violation carries. */
function describeRefusal(refusal: { error: string; [field: string]: unknown }): string {
  const extras = Object.entries(refusal)
    .filter(([key]) => key !== "error" && key !== "detail")
    .map(([key, value]) => `${key}: ${String(value)}`)
    .join(", ");
  return extras ? `${refusal.error} (${extras})` : refusal.error;
}

function HandOverPanel({
  scores,
  teamName,
  client,
  seated,
}: {
  scores: [HandScore, HandScore];
  teamName: (team: number) => string;
  client: GameClient;
  seated: boolean;
}) {
  return (
    <div className="panel">
      <h2>hand over</h2>
      {[0, 1].map((team) => (
        <p key={team}>
          {teamName(team)}: {handScoreTotal(scores[team])} (
          {scores[team].table_cards} table, −{scores[team].hand_cards} hand
          {scores[team].canastra_bonus ? `, ${scores[team].canastra_bonus} canastras` : ""}
          {scores[team].red_three_bonus ? `, ${scores[team].red_three_bonus} red 3s` : ""}
          {scores[team].going_out_bonus ? `, ${scores[team].going_out_bonus} bateu` : ""}
          {scores[team].unopened_penalty ? `, ${scores[team].unopened_penalty} never opened` : ""})
        </p>
      ))}
      {seated && <button onClick={() => client.send({ type: "settle" })}>next hand</button>}
    </div>
  );
}
```

(`TurnControls` arrives in Task 9; same placeholder approach: create `web/src/game/TurnControls.tsx` exporting `export function TurnControls() { return null; }` so this task typechecks, Task 9 overwrites it.)

- [ ] **Step 3: Add table styles to `web/src/styles.css`**

Append:

```css
.panel { border: 1px solid #557; border-radius: 8px; padding: 0.75rem 1rem; margin: 0.5rem 1rem; }
.refusal-banner { background: #5c2e2e; color: #fdd; padding: 0.4rem 1rem; }
.spectate { display: flex; gap: 0.5rem; padding: 1rem; }
```

- [ ] **Step 4: Verify**

Run: `npm run typecheck && npm run smoke`
Expected: both pass (the smoke is unaffected by client work, but it is cheap and guards the server).

Manual: dev-run server + vite, open two windows, sit 0 and 1, start — both windows render the table, hands differ per window (only your own hand visible), the move feed scrolls, bot turns appear with pacing.

- [ ] **Step 5: Commit**

```bash
git add web/
git commit -m "Render the multiplayer table: melds, pile, feed, settlement panel"
```

---

### Task 9: Game client — turn controls

**Files:**
- Modify: `web/src/game/TurnControls.tsx` (replace placeholder)
- Modify: `web/src/styles.css` (selection + action bar styles)

The action bar is driven by `view.phase` when `view.turn === seat`. Selection state lives here (indices into the sorted hand). Card rendering reuses `CardChip`; the sorted hand uses the now-exported `compareCards`.

- [ ] **Step 1: Write `web/src/game/TurnControls.tsx`**

```tsx
import { useState } from "react";
import type { GameClient } from "./client";
import type { Card, MeldTarget, PlayerView, Seat } from "@canastra/bots";
import { CardChip, compareCards } from "../ui/Cards";

/**
 * The phase-driven action bar. The client offers; the engine judges — buttons
 * are enabled by cheap shape checks (3+ cards for a meld, 1 for a discard),
 * never by re-implemented rules, and a wrong guess comes back as a `refused`
 * banner that costs nothing.
 */
export function TurnControls({
  client,
  view,
  seat,
}: {
  client: GameClient;
  view: PlayerView;
  seat: Seat;
}) {
  /** Indices into the sorted hand. */
  const [selected, setSelected] = useState<number[]>([]);
  /** Taking the pile: pick exactly two core cards, then a target. */
  const [pileMode, setPileMode] = useState(false);
  /** Add-to-meld armed: the next click on one of our melds is the target. */
  const [addingToMeld, setAddingToMeld] = useState(false);

  const sorted = [...view.hand].sort(compareCards);
  const cards = selected.map((at) => sorted[at]);
  const myTurn = view.turn === seat;

  const send = (action: Parameters<GameClient["send"]>[0]["action"]) => {
    client.send({ type: "action", action });
    setSelected([]);
    setPileMode(false);
    setAddingToMeld(false);
  };

  const toggle = (at: number) =>
    setSelected((previous) =>
      previous.includes(at) ? previous.filter((each) => each !== at) : [...previous, at],
    );

  if (!myTurn) {
    return (
      <div className="controls">
        <YourHand view={view} sorted={sorted} selected={selected} toggle={toggle} />
        <p className="dim">waiting for the others…</p>
      </div>
    );
  }

  return (
    <div className="controls">
      <YourHand view={view} sorted={sorted} selected={selected} toggle={toggle} />

      {view.frozen.length > 0 && (
        <p className="dim">dimmed cards came from the pile and are frozen this turn (§5)</p>
      )}

      <div className="actions">
        {view.phase === "AwaitingDraw" && !pileMode && (
          <>
            <button onClick={() => send({ type: "Draw" })}>draw from stock</button>
            <button disabled={view.discard.length === 0} onClick={() => setPileMode(true)}>
              take the pile
            </button>
          </>
        )}

        {view.phase === "AwaitingDraw" && pileMode && (
          <>
            <span>core: pick exactly 2 natural cards</span>
            <button
              disabled={cards.length !== 2}
              onClick={() => send({ type: "TakeDiscardPile", core: [cards[0], cards[1]] as [Card, Card], target: { kind: "NewMeld" } })}
            >
              as a new meld
            </button>
            <button disabled={cards.length !== 2} onClick={() => setAddingToMeld(true)}>
              onto an existing meld…
            </button>
            <button onClick={() => { setPileMode(false); setSelected([]); }}>cancel</button>
          </>
        )}

        {view.phase === "AwaitingRefusalChoice" && view.pending_refusal && (
          <>
            <span>offered: <CardChip card={view.pending_refusal} /></span>
            <button onClick={() => send({ type: "KeepDrawnCard" })}>keep it</button>
            <button onClick={() => send({ type: "RefuseDrawnCard" })}>refuse it</button>
          </>
        )}

        {view.phase === "Melding" && (
          <>
            <button disabled={cards.length < 3} onClick={() => send({ type: "LayMeld", cards })}>
              lay as new meld
            </button>
            <button disabled={cards.length === 0} onClick={() => setAddingToMeld(true)}>
              add to a meld…
            </button>
            <button disabled={cards.length !== 1} onClick={() => send({ type: "Discard", card: cards[0] })}>
              discard
            </button>
            {view.stock_count === 0 && view.hand.length === 1 && (
              <button onClick={() => send({ type: "EndTurnWithoutDiscard" })}>
                end without discarding
              </button>
            )}
            <button onClick={() => { client.send({ type: "restartTurn" }); setSelected([]); }}>
              restart turn
            </button>
          </>
        )}
      </div>

      {addingToMeld && (
        <MeldPicker
          view={view}
          seat={seat}
          onPick={(target) => {
            if (pileMode) send({ type: "TakeDiscardPile", core: [cards[0], cards[1]] as [Card, Card], target });
            else send({ type: "AddToMeld", meld: (target as { meld: number }).meld, cards });
          }}
          onCancel={() => setAddingToMeld(false)}
        />
      )}
    </div>
  );
}

function YourHand({
  view,
  sorted,
  selected,
  toggle,
}: {
  view: PlayerView;
  sorted: Card[];
  selected: number[];
  toggle: (at: number) => void;
}) {
  const frozen = [...view.frozen];
  return (
    <div className="hand yours">
      {sorted.map((card, at) => {
        // §5: frozen is a multiset — dim only as many copies as were swept up.
        const frozenAt = frozen.indexOf(card);
        if (frozenAt >= 0) frozen.splice(frozenAt, 1);
        return (
          <button
            key={`${card}-${at}`}
            className={`card-button${selected.includes(at) ? " selected" : ""}`}
            onClick={() => toggle(at)}
          >
            <CardChip card={card} dim={frozenAt >= 0} note={frozenAt >= 0 ? "frozen this turn (§5)" : undefined} />
          </button>
        );
      })}
    </div>
  );
}

function MeldPicker({
  view,
  seat,
  onPick,
  onCancel,
}: {
  view: PlayerView;
  seat: Seat;
  onPick: (target: MeldTarget) => void;
  onCancel: () => void;
}) {
  const team = seat % 2;
  return (
    <div className="meld-picker">
      {view.tables[team].melds.map((_, index) => (
        <button key={index} onClick={() => onPick({ kind: "Existing", meld: index })}>
          meld #{index}
        </button>
      ))}
      <button onClick={onCancel}>cancel</button>
    </div>
  );
}
```

- [ ] **Step 2: Add control styles to `web/src/styles.css`**

Append:

```css
.controls { padding: 0.75rem 1rem; display: grid; gap: 0.5rem; }
.controls .actions { display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; }
.card-button { background: none; border: 2px solid transparent; border-radius: 6px; padding: 2px; cursor: pointer; }
.card-button.selected { border-color: #7c7; }
.meld-picker { display: flex; gap: 0.5rem; }
```

- [ ] **Step 3: Verify**

Run: `npm run typecheck`
Expected: pass.

Manual (the real check for this task — it is all interaction):
- `npm run dev --prefix server` + `npm run dev --prefix web`, one window, sit, start.
- Draw, meld if you can, discard; get a refusal on purpose (e.g. discard two cards — impossible; try `LayMeld` with 2 cards — the button is disabled, so try an illegal meld like 3 mixed suits) and see the banner.
- Open a second window as spectator; take over a bot seat mid-match; the hand arrives.
- Close the first window, reopen: the seat is reclaimed with the same hand.

- [ ] **Step 4: Commit**

```bash
git add web/
git commit -m "Play turns from the browser: draw, take the pile, meld, discard"
```

---

### Task 10: Root scripts — dev, start, smoke

**Files:**
- Modify: `package.json` (root)

- [ ] **Step 1: Add scripts and `concurrently`**

```json
"scripts": {
  ...,
  "dev": "concurrently -n server,web -c blue,green \"npm run dev --prefix server\" \"npm run dev --prefix web\"",
  "start": "npm run build --prefix web && npm run start --prefix server",
  "smoke": "npm run smoke --prefix server"
},
"devDependencies": {
  "concurrently": "^9.0.0"
}
```

- [ ] **Step 2: Verify**

Run: `npm install && npm run dev` (then Ctrl-C)
Expected: both the server (`:3001`) and Vite (`:5173`) boot; `http://localhost:5173/` connects through the proxy.

Run: `npm run smoke`
Expected: `smoke: all good`.

- [ ] **Step 3: Commit**

```bash
git add package.json package-lock.json
git commit -m "One-command dev and production entry points"
```

---

### Task 11: Docs

**Files:**
- Modify: `CLAUDE.md` (repo status item 3, Commands, Architecture)
- Modify: `web/README.md` (the page is now two pages)
- Create: `server/README.md`
- Create: `protocol/README.md`

- [ ] **Step 1: `CLAUDE.md`**

Repo status item 3 becomes:

```markdown
3. **Web app** — lets people play Canastra against each other or against a bot. **Built (MVP), on
   the `web-game` branch.** One authoritative Node server (`server/`) holds the engine and one
   global table; browsers are thin clients over a WebSocket protocol (`protocol/`). Bots fill
   empty seats. The old engine sandbox survives at `/sandbox.html`.
```

Commands — add:

```markdown
Multiplayer dev (server on :3001, Vite on :5173 proxying `/ws`):

```bash
npm run dev
```

Production: `npm start` (builds `web/dist`, serves everything from the Node server on `:3001`).
End-to-end check of the server: `npm run smoke`.
```

Architecture — add `protocol/` and `server/` paragraphs to the JS package list:

```markdown
- **`protocol/`** (`@canastra/protocol`) — the client↔server wire messages, shared by `web/` and
  `server/`. No logic, no runtime dependencies (the wire types come from `@canastra/bots`).
- **`server/`** (`@canastra/server`) — the multiplayer server: one global table, seat binding per
  connection (F6 discharged by construction — `action` messages carry no seat), bot driving through
  the harness `step`, token reclaim, snapshot persistence under `server/data/`.
```

And update the `web/` paragraph to mention the two pages and that the game client never loads the wasm.

- [ ] **Step 2: `web/README.md`**

Add near the top:

```markdown
## Two pages

- `/` — the multiplayer game client. Thin: no engine, no wasm; it renders what `@canastra/server`
  sends and sends actions back. It never sees another seat's hand.
- `/sandbox.html` — the engine sandbox described below, unchanged.
```

- [ ] **Step 3: `server/README.md`**

```markdown
# server/ — the multiplayer Canastra server

One authoritative Node process: it holds the real engine (the same wasm the sandbox and the harness
CLI run), one global table, and the bots that cover empty or abandoned seats. Browsers are thin
clients; the wire protocol lives in `../protocol/`.

## Running

```bash
npm run build:engine   # once, and after any engine change — the wasm is loaded from web/src/engine
npm run dev            # from the repo root: this server on :3001 + vite on :5173
npm start              # production: builds web/dist, serves everything from :3001
```

`PORT` overrides the port. The table is persisted to `server/data/game.json` after every action
and resumed on boot; delete that file to reset the table.

## Checking

```bash
npm run smoke          # fake players over real WebSockets: full match, reclaim, restart resume
```

## What it deliberately does not do

Authenticate, rate-limit, or hide anything beyond the per-seat information boundary the engine's
`observe` already draws (see ADVERSARIAL-REVIEW.md F6). It is for friends, not the open internet.
```

- [ ] **Step 4: `protocol/README.md`**

```markdown
# protocol/ — `@canastra/protocol`

The client↔server wire messages for the multiplayer table (`ClientMessage`, `ServerMessage`,
`TableState`), shared by `server/` and `web/`. No logic and no runtime dependencies — the engine
wire types come from `@canastra/bots`. One design rule: nothing here may be able to carry a
`GameState`, a snapshot, or another seat's hand.
```

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md web/README.md server/README.md protocol/README.md
git commit -m "Document the multiplayer server, protocol, and game client"
```

---

### Task 12: Final gates

- [ ] **Step 1: Rust gates (engine untouched, but confirm)**

Run from `engine/`: `cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --check`
Expected: 181 tests pass, clippy/fmt clean.

- [ ] **Step 2: Full JS typecheck**

Run from repo root: `npm run typecheck`
Expected: all five packages pass.

- [ ] **Step 3: Smoke**

Run: `npm run smoke`
Expected: `smoke: all good`.

- [ ] **Step 4: Production build**

Run: `npm run build --prefix web && npm run start --prefix server` (then Ctrl-C)
Expected: `dist/` builds; the server serves `http://localhost:3001/` (game client) and `/sandbox.html`.

- [ ] **Step 5: Manual pass**

Two windows + one spectator window: lobby, sit, start, play several turns, force a refusal, drop and reclaim, watch a hand settle, confirm the sandbox at `/sandbox.html` still runs a bot match.

- [ ] **Step 6: Final commit (if anything shook loose)**

```bash
git add -A && git commit -m "Final gates for the multiplayer milestone" || true
git log --oneline
```
