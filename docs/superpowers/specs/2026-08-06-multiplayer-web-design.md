# Multiplayer web app — design

Date: 2026-08-06. Branch: `web-game` (worktree `../web-game`).

## Goal

Turn `web/` into a place where people play Canastra against each other (or against
bots) over the network, per the repo's third planned component. The players are
**well-intentioned**: no accounts, no passwords, no rate limiting, no cheat
detection, no TLS in the app layer. The one boundary kept is *information* —
players must not receive other players' cards — because that is game integrity,
not security, and the engine already makes it free (`observe(state, seat)`).

## Decisions (from brainstorming)

- **Bots fill empty seats.** A game runs with 1–4 humans; unclaimed seats are
  played by server-side bots.
- **Single global table.** One table per server process. No rooms, no codes, no
  lobby browser. Everyone who connects arrives at the same table.
- **Pick a seat.** A connecting player types a name and clicks an empty (or
  bot-played) seat. Partnerships are visible before sitting (seats 0+2 vs 1+3).
- **Someone presses start.** Any seated human may start the match; empty seats
  become bots at that moment.
- **Bot covers, can reclaim.** On disconnect a bot immediately takes over the
  seat. The browser holds a session token in `localStorage`; reconnecting with it
  rebinds the seat mid-match.
- **Mid-match takeover stays.** `sit` on a bot seat works while a match is in
  progress — the human takes over the bot's hand. (Confirmed: "why not, sure".)
- **Sandbox kept, separate route.** The existing omniscient sandbox moves to its
  own entry untouched; the multiplayer client becomes the default page.
- **Smoke script kept.** A scripted end-to-end check of the server (~150 lines),
  run on demand. (Confirmed: "I like that".)

## Architecture

Approach A: an **authoritative Node server** holding the real engine wasm;
browsers are thin clients that render server messages and send actions.

### Packages

The npm workspace grows from three packages to five:

- **`protocol/`** (`@canastra/protocol`) — new, tiny (~80 lines), zero
  dependencies. The wire messages shared by client and server. Exists so `web/`
  does not import server code and `bots/` stays on-mission.
- **`server/`** (`@canastra/server`) — new. Node + `ws`, run with `tsx` (same
  pattern as the harness CLI). Depends on `@canastra/protocol`,
  `@canastra/bots` (BOTS registry, rng, wire types), `@canastra/harness`
  (`Match`, `step`, `label`, engine loader).
- **`web/`** — gains a second page (the game client) next to the sandbox.
- **`bots/`**, **`harness/`** — unchanged except two small harness additions
  below.

### Harness additions (small, justified)

1. Export `loadEngine` from `@canastra/harness` index (it exists as
   `load-node.ts` but is not re-exported). The server loads the wasm the Node
   way, same as the CLI.
2. `Match.restore(snapshot, meta)` static. Today `Match` only constructs from a
   seed, which always deals a fresh game. Persistence and reclaim need to wrap a
   `Game.restore`ed handle in a `Match` (log + checkpoint). ~10 lines; the CLI
   benefits too (resume a replay).

### Server modules

```
server/src/
  main.ts         entry: loadEngine → new Table → HTTP+WS on one port;
                  serves web/dist statically when built
  table.ts        the global table: seats, lobby, match lifecycle, bot driving,
                  broadcasting, safeMode bookkeeping
  persistence.ts  write {snapshot, seats, seed, log} to server/data/game.json
                  after every accepted action; restore on boot via Game.restore
  smoke.ts        on-demand end-to-end check (see Verification)
```

**Bot driving.** After every accepted human action, while the turn belongs to a
bot seat, the table runs the harness `step()` on a ~500 ms timer per action (so
humans can follow), broadcasting each move. `safeMode` and `restartTurn`
bookkeeping mirrors `harness/src/cli.ts` exactly. A bot seat's dead-ended turn
restarts automatically, same as the CLI.

**Engine purity is preserved.** All adjudication stays in Rust. The server never
constructs or inspects game state beyond `Match`'s existing API.

## Wire protocol

One WebSocket at `/ws`, JSON messages discriminated by `type`, mirroring the
engine's `Action` tagging style. Types live in `@canastra/protocol`.

### Client → server

| message | payload | meaning |
|---|---|---|
| `hello` | `{name: string, token?: string}` | identify; `token` from `localStorage` reclaims a seat |
| `sit` | `{seat: Seat}` | claim an empty/bot seat; mid-match, takes over the bot's hand |
| `stand` | `{}` | leave the seat; a bot takes over |
| `start` | `{}` | begin the match (any seated human); empty seats become bots |
| `action` | `{action: Action}` | an engine `Action`. **No seat field** — the server passes the connection's bound seat to `Match.apply` (F6 discharged by construction) |
| `restartTurn` | `{}` | escape hatch for a dead-ended turn; `Match.restartTurn` semantics |
| `settle` | `{}` | at `HandOver`, skip the pause and bank the hand now |

### Server → client

| message | payload | meaning |
|---|---|---|
| `welcome` | `{token, seat: Seat \| null, table: TableState}` | token to store; `seat` non-null means a reclaim happened |
| `table` | `TableState` | broadcast lobby/table state after any change |
| `view` | `{view: PlayerView}` | **private**, per seated player, after every action — the engine's `observe(state, seat)` |
| `event` | `{text: string}` | one move-log line (harness `label`), broadcast |
| `refused` | `{violation: RuleViolation}` | the rule that rejected your action, to the actor only |
| `handOver` | `{scores: [HandScore, HandScore]}` | itemised §13 settlement, broadcast at hand end |

`TableState` (public information only):

```ts
interface TableState {
  seats: { occupant: { kind: "human"; name: string; connected: boolean }
                  | { kind: "bot"; botId: string }
                  | { kind: "empty" } }[];  // length 4
  phase: "lobby" | Phase;                   // engine Phase once a match runs
  turn: Seat | null;
  scores: [number, number] | null;          // match scores once dealt
  handNumber: number | null;
}
```

### Information rules (kept deliberately)

- A browser never receives a `GameState`, a snapshot, or another seat's hand.
- **`handScore` is never sent mid-hand** (it sums both partners' hands — the
  leak the web README flags). It is broadcast only at `HandOver`, when physical
  players would be counting together anyway.
- **Spectators** (connected, not seated) receive `table` + `event` only.

### Lifecycle

- **Lobby:** `phase: "lobby"`, no `Match` exists. `sit`/`stand` freely. `start`
  from any seated human deals (fresh random seed from the server) and fills
  empty seats with `random-plus` bots.
- **During a match:** `action`/`restartTurn` from the bound seat only; the
  engine refuses out-of-turn play. Bots driven as above.
- **Hand end:** phase `HandOver` → broadcast `handOver` → wait ~10 s, or any
  seated human sends `settle` → `settleHand()` → next hand, or `MatchOver`.
- **Match end:** broadcast final scores, table returns to lobby. Seats persist
  (humans stay seated); anyone may press `start` again.
- **Disconnect:** seat's occupant becomes `{kind:"bot", ...}` immediately and
  the game never stalls; the seat remembers the token. **Reconnect:** `hello`
  with the stored token rebinds the connection and returns the current
  `PlayerView` via `welcome` + `view`.
- **Persistence:** after every accepted action, write the snapshot + seats +
  seed + log to `server/data/game.json`. On boot, if the file exists and passes
  `Game.restore`'s invariant check, resume; otherwise start in lobby.

## Web client

`web/` becomes a two-page Vite app via MPA `rollupOptions.input` — no router
library. `index.html` → game client (default), `sandbox.html` → the existing
sandbox, moved untouched (it keeps the wasm and harness imports).

```
web/src/
  ui/            existing sandbox (untouched)
  game/
    main.tsx       entry: connect, hello with stored token
    client.ts      WS wrapper: typed send/receive, auto-reconnect with backoff
    Lobby.tsx      name entry, seat picker (partnerships visible), start
    Table.tsx      game screen: your hand, both tables' melds, pile, move feed
    TurnControls.tsx  phase-driven action bar
```

The existing `Cards.tsx` components (`Hand`, `CardChip`, `MeldView`) are reused
by the game client — they already render `PlayerView` shapes.

**Screen flow:** connect → lobby → (start) → table. A `welcome` with a non-null
seat lands directly on the table screen (reclaim).

**Turn interactions** (active when `view.turn === mySeat`):

- **AwaitingDraw:** *draw* button (`Draw`); *take the pile* enters core-picking
  mode — select 2 hand cards, choose *new meld* or an existing meld, confirm →
  `TakeDiscardPile {core, target}`. The client does not pre-validate; the
  engine judges.
- **AwaitingRefusalChoice:** show the offered card with *keep* / *refuse*
  (`KeepDrawnCard` / `RefuseDrawnCard`).
- **Melding:** click-to-select hand cards; *lay as new meld* (`LayMeld`),
  *add to meld* then click a table meld (`AddToMeld`), *discard selected*
  (`Discard`, exactly one card); *restart turn* escape hatch; frozen-cards
  banner when `frozen` is non-empty; `EndTurnWithoutDiscard` offered only in
  the F1 position (stock empty, one card in hand).
- **HandOver:** itemised §13 settlement panel + *next hand* (`settle`).
- **MatchOver:** final scores; table returns to lobby.

**Refusal feedback:** a `refused` message renders inline naming the rule and
its payload numbers (`OpeningMinimumNotMet: laid 45 of 75`). Guessing stays
free — same model as the bots.

**Off-turn:** table updates live via `table`/`event` broadcasts; your hand is
visible; controls inert. No running §13 total mid-hand (see information rules);
match scores always visible.

## Workflow, deploy, verification

**Dev:** root `npm run dev` runs both sides — `tsx watch server/src/main.ts`
(port 3001) and Vite (5173) with `server.proxy = {"/ws": "ws://localhost:3001"}`
in `web/vite.config.ts`. `npm run dev:sandbox` serves the sandbox entry only, if
wanted without the server.

**Deploy:** `vite build` → the Node server serves `web/dist` and `/ws` on one
port. One process, `npm start`. Persistence in `server/data/`.

**Verification:**

- `npm run typecheck` extended to chain `protocol/` and `server/`.
- Engine untouched: `cargo test --workspace`, `clippy -D warnings`, `fmt
  --check` must stay green; `build:engine` not needed (no engine change).
- **Smoke script** (`server/src/smoke.ts`, run on demand via `npm run smoke
  --prefix server`): boots a table in-process with four fake WS clients; plays
  a full match with one scripted "human" seat (draw + legal discard heuristics)
  and three bots; asserts `MatchOver` is reached; kills one client mid-match
  and asserts reclaim via token works.
- Manual: solo vs 3 bots in one window; two windows as partners/opponents.

**Docs to update:** `CLAUDE.md` (repo status: web app no longer "not started";
commands; architecture), `web/README.md` (two pages), new `server/README.md`,
new `protocol/README.md` if warranted.

## Non-goals

Accounts/passwords, rate limiting, TLS in the app layer, cheat detection, turn
timers, chat, multiple rooms/lobbies, matchmaking, ELO, mobile-specific UI,
spectator chat. Deliberately excluded per "well-intentioned players" or YAGNI.

## Risks / open points

- **`Match.restore` checkpoint:** `Game.restore` sets `turn_start = state`, so a
  restore mid-turn loses the true turn start. Acceptable: persistence writes
  after every action, and a server restart then replays at most the current
  turn from a slightly early checkpoint — the same position a `restartTurn`
  would produce. No rules consequence.
- **Bot pacing under reconnect storms:** the 500 ms bot timer is in-memory;
  reconnects do not affect it. Fine.
- **`sit` takeover of a bot mid-turn:** the bot's current turn-in-progress
  belongs to the seat, not the bot; the human simply starts deciding from the
  next action. The new human may inherit a half-laid turn and can `restartTurn`
  if they dislike it.
