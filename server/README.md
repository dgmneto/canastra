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

`PORT` overrides the port; `BOT_DELAY_MS` overrides the bot pacing (500 ms default). The table is
persisted to `server/data/game.json` after every action and resumed on boot; delete that file to
reset the table.

## Checking

```bash
npm run smoke          # fake players over real WebSockets: full match, reclaim, restart resume
```

## What it deliberately does not do

Authenticate, rate-limit, or hide anything beyond the per-seat information boundary the engine's
`observe` already draws (see ADVERSARIAL-REVIEW.md F6). It is for friends, not the open internet.
