# web/ — Canastra web app (Go)

Nothing here yet. This file is the handover from the session that built the Rust engine.

## What already exists

`engine/` is a finished Cargo workspace implementing the complete rulebook in
[canastra-regras-da-casa.md](../canastra-regras-da-casa.md). 158 tests, clippy and rustfmt clean,
builds for `wasm32-unknown-unknown`.

- `crates/canastra-engine` — the rules core. No binding dependencies, no IO, no threads, no clocks.
- `crates/canastra-wasm` — wasm-bindgen glue exposing a `Game` handle to **JavaScript**.

Read [CLAUDE.md](../CLAUDE.md) before writing code against it. The four API entry points:

```rust
new_game(seed: u64) -> GameState
apply(&GameState, Seat, &Action) -> Result<GameState, RuleViolation>   // pure
observe(&GameState, Seat) -> PlayerView                                // redacted
settle_hand(&GameState) -> Result<GameState, RuleViolation>            // end of hand
```

## Read this before choosing an architecture

**`canastra-wasm` is for the browser, not for Go.** It is a wasm-bindgen crate: the output needs
JavaScript glue and a JS host. It will not load in `wazero` or any pure-Go wasm runtime. Assuming
otherwise is the single most likely way to waste a day here.

Go has three real options for reaching the engine:

1. **Rust sidecar over stdio or HTTP.** A small binary wrapping the engine, speaking the JSON shapes
   already pinned in `engine/crates/canastra-engine/tests/boundary.rs`. Simplest, language-clean,
   easy to deploy and to restart. Recommended starting point.
2. **A new `canastra-cabi` crate** built as a `staticlib` exposing `extern "C"` functions that take
   and return JSON strings, linked via cgo. Fastest at runtime, but cgo makes cross-compilation and
   container builds noticeably more annoying.
3. **A new `canastra-wasi` crate** targeting `wasm32-wasip1` with a plain C-ABI export (no
   wasm-bindgen), run under `wazero`. Keeps the Go build pure and sandboxes the engine. More upfront
   work than option 1.

The existing `canastra-wasm` remains useful — for the **frontend**, so the browser can validate moves
and drive the UI without a round trip. That is a separate consumer from the Go server, and both talk
the same serde contract.

## Rules the server has to enforce, not the client

- **Never send a `GameState` to a client.** It holds all four hands, the stock order, and the match
  seed. Send `observe(state, seat)` instead. The redaction is structural, so using it is enough.
- **`Game::snapshot()` in the wasm crate returns the full state**, seed included. It exists for
  server-side persistence. Do not expose it to a browser in a multiplayer game — anyone holding it
  can reconstruct the entire deal.
- **The engine takes `seat` as an explicit parameter** so it rejects out-of-turn moves, but it cannot
  know *who is asking*. Bind the authenticated session to a seat on the server and pass that. Never
  pass a seat the client supplied.
- **Some turns cannot be finished.** §6 requires the opening minimum to be met inside a single turn,
  and that is only knowable at the discard. A player who lays too little simply cannot discard. The
  server must offer a "restart turn" that replays from the state the turn began with — this is why
  `apply` is pure. Keep the turn-start state per game.

## Suggested first slice

A single local hot-seat game, no accounts, no persistence: deal, render four hands, drive a full hand
to scoring. That exercises every part of the engine contract end to end and will surface the shape
of the state you actually want to push to the browser, before any multiplayer or bot work.

## Before you build on it

Read [ADVERSARIAL-REVIEW.md](../ADVERSARIAL-REVIEW.md) first. It lists confirmed bugs in the engine,
including one that hard-locks a game and two that make untrusted input dangerous. Several are things
a server would otherwise hit in production.
