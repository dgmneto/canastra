# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo status

Three planned components:

1. **Rust engine/state machine** — implements Canastra game rules and turn logic. **Built**, in `engine/`.
2. **Bot project** — trains/designs AI bots to play against. **Not started** as a training
   harness, but `bots/` (`@canastra/bots`) is now its own npm package: it holds the toy policies
   behind a `Bot` interface with a per-seat picker so they can play each other, plus the engine
   wire types and seeded `rng`, and it is registered in the `BOTS` registry. Useful for eyeballing
   behaviour; it is not a training harness and does not presume the shape the real project will take.
3. **Web app** — lets people play Canastra against each other or against a bot. **Built (MVP).**
   `server/` holds the real engine and the one global table; `web/` is two pages: the game client
   at `/` (thin — no wasm, no rules, renders what the server sends) and the engine sandbox at
   `/sandbox.html` (unchanged). `protocol/` is their shared wire language. Bots fill empty seats and
   cover disconnected players (reclaim via a localStorage token). Well-intentioned players only: no
   auth, no rate limiting; the one boundary kept is information — a browser never receives another
   seat's cards (F6 discharged by construction: `action` messages carry no seat).

Update this file's "Commands" and "Architecture" sections as each component is scaffolded.

## Source of truth for game rules

[canastra-regras-da-casa.md](canastra-regras-da-casa.md) is the authoritative rules spec (Portuguese) for this specific house variant of Canastra. It differs from generic Canastra rules in ways that matter for implementation — read it before writing any engine logic. Key points to hold in mind when implementing:

- 2 baralhos + 4 coringas = 108 cards, 4 players in fixed partnerships, 15 cards/hand, **no "morto" (dead hand)**.
- Turn = draw (1 from stock, or the entire discard pile under strict conditions) → optionally meld/lay off → discard.
- Taking the discard pile requires forming a contiguous 3-card same-suit sequence (top card + 2 natural cards from hand, no wild cards); the rest of the pile goes to hand and is frozen until the following turn.
- Two wild cards with different behavior: **Joker** (never dirties a meld, works in any meld) and **2** (dirties a meld, only usable in its own suit's sequence or in the Aces meld). Max one wild card per meld.
- Wild card repositioning within a sequence is allowed only while "unlocked" (natural cards on at most one side); once natural cards flank both sides it's locked permanently.
- Red 3s go straight to the table on draw (with an immediate replacement draw) and score ±100 at hand end depending on whether the partnership has a clean canastra; black 3s are worthless blockers that never reach the table.
- Opening minimum (75 or 120 pts depending on partnership score ≥2500) must be met in a single turn.
- Canastra (7+ card meld) bonus tiers: dirty (contains a 2) 200, clean (no 2, joker OK) 500, clean Aces (7-8 natural aces) 1000.
- Going out ("bater") requires at least one clean canastra on the table.
- A partnership that never opened by hand end scores a flat −300: no hand negatives, no red-3 points (§13.3).
- Game ends when a partnership reaches 5000 points at the end of a hand.

## Rules clarifications (resolved ambiguities)

[canastra-regras-da-casa.md](canastra-regras-da-casa.md) doesn't cover these cases explicitly. Decisions below are binding for the engine:

1. **Ace canastra + 2 present:** a 2 in the meld always caps it at the dirty/suja tier (200), regardless of natural-ace count. The 1000-point "limpa de Ases" tier requires zero 2s (Joker still allowed).
2. **Exact tie at 5000+:** if both duplas cross 5000 in the same hand with equal points, play another hand (sudden death) instead of declaring a draw.
3. **Stock depleted via red-3 replacement draw** (not a normal turn-draw): treat identically to section 11.2 — the player whose replacement draw empties the stock completes their turn normally (may lay down/bater), then the hand ends.
4. **First-turn refusal when the first card drawn is a red 3:** the red 3 goes to the table + replacement draw as usual, but the refusal privilege is *not* burned — it carries over and applies to the replacement card.
5. **Cards frozen by taking the discard pile:** §5 says they "não pode ser usada neste turno" without saying whether discarding counts as using. Read "usada" as *melded* — a frozen card may be discarded but not melded. This case is reachable: a player who takes the pile holding only the two core cards melds both and is left holding nothing but frozen cards, yet still owes a discard.
6. **A player who cannot legally discard keeps the card and the hand ends.** Without a clean canastra a partnership must always keep at least one card in hand, so a player holding exactly one card has no legal discard at all. This is reachable: §12's replacement draw returns nothing when the stock has just run out, leaving a one-card hand intact. The player keeps the card, the hand ends under §11.2, nobody takes the going-out bonus, and the retained card scores against them. Modelled as `Action::EndTurnWithoutDiscard`, legal only in exactly that position.
7. **A partnership that never opened at all scores a flat −300 at hand end (§13.3).** "Never opened" is exactly `TeamTable.opened == false`, which is the same as "put down no cards" — the engine already forbids ending a turn having laid cards without meeting §6's minimum. The flat −300 replaces the whole itemized score: hand negatives are not counted, red 3s contribute nothing (not even the usual −100 for lacking a clean canastra), and there is nothing on the table to score. Mid-hand this reads as a running penalty in `score_hand`, but `settle_hand` banks it only when the hand is actually over.

## Commands

All Rust commands run from `engine/`.

```bash
cargo test --workspace
```

```bash
cargo test -p canastra-engine wildcard_is_locked -- --exact --nocapture
```

Gates that must pass before any change is done:

```bash
cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --check
```

The wasm promise is checked by building, not by assertion:

```bash
cargo build -p canastra-wasm --target wasm32-unknown-unknown
```

The JS projects form an npm workspace rooted at the repo root, so install once there. TypeScript for
all three projects (bots, harness, web) is checked with one command:

```bash
npm install && npm run typecheck
```

The sandbox in `web/` runs from the repo root. `build:engine` regenerates `web/src/engine/` from the
Rust crate and has to be re-run after any engine change, or the page keeps the stale wasm:

```bash
npm run build:engine && npm run dev --prefix web
```

The harness is an executable that runs a whole match and prints the moves and final score as JSON
Lines on stdout:

```bash
npx canastra-harness --seed 7 random random-plus random random-plus
```

Multiplayer dev (server on :3001, Vite on :5173 proxying `/ws`):

```bash
npm run dev
```

Production (`vite build`, then the Node server serves everything on :3001) and the server's
end-to-end check:

```bash
npm start
npm run smoke
```

## Architecture

`engine/` is a self-contained Cargo workspace. `crates/canastra-engine` is the rules core and has no
binding dependencies; `crates/canastra-wasm` holds the JavaScript glue.

The JavaScript side is an npm workspace rooted at the repo root, with five packages:

- **`bots/`** (`@canastra/bots`) — the bot policies plus the engine wire types (`PlayerView`,
  `Action`, …), seeded `rng`, and the `BOTS` registry. It is the leaf everything else depends on:
  it has no engine, no wasm, only opinions. Add a bot by writing `src/<name>.ts` and registering it
  in `src/index.ts`.
- **`harness/`** (`@canastra/harness`) — the thing that actually *plays* a game: the `Match` wasm
  wrapper, the `step` driver, and `runMatch`/`series`. Its `canastra-harness` bin (see Commands) is
  the executable that runs a match from a seed + bot names to JSONL. Web and the CLI share this code.
- **`protocol/`** (`@canastra/protocol`) — the client↔server wire messages (`ClientMessage`,
  `ServerMessage`, `TableState`), shared by `server/` and `web/`. No game logic and no runtime
  dependencies; the engine wire types come from `@canastra/bots`. One design rule: nothing here may
  be able to carry a `GameState`, a snapshot, or another seat's hand.
- **`server/`** (`@canastra/server`) — the multiplayer server: one global table, a seat bound to
  each connection (F6 discharged by construction — `action` messages carry no seat), bots driven
  through the harness `step`, token reclaim, and snapshot persistence under `server/data/`.
- **`web/`** (`canastra-web`) — the Vite + React front end, two pages. `/sandbox.html` is the engine
  sandbox described below (unchanged); it imports the bot registry and the harness driver rather than
  owning copies. The game client at `/` is thin — no wasm, no rules — rendering what the server sends
  via `@canastra/protocol`. `web/` owns the generated wasm glue (`web/src/engine/`), which the sandbox,
  the server and the Node harness all load.

The committed wasm lives in `web/src/engine/` (gitignored, rebuilt by `build:engine`). The harness
CLI drives it in Node by compiling the bytes into a `WebAssembly.Module`; the browser loads it by
`fetch`. In both cases the same `Game` class runs the real engine — no reimplementation.

`web/` is a Vite + React page that loads `canastra-wasm` directly in the browser and drives all four
seats with bots. It is a **sandbox for watching the engine**, not the multiplayer app: it holds the
whole `GameState` client-side and renders every hand face up. That is safe only because there is no
opponent to hide anything from, and it is precisely what a networked client must not do — F6 in
[ADVERSARIAL-REVIEW.md](ADVERSARIAL-REVIEW.md) states the obligations that reappear the moment a
second person is involved. See [web/README.md](web/README.md).

**The engine is a pure function.** `apply(&GameState, Seat, &Action) -> Result<GameState, RuleViolation>`
never mutates its input. This is load-bearing, not stylistic: §6 requires a partnership's opening melds
to clear 75 (or 120) within a *single turn*, and that total is only knowable when the player tries to
discard. Rather than staging melds in a side buffer, a player who lays too little simply cannot end
their turn, and the caller backs out by reusing the state it held when the turn began. `canastra-wasm`'s
`Game` handle does exactly this to implement `rewindTurn`. The same purity lets a searching bot clone
positions freely.

`seat` is an explicit parameter rather than being read from `state.turn`, so the engine itself rejects
out-of-turn moves. A multiplayer server must never have to trust a client's claim about which player it is.

**Sequences are contiguous slot arrays.** `Sequence { suit, low, slots }` where `slots[i]` covers rank
`low + i` and every position is filled. This collapses §9's wild-card locking rule — "naturais dos dois
lados" — into one invariant: **a wild is locked exactly when it sits at an interior index.** A free wild
at either end accepts the same future cards, so it is stored canonically high and only drops low when the
Ace is already taken.

**`GameState` is omniscient; `PlayerView` is what goes on the wire.** The state holds all four hands and
the stock order. `observe(state, seat)` redacts it, and the redaction is structural — `PlayerView` has
nowhere to put another player's cards. Never send a `GameState` to a client.

**Randomness only exists at deal time.** Canastra fixes the stock when cards are dealt and never
reshuffles, so no RNG lives in `GameState`. A whole match replays from `seed` plus its action log. The
shuffle is hand-rolled Fisher-Yates over ChaCha8 rather than a helper crate's `shuffle`, because no
shuffle helper promises a stable algorithm across releases and a routine upgrade would silently
invalidate every recorded game. `tests/boundary.rs` pins a literal opening hand to catch exactly that.

**Cross-language boundary.** Other languages never construct Rust types — they send plain data and serde
validates it. Two constraints follow and must be preserved: no `Action` variant may be a *tuple* variant
(serde's internal tagging cannot express them; unit and struct variants are both fine), and `Card`
serializes as a compact string via its `Display`/`FromStr` codec. `tests/boundary.rs` pins these shapes
as literals — changing one changes every consumer.

`testkit::Rig` builds arbitrary positions (`Rig::new().hand(1, "4D 5D").discard("9C 6D").build()`). It
lives in the library rather than a `#[cfg(test)]` module so integration tests and downstream crates can
reproduce a reported position, which is usually a table rather than a seed.

Section references throughout the code (§5, §9, …) point at the rules spec.
