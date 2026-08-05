# Design: `enumerate` in the engine, available to bots (ADVERSARIAL-REVIEW F7)

Date: 2026-08-05

Status: design for approval before implementation

## Problem

ADVERSARIAL-REVIEW.md F7 is still open:

> Bots currently have to guess an action and check the error. `validate` is
> implemented as `apply(...).map(|_| ())`, so every check clones the whole
> state. Enumerating a move list is O(moves) full clones.
>
> **Suggested fix.** When the bot work starts, add a non-cloning `validate` and
> a `legal_actions` that enumerates. Meld enumeration is the combinatorially
> interesting part and deserves its own design pass.

The `Bot` interface (`bots/src/bot.ts`) documents that it is shaped around the
gap: "propose a list" because the engine has no `legal_actions` yet, and "If
`legal_actions` ever lands this interface should be revisited."

Goals:

1. A pure Rust function that, given a `GameState` and a `Seat`, returns every
   action the current player may legally take, one ply, deterministically.
2. The bots must be able to see that list. The bots are pure TypeScript and
   never touch the engine; the harness driver owns the `Game` wasm handle, so
   the list has to cross the wasm boundary and be plumbed into the `Bot`
   interface.

Decisions taken with the project owner:

- **Enumeration scope is bounded, not exhaustive over composites.** Atomic
  moves only. Multi-card `AddToMeld` subsets are excluded because they are
  sequences of single-card adds (the engine cannot accept a set it cannot
  accept one-by-one) and enumerating them buys nothing for search.
- **The `Bot` interface is restructured** so `candidates` receives the legal
  action list directly, rather than the list living only on the wasm handle
  where bots never see it.

## Section A — Rust engine: `enumerate`

New module `engine/crates/canastra-engine/src/enumerate.rs`, re-exported from
`lib.rs`.

```rust
/// Every action `seat` may legally take right now, one ply, in deterministic
/// order. Call it with the player whose turn it is; other seats get an empty
/// list.
pub fn enumerate(state: &GameState, seat: Seat) -> Vec<Action>
```

Semantics by phase:

- `AwaitingDraw` — `Draw`, plus every legal `TakeDiscardPile { core, target }`
  (§5).
- `AwaitingRefusalChoice` — `KeepDrawnCard`, `RefuseDrawnCard`.
- `Melding` — every *distinct* `LayMeld`, every *single-card* `AddToMeld`,
  every *distinct* `Discard`, plus `EndTurnWithoutDiscard` where it is legal
  (clarification #6 corner).
- `HandOver` / `MatchOver`, or a seat that is not the current one — empty.

### Strategy: candidates + `apply` as the judge

The enumerator generates a cheap **superset** of candidate actions, then keeps
only those for which `apply(state, seat, &action)` succeeds.

This buys two guarantees that the hand-rolled bot code never had:

- **Soundness for free.** Nothing illegal can survive, because `apply` is the
  single referee. Unlike the bots' per-bot "guess and check", the check
  happens once, centrally, and is always complete.
- **No rule drift.** The enumerator contains zero re-implemented rules: it
  only has to *cover* legal moves, and validation prunes the rest. It can
  never disagree with the referee.

Cost is O(candidates) `GameState` clones via `apply`. A clone is small (four
hands, the stock, the table, ~1 KB); for a one-ply enumeration this is where
F7's cloning complaint stops mattering. F7's separate "non-cloning `validate`"
optimisation is explicitly **out of scope** here and is not part of this
change.

### Candidate generation

- **`LayMeld`** — for each suit:
  - Collect the natural cards of that suit held in hand, deduped by rank
    (a sequence holds each rank once).
  - Every rank window `i..=j` in 4..A (inclusive) with length ≥ 3:
    - all ranks held → candidate = the naturals in the window;
    - exactly one rank missing and a usable wild held (Joker, or a 2 of that
      suit) → candidate = the naturals + that wild.
  - Ace melds (§7.2): every subset of 3+ natural aces held, each with and
    without one held wild (Joker, or a 2 of any suit — an ace meld accepts any
    2). `Meld::new` decides whether a candidate is a sequence or an aces meld.

  This window+wild scheme covers the full span of `Sequence::build` shapes —
  wild filling an interior gap, wild capping either end, including the
  canonical drop-low case when the Ace is already taken (§9) — without
  re-deriving `assemble`'s canonical wild placement. The action is just the
  card set; `Meld::new` canonicalizes.

- **`AddToMeld`** — each own-team meld index × each distinct hand card, one
  card at a time. `apply` filters the §5 cases (frozen cards, wild in the
  pile-core meld) and ordinary invalid placements.

- **`Discard`** — each distinct hand card. `apply` filters the corollary
  cases (`NoCleanCanastra` on the last card, etc.).

- **`TakeDiscardPile`** — each unordered pair of distinct *natural* cards held
  × (`NewMeld` + each own-team meld index). Wilds are excluded statically
  (they can never be in the core, §5), and `apply` filters blocked tops
  (black 3 / wild, §5), frozen cores, §6 reachability, and invalid joins.

- **Phase moves** — `Draw`, `KeepDrawnCard`, `RefuseDrawnCard`,
  `EndTurnWithoutDiscard`: one candidate each, pushed only when the phase
  admits them; `apply` still filters (empty stock, cornered-only
  `EndTurnWithoutDiscard`, ...).

### Determinism

Deduplicate the retained list by value and sort by a canonical key (the
`Debug` representation is sufficient and stable). Deterministic order matters:
seed + action log must keep replaying, and bots keyed on the list need a stable
input.

## Section B — wasm binding

In `engine/crates/canastra-wasm/src/lib.rs`, on `Game`:

```rust
#[wasm_bindgen(js_name = legalActions)]
pub fn legal_actions(&self, seat: u8) -> Result<JsValue, JsValue>
```

Uses the existing `serde_wasm_bindgen` path. `Vec<Action>` serializes as a
JS array of internally-tagged `{type: ...}` objects — exactly the shape the TS
`Action` union in `bots/src/types.ts` already describes, so no new TS type is
needed.

## Section C — restructured `Bot` interface

### `bots/src/bot.ts`

```ts
export interface Bot {
  readonly id: string;
  readonly name: string;
  readonly blurb: string;
  candidates(view: PlayerView, legal: Action[], context: BotContext): Action[];
}
```

Contract: `legal` is the engine's complete, deterministic list of legal
actions for the current seat and phase. The bot returns the legal moves ranked
best-first — its body becomes pure *preference*; it no longer has to guess at
legality. An empty return concedes the turn (the driver restarts it, as today).
The F7 note in this file's header is rewritten to reflect the closure.

`BotContext` (`rng`, `safeMode`) is unchanged.

### `harness/src/match.ts`

Add a thin passthrough:

```ts
legalActions(seat: Seat): Action[] {
  return this.game.legalActions(seat) as Action[];
}
```

### `harness/src/driver.ts`

Each step now resolves the legal list once and hands it to the bot:

```ts
const seat = view.turn;
const legal = match.legalActions(seat);
for (const action of bot.candidates(view, legal, context)) {
  const refused = match.apply(seat, action);   // engine stays the judge
  if (!refused) return { action, refusals };
  refusals.push(describe(action, refused.error));
}
match.restartTurn(seat);
return { action: "restartTurn", refusals };
```

`apply` remains the referee, so a bot that returns something outside `legal`
is still handled exactly as today (refused, then restart). Refusals become
rare diagnostics for well-behaved bots rather than the normal path.

### Existing bot rewrites

Each bot is rewritten to *rank `legal`* rather than construct action guesses:

- **`random.ts`** — same heuristics (rng gates, cheapest-discard-first, lay
  what it finds), but reads the move set from `legal`.
- **`random-plus.ts`** — its `opening` / `layOffs` / `discards` /
  `usefulness` heuristics become *ranking criteria* applied to `legal`; the
  helpers stop building `Action`s for the engine to judge.
- **`random-discard-hungry.ts`** — the hand-rolled `pileTakes()` (§5 shape
  math) is deleted. The bot becomes "put every `TakeDiscardPile` from `legal`
  first, `Draw` last." The engine now answers §5 exactly.
- **`melds.ts`** — `findMelds` / `enumerateMelds` stop being move generators;
  `meldCards` / `meldValue` remain useful to inspect and rank legal melds.
  Unused code is removed during implementation.

This is the payoff of closing F7: the bots shed hand-rolled rules in exactly
the areas F7 called the combinatorially scary part.

## Section D — Tests and verification

Rust (TDD — tests written first, watched fail):

- New `tests/enumerate.rs` plus unit tests in `enumerate.rs`:
  - `AwaitingDraw` with no takeable pile yields exactly `[Draw]`.
  - The §5 worked example: with the 6♦ on top, `enumerate` yields `Draw` plus
    every `TakeDiscardPile` for cores `4D 5D`, `5D 7D`, `7D 8D`, each into
    `NewMeld` and each own meld.
  - `AwaitingRefusalChoice` yields exactly `KeepDrawnCard` + `RefuseDrawnCard`.
  - LayMeld windows: a hand `4H 5H 6H 7H` yields `4H 5H 6H`, `5H 6H 7H`,
    `4H 5H 6H 7H`.
  - LayMeld wilds: `6H 7H JOKER` yields `6H 7H + JOKER`; a hand with
    `4H 5H 6H` and a Joker yields the plain run and the wild-capped run.
  - Ace subsets: four aces in hand yield every 3- and 4-ace subset, with and
    without a held wild.
  - Frozen exclusion: after taking the pile, frozen cards appear in no
    `AddToMeld` / `LayMeld` candidate.
  - Cornered position (clarification #6): `EndTurnWithoutDiscard` is the only
    way to end, and is present.
  - Soundness invariant: for the crafted hands, every returned action passes
    `apply(state, seat, &action)`.
  - Completeness: the expected actions named above are present.
  - Determinism: two calls return equal vectors.
- Cross-language: the `Action` wire shape is already pinned in
  `tests/boundary.rs`; no shape change is expected, only the new wasm method.

TypeScript / harness:

- `npm run build:engine` first (the harness and web load the generated wasm in
  `web/src/engine/`, which is regenerated from the Rust crate).
- `npm run typecheck` across all three projects.
- Smoke test: `npx canastra-harness --seed 7 random random-plus random
  random-plus` runs a full match to completion.

Gates that must pass:

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --check
cargo build -p canastra-wasm --target wasm32-unknown-unknown
npm install && npm run typecheck
```

## Out of scope

- F7's "non-cloning `validate`" — orthogonal optimisation, not needed for a
  one-ply enumeration.
- A `PlayerView`-based enumerate in the engine — the bots receive the list via
  the wasm `Game` handle; no second entry point is needed.
- Changing the replay-log format or bot id contract.
