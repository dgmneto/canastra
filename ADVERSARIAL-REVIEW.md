# Adversarial review — `engine/`

Written at the end of the session that built the engine, reviewing its own work. Every finding below
was **reproduced with a throwaway test** before being written down, except F5 and F6 which are marked
as reached by code reading. One hypothesis was investigated and refuted; it is recorded at the bottom
so nobody re-opens it.

State at time of review: commit `261e6bf`, 158 tests passing, clippy and rustfmt clean.

---

## F1 — Unwinnable deadlock when a red 3 empties the stock (critical)

**Confirmed by reproduction.**

A player holding one card draws a red 3 as the last card of the stock. Per §12 the 3 goes to the table
and they draw a replacement — but the stock is now empty, so no replacement comes. Their hand is back
to one card. They must discard, and that discard would empty their hand, which `apply` treats as going
out. With no clean canastra on the table it is refused:

```
hand after draw = 1, stock = 0
discard 4H -> Err(NoCleanCanastra)
```

There is no other legal action. Rewinding the turn does not help: the deal is deterministic, so the
replay produces the identical position. **The game is stuck forever.**

Reproduce:

```rust
let state = Rig::new()
    .hand(1, "4H")
    .meld(1, "5S 6S 7S 8S 9S TS 2S")   // dirty canastra only
    .stock("3H")
    .turn(1).build();
let drawn = apply(&state, seat(1), &Action::Draw).unwrap();
apply(&drawn, seat(1), &Action::Discard { card: card("4H") });  // NoCleanCanastra
```

**Cause.** `apply.rs` treats *any* discard that empties the hand as going out (§11.1) and applies the
clean-canastra gate. But when the stock is already empty the hand is ending under §11.2 regardless, and
the player is not "batendo" — the hand simply stops.

**Suggested fix.** In `discard`, when `state.stock.is_empty()`, let the discard through and end the
hand under §11.2 with `went_out = None` — no bonus, no clean-canastra requirement. Read §11.2's "a mão
termina após o descarte dele e ninguém leva o bônus de batida" as covering this case. Worth confirming
as a rules decision and recording as a CLAUDE.md clarification, since the spec does not address it.

The neighbouring case already behaves correctly: the same position *with* a clean canastra ends the
hand with `went_out = Some(seat)`.

---

## F2 — Malformed JSON builds invalid melds, and reading them panics (high)

**Confirmed by reproduction.**

`Sequence` derives `Deserialize` over its private fields, so serde bypasses every invariant the
constructors enforce. This parses cleanly:

```json
{"kind":"Sequence","meld":{"suit":"Hearts","low":250,"slots":[]}}
```

The result is a zero-length sequence whose `low` is outside the 4..=A range. `Sequence::low()` then
panics at `meld.rs:126` on the `expect`, and `high()` computes `low + len - 1` on an empty slot vector,
which underflows.

**Why it matters.** `Game::restore` in `canastra-wasm` and any server that accepts a stored snapshot
take this input. A crafted or merely corrupt payload is a remote panic — a denial of service in a
multiplayer server, and a crashed tab in the browser.

**Suggested fix.** Deserialize `Sequence` and `AcesMeld` through a validated intermediate
(`#[serde(try_from = "RawSequence")]`) that runs the same checks `Meld::new` does, so an invalid meld
can never exist as a value. Same treatment as `Seat`, which already does this correctly.

---

## F3 — `GameState` deserializes with impossible contents (high)

**Confirmed by reproduction.** A hand-edited snapshot with ten copies of `6H` in one hand and a red 3
sitting in the discard pile round-trips without complaint.

Neither is possible in a real game: the deck holds exactly two of each card, and §12 guarantees a red 3
never reaches a hand and therefore never reaches the pile. Nothing validates deck conservation on the
way in.

**Why it matters.** Same entry points as F2. A client that can influence a restored snapshot can invent
cards, and the engine will happily score them.

**Suggested fix.** A `GameState::check_invariants() -> Result<(), &'static str>` verifying that the
union of every zone is exactly the 108-card deck, that no red 3 is in a hand or the pile, and that
`turn_context.frozen` is a sub-multiset of the current player's hand. Call it on every restore path.
`tests/boundary.rs` already has the "conserves the whole deck" helper to build on.

---

## F4 — An empty `AddToMeld` poisons the turn (medium)

**Confirmed by reproduction.** `Action::AddToMeld { meld: 0, cards: [] }` is accepted and sets
`laid_anything = true` with `laid_value = 0`.

For a partnership that has not yet opened, that is a trap: `commit_opening` now believes something was
laid, so the player cannot discard (0 < 75) and must abandon the turn. A UI that sends an empty
lay-off on a mis-click hands the player a dead turn with no explanation.

**Suggested fix.** Reject empty `cards` in `add_to_meld`, or only set `laid_anything` when at least one
card actually moved. `LayMeld` is already safe — `Meld::new` rejects fewer than three cards.

---

## F5 — `rewindTurn` can revert to the wrong turn (medium)

**By code reading**, not reproduced — exercising it needs a JS host.

`canastra-wasm/src/lib.rs` refreshes `turn_start` inside `apply`, but only when the phase *before* the
action is `AwaitingDraw`. So the checkpoint is written on the first action of a turn. Call
`rewindTurn()` before taking any action in a turn — a player clicking "restart turn" as their opening
move — and `turn_start` still holds the *previous* player's turn start. The game silently rewinds a
whole turn too far.

**Suggested fix.** Also refresh `turn_start` after applying, whenever the resulting phase is
`AwaitingDraw`. Then the checkpoint is always the current turn's start regardless of call order. Worth
a test once there is a JS or `wasm-bindgen-test` harness.

---

## F6 — Trust boundary is undefended by construction (medium, by design but unenforced)

**By code reading.**

- `Game::snapshot()` returns the entire `GameState` as JSON — all four hands, the stock order, and the
  match seed. Anyone holding it can reconstruct the whole deal. It exists for server-side persistence;
  nothing stops a browser build from exposing it.
- `Game::apply(seat, action)` takes the seat from the caller. In a browser the caller is the player, so
  client-side wasm cannot enforce identity at all.

Neither is a bug in the engine — `apply` correctly rejects out-of-turn moves, and `observe` correctly
redacts. But both are server obligations that nothing in the code reminds you of.

**Suggested fix.** Document at the `Game` type. Consider splitting the wasm surface so a browser build
cannot reach `snapshot` at all, and make the server bind an authenticated session to a seat.

---

## F7 — No `legal_actions`, and `validate` is expensive (low)

Deferred deliberately to the bot milestone, but worth stating: bots currently have to guess an action
and check the error. `validate` is implemented as `apply(...).map(|_| ())`, so every check clones the
whole state. Enumerating a move list is O(moves) full clones.

**Suggested fix.** When the bot work starts, add a non-cloning `validate` and a `legal_actions` that
enumerates. Meld enumeration is the combinatorially interesting part and deserves its own design pass.

---

## F8 — Meld JSON is awkward for other languages (low)

`Sequence` serializes its private `low` field as a raw sequence index where `4` is `0`. A TypeScript or
Go consumer has to know that encoding to render a meld. Everything else on the wire is self-describing
(cards are `"6D"`, actions are tagged objects), so this is the one rough edge.

**Suggested fix.** A custom `Serialize` emitting explicit ranks, or a derived view type for clients.
Whatever is chosen should be pinned in `tests/boundary.rs` like the rest of the contract.

---

## F9 — One weak test (low)

`going_out_does_not_excuse_the_opening_minimum` in `apply.rs` asserts with
`matches!(..., Err(NoCleanCanastra | OpeningMinimumNotMet { .. }))`. It passes whichever of the two
rules fires, so it would not notice if the wrong one did. Split it into two tests with unambiguous
setups.

---

## F10 — Two things the plan promised and this session did not deliver (low)

- **A `proptest` case asserting `apply` never panics on arbitrary input.** F2 is exactly the class of
  bug it would have caught, which is an argument for adding it early rather than late.
- **`ts-rs` TypeScript generation.** The interoperability guarantee is the serde contract and that *is*
  pinned by `tests/boundary.rs`; ts-rs is the compile-time convenience on top of it. Tracked separately.

**Process note.** For the serialization work the serde derives were written before `tests/boundary.rs`,
so those 13 tests passed on their first run rather than failing first. Derives are generated code, but
the shape decisions they encode — `tag = "type"`, cards as strings — are real design and should have
been driven by tests. They are pinned now; they were not pinned when they were chosen.

---

## Investigated and refuted

**"A player can be forced to empty their hand on an ordinary discard."** Not reachable. A hand at the
start of a turn always holds at least one card, every turn begins with a draw, so the hand is at least
two before the discard and at least one after. Melding down to a position with no legal discard is
possible but self-inflicted within the turn, and rewinding fixes it. The comment to this effect in
`maybe_go_out` is correct.

The exception is F1, where the draw returns no card because the stock ran out mid red-3 replacement —
which is why F1 is a real bug and this is not.
