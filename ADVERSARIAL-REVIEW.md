# Adversarial review — `engine/`

Written at the end of the session that built the engine, reviewing its own work. Every finding below
was **reproduced with a throwaway test** before being written down, except F5 and F6 which are marked
as reached by code reading. One hypothesis was investigated and refuted; it is recorded at the bottom
so nobody re-opens it.

State at time of review: commit `261e6bf`, 158 tests passing, clippy and rustfmt clean.

**Update.** F1–F4, F9 and F11 have since been fixed with regression tests. Fixed sections are kept in
full: the reproduction is the useful part, and it documents what the regression test is defending.

| | Finding | Severity | Status |
|---|---|---|---|
| F1 | Unwinnable deadlock when a red 3 empties the stock | critical | **fixed** |
| F2 | Malformed JSON builds invalid melds; accessors panic | high | **fixed** |
| F3 | `GameState` deserializes with impossible contents | high | **fixed** |
| F4 | Empty `AddToMeld` poisons the turn | medium | **fixed** |
| F5 | `rewindTurn` can revert to the wrong turn | medium | **needs a design decision first** |
| F6 | Trust boundary undefended by construction | medium | open — *not* covered by F2/F3 |
| F7 | No `legal_actions`; `validate` clones | low | open, deliberately |
| F8 | Meld JSON awkward for other languages | low | open |
| F9 | One weak test | low | **fixed** |
| F10 | proptest and ts-rs not delivered | low | closed, not relevant |
| F11 | A lay could strand the player's last card | medium | **fixed** |

---

## F1 — Unwinnable deadlock when a red 3 empties the stock (critical) — FIXED

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

**Fix applied.** The rule was settled by the project owner and is now CLAUDE.md clarification #6:
without a clean canastra a partnership must always keep at least one card in hand, so a player who
cannot discard **keeps the card and the hand ends**. My original suggestion — letting the discard
through — was wrong, and would have thrown away a card the rules say is retained and scored against
them.

Modelled as `Action::EndTurnWithoutDiscard`, legal only when the stock is empty, the hand holds exactly
one card, and the partnership has no clean canastra. The card stays in hand, `went_out` stays `None`,
and the retained card counts against the partnership at scoring.

An earlier idea — auto-ending the hand as soon as the position was detected — was rejected on
inspection: a player holding one card may be able to lay it off to *complete* a clean canastra, which
makes going out legal after all. Ending automatically would silently deny them that.

The neighbouring case already behaved correctly and still does: the same position *with* a clean
canastra ends the hand with `went_out = Some(seat)`.

---

## F2 — Malformed JSON builds invalid melds, and reading them panics (high) — FIXED

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

**Fix applied.** `Sequence` and `AcesMeld` now deserialize through a private `RawSequence` /
`RawAcesMeld` intermediate via `#[serde(try_from = ...)]`. The conversion checks length, that the run
fits inside 4..=A, that each natural card really is the rank its slot position claims, suit agreement,
the §8 one-wild limit, and the §8 own-suit rule for a 2. An invalid meld can no longer exist as a
value, which is the same treatment `Seat` already had.

Bounding the top of the run is what removes the panic specifically: it bounds `low` as a side effect,
so `Sequence::low` and `Sequence::high` can no longer index off the rank table.

---

## F3 — `GameState` deserializes with impossible contents (high) — FIXED

**Confirmed by reproduction.** A hand-edited snapshot with ten copies of `6H` in one hand and a red 3
sitting in the discard pile round-trips without complaint.

Neither is possible in a real game: the deck holds exactly two of each card, and §12 guarantees a red 3
never reaches a hand and therefore never reaches the pile. Nothing validates deck conservation on the
way in.

**Why it matters.** Same entry points as F2. A client that can influence a restored snapshot can invent
cards, and the engine will happily score them.

**Fix applied.** `GameState::check_invariants() -> Result<(), StateError>` verifies that the union of
every zone is exactly the 108-card deck, that no red 3 sits in a hand or the pile, that each
partnership's red-3 zone holds only red 3s, that `turn_context.frozen` is a sub-multiset of the current
player's hand, and that `pile_core_meld` points at a meld that exists.

`Game::restore` in `canastra-wasm` now calls it, so a snapshot that parses but describes an unreachable
position is refused rather than fed to the engine. It is deliberately a *soundness* check, not a rules
check — it asks whether a position is possible, not whether it was reached legally.

Note that this is separate from F2 and neither subsumes the other: a state can be built entirely from
individually valid melds and still invent cards.

---

## F4 — An empty `AddToMeld` poisons the turn (medium) — FIXED

**Confirmed by reproduction.** `Action::AddToMeld { meld: 0, cards: [] }` is accepted and sets
`laid_anything = true` with `laid_value = 0`.

For a partnership that has not yet opened, that is a trap: `commit_opening` now believes something was
laid, so the player cannot discard (0 < 75) and must abandon the turn. A UI that sends an empty
lay-off on a mis-click hands the player a dead turn with no explanation.

**Fix applied.** `add_to_meld` refuses an empty `cards` list with `RuleViolation::NoCardsGiven`.
`LayMeld` was already safe — `Meld::new` rejects fewer than three cards.

---

## F5 — `rewindTurn` reverts too far — and should it exist at all? (medium, blocked on a decision)

**By code reading**, not reproduced — exercising it needs a JS host.

`canastra-wasm/src/lib.rs` refreshes `turn_start` inside `apply`, but only when the phase *before* the
action is `AwaitingDraw`, so the checkpoint is written on the first action of a turn. Call
`rewindTurn()` before taking any action — a player clicking "restart turn" as their opening move — and
`turn_start` still holds the *previous* player's turn start, silently rewinding a whole turn too far.
Fixing the bug is two lines: also refresh after applying, whenever the resulting phase is
`AwaitingDraw`.

**But the prior question is whether rewind belongs in this game at all.** The project's stated
philosophy is that every action is final and nothing can be taken back, and a general undo plainly
violates that.

It was never meant as an undo. It exists because §6 cannot be judged until the discard: the opening
minimum has to be met across a single turn, and whether it was met is unknowable until the player tries
to end the turn. A player who lays 45 toward a 75 minimum has made a move the rules will not let them
complete, and without some escape the game stops. It is closer to "that turn was never legal" than to
"I changed my mind".

That said, F11 has since removed the *other* route into a stuck turn, and the same treatment could
remove this one. Two ways to get there:

1. **Make opening atomic.** A single `Open { melds }` action, validated as a whole, so a partnership
   can never lay part of an opening. Matches how the opening is actually declared at a physical table
   ("I'm opening with these"), and rewind disappears entirely. The complication is §5: taking the pile
   can be part of the same opening, so `TakeDiscardPile` would have to fold into the same action or be
   allowed to precede it.
2. **Validate eagerly.** Refuse a lay if the minimum can no longer be reached from what remains in
   hand. Keeps the API as it is, but requires enumerating meld combinations over the remaining hand at
   every lay — the same combinatorial problem F7 defers, dragged into the hot path.

**Recommendation: option 1**, and drop `rewindTurn` when it lands. Until then the escape hatch is
load-bearing, so the two-line bug is worth fixing regardless of which way this goes.

---

## F6 — Trust boundary undefended by construction (medium, open)

**Not addressed by F2/F3 — they solve the opposite direction.** F2 and F3 validate what comes *in*:
they stop a malformed or impossible state from being loaded. F6 is about what goes *out*, and about who
is allowed to ask. `check_invariants` will happily accept a perfectly valid game and then hand every
card in it to whoever called `snapshot()`.

Two specifics, both by code reading:

- `Game::snapshot()` returns the entire `GameState` as JSON — all four hands, the stock order, and the
  match seed. The seed is the worst of it: with it, the whole deal is reconstructible from scratch.
  It exists for server-side persistence, and nothing stops a browser build from calling it.
- `Game::apply(seat, action)` takes the seat from the caller. The engine correctly rejects moves made
  out of turn, but it cannot know *who is asking*. In a browser the caller is the player, so client-side
  wasm cannot enforce identity at all.

Neither is a bug in the rules core — `apply` guards the turn order and `observe` redacts properly. They
are obligations that land on whoever embeds the engine, and nothing in the code says so at the point
where it matters.

**Suggested fix.** Document both at the `Game` type. Better, split the wasm surface so a browser build
cannot reach `snapshot` at all, and have the server bind an authenticated session to a seat and pass
that rather than anything the client sent. See `web/README.md`, which states these as server
obligations for whoever builds the Go app.

---

## F7 — No `legal_actions`, and `validate` is expensive (low)

Deferred deliberately to the bot milestone, but worth stating: bots currently have to guess an action
and check the error. `validate` is implemented as `apply(...).map(|_| ())`, so every check clones the
whole state. Enumerating a move list is O(moves) full clones.

**Suggested fix.** When the bot work starts, add a non-cloning `validate` and a `legal_actions` that
enumerates. Meld enumeration is the combinatorially interesting part and deserves its own design pass.

---

## F8 — Meld JSON is awkward for other languages (low, open)

Everything else on the wire is self-describing. Cards are `"6D"`. Actions are tagged objects a
TypeScript author can write from memory. Melds are the one exception.

What the engine emits today for `6♥ 7♥ 8♥`:

```json
{"kind":"Sequence","meld":{"suit":"Hearts","low":2,"slots":[
  {"kind":"Natural","card":"6H"},
  {"kind":"Natural","card":"7H"},
  {"kind":"Natural","card":"8H"}]}}
```

`low` is `2` because it is an internal index into the 4..=A range, where `4` is `0`. Nothing in the
payload says so. A client rendering this has to hardcode that offset, and get it right again for every
language that consumes the engine.

It is worse for a wild card. `Coringa-Q♠-K♠-A♠`:

```json
{"kind":"Sequence","meld":{"suit":"Spades","low":7,"slots":[
  {"kind":"Wild","card":"JOKER"},
  {"kind":"Natural","card":"QS"},
  {"kind":"Natural","card":"KS"},
  {"kind":"Natural","card":"AS"}]}}
```

The single most important thing a UI needs here is **which rank the Joker is standing in for** — it is
the Jack — and it is not in the payload at all. The client can only recover it by computing
`low + index` and then inverting the same undocumented offset. Whether the wild is locked (§9), which
decides whether the player may still slide it, is likewise absent and has to be re-derived from its
position.

An ideal shape says what it means and carries the two facts the client cannot cheaply compute:

```json
{"kind":"Sequence","suit":"Spades","cards":[
  {"card":"JOKER","standingIn":"J","wild":true,"locked":false},
  {"card":"QS","wild":false},
  {"card":"KS","wild":false},
  {"card":"AS","wild":false}]}
```

Ace melds are already fine, since there is no ordering to encode:

```json
{"kind":"Aces","meld":{"aces":["AH","AD","AS"],"wild":null}}
```

**Suggested fix.** A hand-written `Serialize` for `Sequence` emitting explicit ranks plus
`standingIn` and `locked`, with `Deserialize` continuing to accept it through the validated `TryFrom`
added in F2. Whatever shape is chosen should be pinned in `tests/boundary.rs` alongside the rest of the
contract. Worth doing before the first client is written, since it is a breaking change afterwards.

---

## F9 — One weak test (low) — FIXED

`going_out_does_not_excuse_the_opening_minimum` asserted with
`matches!(..., Err(NoCleanCanastra | OpeningMinimumNotMet { .. }))`, passing whichever of the two rules
fired. Worse, the setup put the clean canastra on the *opposing* partnership, so the rule it actually
exercised was `NoCleanCanastra` — not the one in its name.

**Fix applied.** The test now lays a clean canastra straight out of hand, which satisfies §11.1 so that
rule cannot be what fires, and picks `5♥..J♥` because it is only 55 in card value — short of the 75
needed to open. `OpeningMinimumNotMet { laid: 55, required: 75 }` is asserted exactly. A companion test
runs the identical lay from an already-open partnership and checks it goes out cleanly, which is what
pins down that the first test fails on the minimum and nothing else.

---

## F10 — proptest and ts-rs not delivered (low, closed)

Closed as not relevant by the project owner.

For the record: the plan promised a `proptest` case asserting `apply` never panics on arbitrary input,
and `ts-rs` TypeScript generation. F2 was exactly the class of bug the proptest would have caught, so
if fuzzing ever does get added, deserialization is where to point it. The interoperability guarantee
itself does not depend on ts-rs — that is the serde contract, and it is pinned by `tests/boundary.rs`.

**Process note, still worth keeping.** For the serialization work the serde derives were written before
`tests/boundary.rs`, so those tests passed on their first run rather than failing first. Derives are
generated code, but the shape decisions they encode — `tag = "type"`, cards as strings — are real design
and should have been driven by tests. They are pinned now; they were not pinned when they were chosen.

---

## F11 — A lay could strand the player's last card (medium) — FIXED

**Found while working through F5**, and fixed in the same pass.

A player could lay down to exactly one card with no clean canastra behind them. That position is
already lost: the compulsory discard would empty their hand, which is going out, which §11.1 refuses
without a clean canastra. There was no legal move left, and the only escape was the rewind whose
existence F5 questions.

Three existing tests turned out to be sitting in exactly that position and were corrected rather than
the rule relaxed — each was testing the opening minimum and happened to lay down to one card on the way.

**Fix applied.** `maybe_go_out` now also refuses a lay that would leave a single card when the
partnership has no clean canastra and the stock still holds cards. The player is told their lay strands
them while they still have a choice, instead of finding out a move later that nothing is legal.

The stock-empty case is deliberately excluded: no discard is owed there, so
`Action::EndTurnWithoutDiscard` carries the turn under clarification #6.

This is a rules *inference*, not something the spec states — it follows from "must discard" plus "may
not go out without a clean canastra". Worth confirming, since it forbids a move players might expect to
be able to make.

---

## Investigated and refuted

**"A player can be forced to empty their hand on an ordinary discard."** Not reachable. A hand at the
start of a turn always holds at least one card, every turn begins with a draw, so the hand is at least
two before the discard and at least one after. Melding down to a position with no legal discard is
possible but self-inflicted within the turn, and rewinding fixes it. The comment to this effect in
`maybe_go_out` is correct.

The exception is F1, where the draw returns no card because the stock ran out mid red-3 replacement —
which is why F1 is a real bug and this is not.
