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
| F5 | Opening minimum only judged at the discard | medium | **fixed** (eager check) |
| F6 | Trust boundary undefended by construction | medium | open — a server obligation |
| F7 | No `legal_actions`; `validate` clones | low | open, deliberately |
| F8 | Meld JSON awkward for other languages | low | **fixed** |
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

## F5 — The opening minimum was only judged at the discard (medium) — FIXED

The original finding was a bug in `rewindTurn`. The real issue turned out to be
why `rewindTurn` existed at all.

§6 requires the opening minimum to be met across a single turn, and the engine only judged it when the
player tried to discard. A player who laid 45 toward a 75 minimum had made a move the rules would not
let them complete, and the only escape was to abandon the turn. That is undo by another name, and this
game's moves are final.

**Fix applied.** `check_opening_reachable` now runs after every lay. It bounds what the player could
still add — every card still usable, at full face value — and refuses the lay outright if even that
cannot reach the minimum. The error carries `laid`, `best_possible` and `required`, so a UI can say
exactly how short they would be. §5's frozen cards are excluded, since they cannot be melded this turn.

**The bound is deliberately optimistic, and that is the safe direction.** It asks whether the points
could still be laid, not whether they can be arranged into legal melds. A player holding 95 points of
cards that cannot form a single meld will still be allowed to lay, and will still fail at the discard.

Erring the other way would be much worse. A check that is too generous leaves a turn that has to be
abandoned — an annoyance. A check that is too strict refuses a move the rules permit — a broken game.
So the check only ever fires when it is *certain*.

**What is left open.** Closing the gap completely means computing the best possible melding of an
arbitrary hand: choosing per suit which runs to form from up to two copies of each rank, deciding
whether each ace serves a run or the ace meld, and allocating wild cards one per meld. That is the same
combinatorial problem `legal_actions` faces (F7) — which is the connection between the two, though the
version needed here is strictly easier, since it only needs the best achievable *value* rather than the
full list of moves.

Until that exists, `rewindTurn` remains as a backstop for the residual case. With F11 closing the other
route into a stuck turn and this check closing most of this one, it should be reachable far less often.
The original two-line bug — `turn_start` refreshed only when the phase *before* an action is
`AwaitingDraw`, so calling `rewindTurn` as the first move of a turn reverts a whole turn too far — is
still worth fixing while the method exists.

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

## F8 — Meld JSON was awkward for other languages (low) — FIXED

Everything else on the wire was self-describing. Cards are `"6D"`. Actions are tagged objects a
TypeScript author can write from memory. Melds were the exception.

What the engine used to emit for `6♥ 7♥ 8♥`:

```json
{"kind":"Sequence","meld":{"suit":"Hearts","low":2,"slots":[
  {"kind":"Natural","card":"6H"}, {"kind":"Natural","card":"7H"}, {"kind":"Natural","card":"8H"}]}}
```

`low` was `2` because it is an internal index into the 4..=A range where `4` is `0`. Nothing in the
payload said so. Worse, for `Coringa-Q♠-K♠-A♠` the single thing a UI most needs — which rank the Joker
stands in for — was absent entirely, recoverable only by computing `low + index` and inverting that
same undocumented offset. Whether the wild was locked (§9), which decides whether the player may still
slide it, was missing too.

**Fix applied.** Melds are now internally tagged and say what they mean:

```json
{"kind":"Sequence","suit":"Spades","cards":[
  {"card":"JOKER","standingInRank":"J","locked":false},
  {"card":"QS"},{"card":"KS"},{"card":"AS"}]}
```

```json
{"kind":"Sequence","suit":"Hearts","cards":[
  {"card":"6H"},{"card":"7H"},
  {"card":"2H","standingInRank":"8","locked":true},
  {"card":"9H"}]}
```

```json
{"kind":"Aces","aces":["AH","AD","AS"],"wild":null}
```

A natural card carries nothing but itself, since its rank is in the code. A wild carries the two facts
a client cannot cheaply compute. `locked` is derived on the way out and recomputed on the way in, so a
payload cannot lie about it.

**`standingInRank` is a bare rank (`"J"`), not a card (`"JS"`).** The deck holds two real J♠, and
emitting a third would invite a client to render an actual Jack of Spades, or to trip over it when
matching cards for animation. The field name carries the type, so nothing is ambiguous. `Rank` now
serializes as the same single character cards use, rather than as `"Jack"`.

Deserialization goes through the F2 validation, so the new shape cannot be used to smuggle in an
invalid meld either. All of it is pinned in `tests/boundary.rs`.

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
