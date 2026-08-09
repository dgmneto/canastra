# F7: Enumerate Legal Actions — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Work test-first (@superpowers:test-driven-development): failing test, watch it fail, minimal implementation, watch it pass, commit.

**Goal:** Close ADVERSARIAL-REVIEW.md F7 — give the engine a `legal_actions` enumeration and restructure the TS `Bot` interface around it (milestone M0 of the bot-training spec).

**Architecture:** A new pure-Rust `enumerate` module in `canastra-engine` generates a cheap *superset* of candidate actions per phase and keeps only those `apply` accepts — zero re-implemented rules, no drift possible. A `legalActions` wasm binding carries it to JS, and the three TS bots are rewritten to *rank* the legal list instead of guessing moves.

**Tech Stack:** Rust (engine), wasm-pack (bindings), TypeScript (bots/harness).

**Authoritative references:**
- Spec being executed: `docs/superpowers/specs/2026-08-05-f7-enumerate-design.md` (approved). If anything below conflicts with it, the spec wins — stop and surface the conflict.
- Parent design: `docs/superpowers/specs/2026-08-06-bot-training-design.md`.
- **All work happens in the worktree `/Users/dgmneto/canastra-bot-training` on branch `bot-training`.** Every command below runs there. Rust commands run from `engine/`.

---

### Task 1: Worktree baseline

The worktree shares git history but not build artifacts: `node_modules/` and the gitignored generated wasm (`web/src/engine/`) do not exist there yet.

**Files:** none (setup only).

- [ ] **Step 1: Install JS dependencies**

Run: `npm install` (from `/Users/dgmneto/canastra-bot-training`)
Expected: workspace install completes, no errors.

- [ ] **Step 2: Verify the Rust baseline is green before touching anything**

Run: `cargo test --workspace` (from `/Users/dgmneto/canastra-bot-training/engine`)
Expected: all tests pass, 0 failed.

- [ ] **Step 3: Generate the wasm bindings the harness and web load**

Run: `npm run build:engine` (from `/Users/dgmneto/canastra-bot-training`)
Expected: `wasm-pack build` succeeds; `web/src/engine/canastra.js` + `canastra_bg.wasm` exist.

- [ ] **Step 4: Smoke the harness end-to-end**

Run: `npx canastra-harness --seed 7 random random-plus random random-plus | head -1`
Expected: one JSON line with `"type":"result"` and `"unfinished":false`.

No commit — nothing trackable changed.

---

### Task 2: `enumerate` scaffold + first failing test

**Files:**
- Create: `engine/crates/canastra-engine/src/enumerate.rs`
- Modify: `engine/crates/canastra-engine/src/lib.rs`
- Test: `engine/crates/canastra-engine/tests/enumerate.rs`

- [ ] **Step 1: Write the failing test**

Create `engine/crates/canastra-engine/tests/enumerate.rs`:

```rust
//! `enumerate` is the answer to "what may I do right now?" (F7). These tests
//! pin coverage (every legal move is offered), soundness (nothing illegal
//! survives), and determinism (same position, same list).

use std::collections::HashSet;

use canastra_engine::action::MeldTarget;
use canastra_engine::state::Phase;
use canastra_engine::testkit::Rig;
use canastra_engine::{Action, GameState, Seat, apply, enumerate, new_game, settle_hand};

fn seat(index: u8) -> Seat {
    Seat::new(index).expect("0..=3")
}

/// The `LayMeld` payloads in a list, as sorted card strings for
/// order-free comparison.
fn lay_melds(actions: &[Action]) -> HashSet<Vec<String>> {
    actions
        .iter()
        .filter_map(|action| match action {
            Action::LayMeld { cards } => {
                Some(cards.iter().map(|card| card.to_string()).collect())
            }
            _ => None,
        })
        .collect()
}

#[test]
fn awaiting_draw_with_no_takeable_pile_yields_exactly_draw() {
    let state = Rig::new().stock("8C 9D").discard("9C").hand(1, "4D 5D").build();
    assert_eq!(enumerate(&state, seat(1)), vec![Action::Draw]);
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test -p canastra-engine --test enumerate` (from `engine/`)
Expected: FAIL — `enumerate` is not found / unresolved import.

- [ ] **Step 3: Write the minimal implementation**

Create `engine/crates/canastra-engine/src/enumerate.rs`:

```rust
//! Legal-move enumeration (ADVERSARIAL-REVIEW.md F7).
//!
//! `enumerate` answers the question every bot has to ask — "what am I allowed
//! to do right now?" — so policies rank real moves instead of guessing and
//! checking `apply` errors. It generates a cheap superset of candidates and
//! lets `apply` judge every one: the rules stay in exactly one place, and the
//! two can never drift apart.

use crate::action::{Action, MeldTarget};
use crate::apply::apply;
use crate::card::{Card, Rank, Suit};
use crate::state::{GameState, Phase, Seat};

/// Every action `seat` may legally take right now, one ply, in deterministic
/// order. Call it with the player whose turn it is; any other seat, or a
/// finished hand or match, gets an empty list.
pub fn enumerate(state: &GameState, seat: Seat) -> Vec<Action> {
    if seat != state.turn {
        return Vec::new();
    }
    let candidates = match state.phase {
        Phase::AwaitingDraw => draw_candidates(state, seat),
        Phase::AwaitingRefusalChoice => refusal_candidates(),
        Phase::Melding => melding_candidates(state, seat),
        Phase::HandOver | Phase::MatchOver => Vec::new(),
    };

    let mut actions: Vec<Action> = candidates
        .into_iter()
        .filter(|action| apply(state, seat, action).is_ok())
        .collect();
    // Deterministic order matters: seed + action log must keep replaying, and
    // bots keyed on the list need a stable input. Candidates are already
    // emitted with their cards in canonical (string) order, so equivalent
    // candidates are equal values and dedup collapses them.
    actions.sort_by_key(|action| format!("{action:?}"));
    actions.dedup();
    actions
}

/// §3: the lead player's one-time choice about the card they were shown.
fn refusal_candidates() -> Vec<Action> {
    vec![Action::KeepDrawnCard, Action::RefuseDrawnCard]
}

/// §4.1 / §5: draw from the stock, or capture the pile with a natural core.
fn draw_candidates(state: &GameState, seat: Seat) -> Vec<Action> {
    let mut candidates = vec![Action::Draw];
    let meld_count = state.table(seat.team()).melds.len();
    for core in natural_pairs(state.hand(seat)) {
        candidates.push(Action::TakeDiscardPile {
            core,
            target: MeldTarget::NewMeld,
        });
        for meld in 0..meld_count {
            candidates.push(Action::TakeDiscardPile {
                core,
                target: MeldTarget::Existing { meld },
            });
        }
    }
    candidates
}

/// §5 cores: every unordered pair of natural cards held. Multiset-based — a
/// same-value pair (e.g. two A♥) is a candidate when two copies are held,
/// since a pair of aces with an ace on top is a legal capture. Wilds are
/// excluded statically; `apply` filters blocked tops, frozen cores, §6
/// reachability, and invalid joins.
fn natural_pairs(hand: &[Card]) -> Vec<[Card; 2]> {
    let mut uniques: Vec<Card> = hand.iter().copied().filter(|card| card.is_natural()).collect();
    uniques.sort_by_key(|card| card.to_string());
    uniques.dedup();

    let mut pairs = Vec::new();
    for (i, &a) in uniques.iter().enumerate() {
        for &b in &uniques[i..] {
            if a == b && hand.iter().filter(|&&card| card == a).count() < 2 {
                continue;
            }
            pairs.push([a, b]);
        }
    }
    pairs
}

/// §4.2–§4.3: meld, lay off, discard, or (in exactly one corner) end the
/// turn holding a card that cannot legally be thrown.
fn melding_candidates(state: &GameState, seat: Seat) -> Vec<Action> {
    let hand = state.hand(seat);
    let mut candidates = lay_meld_candidates(hand);

    let mut distinct: Vec<Card> = hand.to_vec();
    distinct.sort_by_key(|card| card.to_string());
    distinct.dedup();

    for meld in 0..state.table(seat.team()).melds.len() {
        for &card in &distinct {
            candidates.push(Action::AddToMeld {
                meld,
                cards: vec![card],
            });
        }
    }
    for &card in &distinct {
        candidates.push(Action::Discard { card });
    }
    candidates.push(Action::EndTurnWithoutDiscard);
    candidates
}

/// §7: every distinct meld this hand can lay — sequence windows plus ace
/// sub-multisets. Cards within each candidate are in canonical (string)
/// order so equivalent candidates dedup to one.
fn lay_meld_candidates(hand: &[Card]) -> Vec<Action> {
    let mut candidates = Vec::new();

    for suit in Suit::ALL {
        // One card per rank: a sequence covers each rank once.
        let mut held: Vec<Card> = hand
            .iter()
            .copied()
            .filter(|card| card.suit() == Some(suit))
            .filter(|card| card.rank().and_then(Rank::sequence_index).is_some())
            .collect();
        held.sort_by_key(|card| card.rank().and_then(Rank::sequence_index));
        held.dedup();

        // §8: wilds usable in this suit's sequences — the Joker anywhere, a
        // 2 only in its own suit. Each wild held yields its own candidate.
        let mut wilds: Vec<Card> = hand
            .iter()
            .copied()
            .filter(|card| {
                card.is_joker()
                    || (card.suit() == Some(suit) && card.rank() == Some(Rank::Two))
            })
            .collect();
        wilds.sort_by_key(|card| card.to_string());
        wilds.dedup();

        // Every rank window of length ≥ 3 in 4..A (indices 0..=10).
        for start in 0..11u8 {
            for len in 3..=(11u8 - start) {
                let present: Vec<Card> = (start..start + len)
                    .filter_map(|index| {
                        held.iter()
                            .find(|card| {
                                card.rank().and_then(Rank::sequence_index) == Some(index)
                            })
                            .copied()
                    })
                    .collect();
                match (start + len - start) as usize - present.len() {
                    0 => candidates.push(lay(present)),
                    1 => {
                        for &wild in &wilds {
                            let mut cards = present.clone();
                            cards.push(wild);
                            candidates.push(lay(cards));
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // §7.2: every sub-multiset of 3+ natural aces held, each with and without
    // one held wild (an ace meld takes a Joker or any 2). Sub-multiset, not
    // subset: the deck holds two copies of each ace and `AcesMeld` accepts
    // duplicates, so `AH AH AD` is a legal meld.
    let mut aces: Vec<Card> = hand
        .iter()
        .copied()
        .filter(|card| card.rank() == Some(Rank::Ace))
        .collect();
    aces.sort_by_key(|card| card.to_string());

    let mut wilds: Vec<Card> = hand.iter().copied().filter(|card| card.is_wild()).collect();
    wilds.sort_by_key(|card| card.to_string());
    wilds.dedup();

    for size in 3..=aces.len() {
        for combo in combinations(&aces, size) {
            candidates.push(lay(combo.clone()));
            for &wild in &wilds {
                let mut cards = combo.clone();
                cards.push(wild);
                candidates.push(lay(cards));
            }
        }
    }

    candidates
}

/// Every `size`-card combination of `cards`, deduplicated by value (the hand
/// may hold two copies of the same card).
fn combinations(cards: &[Card], size: usize) -> Vec<Vec<Card>> {
    fn pick(
        cards: &[Card],
        size: usize,
        next: usize,
        stack: &mut Vec<usize>,
        seen: &mut HashSet<Vec<Card>>,
        out: &mut Vec<Vec<Card>>,
    ) {
        if stack.len() == size {
            let mut combo: Vec<Card> = stack.iter().map(|&i| cards[i]).collect();
            combo.sort_by_key(|card| card.to_string());
            if seen.insert(combo.clone()) {
                out.push(combo);
            }
            return;
        }
        for i in next..cards.len() {
            stack.push(i);
            pick(cards, size, i + 1, stack, seen, out);
            stack.pop();
        }
    }

    use std::collections::HashSet;
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    pick(cards, size, 0, &mut Vec::new(), &mut seen, &mut out);
    out
}

/// A `LayMeld` candidate with its cards in canonical (string) order.
fn lay(mut cards: Vec<Card>) -> Action {
    cards.sort_by_key(|card| card.to_string());
    Action::LayMeld { cards }
}
```

Note: this is the **complete** module — later tasks only add tests, never restructure it. If a test in a later task forces a code change here, that is a signal the test (or this design) is wrong: re-check against the F7 spec before editing.

Modify `engine/crates/canastra-engine/src/lib.rs`: add `pub mod enumerate;` to the module list (after `pub mod deal;` keeps alphabetical order: action, apply, card, deal, enumerate, meld, …) and add `pub use enumerate::enumerate;` to the re-exports.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p canastra-engine --test enumerate`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add engine/crates/canastra-engine/src/enumerate.rs engine/crates/canastra-engine/src/lib.rs engine/crates/canastra-engine/tests/enumerate.rs
git commit -m "Enumerate legal actions: scaffold, Draw-only in AwaitingDraw"
```

---

### Task 3: Refusal phase, wrong seat, terminal phases

**Files:**
- Modify: `engine/crates/canastra-engine/src/testkit.rs` (add `pending_refusal`)
- Test: `engine/crates/canastra-engine/tests/enumerate.rs`

- [ ] **Step 1: Extend the Rig with a pending refusal card**

Add to `impl Rig` in `engine/crates/canastra-engine/src/testkit.rs`, next to `refusal_available`:

```rust
    /// §3: the card the lead player is currently deciding whether to keep.
    pub fn pending_refusal(mut self, spec: &str) -> Rig {
        self.state.turn_context.pending_refusal = Some(card(spec));
        self
    }
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/enumerate.rs`:

```rust
#[test]
fn refusal_choice_yields_keep_and_refuse() {
    // The stock must be non-empty: refusing means drawing a replacement.
    let state = Rig::new()
        .phase(Phase::AwaitingRefusalChoice)
        .pending_refusal("KH")
        .refusal_available()
        .stock("8C 9D")
        .hand(1, "4D 5D")
        .build();
    let actions = enumerate(&state, seat(1));
    assert!(actions.contains(&Action::KeepDrawnCard));
    assert!(actions.contains(&Action::RefuseDrawnCard));
    assert_eq!(actions.len(), 2);
}

#[test]
fn refusal_without_stock_leaves_only_keep() {
    // Refusing means drawing a replacement, so an empty stock takes the
    // option off the table even mid-decision.
    let state = Rig::new()
        .phase(Phase::AwaitingRefusalChoice)
        .pending_refusal("KH")
        .refusal_available()
        .hand(1, "4D 5D")
        .build();
    assert_eq!(enumerate(&state, seat(1)), vec![Action::KeepDrawnCard]);
}

#[test]
fn other_seats_and_terminal_phases_get_nothing() {
    let state = Rig::new().stock("8C 9D").discard("9C").hand(1, "4D 5D").build();
    assert!(enumerate(&state, seat(0)).is_empty(), "not seat 0's turn");

    for phase in [Phase::HandOver, Phase::MatchOver] {
        let state = Rig::new().phase(phase).build();
        assert!(enumerate(&state, seat(1)).is_empty(), "{phase:?} decides nothing");
    }
}
```

- [ ] **Step 3: Run to verify they pass**

The Task 2 implementation already covers these (refusal candidates filtered by `apply`). Run: `cargo test -p canastra-engine --test enumerate`
Expected: 4 passed. If `refusal_without_stock_leaves_only_keep` fails, check how `apply.rs` handles `RefuseDrawnCard` against an empty stock and reconcile with the F7 spec before changing anything.

- [ ] **Step 4: Commit**

```bash
git add engine/crates/canastra-engine/src/testkit.rs engine/crates/canastra-engine/tests/enumerate.rs
git commit -m "Enumerate: refusal phase, wrong-seat and terminal-phase guards"
```

---

### Task 4: LayMeld sequence windows

**Files:**
- Test: `engine/crates/canastra-engine/tests/enumerate.rs`

- [ ] **Step 1: Write the failing test**

Append:

```rust
#[test]
fn lay_meld_windows_from_a_four_card_run() {
    // §6 note: the partnership is marked open so the eager opening-minimum
    // check does not filter the small melds this test is about.
    let state = Rig::new()
        .phase(Phase::Melding)
        .opened(1)
        .hand(1, "4H 5H 6H 7H QS")
        .discard("9C")
        .build();
    let melds = lay_melds(&enumerate(&state, seat(1)));
    let expected: HashSet<Vec<String>> = [
        vec!["4H", "5H", "6H"],
        vec!["5H", "6H", "7H"],
        vec!["4H", "5H", "6H", "7H"],
    ]
    .into_iter()
    .map(|meld| meld.into_iter().map(String::from).collect())
    .collect();
    assert_eq!(melds, expected);
}
```

- [ ] **Step 2: Run to verify it passes**

Task 2's `lay_meld_candidates` already implements this. Run: `cargo test -p canastra-engine --test enumerate lay_meld_windows`
Expected: PASS. If it fails, debug against the F7 spec's window scheme (§"Candidate generation / LayMeld").

- [ ] **Step 3: Commit**

```bash
git add engine/crates/canastra-engine/tests/enumerate.rs
git commit -m "Enumerate: sequence window LayMelds"
```

---

### Task 5: LayMeld wild cards

**Files:**
- Test: `engine/crates/canastra-engine/tests/enumerate.rs`

- [ ] **Step 1: Write the failing tests**

Append:

```rust
#[test]
fn lay_meld_with_a_joker_filling_the_gap() {
    // The spare QS keeps the lay from emptying the hand: going out needs a
    // clean canastra (§11.1), which this table does not have.
    let state = Rig::new()
        .phase(Phase::Melding)
        .opened(1)
        .hand(1, "6H 7H JOKER QS")
        .discard("9C")
        .build();
    let melds = lay_melds(&enumerate(&state, seat(1)));
    assert_eq!(
        melds,
        HashSet::from([vec!["6H".to_string(), "7H".to_string(), "JOKER".to_string()]])
    );
}

#[test]
fn lay_meld_plain_and_wild_capped_variants() {
    // Spare QS as above: a four-card lay must not empty the hand.
    let state = Rig::new()
        .phase(Phase::Melding)
        .opened(1)
        .hand(1, "4H 5H 6H JOKER QS")
        .discard("9C")
        .build();
    let melds = lay_melds(&enumerate(&state, seat(1)));
    let expected: HashSet<Vec<String>> = [
        vec!["4H", "5H", "6H"],
        vec!["4H", "5H", "6H", "JOKER"],
    ]
    .into_iter()
    .map(|meld| meld.into_iter().map(String::from).collect())
    .collect();
    assert_eq!(melds, expected);
}

#[test]
fn each_usable_wild_yields_its_own_meld() {
    // §8: a 2 works only in its own suit. With both a Joker and the 2♥ in
    // hand, the same window produces two distinct melds; the 2♦ is unusable.
    let with_two_wilds = Rig::new()
        .phase(Phase::Melding)
        .opened(1)
        .hand(1, "4H 5H 6H JOKER 2H")
        .discard("9C")
        .build();
    let melds = lay_melds(&enumerate(&with_two_wilds, seat(1)));
    let expected: HashSet<Vec<String>> = [
        vec!["4H", "5H", "6H"],
        vec!["4H", "5H", "6H", "JOKER"],
        vec!["2H", "4H", "5H", "6H"],
    ]
    .into_iter()
    .map(|meld| meld.into_iter().map(String::from).collect())
    .collect();
    assert_eq!(melds, expected);

    let foreign_two = Rig::new()
        .phase(Phase::Melding)
        .opened(1)
        .hand(1, "4H 5H 6H 2D")
        .discard("9C")
        .build();
    let melds = lay_melds(&enumerate(&foreign_two, seat(1)));
    assert_eq!(
        melds,
        HashSet::from([vec!["4H".to_string(), "5H".to_string(), "6H".to_string()]])
    );
}
```

- [ ] **Step 2: Run to verify they pass**

Run: `cargo test -p canastra-engine --test enumerate lay_meld`
Expected: all `lay_meld_*` tests pass. The `foreign_two` case is the sharp one: a 2 of another suit must produce no candidate, and it must also not accidentally appear via the aces path (no aces in hand here).

- [ ] **Step 3: Commit**

```bash
git add engine/crates/canastra-engine/tests/enumerate.rs
git commit -m "Enumerate: wild-capped LayMelds, per-wild candidates"
```

---

### Task 6: Ace meld sub-multisets

**Files:**
- Test: `engine/crates/canastra-engine/tests/enumerate.rs`

- [ ] **Step 1: Write the failing test**

Append:

```rust
#[test]
fn ace_melds_are_sub_multisets_with_and_without_a_wild() {
    // §7.2 with duplicated aces: AH AH AD AS yields every 3- and 4-ace
    // sub-multiset, each plain and each capped with the held Joker. The spare
    // QS keeps even the five-card lay from emptying the hand (§11.1).
    let state = Rig::new()
        .phase(Phase::Melding)
        .opened(1)
        .hand(1, "AH AH AD AS JOKER QS")
        .discard("9C")
        .build();
    let melds = lay_melds(&enumerate(&state, seat(1)));
    let expected: HashSet<Vec<String>> = [
        vec!["AD", "AH", "AH"],
        vec!["AH", "AH", "AS"],
        vec!["AD", "AH", "AS"],
        vec!["AD", "AH", "AH", "AS"],
        vec!["AD", "AH", "AH", "JOKER"],
        vec!["AH", "AH", "AS", "JOKER"],
        vec!["AD", "AH", "AS", "JOKER"],
        vec!["AD", "AH", "AH", "AS", "JOKER"],
    ]
    .into_iter()
    .map(|meld| meld.into_iter().map(String::from).collect())
    .collect();
    assert_eq!(melds, expected);
}
```

- [ ] **Step 2: Run to verify it passes**

Run: `cargo test -p canastra-engine --test enumerate ace_melds`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add engine/crates/canastra-engine/tests/enumerate.rs
git commit -m "Enumerate: ace sub-multiset LayMelds with duplicate aces"
```

---

### Task 7: AddToMeld, Discard, EndTurnWithoutDiscard — and the frozen/final-card corners

**Files:**
- Test: `engine/crates/canastra-engine/tests/enumerate.rs`

- [ ] **Step 1: Write the failing tests**

Append:

```rust
#[test]
fn discards_dedup_identical_cards() {
    let state = Rig::new()
        .phase(Phase::Melding)
        .hand(1, "6D 6D KH")
        .meld(1, "4H 5H 6H")
        .discard("9C")
        .build();
    let actions = enumerate(&state, seat(1));
    let discards: Vec<&Action> = actions
        .iter()
        .filter(|a| matches!(a, Action::Discard { .. }))
        .collect();
    assert_eq!(
        discards,
        [&Action::Discard { card: "6D".parse().unwrap() }, &Action::Discard { card: "KH".parse().unwrap() }]
    );
    // A legal discard exists, so the no-discard escape hatch stays shut.
    assert!(!actions.contains(&Action::EndTurnWithoutDiscard));
}

#[test]
fn frozen_cards_can_be_discarded_but_not_melded() {
    // §5 + clarification #5: cards swept from the pile are frozen this turn —
    // unmeldable, still discardable.
    let state = Rig::new()
        .phase(Phase::Melding)
        .hand(1, "7D 8D 9D 4H 5H 6H")
        .frozen("7D 8D 9D")
        .meld(1, "4D 5D 6D")
        .discard("9C")
        .build();
    let actions = enumerate(&state, seat(1));

    let frozen = ["7D", "8D", "9D"];
    let melding_with_frozen = actions.iter().any(|action| match action {
        Action::LayMeld { cards } => cards.iter().any(|c| frozen.contains(&c.to_string().as_str())),
        Action::AddToMeld { cards, .. } => {
            cards.iter().any(|c| frozen.contains(&c.to_string().as_str()))
        }
        _ => false,
    });
    assert!(!melding_with_frozen, "frozen cards must not be melded: {actions:?}");

    // The natural extension that would be legal without the freeze.
    assert!(!actions.contains(&Action::AddToMeld { meld: 0, cards: vec!["7D".parse().unwrap()] }));
    // …while discarding a frozen card stays legal (clarification #5).
    assert!(actions.contains(&Action::Discard { card: "7D".parse().unwrap() }));
    // And the unfrozen run is still offered.
    assert!(lay_melds(&actions).contains(&vec!["4H".to_string(), "5H".to_string(), "6H".to_string()]));
}

#[test]
fn cornered_player_may_only_end_without_discarding() {
    // CLAUDE.md clarification #6: one card in hand, no clean canastra — no
    // legal discard exists, so EndTurnWithoutDiscard is the only way out.
    let state = Rig::new()
        .phase(Phase::Melding)
        .hand(1, "KH")
        .meld(1, "4H 5H 6H")
        .discard("9C")
        .build();
    assert_eq!(enumerate(&state, seat(1)), vec![Action::EndTurnWithoutDiscard]);
}
```

- [ ] **Step 2: Run to verify they pass**

Run: `cargo test -p canastra-engine --test enumerate`
Expected: all pass. The cornered test pins the exact list; if it also contains a `Discard`, re-read `RuleViolation::NoCleanCanastra` / `WouldStrandLastCard` handling in `apply.rs` — one of them should be refusing that discard.

- [ ] **Step 3: Commit**

```bash
git add engine/crates/canastra-engine/tests/enumerate.rs
git commit -m "Enumerate: melding-phase moves, frozen and cornered corners"
```

---

### Task 8: TakeDiscardPile — the §5 worked example

**Files:**
- Test: `engine/crates/canastra-engine/tests/enumerate.rs`

- [ ] **Step 1: Write the failing tests**

Append:

```rust
#[test]
fn taking_the_pile_the_rules_spec_worked_example() {
    // §5: with the 6♦ on top, the legal cores are 4D 5D, 5D 7D and 7D 8D —
    // each into a new meld, and into whichever existing meld the three join.
    let state = Rig::new()
        .stock("8C 9D")
        .discard("9C 6D")
        .hand(1, "4D 5D 7D 8D KH")
        .meld(1, "7D 8D 9D")
        .meld(1, "8D 9D TD")
        .build();
    let takes: HashSet<Action> = enumerate(&state, seat(1))
        .into_iter()
        .filter(|a| matches!(a, Action::TakeDiscardPile { .. }))
        .collect();
    let take = |a: &str, b: &str, target: MeldTarget| Action::TakeDiscardPile {
        core: [a.parse().unwrap(), b.parse().unwrap()],
        target,
    };
    let expected = HashSet::from([
        take("4D", "5D", MeldTarget::NewMeld),
        take("4D", "5D", MeldTarget::Existing { meld: 0 }),
        take("5D", "7D", MeldTarget::NewMeld),
        take("5D", "7D", MeldTarget::Existing { meld: 1 }),
        take("7D", "8D", MeldTarget::NewMeld),
    ]);
    assert_eq!(takes, expected);
}

#[test]
fn taking_the_pile_with_a_same_value_core() {
    // A pair of aces with an ace on top is a legal capture (aces meld).
    let state = Rig::new()
        .stock("8C 9D")
        .discard("9C AS")
        .hand(1, "AH AH KD QC")
        .opened(1)
        .build();
    let actions = enumerate(&state, seat(1));
    assert!(actions.contains(&Action::TakeDiscardPile {
        core: ["AH".parse().unwrap(), "AH".parse().unwrap()],
        target: MeldTarget::NewMeld,
    }));
    assert_eq!(actions.len(), 2, "just Draw and the one capture: {actions:?}");
}

#[test]
fn a_blocked_pile_offers_no_takes() {
    // §5: a black 3 on top puts the pile out of reach.
    let state = Rig::new()
        .stock("8C 9D")
        .discard("9C 3S")
        .hand(1, "4S 5S KD QC")
        .build();
    assert_eq!(enumerate(&state, seat(1)), vec![Action::Draw]);
}
```

- [ ] **Step 2: Run to verify they pass**

Run: `cargo test -p canastra-engine --test enumerate`
Expected: all pass. If the worked example offers extra/missing `Existing` targets, check whether the three cards really join the named meld — `apply` is the judge and this test encodes its verdicts, not the other way around.

- [ ] **Step 3: Commit**

```bash
git add engine/crates/canastra-engine/tests/enumerate.rs
git commit -m "Enumerate: discard-pile captures, blocked and same-value cores"
```

---

### Task 9: Determinism + soundness fuzz

**Files:**
- Test: `engine/crates/canastra-engine/tests/enumerate.rs`

- [ ] **Step 1: Write the failing tests**

Append:

```rust
#[test]
fn enumeration_is_deterministic() {
    let state = Rig::new()
        .stock("8C 9D")
        .discard("9C 6D")
        .hand(1, "4D 5D 7D 8D KH")
        .meld(1, "7D 8D 9D")
        .build();
    assert_eq!(enumerate(&state, seat(1)), enumerate(&state, seat(1)));
}

/// SplitMix64 finalizer — a deterministic pseudo-random pick without pulling
/// an rng dependency into the test suite.
fn mix(seed: u64, ply: u64) -> usize {
    let mut x = seed ^ ply.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (x ^ (x >> 31)) as usize
}

#[test]
fn everything_enumerated_is_legal_across_whole_matches() {
    for seed in 0..20u64 {
        let mut state: GameState = new_game(seed);
        let mut turn_start = state.clone();
        let mut safe = false;
        for ply in 0..200_000u64 {
            match state.phase {
                Phase::HandOver => {
                    state = settle_hand(&state).expect("settle");
                    continue;
                }
                Phase::MatchOver => break,
                _ => {}
            }
            // A turn can dead-end (§6's eager check is optimistic), so keep
            // the position it began from — the harness driver does the same.
            if matches!(state.phase, Phase::AwaitingDraw | Phase::AwaitingRefusalChoice) {
                turn_start = state.clone();
                safe = false;
            }
            let turn = state.turn;
            let actions = enumerate(&state, turn);
            for action in &actions {
                assert!(apply(&state, turn, action).is_ok(), "offered an illegal {action:?}");
            }
            if actions.is_empty() {
                // The residual self-strand: back out and finish the turn
                // plainly, which is exactly the driver's safeMode path.
                assert!(!safe, "even the plain retry dead-ended (seed {seed})");
                state = turn_start.clone();
                safe = true;
                continue;
            }
            let pick = if safe { 0 } else { mix(seed, ply) % actions.len() };
            state = apply(&state, turn, &actions[pick]).expect("enumerated action applies");
        }
    }
}
```

- [ ] **Step 2: Run to verify they pass**

Run: `cargo test -p canastra-engine --test enumerate -- --nocapture`
Expected: all pass. The fuzz is the load-bearing test of the whole task list: it plays 20 seeded matches choosing random enumerated actions and asserts every offered action applies. A failure here means candidate generation missed a rule that `apply` knows — find the mismatch and fix the generator, never the filter.

- [ ] **Step 3: Run the full engine suite plus the lint gates**

Run (from `engine/`):
```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```
Expected: all green. Fix any clippy/fmt findings in `enumerate.rs` before proceeding.

- [ ] **Step 4: Commit**

```bash
git add engine/crates/canastra-engine/tests/enumerate.rs engine/crates/canastra-engine/src/enumerate.rs
git commit -m "Enumerate: determinism and whole-match soundness fuzz"
```

---

### Task 10: wasm `legalActions` + rebuild

**Files:**
- Modify: `engine/crates/canastra-wasm/src/lib.rs`

- [ ] **Step 1: Add the binding**

In `engine/crates/canastra-wasm/src/lib.rs`, add `enumerate` to the `canastra_engine` import list, then add this method inside `impl Game`, after `view`:

```rust
    /// F7: every action `seat` may legally take right now, one ply, in
    /// deterministic order. Serializes as the same `{type: ...}` objects the
    /// TS `Action` union already describes.
    #[wasm_bindgen(js_name = legalActions)]
    pub fn legal_actions(&self, seat: u8) -> Result<JsValue, JsValue> {
        let seat = Seat::new(seat).ok_or_else(|| message("seat must be 0, 1, 2 or 3"))?;
        serde_wasm_bindgen::to_value(&enumerate(&self.state, seat))
            .map_err(|error| message(&error.to_string()))
    }
```

- [ ] **Step 2: Verify the wasm promise**

Run: `cargo build -p canastra-wasm --target wasm32-unknown-unknown` (from `engine/`)
Expected: builds clean.

- [ ] **Step 3: Regenerate the committed bindings**

Run: `npm run build:engine` (from the worktree root)
Expected: `web/src/engine/canastra.d.ts` now declares `legalActions(seat: number): any`.

- [ ] **Step 4: Smoke the binding in Node**

Run (from the worktree root):

```bash
node --experimental-wasm-modules --input-type=module -e "
import init, { Game } from './web/src/engine/canastra.js';
import { readFileSync } from 'node:fs';
await init({ module_or_path: new WebAssembly.Module(readFileSync('./web/src/engine/canastra_bg.wasm')) });
const game = new Game(7n);
console.log(JSON.stringify(game.legalActions(1)));
game.apply(1, { type: 'Draw' });
console.log(JSON.stringify(game.legalActions(1)));
"
```

Expected: a fresh match starts in `AwaitingDraw` (the refusal choice only
appears once the first card is in hand), so the first line is exactly
`[{"type":"Draw"}]`. After the draw, seat 1 faces the §3 decision, so the
second line contains `{"type":"KeepDrawnCard"}` and `{"type":"RefuseDrawnCard"}`
(if the drawn card was a red 3, clarification #4 carries the privilege to the
replacement card — the pair still appears once a non-red-3 is shown).

- [ ] **Step 5: Commit**

```bash
git add engine/crates/canastra-wasm/src/lib.rs
git commit -m "Expose legalActions across the wasm boundary"
```

(`web/src/engine/` is gitignored — the build artifacts are not committed.)

---

### Task 11: Restructure the `Bot` interface

**Files:**
- Modify: `bots/src/bot.ts`
- Modify: `harness/src/match.ts`
- Modify: `harness/src/driver.ts`

Typecheck stays red until Task 14 — do Tasks 12–14 before running it.

- [ ] **Step 1: Rewrite `bots/src/bot.ts`**

Full replacement:

```ts
/**
 * What a bot is.
 *
 * A bot is a *policy* and nothing else: given a position and the complete
 * list of moves the engine allows in it, rank those moves best first. It does
 * not touch the engine, does not know whether a move was accepted, and cannot
 * end a turn on its own — the harness's driver does all of that, identically
 * for every bot, so two bots are always judged on the same terms.
 *
 * `legal` is the engine's own enumeration (ADVERSARIAL-REVIEW.md F7, closed).
 * The interface used to be "propose a list and let `apply` judge" because the
 * engine could not say what was legal; now that it can, a bot's body is pure
 * preference. The engine remains the referee: anything returned outside
 * `legal` is refused, and running out of ideas concedes the turn (the driver
 * restarts it), so a well-behaved bot returns every legal move, ranked.
 */

import type { Action, PlayerView } from "./types";
import type { Rng } from "./rng";

export interface BotContext {
  /** Seeded, so a match with bots in it still replays. */
  rng: Rng;
  /**
   * The previous attempt at this turn dead-ended and was restarted, so the
   * bot must not walk into the same wall — draw and discard, nothing clever.
   *
   * The deal is deterministic: an unmodified retry reproduces the position
   * exactly, so a bot that ignores this will loop forever.
   */
  safeMode: boolean;
}

export interface Bot {
  /** Stable across versions — it goes in the replay log. */
  readonly id: string;
  readonly name: string;
  /** One line, shown in the UI. */
  readonly blurb: string;

  /**
   * The legal moves in this position, best first.
   *
   * Ordering *is* the policy: the driver plays the first one. Returning an
   * empty list concedes the turn — the driver will restart it, which is a
   * real cost — so the list should always contain every legal move, however
   * low the tail ranks.
   */
  candidates(view: PlayerView, legal: Action[], context: BotContext): Action[];
}

/** The legal actions of one variant, narrowed for the caller. */
export function ofType<T extends Action["type"]>(
  legal: Action[],
  type: T,
): Extract<Action, { type: T }>[] {
  return legal.filter((a): a is Extract<Action, { type: T }> => a.type === type);
}
```

- [ ] **Step 2: Add the passthrough to `harness/src/match.ts`**

Add this method to `class Match`, after `views()`:

```ts
  /** F7: every action `seat` may legally take right now, straight from the engine. */
  legalActions(seat: Seat): Action[] {
    return this.game.legalActions(seat) as Action[];
  }
```

- [ ] **Step 3: Resolve the legal list in `harness/src/driver.ts`**

In `step`, replace:

```ts
  const seat = view.turn;
  const refusals: string[] = [];

  for (const action of bot.candidates(view, context)) {
```

with:

```ts
  const seat = view.turn;
  const legal = match.legalActions(seat);
  const refusals: string[] = [];

  for (const action of bot.candidates(view, legal, context)) {
```

Also update the comment above the trailing `restartTurn` call: refusals are now rare diagnostics for a bot that returns moves outside `legal`, not the normal path.

- [ ] **Step 4: Commit**

```bash
git add bots/src/bot.ts harness/src/match.ts harness/src/driver.ts
git commit -m "Restructure the Bot interface around the engine's legal list"
```

(Typecheck will fail until the bots are rewritten — that is expected; the next three tasks fix it.)

---

### Task 12: Rewrite `random.ts` as a ranker

**Files:**
- Modify: `bots/src/random.ts`

- [ ] **Step 1: Rewrite the bot**

Full replacement of `bots/src/random.ts`:

```ts
/**
 * Random — the baseline.
 *
 * Lays whatever it can find, adds single cards to its own melds, and throws
 * away its cheapest card. It never takes the discard pile (§5), never holds a
 * black 3 as a blocker, and never plays toward a canastra — so a partnership of
 * these two builds wide and shallow, and is punished by §12's red 3s for it.
 *
 * That is the point: it is the floor a real bot has to beat.
 */

import type { Action, Card } from "./types";
import { cardValue } from "./types";
import type { Bot, BotContext } from "./bot";
import { ofType } from "./bot";
import { findMelds, meldCards, meldValue } from "./melds";

export const randomBot: Bot = {
  id: "random",
  name: "Random",
  blurb: "Lays what it finds, discards its cheapest card. The floor.",

  candidates(view, legal, context: BotContext): Action[] {
    switch (view.phase) {
      case "AwaitingRefusalChoice": {
        // §3: the once-per-hand refusal. Cheap cards are worth throwing back.
        const refuse = ofType(legal, "RefuseDrawnCard");
        const keep = ofType(legal, "KeepDrawnCard");
        return view.pending_refusal && cardValue(view.pending_refusal) <= 5 && context.rng() < 0.5
          ? [...refuse, ...keep]
          : [...keep, ...refuse];
      }

      case "AwaitingDraw":
        // Never reaches for the pile, but the tail stays ranked so the list
        // is always complete.
        return [...ofType(legal, "Draw"), ...legal.filter((a) => a.type !== "Draw")];

      case "Melding": {
        const moves: Action[] = [];
        const table = view.tables[view.seat % 2];

        if (!context.safeMode) {
          // §6: the opening minimum has to be met inside one turn. The engine's
          // eager check is optimistic — it counts every remaining card at face
          // value — so it will happily allow a 45-point lay that this hand can
          // never grow to 75, leaving a turn that cannot be discarded out of.
          // Only rank lays at all if hand plus table actually clears the bar.
          //
          // What is already down counts. A partnership that is not open yet but
          // has melds on the table laid them earlier in *this* turn — that is
          // the only way to be in that position — so their value is this turn's
          // progress, and `PlayerView` carries no `laid_value` to read instead.
          const playable = view.hand.filter((card) => !view.frozen.includes(card));
          const inHand = findMelds(playable).reduce((sum, meld) => sum + meldValue(meld), 0);
          const alreadyLaid = table.opened
            ? 0
            : table.melds.reduce((sum, meld) => sum + meldValue(meldCards(meld)), 0);
          const layable = table.opened || alreadyLaid + inHand >= view.opening_minimum;

          if (layable && context.rng() < 0.85) moves.push(...ofType(legal, "LayMeld"));
          if (context.rng() < 0.7) moves.push(...ofType(legal, "AddToMeld"));
        }

        // §4.3: the turn ends with a discard. Cheapest first, so it keeps the
        // cards worth points.
        moves.push(...ofType(legal, "Discard").sort((a, b) => cardValue(a.card) - cardValue(b.card)));
        moves.push(...ofType(legal, "EndTurnWithoutDiscard"));

        // Ranking is complete: anything not yet listed trails the discards.
        for (const action of legal) if (!moves.includes(action)) moves.push(action);
        return moves;
      }

      default:
        return [...legal];
    }
  },
};
```

(The `Card` import may be unused depending on whether `playable` typing needs it — drop it if `tsc` flags `noUnusedLocals`… the project tsconfig does not enable it, so an unused import is tolerated but sloppy: remove it if unused.)

- [ ] **Step 2: Commit**

```bash
git add bots/src/random.ts
git commit -m "Random ranks the legal list instead of guessing moves"
```

---

### Task 13: Rewrite `random-plus.ts` as a ranker

**Files:**
- Modify: `bots/src/random-plus.ts`

- [ ] **Step 1: Rewrite the bot**

Full replacement of `bots/src/random-plus.ts`. The four opinions survive unchanged — what changes is that they *rank* `legal` rather than construct guesses:

```ts
/**
 * Random Plus — Random with four opinions.
 *
 * All four come from the same observation: §13 pays for *depth*, not breadth.
 * Random builds six shallow melds, scores 420 in table cards and still only
 * banks 90, because it earns no canastra bonus and its red 3s turn negative.
 * The bonuses are the game.
 *
 *  1. **Hoard 2s.** §8 lets a 2 into a meld; §10 then caps that meld at the
 *     dirty tier forever — 200 instead of 500. Worse, §12 pays red 3s ±100
 *     each on whether a *clean* canastra exists, so one careless 2 can cost
 *     300 in bonus and flip up to 400 more in red 3s. A Joker does none of
 *     that, so Jokers are spent freely and 2s are held back. The exception is
 *     opening: §6's minimum has to be met in one turn, and a partnership that
 *     never opens scores nothing at all.
 *
 *  2. **Deepen before widening.** A seventh card on a six-card meld is worth
 *     500; a fresh three-card meld is worth about 15. Lay-offs are ranked
 *     before new melds, longest meld first.
 *
 *  3. **Discard what has no future.** Random throws its cheapest card, which
 *     is exactly the 4–7 that runs are built from. This one scores every card
 *     by whether it extends a meld or has same-suit neighbours in hand, then
 *     throws the least useful — breaking ties toward *high* cards, since
 *     §13.2 charges for whatever is still in hand at the end.
 *
 *  4. **Use black 3s as blockers.** §5 says a black 3 on top puts the pile out
 *     of reach. They are worth nothing wherever they sit, so holding one is
 *     free and throwing one onto a fat pile denies it to the next player.
 */

import type { Action, Card, Meld, PlayerView } from "./types";
import { cardValue } from "./types";
import type { Bot, BotContext } from "./bot";
import { ofType } from "./bot";
import { findMelds, meldCards, meldValue } from "./melds";

const SEQUENCE_RANKS = "456789TJQKA";
const SUIT_CODE: Record<string, string> = {
  Clubs: "C",
  Diamonds: "D",
  Hearts: "H",
  Spades: "S",
};

/** A pile worth denying. Below this, a black 3 is worth keeping for later. */
const PILE_WORTH_BLOCKING = 5;

export const randomPlusBot: Bot = {
  id: "random-plus",
  name: "Random Plus",
  blurb: "Hoards 2s for clean canastras, deepens melds, discards dead cards.",

  candidates(view, legal, context: BotContext): Action[] {
    switch (view.phase) {
      case "AwaitingRefusalChoice": {
        // §3: one refusal per hand. Spend it on a card with no future.
        const card = view.pending_refusal;
        const dead = card !== null && !isWild(card) && usefulness(card, view) < 0;
        const refuse = ofType(legal, "RefuseDrawnCard");
        const keep = ofType(legal, "KeepDrawnCard");
        return dead ? [...refuse, ...keep] : [...keep, ...refuse];
      }

      case "AwaitingDraw":
        return [...ofType(legal, "Draw"), ...legal.filter((a) => a.type !== "Draw")];

      case "Melding": {
        const moves: Action[] = [];
        const table = view.tables[view.seat % 2];

        if (!context.safeMode) {
          if (table.opened) {
            moves.push(...rankLayOffs(view, legal));
            // Clean melds only once open: there is no deadline any more, so a
            // 2 spent here buys nothing that waiting would not.
            moves.push(...ofType(legal, "LayMeld").filter((a) => !a.cards.some(isTwo)));
          } else {
            moves.push(...opening(view, legal));
          }
        }

        moves.push(...rankDiscards(view, legal, context));
        moves.push(...ofType(legal, "EndTurnWithoutDiscard"));
        for (const action of legal) if (!moves.includes(action)) moves.push(action);
        return moves;
      }

      default:
        return [...legal];
    }
  },
};

/**
 * §6: the opening lay.
 *
 * Clean first — if the minimum is reachable without touching a 2, that is
 * strictly better. Only when it is not does it spend them, because a
 * partnership that never opens scores nothing, and that dwarfs the tier.
 *
 * The reachability test is unchanged: the engine's eager check is optimistic
 * (it counts every remaining card at face value), so a bot that trusts it can
 * lay 45 toward 75 and then be unable to discard.
 */
function opening(view: PlayerView, legal: Action[]): Action[] {
  const minimum = view.opening_minimum;
  // Melds already down while `opened` is false were laid earlier in this same
  // turn — the only way to be in that position — so they are progress.
  const table = view.tables[view.seat % 2];
  const playable = view.hand.filter((card) => !view.frozen.includes(card));
  const laid = table.melds.reduce((sum, meld) => sum + meldValue(meldCards(meld)), 0);

  const lays = ofType(legal, "LayMeld");
  const clean = lays.filter((a) => !a.cards.some(isTwo));
  const cleanReachable =
    laid + findMelds(playable.filter((card) => !isTwo(card))).reduce((s, m) => s + meldValue(m), 0) >=
    minimum;
  if (cleanReachable) return [...rankLayOffs(view, legal), ...clean];

  const anyReachable = laid + findMelds(playable).reduce((s, m) => s + meldValue(m), 0) >= minimum;
  if (anyReachable) return [...rankLayOffs(view, legal), ...lays];

  // Out of reach this turn either way. Laying anything now would only strand
  // the turn, so rank nothing and let the discard close it.
  return [];
}

/**
 * §4.2: single-card lay-offs, longest meld first.
 *
 * Longest first is the whole point — the seventh card is worth 200 or 500, and
 * every card after that is worth its face value on a meld that already paid.
 * A 2 is only ever added to a meld §10 has already spoiled.
 */
function rankLayOffs(view: PlayerView, legal: Action[]): Action[] {
  const table = view.tables[view.seat % 2];
  return ofType(legal, "AddToMeld")
    .filter((a) => !isTwo(a.cards[0]) || meldCards(table.melds[a.meld]).some(isTwo))
    .sort(
      (a, b) =>
        meldCards(table.melds[b.meld]).length - meldCards(table.melds[a.meld]).length,
    );
}

/**
 * §4.3: the compulsory discard, least useful card first.
 */
function rankDiscards(view: PlayerView, legal: Action[], context: BotContext): Action[] {
  const blocking = view.discard.length >= PILE_WORTH_BLOCKING;

  const ranked = ofType(legal, "Discard").sort((a, b) => {
    // §5: a black 3 on top freezes the pile. Free to hold, so it is thrown
    // when there is something worth denying and kept when there is not.
    const scoreA = isBlackThree(a.card) ? (blocking ? -100 : 40) : usefulness(a.card, view);
    const scoreB = isBlackThree(b.card) ? (blocking ? -100 : 40) : usefulness(b.card, view);
    if (scoreA !== scoreB) return scoreA - scoreB;
    // §13.2 charges for whatever is still in hand, so dump the dearer card.
    return cardValue(b.card) - cardValue(a.card);
  });

  // A little noise so two Random Plus seats do not play in lockstep.
  if (ranked.length > 2 && context.rng() < 0.15) {
    [ranked[0], ranked[1]] = [ranked[1], ranked[0]];
  }
  return ranked;
}

/**
 * How much this card is worth keeping. Higher means hold.
 *
 * Wilds are never voluntarily discarded: a Joker is a free slot in any future
 * canastra, and a 2 is being saved deliberately.
 */
function usefulness(card: Card, view: PlayerView): number {
  if (isWild(card)) return 1000;

  const ours = view.tables[view.seat % 2].melds;
  if (ours.some((meld) => extendsMeld(card, meld))) return 500;

  const rank = SEQUENCE_RANKS.indexOf(card[0]);
  if (rank < 0) return 0;

  // Same-suit cards within two ranks are the makings of a future run.
  let neighbours = 0;
  for (const other of view.hand) {
    if (other === card || other.length !== 2 || other[1] !== card[1]) continue;
    const distance = Math.abs(SEQUENCE_RANKS.indexOf(other[0]) - rank);
    if (distance >= 1 && distance <= 2) neighbours += 1;
  }

  // With no neighbours the card is dead weight, and the dearer it is the more
  // it costs to keep holding it.
  return neighbours * 30 - cardValue(card);
}

/** Whether this card would sit on either end of that meld. */
function extendsMeld(card: Card, meld: Meld): boolean {
  if (meld.kind === "Aces") return card.length === 2 && card[0] === "A";
  if (card.length !== 2 || card[1] !== SUIT_CODE[meld.suit]) return false;

  // Slots run low to high, and a wild reports the rank it stands in for, so
  // the run's ends are the first and last effective ranks.
  const effective = meld.cards.map((slot) => SEQUENCE_RANKS.indexOf(slot.standingInRank ?? slot.card[0]));
  const low = Math.min(...effective);
  const high = Math.max(...effective);
  const rank = SEQUENCE_RANKS.indexOf(card[0]);
  return rank === low - 1 || rank === high + 1;
}

function isTwo(card: Card): boolean {
  return card.length === 2 && card[0] === "2";
}

function isWild(card: Card): boolean {
  return card === "JOKER" || isTwo(card);
}

function isBlackThree(card: Card): boolean {
  return card.length === 2 && card[0] === "3" && (card[1] === "S" || card[1] === "C");
}
```

- [ ] **Step 2: Commit**

```bash
git add bots/src/random-plus.ts
git commit -m "Random Plus ranks the legal list instead of guessing moves"
```

---

### Task 14: Rewrite `random-discard-hungry.ts` + trim check

**Files:**
- Modify: `bots/src/random-discard-hungry.ts`
- Modify: `bots/src/melds.ts` (only if something is now unused)

- [ ] **Step 1: Rewrite the bot**

Full replacement of `bots/src/random-discard-hungry.ts` — the hand-rolled §5 shape math is deleted; the engine now answers §5 exactly:

```ts
/**
 * Random Discard Hungry — Random, but it always reaches for the pile.
 *
 * Deliberately **identical to Random everywhere except the draw**: it delegates
 * the whole melding and discarding phase to `randomBot`. That is what makes the
 * comparison mean something — any difference in results is §5 pile-taking and
 * nothing else.
 *
 * §5 makes this an expensive habit, which is the interesting part:
 *
 *  - The three cards that capture the pile must be the top card plus **two
 *    natural cards from hand of the same suit**, forming a contiguous run. No
 *    wild may stand in, and all three land in one meld.
 *  - Everything else in the pile goes to hand and is **frozen** — unmeldable
 *    until the next turn (CLAUDE.md clarification #5 reads "usada" as *melded*,
 *    so a frozen card may still be discarded).
 *  - The capturing meld takes no wild for the rest of the turn.
 *
 * So a big pile is a big hand of dead cards, and §13.2 charges for every card
 * still held at the end. Hunger is not obviously good, which is why it is worth
 * measuring rather than assuming.
 */

import type { Action } from "./types";
import type { Bot, BotContext } from "./bot";
import { ofType } from "./bot";
import { randomBot } from "./random";

export const randomDiscardHungryBot: Bot = {
  id: "random-hungry",
  name: "Random Discard Hungry",
  blurb: "Random, but grabs the discard pile whenever §5 allows it.",

  candidates(view, legal, context: BotContext): Action[] {
    if (view.phase === "AwaitingDraw") {
      // §5 taking is exactly what dead-ends a turn: the captured pile arrives
      // frozen, so a partnership that has not opened often cannot reach §6's
      // minimum and cannot then discard. `safeMode` says that already happened
      // — reaching for the pile again would reproduce it, forever, because the
      // deal is deterministic.
      if (context.safeMode) return ofType(legal, "Draw");
      // Every legal capture first, the ordinary draw last.
      return [...ofType(legal, "TakeDiscardPile"), ...ofType(legal, "Draw")];
    }
    return randomBot.candidates(view, legal, context);
  },
};
```

- [ ] **Step 2: Trim `melds.ts` if anything went unused**

Run: `grep -n "enumerateMelds\|findMelds\|meldCards\|meldValue" bots/src/*.ts`
Expected: `findMelds`, `meldCards`, `meldValue` are all still imported by `random.ts`/`random-plus.ts`. `enumerateMelds` is used inside `findMelds`; if nothing external imports it, drop the `export` keyword. Remove nothing else — all four helpers are still load-bearing (as evaluators, not generators).

- [ ] **Step 3: Commit**

```bash
git add bots/src/random-discard-hungry.ts bots/src/melds.ts
git commit -m "Random Discard Hungry takes from the legal list; drop §5 shape math"
```

---

### Task 15: Typecheck + behavioral smoke

**Files:** none changed.

- [ ] **Step 1: Typecheck all three TS projects**

Run: `npm run typecheck` (from the worktree root)
Expected: bots, harness, web all clean. `web/` needed no changes — it drives bots through `step`, whose signature is unchanged.

- [ ] **Step 2: Smoke a full match**

Run: `npx canastra-harness --seed 7 random random-plus random random-plus | head -1`
Expected: `"type":"result"`, `"unfinished":false`, and low `"restarts"` (0 is typical; the residual self-strand makes a small nonzero count legitimate).

- [ ] **Step 3: Sanity-check the pecking order survived the rewrite**

The rewritten bots rank rather than guess, so exact replays differ — but the ordering of strength must not: Random Plus should clearly beat Random over a small series.

Run (from the worktree root):

```bash
npx tsx -e "
import { loadEngine } from './harness/src/load-node.ts';
import { headToHead } from './harness/src/series.ts';
await loadEngine();
const report = headToHead('random', 'random-plus', 40);
console.log(JSON.stringify(report));
"
```

Expected: `winsEles` (random-plus) comfortably above `winsNos` (random) across the 40 matches. If the ordering inverts or lands near 50/50, a rewrite bug is far more likely than a rules surprise — diff the ranking logic against the old bots on `main` (`git show main:bots/src/random-plus.ts`) before touching anything else.

No commit — verification only.

---

### Task 16: Close F7 in the adversarial review + final gates

**Files:**
- Modify: `ADVERSARIAL-REVIEW.md` (F7 entry only)

- [ ] **Step 1: Mark F7 closed**

Read the F7 entry in `ADVERSARIAL-REVIEW.md` ("No `legal_actions`, and `validate` is expensive") and update its status to record the closure: `enumerate` in `canastra-engine`, `legalActions` on the wasm `Game`, bots now rank the legal list. Do not touch the "non-cloning `validate`" half-sentence's substance — that optimisation stays explicitly out of scope per the F7 spec.

- [ ] **Step 2: Full gate sweep**

Run (from `engine/` unless noted):

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
cargo build -p canastra-wasm --target wasm32-unknown-unknown
npm run typecheck            # from the worktree root
npx canastra-harness --seed 7 random random-plus random random-plus | head -1
```

Expected: every command green.

- [ ] **Step 3: Commit**

```bash
git add ADVERSARIAL-REVIEW.md
git commit -m "Close F7: legal-move enumeration, exposed to bots"
```

---

## Done criteria for M0

- `enumerate` lives in the engine with the full test list from the F7 spec's Section D, including the whole-match soundness fuzz.
- `legalActions` crosses the wasm boundary; `Match`, `step`, and all three bots speak the new `candidates(view, legal, context)` interface.
- All gates green; the pecking-order sanity check holds.
- F7 marked closed in `ADVERSARIAL-REVIEW.md`.

Then: write the M1 plan (PlayerView extension + `canastra-encode` + PyO3 scaffold) from `docs/superpowers/specs/2026-08-06-bot-training-design.md`.
