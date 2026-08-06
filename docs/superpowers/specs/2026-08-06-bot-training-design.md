# Design: neural-network bot training (bot-training milestone)

Date: 2026-08-06

Status: design for approval before implementation

Branch: all work happens in the `bot-training` worktree/branch. `main` is untouched.

## Problem

The repo has a rules-complete Rust engine, toy heuristic bots in TypeScript, and a
harness that plays matches. The next component is the **bot project**: train
neural networks to play Canastra. This spec covers the first iteration —
ladder stage 1 (MLP + genetic algorithm), tabula rasa — plus the two pieces of
shared infrastructure it needs: legal-move enumeration (F7, already spec'd in
`2026-08-05-f7-enumerate-design.md`) and a single-sourced encoding layer.

Design principles adopted from prior discussion (binding):

1. Fixed-length observation, asserted in `encode`.
2. Thermometer encoding over raw magnitudes.
3. Everything relative to the acting seat; one network plays all four seats.
4. Illegal states/actions must be structurally unrepresentable by the policy.
5. Derivable facts (locked wild, is-canastra, extendable) get their own units.
6. Table melds are a shared pool of 33 tokens with an ownership bit.
7. Cheap redundancy is fine (censuses duplicate token info, order-free).
8. Tabula rasa: no shaped rewards, no hand-coded heuristic features-as-rewards.
9. Duplicate-deal evaluation is built **before** training anything.
10. Env stepping is the throughput ceiling: batch across the population, keep
    FFI crossings O(plies), write encodings into pre-allocated buffers.

Decisions taken with the project owner:

- **Training stack is Python + PyTorch**, driving the existing Rust engine
  through a thin PyO3/maturin crate. The engine stays the single referee; no
  rules knowledge exists in Python. Chosen over a pure-Rust trainer for
  ecosystem familiarity and the PPO endgame.
- **The action space is score-the-legal-list**, not a fixed flat head. F7's
  `enumerate` yields the complete legal action list every ply; the policy
  encodes each legal action into a feature vector and scores them with a
  masked softmax. This replaces the earlier "163-unit staged micro-decision
  head" sketch: the engine's `LayMeld { cards }` is atomic and combinatorial,
  so a fixed head over constructed actions would need a staging decoder that
  re-derives meld assembly outside the engine — exactly the rule-knowledge
  split we are avoiding. Masking is automatic and total; self-stranding at the
  opening minimum is covered by the engine's own `CannotReachOpeningMinimum`
  guard plus the restart-turn fallback.
- **The encoder and action featurizer live in a new pure-Rust crate
  `canastra-encode`**, consumed by both the PyO3 bindings (training) and the
  wasm bindings (TS play). One implementation, two consumers: the observation
  layout cannot drift between training and deployment.
- **Trained weights export to JSON**; a `JSONWeightsBot` in the BOTS registry
  runs the forward pass in TypeScript so trained networks play in the existing
  harness and web sandbox.

## Section A — repository layout

```
canastra/  (worktree on branch `bot-training`)
├── engine/crates/
│   ├── canastra-engine/     + enumerate.rs        ← M0: execute the F7 spec
│   ├── canastra-encode/     NEW pure-Rust crate   ← M1
│   └── canastra-wasm/       + legalActions (M0), + encodeState (M1)
├── training/                NEW top-level, training-only (M1–M3)
│   ├── Cargo.toml           canastra-py: PyO3 bindings (maturin)
│   ├── pyproject.toml       maturin project + ruff/mypy config
│   ├── README.md
│   ├── python/canastra_train/
│   │   ├── genome.py        flat parameter vector ↔ torch modules; JSON I/O
│   │   ├── policy.py        torch module: trunk + scoring head, masked softmax
│   │   ├── pool.py          wrapper over the PyO3 game pool
│   │   ├── evaluate.py      duplicate-deal eval runner
│   │   ├── ga.py            population, selection, mutation, checkpointing
│   │   └── train.py         CLI entry point
│   ├── tests/               pytest
│   └── fixtures/            random-init weights JSON for smoke tests
└── bots/src/json-weights.ts  NEW: JSONWeightsBot + TS forward pass (M2)
```

`canastra-encode` joins the engine Cargo workspace (pure Rust, no heavy deps,
covered by the workspace gates). `training/` is a **separate** maturin project
with its own `Cargo.toml` path-depending on `canastra-engine` and
`canastra-encode`; it is deliberately outside the engine workspace so the
engine's gates stay Python-free and fast. The heuristic bots
(`random`, `random-plus`, `random-hungry`) stay in TypeScript; nothing ports
them.

## Section B — M0: legal-move enumeration (F7)

Executed exactly per `2026-08-05-f7-enumerate-design.md`, which is already
approved and test-planned: engine `enumerate`, wasm `legalActions`, the
restructured `Bot` interface (`candidates(view, legal, context)`), the
existing-bot rewrites, and that spec's Section D test list. No deviations.
If implementation surfaces a conflict between that spec and this one, stop and
revisit the design rather than patching locally.

## Section C — `canastra-encode`: observation encoding

Pure Rust, depends only on `canastra-engine`. API:

```rust
pub const OBS_DIM: usize = 2002;
pub const ACT_DIM: usize = 101;

/// Write the acting seat's observation into `out` (length OBS_DIM).
/// Asserts the final length — principle 1.
pub fn encode_observation(view: &PlayerView, out: &mut [f32]);

/// Write one feature vector per legal action. `legal` must be the output of
/// `enumerate(state, seat)` for the acting seat; `table_tokens` is the same
/// canonical token ordering used by encode_observation (target indices align).
pub fn encode_actions(view: &PlayerView, legal: &[Action], out: &mut [f32]);
```

The encoder takes `PlayerView`, never `GameState`: training must see exactly
what a deployed seat may see (`view.rs` doc-comment principle; F6 hygiene).
`canastra-encode` internally calls `observe(state, seat)` where handed a state.

**`PlayerView` extension (part of M1, first task).** Three observation
segments — `laid_value`, `took_pile`, `refusal_available` — describe the
current turn and live in `TurnContext`, which `PlayerView` does not expose.
They are not derivable from the view (the table carries no per-turn
attribution), and they are decision-critical (opening-minimum progress).
The fix is to extend `PlayerView` with `laid_value: u32`, `took_pile: bool`,
`refusal_available: bool`, populated from the current turn's `TurnContext`.
This is not a leak: at a real table everyone watches the acting player lay
melds and take the pile, so these are public facts about the current turn
(the per-observer redaction of `pending_refusal` is unchanged). It **is** a
wire-shape change: `tests/boundary.rs` pins and the hand-written TS types in
`bots/src/types.ts` follow, and M1 includes updating both, plus the wasm
boundary and any view consumers in `web/`.

**Card identity indexing.** Standard cards: `suit_index * 13 + rank_index`,
suits in the engine's `Suit` declaration order, ranks in *game* order
`4,5,6,7,8,9,T,J,Q,K,A,2,3` (sequence-adjacent ranks are encoding-adjacent).
Joker = 52. One-hot identity vectors are 53 wide.

**Segment layout** (offsets pinned; sum = `OBS_DIM` = 2002):

| Segment | Width | Contents |
|---|---|---|
| phase | 5 | one-hot over `Phase` |
| laid_value | 13 | thermometer ≥10…≥130 step 10 |
| took_pile | 1 | bit |
| refusal_available | 1 | bit |
| pending card | 53 | one-hot, all-zero when none |
| hand_number | 6 | thermometer ≥2…≥12 step 2 |
| my hand census | 104 | per standard identity: ≥1, ≥2 |
| my hand jokers | 4 | thermometer ≥1…≥4 |
| frozen census | 108 | same 104+4 shape, frozen cards only |
| my hand size | 8 | thermometer ≥16…≥30 step 2 |
| other hand counts | 36 | right/partner/left, each ≥2…≥24 step 2 |
| stock count | 11 | thermometer ≥4…≥44 step 4 |
| my score | 20 | thermometer ≥250…≥5000 step 250 |
| their score | 20 | same |
| ≥2500 bits | 2 | mine, theirs (opening-threshold proximity) |
| opening minimum | 3 | one-hot {already opened, 75, 120} |
| opened bits | 2 | mine, theirs |
| clean canastra bits | 2 | mine, theirs |
| red threes | 8 | per team thermometer ≥1…≥4 |
| pile top | 53 | one-hot, all-zero when empty |
| pile size | 15 | thermometer ≥2…≥30 step 2 |
| pile census | 108 | order-free, per identity ≥1/≥2 + jokers |
| meld tokens | 1419 | 33 tokens × 43 features |

**Meld token features (43).** present, my-team, kind one-hot {sequence,
aces} (2), suit one-hot (4, zero for aces), low-rank one-hot over the 11
sequence ranks 4…A (11, zero for aces), length thermometer ≥3…≥12 (10),
wild-present, wild-locked (derived: wild at interior index — supplied, never
inferred, principle 5), is-canastra, tier one-hot {none, dirty, clean,
clean-aces} (4), extendable-low, extendable-high, points thermometer
≥25/≥50/≥75/≥100/≥150 (5).

**Canonical token sort.** `(my-team first, kind, suit, low rank, length)`;
aces melds after sequences within a team; empty tokens at the end. This is the
mitigation for slot-order sensitivity (known open risk). If learning plateaus,
the documented swap is Deep Sets pooling over the same token features —
observation-side only; with legal-list scoring there is no WHICH RUN head to
break.

**Why 33 tokens is the exact bound.** The 8 threes (4 red, 4 black) can never
appear in a meld, leaving 100 meldable cards; every meld holds ≥3 cards, so
the table can never exceed `floor(100/3) = 33` melds. Overflow is therefore
unreachable in a legal game; as belt-and-braces the encoder drops tail tokens
beyond 33 deterministically (sort order) and the fuzz test asserts it never
happens.

**Seat relativity.** All segments are encoded from the acting seat: "my",
"right", "partner", "left" are computed from `view.seat`. The same network
plays all four seats; partners share a genome.

**Tests** (Rust, TDD): constant length over all `testkit::Rig` positions and a
fuzz over 1k random games; four-seat relativity (encode one position from each
seat, check the my/right/partner/left permutation); thermometer monotonicity;
token sort stability; no NaN; redaction honored (encoding two states that
differ only in hidden information yields identical vectors — this is the F6
regression test).

## Section D — `canastra-encode`: action features + legal-list scoring

Each legal action from `enumerate` becomes a 101-wide feature vector:

| Block | Width | Contents |
|---|---|---|
| kind | 8 | one-hot over the `Action` variants |
| primary card | 18 | rank one-hot 13 + suit one-hot 4 + is-joker (discard, add-to-meld, keep/refuse) |
| meld descriptor | 28 | suit 4 + low-rank 11 + length therm 10 + wild-present + wild-is-joker + is-aces (LayMeld; TakeDiscardPile core+top shape) |
| target | 34 | meld-token index one-hot 33 + new-meld bit (AddToMeld, TakeDiscardPile) |
| points | 10 | thermometer ≥5…≥50 step 5 over involved cards |
| opening context | 3 | reaches-minimum, exceeds-by-≥25, exceeds-by-≥50 |

The target block's token index uses the **same canonical token ordering** as
the observation's table segment, so the policy can attend to "the meld this
action touches" by index. The translation is explicit: the engine's
`AddToMeld { meld }` / `MeldTarget::Existing { meld }` address melds
**per-partnership**, while the target block indexes the shared 33-token pool —
`canastra-encode` maps team-local index → canonical pool index when featurizing,
and that mapping is pinned by a test (featurize an `AddToMeld` against a rigged
two-team table, assert the one-hot lands on the right token). Distinct legal
actions may rarely collide to identical features; that is acceptable (either
choice is fine) and noted, not papered over.

Policy forward pass (PyTorch, mirrored exactly in TypeScript):

```
emb    = tanh(Linear(512→256)(tanh(Linear(2002→512)(obs))))   # trunk
logit  = Linear(128→1)(tanh(Linear(357→128)([emb; feats])))   # per action
scores = masked_softmax(logits, menu_mask)                    # −inf on padding
```

Menus are ragged: pad each batch to the batch's maximum menu length with a
mask. **Never truncate a menu** — truncation can remove every turn-ending
action. Menu length is asserted ≥ 1 in decision phases.

Playing: argmax over masked scores. Training: sample from the masked softmax
(exploration). `JSONWeightsBot.candidates` returns the legal list sorted by
descending score.

## Section E — weights JSON format (pinned)

```json
{
  "format": "canastra-weights@1",
  "arch": { "obs": 2002, "act": 101, "trunk": [512, 256], "head": [128],
            "activation": "tanh" },
  "params": { "<layer>.weight": { "shape": [m, n], "data": [/* flat row-major */] },
              "<layer>.bias":   { "shape": [m],    "data": [...] } }
}
```

Versioned by `format`. The TS forward pass is a generic tanh-MLP reader driven
by `arch`, so architecture changes do not require TS changes — only the format
version check. A committed fixture (`training/fixtures/random-init.json`) holds
small seeded-random weights for smoke tests.

## Section F — PyO3 boundary (`canastra-py`)

Thin bindings, batch-oriented, one crossing per ply:

```python
class Pool:
    def __init__(self, seeds: list[int]) -> None: ...
    def has_live(self) -> bool: ...
    def encode(self) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """(obs [N,2002] f32, acts [N,M,101] f32, mask [N,M] bool) for all
        seats awaiting a decision, written into pre-allocated buffers."""
    def apply(self, picks: list[int]) -> None:
        """menu index per pending seat; engine remains the referee."""
    def results(self) -> list[MatchResult]: ...  # settled matches
```

Inside Rust: `enumerate` + `encode_*` + `apply` parallelized with Rayon;
hand-over → `settle_hand` and match-over bookkeeping handled in the pool.
Restart-turn fallback: if a seat's menu is empty mid-turn (the residual
self-strand), the pool rewinds that game to its turn-start snapshot and
counts a restart, mirroring the harness driver. A `dump_log` flag appends
JSONL action logs (seed + action log ⇒ full replay, engine guarantee).

M1 includes a benchmark binary measuring plies/sec; results recorded in the
training README. No fixed target is pinned here — if throughput makes a
96-genome generation impractical, revisit population/opponent defaults, not
the architecture.

## Section G — GA trainer (M3)

Tabula rasa: random init, self-play only, scalar reward is the match result.
No shaped rewards, no heuristic features, no discard-history inputs.

Defaults (all CLI-configurable):

- Population 96; elites 8 carried unmutated; tournament selection size 4.
- Each genome plays 4 sampled opponents per generation, each pairing as
  **duplicate-deal paired seatings**: two matches on the same seed,
  `[A,B,A,B]` and `[B,A,B,A]`. Fitness = mean score differential (final
  scores; unfinished matches at the 200k-action cap count as-is; no win
  bonus — score diff only). Common seed set per generation via
  `splitmix64(run_seed, generation)`.
- Hall of fame: the champion is archived every 5 generations; 1 of each
  genome's 4 opponents is drawn from the hall of fame (anti-cycling).
- Mutation: Gaussian σ=0.02, decay ×0.995/generation, floor 0.002. Crossover
  off by default (flag).
- Match = full game to 5000 (multiple hands), same as the harness.
- Checkpoints: genomes + config + generation seeds + fitness table as
  compressed `.npz`, every 5 generations plus on improvement; keep last 10.
- Every training game is replayable from seed + action log.

OpenAI-ES with antithetic pairs is the documented variant if GA plateaus —
same fitness, same eval; not built now.

## Section H — evaluation (M2, built before the trainer)

Two runners, one protocol: N seeds, each played twice with swapped seatings,
report mean differential with 95% CI, win counts, mean hands.

1. **Python**: `evaluate.py` runs genome-vs-genome through the pool.
2. **TypeScript**: `harness/src/eval-nn.ts` (run with tsx) takes a weights
   JSON path + an opponent bot id + seed count and runs the existing
   `headToHead` machinery with a `JSONWeightsBot`. This is the external
   validation path against `random` / `random-plus` / `random-hungry`.

`JSONWeightsBot` (`bots/src/json-weights.ts`): constructed via
`makeJsonWeightsBot(weightsJson, id)`; implements the post-F7 `Bot` interface
by scoring `legal` and returning it sorted. Registered in `BOTS` as
`nn-random` backed by the committed fixture, so smoke tests and the sandbox
can use it without a training run. Harness CLI changes: none.

Sanity gates that must pass before M3 starts:

- random genome vs random genome over 1k paired seeds: differential ≈ 0
  within CI.
- `nn-random` plays full legal matches through the harness with zero
  restarts beyond the residual-strand fallback.

## Section I — milestones and gates

- **M0 — F7 enumeration.** Per the F7 spec. Gates: `cargo test --workspace`,
  `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --check`,
  wasm build, `npm install && npm run typecheck`, harness smoke match.
- **M1 — `canastra-encode` + bindings.** First the `PlayerView` extension
  (`laid_value`, `took_pile`, `refusal_available`) with its `boundary.rs` and
  `bots/src/types.ts` sync; then encoder + featurizer + tests; wasm
  `encodeState`; `training/` scaffold (maturin, pytest, ruff, mypy); pool
  benchmark. Gates: all of M0 plus `maturin develop`, `pytest`, `ruff check`,
  `mypy`, and the F6-redaction encoder test.
- **M2 — eval harness.** Weights JSON + `JSONWeightsBot` + `eval-nn.ts` +
  Python runner. Gates: both sanity checks above.
- **M3 — GA trainer.** `ga.py`/`train.py`, checkpointing, JSONL generation
  logs. Success gate: a trained champion beats `random` decisively and reaches
  at least parity with `random-plus` (CI overlapping or better) in the TS
  duplicate-deal eval at ~10k hands. This gate is a target backed by
  precedent, not a guarantee; named plateau mitigations are the ES variant and
  the Deep Sets observation swap.

`CLAUDE.md` (repo status, commands, architecture) is updated as each
milestone lands.

## Out of scope

- Ladder stages 2–4: frame stacking, GRU/recurrence, PPO.
- Porting heuristic bots to Rust/Python as training opponents.
- Hyperparameter search frameworks; distributed training; training UI.
- Non-cloning `validate` (F7's orthogonal optimisation).
- Changes to the harness CLI argument format.

## Risks

- **Plateau at stage 1.** Mitigations named above (ES variant; Deep Sets).
- **Throughput.** Env stepping dominates; measured in M1 before M3 commits to
  population sizes. Batch shape keeps torch on the happy path.
- **Encoder/deployment drift.** Eliminated structurally by single-sourcing
  `canastra-encode` across PyO3 and wasm; the TS side contains only a generic
  MLP reader.
- **Partnership credit assignment** (partners share one genome; team reward).
  Accepted as-is for stage 1; per-seat value heads belong to the PPO stage.
