# web/ — Canastra engine sandbox

A single page that deals a match, drives all four seats with bots, and shows every hand face up.
It exists to exercise the engine and to watch bot behaviour, **not** to play against other people —
there is no server, no accounts, and no networking.

## Two pages

- `/` — the multiplayer game client. Thin: no engine, no wasm; it renders what `@canastra/server`
  sends and sends actions back. It never sees another seat's hand.
- `/sandbox.html` — the engine sandbox described below, unchanged.

## Running it

The JS projects are an npm workspace rooted at the repo root, so install once there:

```bash
npm install
```

```bash
npm run build:engine
```

```bash
npm run dev --prefix web
```

`build:engine` runs `wasm-pack` over `engine/crates/canastra-wasm` into `web/src/engine/`, which is
generated and git-ignored. Re-run it after any change to the Rust engine, or the page keeps using the
previously built wasm. It needs `wasm-pack` on the PATH (`npm i -g wasm-pack`) and the
`wasm32-unknown-unknown` target installed.

Controls: **step** plays one action, **play** runs continuously at the chosen speed, **seed** fixes the
deal, **refusals** shows the moves each bot tried before finding a legal one, **stop on hand end**
pauses at `HandOver` before the hand is banked so the deciding position is still on the table, and
**download log** saves the replay.

Each partnership's table carries its §13 score *as it stands*, with the arithmetic on hover. Mid-hand
it is a running total, not a result: going out is not yet paid, and cards still in hand are still
counting against them. It comes from the engine's `score_hand` via `Game::handScore` — the canastra
tiers, the red-3 sign flip and black 3s scoring nothing are all §13, and a second copy of those rules
here would be a copy that drifts.

Seats are labelled **nós** and **eles** throughout. §2 seats partners opposite each other, so seats 0
and 2 are one partnership; the sandbox watches from seat 0, which is what makes those names mean
anything.

## How it is put together

The engine driver and the bots do not live here — they are workspace packages the page imports:

- `../harness/` (`@canastra/harness`) — `Match` wraps the wasm `Game` handle (the only code that
  touches wasm), `step` runs a bot against the engine, and `runMatch`/`series` are the headless
  runs. `src/match.ts`, `src/driver.ts` and `src/lab.ts` here are thin re-exports of it.
- `../bots/` (`@canastra/bots`) — the bots, the `BOTS` registry, and the engine wire types. See
  below.
- `src/loadEngine.ts` — initialises the wasm the browser way. The harness CLI initialises the same
  wasm the Node way; each environment brings its own loader.
- `src/types.ts` — re-exports the engine wire shapes from `@canastra/bots`, which stay pinned
  against `engine/crates/canastra-engine/tests/boundary.rs` on the Rust side.
- `src/ui/` — React components. They read a `PlayerView` and render it; they hold no game rules.

## Writing a bot

A bot is a **policy and nothing else**: given a position, name the moves worth trying, best first.

```ts
export const myBot: Bot = {
  id: "greedy",
  name: "Greedy",
  blurb: "One line, shown in the UI.",
  candidates(view, context) {
    return [/* Actions, best first */];
  },
};
```

Add the file to `../bots/src/`, then add it to `BOTS` in `../bots/src/index.ts`. Nothing else changes — the
driver, the per-seat pickers and the replay log all read from that list. Each seat picks its own bot,
so bots compete by sitting at the same table; a change takes effect on the next match, since swapping
a policy mid-hand would make one log describe two different matches.

**Why "propose a list" rather than "return the move".** The engine has no `legal_actions` (F7), so a
bot cannot know which of its ideas are legal. It offers several and lets `apply` be the judge —
refusal leaves the position untouched, so guessing is free. If `legal_actions` ever lands, this
interface deserves revisiting; until then it is the honest shape.

Three rules, and the second is the one that actually bites:

- **Everything that is not policy lives in `driver.ts`** — settling a finished hand, restarting a
  dead-ended turn, recording refusals. A bot that did its own book-keeping would not be comparable.
- **`context.safeMode` must be respected.** It is set after a turn dead-ended and was restarted. The
  deal is deterministic, so a retry that repeats the same reasoning reproduces the same dead end
  *forever*. In safe mode, draw and discard — nothing clever.

  This is not theoretical. Random Discard Hungry was written ignoring `safeMode` in its draw phase and
  promptly hung: 1884 turn restarts in three hands, because taking the pile freezes the captured cards
  (§5), a partnership that has not opened then cannot reach §6's minimum, cannot discard, restarts —
  and reaches straight for the pile again. Honouring `safeMode` took it to single-digit restarts per
  match. **If a bot makes matches slow or `unfinished`, suspect this first.**
- **A bot must not assume its proposals are legal.** The engine is the judge, and refusal is normal.
  A candidate list that runs out is a restarted turn, which is a real cost, so always end it with
  something that can finish a turn.

`context.rng` is seeded, not `Math.random`, so a match with bots in it still replays.

`../bots/src/melds.ts` holds the shared meld search (which combinations are worth proposing). It is not
rules — the engine remains the only judge of a legal meld — but it is the expensive part to get right,
so bots share it.

Bots on the roster:

| bot | notes |
|---|---|
| **Random** | Lays what it finds, adds single cards to its own melds, discards its cheapest. Never takes the pile, never holds a black 3, never plays toward a canastra. The floor a real bot has to beat. |
| **Random Plus** | Hoards 2s so its canastras stay clean, deepens melds before starting new ones, discards cards with no future, and throws black 3s to block a fat pile. |
| **Random Discard Hungry** | Random in every phase but the draw, where it always reaches for the pile (§5). A controlled experiment: melding and discarding are delegated to `randomBot` itself, so any difference is pile-taking and nothing else. |

### Results

Round robin, 60 seeds per seating, both seatings per pairing — 360 matches, no unfinished.

| pairing | winner | record | mean score |
|---|---|---|---|
| Random Plus vs Random | **Random Plus** | 93–27 (**77.5%**) | 5086 vs 3610 |
| Random Discard Hungry vs Random | **Random Discard Hungry** | 90–30 (**75.0%**) | 5027 vs 3626 |
| Random Plus vs Random Discard Hungry | *neither* | 59–61 (**49.2%**) | 4436 vs 4518 |

Two unrelated strategies — build deep and clean, or take every pile — each beat the baseline by about
the same margin, and are **indistinguishable from each other**. 59–61 out of 120 is a coin flip; the
interval on that sample is roughly ±9%, so this says the two are close, not which is better.

The cost shows up elsewhere. Pile-taking dead-ends turns constantly: pairings involving Hungry logged
**348–482 turn restarts** per 120 matches against **58–74** for the others, because §5's frozen cards
so often leave a turn that cannot reach §6's minimum. It buys its wins expensively.

Matches between the two strong bots also end sooner — about 8 hands against 10 for pairings involving
Random — which is what reaching 5000 faster looks like.

### Why Random Plus wins

All four of its rules come from one observation: §13 pays for *depth*, not breadth. Random routinely
puts more card value on the table than its opponent and still loses, because it earns no canastra
bonus and its red 3s turn negative.

1. **Hoard 2s.** §8 lets a 2 into a meld and §10 then caps that meld at the dirty tier forever — 200
   instead of 500. Worse, §12 pays red 3s ±100 each on whether a *clean* canastra exists, so one
   careless 2 can cost 300 in bonus and swing up to 800 more in red 3s. A Joker does none of this, so
   Jokers are spent freely and 2s are held. The exception is §6: opening beats purity, because a
   partnership that never opens takes a flat −300 (§13.3).
2. **Deepen before widening.** A seventh card on a six-card meld is worth 500; a fresh three-card meld
   is worth about 15. Lay-offs are proposed before new melds, longest meld first.
3. **Discard what has no future.** Random throws its cheapest card — which is exactly the 4–7 that
   runs are built from. Random Plus scores each card by whether it extends a meld or has same-suit
   neighbours in hand, and breaks ties toward *high* cards, since §13.2 charges for whatever is left
   in hand.
4. **Black 3s as blockers.** §5 puts the pile out of reach when a black 3 is on top. They score zero
   wherever they sit, so holding one is free and throwing one onto a fat pile denies it.

## Measuring a bot

One match proves nothing — a single deal swings hundreds of points on which side the red 3s landed.
`globalThis.lab` runs matches headlessly, without React:

```js
lab.headToHead("random-plus", "random", 100)   // both seatings, 100 seeds each
lab.series(["random-plus", "random", "random-plus", "random"], 100)
lab.runMatch(7n, ["random", "random", "random", "random"])
```

`headToHead` plays each pairing in both seatings on purpose: seats 0 and 2 are one partnership (§2),
the lead rotates, and a bot that only won from one side has not shown anything. Roughly one match per
250 ms, so a 200-match run takes about a minute — run it from an `await`-friendly chunked loop rather
than one synchronous call, or the tab will hang.

`unfinished` counts matches that hit the action cap without anyone reaching 5000. That is a bug in a
bot, not a draw.

**This page is deliberately omniscient.** It calls `observe(state, seat)` for all four seats and shows
every hand, which is exactly what a multiplayer client must never do. Nothing here should be lifted
into a networked client without re-imposing that boundary. `Game::handScore` is part of that: its
`hand_cards` sums *both* partners' hands, and partners do not see each other's cards, so it answers a
question no seat is entitled to ask. It sits on the same side of the line as `Game::snapshot`. See F6 in
[ADVERSARIAL-REVIEW.md](../ADVERSARIAL-REVIEW.md), which lists the server obligations: bind a session
to a seat rather than trusting a client's claim, and never let `Game::snapshot()` (which carries the
seed, and so the whole deal) reach a browser.

**The bots guess.** The engine has no `legal_actions` yet (F7), so `bot.ts` proposes candidate moves in
priority order and lets `apply` refuse them — a refusal leaves the position untouched, so guessing costs
nothing but time. Turning on **refusals** shows which rule did the work, which is the most useful view
in here when something looks wrong.

**Turn restarts do not use `rewindTurn()`.** `Match` keeps its own snapshot of the position each turn
began from. The wasm `rewindTurn` refreshes its checkpoint only when the phase *before* an action is
`AwaitingDraw`, so calling it as the first move of a turn reverts a whole turn too far (F5).

## The replay log

`download log` writes JSON Lines: a header carrying the seed, then one entry per accepted action.

```
{"seed":"7","startedAt":"…","bots":["random","random","random","random"]}
{"seat":1,"action":{"type":"Draw"}}
{"seat":1,"action":{"type":"LayMeld","cards":["AH","AH","AS"]}}
{"seat":1,"action":{"type":"Discard","card":"6H"}}
{"settleHand":true}
```

The engine fixes the stock at deal time and never reshuffles, so the seed plus this list reproduces a
whole match. The header names which bot sat in which seat, because the log is a record of a *match*
and that is the thing you want to know when comparing two of them. Abandoned turns are recorded as `{"seat":n,"restartTurn":true}` rather than being cut out
of the log, so a replayer has to honour them — with the same checkpoint rule `Match` uses.

`globalThis.canastra` is the live `Match`, for poking at a position from the console.

## Known limits

- Random Discard Hungry takes the pile whenever §5 allows, but no bot decides *whether it is worth
  taking* — pile size, what is in it, and whether the turn can still be finished are all ignored. On
  the evidence above, that judgement is the obvious next bot.
- A match ends at 5000 points (§14). Bots get there slowly, and a run of bad hands can push scores
  negative, so `MatchOver` can take a long time to appear.
- No tests. The verification so far is manual: matches run to completion, hands settle and score, and
  the engine never refuses a position it created.
