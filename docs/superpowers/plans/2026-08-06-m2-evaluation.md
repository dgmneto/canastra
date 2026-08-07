# M2: Weights Format, JSONWeightsBot, Evaluation Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Work test-first (@superpowers:test-driven-development) where a test framework exists.

**Goal:** Make trained networks playable and measurable: pin the weights-JSON format, add the TS `JSONWeightsBot` (forward pass in TypeScript over the engine's `encodeState`), extend the PyO3 `Pool` with row metadata, and build both evaluation runners (Python duplicate-deal, TS head-to-head) with their sanity gates — milestone M2 of the bot-training spec.

**Architecture:** Weights travel as versioned JSON (`canastra-weights@1`); the TS side implements a generic tanh-MLP reader driven by the `arch` field (architecture changes never require TS changes — only the format version check). The bot gets the engine's encoding lazily through a new `BotContext.encode` hook wired by the callers (`runMatch`, the sandbox) to `Match.encodeState` — the encoding itself stays single-sourced in `canastra-encode`. The Python side gets torch, a flat-genome representation, and a duplicate-deal paired-seatings evaluator over the Pool.

**Tech Stack:** TypeScript (bots/harness, no new deps), Python 3.13 + torch (CPU) in `training/.venv`, the existing PyO3 Pool.

**Authoritative references:**
- Spec: `docs/superpowers/specs/2026-08-06-bot-training-design.md` Sections E (weights format) and H (evaluation).
- **All work happens in `/Users/dgmneto/canastra-bot-training` on branch `bot-training`.** Cargo from `engine/`, npm/npx from the root, Python from `training/`.

**Standing facts (verified):**
- `Pool.encode()` currently returns `(obs, acts, mask)`; row identity (which game/seat) is NOT exposed — Task 1 adds it.
- `runMatch(seed, botIds, maxActions)` and `headToHead(a, b, count)` resolve bots by **id** through `botById` (bots/src/index.ts); `BOTS` is a mutable exported array.
- `BotContext` is `{ rng, safeMode }` (bots/src/bot.ts); callers construct it: `harness/src/series.ts` (`runMatch`) and `web/src/ui/App.tsx` (`advance`).
- `Match` (harness/src/match.ts) wraps the wasm `Game`; `Game.encodeState(seat)` exists (M1-C) returning `{ obs: number[], actions: number[][], legal: Action[] }`.
- bots tsconfig: strict, `moduleResolution: bundler`, no `resolveJsonModule` yet.
- The training arch is trunk `[512, 256]`, head `[128]`, activation tanh, obs 2002, act 101 (~1.2M params). The committed fixture uses a SMALL arch (the reader is generic) to keep the repo light.

---

### Task 1: Pool row metadata

**Files:** `training/src/lib.rs`, `training/python/canastra_py/__init__.pyi`, `training/tests/test_pool.py`

The evaluator (Task 4) must route picks to the right genome per seat; for that it needs to know, per row, which game and which seat the row belongs to.

- [ ] **Step 1: Extend `encode` to a 4-tuple**

In `Pool::encode`, build a rows array alongside the existing buffers: `rows` is `[N, 2] i64` (numpy `PyArray2<i64>`), row `k` = `[game_index, seat_index]` matching `self.pending[k]` and that game's `state.turn`. Return `(obs, acts, mask, rows)`. Fill `rows` in the same sequential fill loop (it's cheap).

Update the doc comment: the fourth element is "per-row `(game index, seat)` so callers can route picks to per-seat policies".

- [ ] **Step 2: Update the `.pyi` stub**

Type the stub properly while touching it (final-M1-review note): `encode() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]` with a comment on shapes, `results() -> list[tuple[int, tuple[int, int], int | None, int, bool]]`, `menu_kinds() -> list[list[str]]`, `has_live() -> bool`, `apply(picks: list[int]) -> None`, `__init__(seeds: list[int], max_actions_per_game: int | None = None)`. Add `import numpy as np` to the stub.

- [ ] **Step 3: Update the tests**

In `test_pool.py`, unpack 4-tuples everywhere. Add to the full-match test: for every row in `rows`, assert `0 <= game_index < 4` and `0 <= seat_index <= 3`; and across one encode call, assert no `(game_index, seat_index)` pair repeats. In the cap test, just unpack the 4-tuple.

- [ ] **Step 4: Build + gates + commit**

```bash
cd training && .venv/bin/maturin develop --release && .venv/bin/pytest -q && .venv/bin/ruff check . && .venv/bin/mypy python/canastra_train tests
```

Commit: "training: expose per-row game and seat identity from the pool".

---

### Task 2: torch, genomes, and the policy forward pass

**Files:** `training/pyproject.toml` (torch dep), `training/python/canastra_train/genome.py`, `training/python/canastra_train/policy.py`, `training/tests/test_genome.py`, `training/tests/test_policy.py`

- [ ] **Step 1: Add torch**

Add `"torch"` to `[project] dependencies` in `training/pyproject.toml`, then `.venv/bin/pip install -e ".[dev]"` (CPU wheel). Record the resolved torch version.

- [ ] **Step 2: `genome.py` — flat vectors ↔ modules ↔ JSON**

The pinned format (spec Section E):

```json
{
  "format": "canastra-weights@1",
  "arch": { "obs": 2002, "act": 101, "trunk": [512, 256], "head": [128], "activation": "tanh" },
  "params": { "<name>.weight": { "shape": [out, in], "data": [/* flat row-major */] },
              "<name>.bias":   { "shape": [out],     "data": [...] } }
}
```

Layer naming contract: `trunk.{i}` for trunk layers (input `obs`, then chained), `head.{i}` for hidden head layers (input = last trunk width + `act`), `head.out` for the final `1 × last` layer. Implement:

```python
"""Flat parameter genomes ↔ torch modules, and the pinned weights-JSON format."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch

FORMAT = "canastra-weights@1"

Arch = dict[str, Any]  # {"obs": int, "act": int, "trunk": list[int], "head": list[int], "activation": "tanh"}


def layer_shapes(arch: Arch) -> list[tuple[str, int, int]]:
    """(name, out, in) for every layer, in genome order."""
    shapes: list[tuple[str, int, int]] = []
    prev = int(arch["obs"])
    for i, width in enumerate(arch["trunk"]):
        shapes.append((f"trunk.{i}", int(width), prev))
        prev = int(width)
    prev += int(arch["act"])
    for i, width in enumerate(arch["head"]):
        shapes.append((f"head.{i}", int(width), prev))
        prev = int(width)
    shapes.append(("head.out", 1, prev))
    return shapes


def genome_size(arch: Arch) -> int:
    return sum(out * inn + out for _, out, inn in layer_shapes(arch))


def random_genome(arch: Arch, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.1, genome_size(arch)).astype(np.float32)


def to_modules(vec: np.ndarray, arch: Arch) -> tuple[torch.nn.ModuleList, torch.nn.ModuleList]:
    """Split the flat genome into (trunk, head) Linear stacks. Deterministic."""
    trunk = torch.nn.ModuleList()
    head = torch.nn.ModuleList()
    offset = 0
    for name, out, inn in layer_shapes(arch):
        weight = torch.from_numpy(vec[offset : offset + out * inn].reshape(out, inn).copy())
        offset += out * inn
        bias = torch.from_numpy(vec[offset : offset + out].copy())
        offset += out
        layer = torch.nn.Linear(inn, out)
        with torch.no_grad():
            layer.weight.copy_(weight)
            layer.bias.copy_(bias)
        (trunk if name.startswith("trunk") else head).append(layer)
    assert offset == vec.size
    return trunk, head


def from_modules(trunk: torch.nn.ModuleList, head: torch.nn.ModuleList) -> np.ndarray:
    parts = []
    for layer in [*trunk, *head]:
        parts.append(layer.weight.detach().numpy().ravel())
        parts.append(layer.bias.detach().numpy().ravel())
    return np.concatenate(parts).astype(np.float32)


def save_json(path: str, arch: Arch, vec: np.ndarray) -> None:
    params: dict[str, dict[str, Any]] = {}
    offset = 0
    for name, out, inn in layer_shapes(arch):
        w = vec[offset : offset + out * inn]
        offset += out * inn
        b = vec[offset : offset + out]
        offset += out
        params[f"{name}.weight"] = {"shape": [out, inn], "data": np.round(w, 6).tolist()}
        params[f"{name}.bias"] = {"shape": [out], "data": np.round(b, 6).tolist()}
    with open(path, "w") as handle:
        json.dump({"format": FORMAT, "arch": arch, "params": params}, handle)


def load_json(path: str) -> tuple[Arch, np.ndarray]:
    with open(path) as handle:
        payload = json.load(handle)
    if payload.get("format") != FORMAT:
        raise ValueError(f"unsupported weights format: {payload.get('format')!r}")
    arch = payload["arch"]
    if arch.get("activation") != "tanh":
        raise ValueError("only tanh weights are supported")
    vec = np.zeros(genome_size(arch), dtype=np.float32)
    offset = 0
    for name, out, inn in layer_shapes(arch):
        for key, size in ((f"{name}.weight", out * inn), (f"{name}.bias", out)):
            entry = payload["params"][key]
            if entry["shape"] != ([out, inn] if "weight" in key else [out]):
                raise ValueError(f"{key}: shape {entry['shape']} does not match the arch")
            if len(entry["data"]) != size:
                raise ValueError(f"{key}: {len(entry['data'])} values, expected {size}")
            vec[offset : offset + size] = np.asarray(entry["data"], dtype=np.float32)
            offset += size
    return arch, vec
```

- [ ] **Step 3: `policy.py` — batched scoring**

```python
"""Batched policy scoring over pool rows."""

from __future__ import annotations

import numpy as np
import torch


def _forward(stack: torch.nn.ModuleList, x: torch.Tensor, final: bool) -> torch.Tensor:
    last = len(stack) - 1
    for index, layer in enumerate(stack):
        x = layer(x)
        if index < last or not final:
            x = torch.tanh(x)
    return x


def logits(
    trunk: torch.nn.ModuleList,
    head: torch.nn.ModuleList,
    obs: torch.Tensor,
    acts: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """[N, M] action logits, -inf on padded columns.

    obs: [N, OBS] float32; acts: [N, M, ACT]; mask: [N, M] bool.
    """
    emb = _forward(trunk, obs, final=False)                    # [N, E]
    width = acts.shape[1]
    emb = emb.unsqueeze(1).expand(-1, width, -1)               # [N, M, E]
    x = torch.cat([emb, acts], dim=2)                          # [N, M, E+ACT]
    scores = _forward(head, x, final=True).squeeze(-1)         # [N, M]
    return scores.masked_fill(~mask, float("-inf"))


def pick_argmax(scores: torch.Tensor) -> list[int]:
    return scores.argmax(dim=1).tolist()


def pick_sample(scores: torch.Tensor, rng: np.random.Generator) -> list[int]:
    """Sample per row from the masked softmax (exploration; used by the GA)."""
    probs = torch.softmax(scores, dim=1)
    picks: list[int] = []
    for row in probs:
        picks.append(int(rng.choice(len(row), p=row.numpy())))
    return picks
```

Note the `final` flag: trunk layers are all tanh; the head's last layer (`head.out`) is linear — matching the TS reader in Task 5.

- [ ] **Step 4: Tests**

`training/tests/test_genome.py`:

```python
"""Genome round-trips and the pinned JSON format."""

import numpy as np
from canastra_train import genome

ARCH = {"obs": 2002, "act": 101, "trunk": [64, 32], "head": [16], "activation": "tanh"}


def test_round_trip_through_modules_is_exact() -> None:
    vec = genome.random_genome(ARCH, seed=1)
    trunk, head = genome.to_modules(vec, ARCH)
    assert np.array_equal(genome.from_modules(trunk, head), vec)


def test_round_trip_through_json_survives_the_precision_cut(tmp_path) -> None:
    vec = genome.random_genome(ARCH, seed=2)
    path = tmp_path / "weights.json"
    genome.save_json(str(path), ARCH, vec)
    arch, loaded = genome.load_json(str(path))
    assert arch == ARCH
    assert np.allclose(loaded, vec, atol=1e-6)


def test_load_rejects_foreign_formats(tmp_path) -> None:
    vec = genome.random_genome(ARCH, seed=3)
    path = tmp_path / "weights.json"
    genome.save_json(str(path), ARCH, vec)
    raw = path.read_text().replace(genome.FORMAT, "something-else@9")
    path.write_text(raw)
    try:
        genome.load_json(str(path))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
```

`training/tests/test_policy.py`:

```python
"""Masked scoring: padding never wins, determinism holds."""

import numpy as np
import torch
from canastra_train import genome, policy

ARCH = {"obs": 2002, "act": 101, "trunk": [32], "head": [16], "activation": "tanh"}


def test_padding_is_never_picked() -> None:
    vec = genome.random_genome(ARCH, seed=4)
    trunk, head = genome.to_modules(vec, ARCH)
    obs = torch.zeros(3, 2002)
    acts = torch.randn(3, 5, 101)
    mask = torch.tensor([[True, False, False, False, False],
                         [True, True, True, False, False],
                         [True, True, True, True, True]])
    scores = policy.logits(trunk, head, obs, acts, mask)
    picks = policy.pick_argmax(scores)
    assert picks[0] == 0
    assert picks[1] in (0, 1, 2)
    assert 0 <= picks[2] < 5
    assert torch.isneginf(scores[0, 1:]).all()


def test_scoring_is_deterministic() -> None:
    vec = genome.random_genome(ARCH, seed=5)
    trunk, head = genome.to_modules(vec, ARCH)
    obs = torch.randn(2, 2002)
    acts = torch.randn(2, 4, 101)
    mask = torch.ones(2, 4, dtype=torch.bool)
    first = policy.logits(trunk, head, obs, acts, mask)
    second = policy.logits(trunk, head, obs, acts, mask)
    assert torch.equal(first, second)
```

- [ ] **Step 5: Gates + commit**

```bash
cd training && .venv/bin/pip install -e ".[dev]" && .venv/bin/maturin develop --release && .venv/bin/pytest -q && .venv/bin/ruff check . && .venv/bin/mypy python/canastra_train tests
```

Commit: "training: genomes, JSON weights format, and batched policy scoring".

---

### Task 3: The committed random-init fixture

**Files:** `training/scripts/make_fixture.py`, `bots/src/fixtures/random-init.json` (generated), `bots/tsconfig.json`

Note: the spec's layout sketch placed the fixture at `training/fixtures/`; it lives in `bots/src/fixtures/` instead because the TS registry must import it directly (Vite and tsx both resolve JSON imports from `bots/`). Deliberate deviation, same artifact.

Layout note: the spec's Section A sketch put the fixture under `training/fixtures/`; it lives in `bots/src/fixtures/` instead, deliberately — the TS registry must `import` it (Vite and tsx both bundle JSON imports), and `bots/` is the package the harness and sandbox consume. The generator script still lives in `training/`.

- [ ] **Step 1: The generator**

`training/scripts/make_fixture.py`:

```python
"""Generate the seeded-random weights fixture behind the `nn-random` bot.

Deliberately tiny arch — the fixture exists so the harness and sandbox can
play *a* network without a training run, not to be good at the game.
"""

from pathlib import Path

from canastra_train import genome

ARCH = {"obs": 2002, "act": 101, "trunk": [32], "head": [16], "activation": "tanh"}
SEED = 20260806
TARGET = Path(__file__).resolve().parents[2] / "bots" / "src" / "fixtures" / "random-init.json"


def main() -> None:
    vec = genome.random_genome(ARCH, seed=SEED)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    genome.save_json(str(TARGET), ARCH, vec)
    print(f"wrote {TARGET} ({genome.genome_size(ARCH)} params)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and enable JSON imports in bots**

```bash
cd training && .venv/bin/python scripts/make_fixture.py
```

In `bots/tsconfig.json`, add `"resolveJsonModule": true` to `compilerOptions` (needed for the fixture import in Task 5; Vite and tsx both handle JSON already).

- [ ] **Step 3: Commit**

```bash
git add training/scripts/make_fixture.py bots/src/fixtures/random-init.json bots/tsconfig.json
git commit -m "Add the seeded random-init weights fixture"
```

---

### Task 4: Duplicate-deal evaluation in Python

**Files:** `training/python/canastra_train/evaluate.py`, `training/python/canastra_train/sanity.py`, `training/tests/test_evaluate.py`, `training/python/canastra_train/__init__.py`

- [ ] **Step 1: `evaluate.py`**

```python
"""Duplicate-deal evaluation: same seeds, swapped seatings, score differentials.

Canastra's variance is dominated by the deal, so two genomes are compared by
playing each seed TWICE — once with genome A in seats 0/2, once with the
seats swapped — and averaging A's score differential over the pair. The deal
cancels; what remains is policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from canastra_py import Pool

from canastra_train import genome as genome_mod
from canastra_train import policy


@dataclass
class PairReport:
    pairs: int
    mean_diff: float
    ci95: float
    wins_a: int
    wins_b: int
    unfinished: int


def evaluate_pair(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    arch: genome_mod.Arch,
    seeds: list[int],
    cap: int = 200_000,
) -> PairReport:
    """A vs B over `seeds`, each seed played in both seatings."""
    count = len(seeds)
    pool_seeds = seeds + seeds  # first half: A in seats 0/2; second half: swapped
    pool = Pool(pool_seeds, max_actions_per_game=cap)
    trunk_a, head_a = genome_mod.to_modules(vec_a, arch)
    trunk_b, head_b = genome_mod.to_modules(vec_b, arch)

    while pool.has_live():
        obs, acts, mask, rows = pool.encode()
        obs_t = torch.from_numpy(obs)
        acts_t = torch.from_numpy(acts)
        mask_t = torch.from_numpy(mask)
        scores_a = policy.logits(trunk_a, head_a, obs_t, acts_t, mask_t)
        scores_b = policy.logits(trunk_b, head_b, obs_t, acts_t, mask_t)
        # Route each row to its genome: in the first half of the pool, genome A
        # owns the even seats (team 0); in the second half, the odd ones.
        use_a = torch.zeros(mask_t.shape[0], dtype=torch.bool)
        for row, (game, seat) in enumerate(rows):
            a_is_team_zero = game < count
            owns = (seat % 2 == 0) == a_is_team_zero
            use_a[row] = owns
        scores = torch.where(use_a.unsqueeze(1), scores_a, scores_b)
        pool.apply(policy.pick_argmax(scores))

    results = pool.results()
    assert len(results) == 2 * count

    diffs: list[float] = []
    unfinished = 0
    wins_a = wins_b = 0
    by_seed: dict[int, list[float]] = {}
    for index, (_seed, scores, winner, _hands, is_unfinished) in enumerate(results):
        a_is_team_zero = index < count
        a_score = scores[0] if a_is_team_zero else scores[1]
        b_score = scores[1] if a_is_team_zero else scores[0]
        by_seed.setdefault(_seed, []).append(a_score - b_score)
        if is_unfinished:
            unfinished += 1
        elif winner is not None:
            won_a = (winner == 0) == a_is_team_zero
            if won_a:
                wins_a += 1
            else:
                wins_b += 1

    for seed in seeds:
        pair = by_seed[seed]
        assert len(pair) == 2, f"seed {seed} did not produce both seatings"
        diffs.append((pair[0] + pair[1]) / 2)

    arr = np.asarray(diffs)
    mean = float(arr.mean())
    ci95 = float(1.96 * arr.std(ddof=1) / np.sqrt(count)) if count > 1 else float("inf")
    return PairReport(count, mean, ci95, wins_a, wins_b, unfinished)
```

(The per-row Python loop building `use_a` is fine at evaluation scale; if it ever shows up in profiles, vectorize from the `rows` array.)

- [ ] **Step 2: `sanity.py` — the spec's first sanity gate**

```python
"""Sanity gate: two random genomes must be indistinguishable.

Same architecture, different seeds — neither has learned anything, so the
duplicate-deal differential must sit inside its confidence interval around
zero. If it does not, the evaluator (not the genomes) is broken.

Spec default is 1000 paired seeds; run that on the training machine. The
`--pairs` flag keeps local runs affordable.
"""

from __future__ import annotations

import argparse

from canastra_train import evaluate, genome

ARCH = {"obs": 2002, "act": 101, "trunk": [512, 256], "head": [128], "activation": "tanh"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=1000)
    parser.add_argument("--first-seed", type=int, default=1)
    args = parser.parse_args()

    vec_a = genome.random_genome(ARCH, seed=101)
    vec_b = genome.random_genome(ARCH, seed=202)
    seeds = list(range(args.first_seed, args.first_seed + args.pairs))
    report = evaluate.evaluate_pair(vec_a, vec_b, ARCH, seeds)
    print(
        f"{report.pairs} pairs: mean diff {report.mean_diff:+.1f} "
        f"(95% CI ±{report.ci95:.1f}), wins {report.wins_a}/{report.wins_b}, "
        f"unfinished {report.unfinished}"
    )
    if abs(report.mean_diff) > report.ci95:
        raise SystemExit("FAIL: random genomes separated — the evaluator is biased")
    print("OK: random genomes are indistinguishable")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Test (small-scale)**

`training/tests/test_evaluate.py`:

```python
"""The evaluator, at smoke scale: random genomes, a handful of pairs."""

from canastra_train import evaluate, genome

ARCH = {"obs": 2002, "act": 101, "trunk": [32], "head": [16], "activation": "tanh"}


def test_random_genomes_are_indistinguishable_at_smoke_scale() -> None:
    vec_a = genome.random_genome(ARCH, seed=11)
    vec_b = genome.random_genome(ARCH, seed=22)
    report = evaluate.evaluate_pair(vec_a, vec_b, ARCH, seeds=[1, 2, 3], cap=200_000)
    assert report.pairs == 3
    assert report.unfinished == 0
    # Three pairs cannot prove equality; they can prove the machinery runs
    # and produces finite, paired differentials.
    assert abs(report.mean_diff) < 5000
```

- [ ] **Step 4: Export + gates + commit**

Add to `training/python/canastra_train/__init__.py`: `from canastra_train import evaluate, genome, policy` and extend `__all__`.

```bash
cd training && .venv/bin/pytest -q && .venv/bin/ruff check . && .venv/bin/mypy python/canastra_train tests
```

Commit: "training: duplicate-deal evaluation and the random-genome sanity gate".

---

### Task 5: The TS forward pass and `JSONWeightsBot`

**Files:** `bots/src/forward.ts`, `bots/src/json-weights.ts`, `bots/src/types.ts`, `bots/src/bot.ts`, `bots/src/index.ts`

- [ ] **Step 1: Wire types**

In `bots/src/types.ts`, add:

```ts
/** The engine's encoding of one seat's position (wasm `encodeState`). */
export interface EncodedState {
  obs: number[];
  actions: number[][];
  legal: Action[];
}
```

In `bots/src/bot.ts`, extend `BotContext`:

```ts
  /**
   * The engine's encoding of this position, for neural bots. Lazy — called
   * only if a bot actually needs it, so toy policies never pay for it.
   * Callers wire it to `Match.encodeState` for the acting seat.
   */
  encode?: () => EncodedState;
```

(importing `EncodedState` from `./types`).

- [ ] **Step 2: `bots/src/forward.ts` — the generic reader**

```ts
/**
 * The weights-JSON forward pass.
 *
 * Generic over the arch: layer widths come from the file, so architecture
 * changes never touch this code — only the format string does. Mirrors
 * `training/python/canastra_train/policy.py` exactly: trunk layers all tanh,
 * head hidden layers tanh, the final `head.out` layer linear.
 *
 * Format contract (pinned, spec Section E):
 *   { "format": "canastra-weights@1",
 *     "arch": { "obs", "act", "trunk": [...], "head": [...], "activation": "tanh" },
 *     "params": { "<name>.weight": { "shape": [out, in], "data": [...] }, ... } }
 * Layer names: trunk.{i}, head.{i}, head.out.
 */

export interface WeightsArch {
  obs: number;
  act: number;
  trunk: number[];
  head: number[];
  activation: "tanh";
}

export interface WeightsJson {
  format: string;
  arch: WeightsArch;
  params: Record<string, { shape: number[]; data: number[] }>;
}

export const WEIGHTS_FORMAT = "canastra-weights@1";

interface Layer {
  weight: number[]; // row-major, shape [out][in]
  bias: number[];
  out: number;
  inn: number;
}

export function validateWeights(weights: WeightsJson): void {
  if (weights.format !== WEIGHTS_FORMAT) {
    throw new Error(`unsupported weights format: ${weights.format}`);
  }
  if (weights.arch.activation !== "tanh") {
    throw new Error("only tanh weights are supported");
  }
  const names = layerNames(weights.arch);
  for (const name of names) {
    for (const part of ["weight", "bias"]) {
      const key = `${name}.${part}`;
      if (!(key in weights.params)) throw new Error(`missing params: ${key}`);
    }
  }
}

function layerNames(arch: WeightsArch): string[] {
  const names: string[] = [];
  for (let i = 0; i < arch.trunk.length; i += 1) names.push(`trunk.${i}`);
  for (let i = 0; i < arch.head.length; i += 1) names.push(`head.${i}`);
  names.push("head.out");
  return names;
}

function layer(weights: WeightsJson, name: string, expectedIn: number): Layer {
  const weight = weights.params[`${name}.weight`];
  const bias = weights.params[`${name}.bias`];
  const [out, inn] = weight.shape;
  if (inn !== expectedIn) {
    throw new Error(`${name}: weight expects input ${inn}, got ${expectedIn}`);
  }
  if (weight.data.length !== out * inn) throw new Error(`${name}: weight data length`);
  if (bias.data.length !== out) throw new Error(`${name}: bias data length`);
  return { weight: weight.data, bias: bias.data, out, inn };
}

function apply(layer: Layer, input: number[]): number[] {
  const out = new Array<number>(layer.out);
  for (let o = 0; o < layer.out; o += 1) {
    let acc = layer.bias[o];
    const base = o * layer.inn;
    for (let i = 0; i < layer.inn; i += 1) acc += layer.weight[base + i] * input[i];
    out[o] = acc;
  }
  return out;
}

const tanh = (xs: number[]) => xs.map(Math.tanh);

/** The observation embedding (trunk, all-tanh). */
export function embed(weights: WeightsJson, obs: number[]): number[] {
  if (obs.length !== weights.arch.obs) {
    throw new Error(`observation is ${obs.length} wide, weights expect ${weights.arch.obs}`);
  }
  let x = obs;
  let inn = weights.arch.obs;
  for (let i = 0; i < weights.arch.trunk.length; i += 1) {
    const l = layer(weights, `trunk.${i}`, inn);
    x = tanh(apply(l, x));
    inn = l.out;
  }
  return x;
}

/** One action's score: head over [embedding; features], final layer linear. */
export function scoreAction(weights: WeightsJson, emb: number[], feats: number[]): number {
  if (feats.length !== weights.arch.act) {
    throw new Error(`action row is ${feats.length} wide, weights expect ${weights.arch.act}`);
  }
  let x = [...emb, ...feats];
  let inn = emb.length + weights.arch.act;
  for (let i = 0; i < weights.arch.head.length; i += 1) {
    const l = layer(weights, `head.${i}`, inn);
    x = tanh(apply(l, x));
    inn = l.out;
  }
  const out = layer(weights, "head.out", inn);
  return apply(out, x)[0];
}
```

- [ ] **Step 3: `bots/src/json-weights.ts`**

```ts
/**
 * A bot that plays trained weights.
 *
 * The network sees exactly what training saw — the engine's own
 * `encodeState` — and ranks the legal list by score. Deterministic: no rng,
 * so a weights file always plays the same game from the same seed.
 */

import type { Action, PlayerView } from "./types";
import type { Bot, BotContext } from "./bot";
import { embed, scoreAction, validateWeights, type WeightsJson } from "./forward";

export function makeJsonWeightsBot(weights: WeightsJson, id: string): Bot {
  validateWeights(weights);
  return {
    id,
    name: `NN ${id}`,
    blurb: "Policy network loaded from JSON weights.",

    candidates(view: PlayerView, legal: Action[], context: BotContext): Action[] {
      const encoded = context.encode?.();
      if (!encoded) {
        throw new Error(`${id}: neural bots need context.encode — the caller must wire encodeState`);
      }
      if (encoded.actions.length !== legal.length) {
        throw new Error(`${id}: encoded rows (${encoded.actions.length}) != legal moves (${legal.length})`);
      }
      if (encoded.obs.length !== weights.arch.obs) {
        throw new Error(`${id}: observation width ${encoded.obs.length} != ${weights.arch.obs}`);
      }
      const emb = embed(weights, encoded.obs);
      const scored = legal.map((action, index) => ({
        action,
        score: scoreAction(weights, emb, encoded.actions[index]),
      }));
      scored.sort((a, b) => b.score - a.score);
      return scored.map((entry) => entry.action);
    },
  };
}
```

- [ ] **Step 4: Registry — `registerBot` + `nn-random`**

In `bots/src/index.ts`:

```ts
import type { Bot } from "./bot";
import { randomBot } from "./random";
import { randomPlusBot } from "./random-plus";
import { randomDiscardHungryBot } from "./random-discard-hungry";
import { makeJsonWeightsBot } from "./json-weights";
import type { WeightsJson } from "./forward";
import randomInit from "./fixtures/random-init.json";

export const BOTS: Bot[] = [randomBot, randomPlusBot, randomDiscardHungryBot];

/**
 * Register a bot that cannot be a static constant — e.g. one built from a
 * weights file. Idempotent by id. Registered bots join the harness CLI, the
 * sandbox pickers, and anything else that reads `BOTS`.
 */
export function registerBot(bot: Bot): void {
  if (!BOTS.some((existing) => existing.id === bot.id)) BOTS.push(bot);
}

/** The committed random-init fixture, so a network can play without training. */
export const nnRandomBot = makeJsonWeightsBot(randomInit as WeightsJson, "nn-random");
registerBot(nnRandomBot);

export const DEFAULT_BOT = randomBot;

export function botById(id: string): Bot {
  return BOTS.find((bot) => bot.id === id) ?? DEFAULT_BOT;
}

export * from "./bot";
export * from "./types";
export * from "./forward";
export { makeJsonWeightsBot } from "./json-weights";
export { makeRng, type Rng } from "./rng";
```

- [ ] **Step 5: Typecheck + commit**

`npm run typecheck` from the worktree root — all three packages clean.

Commit: "Bots: JSON weights forward pass and the JSONWeightsBot".

---

### Task 6: Wiring `encode` through the harness and sandbox

**Files:** `harness/src/match.ts`, `harness/src/series.ts`, `web/src/ui/App.tsx`

- [ ] **Step 1: `Match.encodeState` passthrough**

In `harness/src/match.ts`, after `legalActions`:

```ts
  /** F7/M1: the full policy encoding of a seat — observation, per-action rows, legal list. */
  encodeState(seat: Seat): EncodedState {
    return this.game.encodeState(seat) as EncodedState;
  }
```

(importing `EncodedState` from `@canastra/bots`).

- [ ] **Step 2: Wire the hook in `runMatch`**

In `harness/src/series.ts`, the `step` call becomes:

```ts
    const acting = view.turn;
    const result = step(match, match.views()[acting], botById(botIds[acting]), {
      rng,
      safeMode,
      encode: () => match.encodeState(acting),
    });
```

- [ ] **Step 3: Wire the hook in the sandbox**

In `web/src/ui/App.tsx`'s `advance`, the `step` call becomes:

```ts
    const result = step(current, current.views()[acting], bot, {
      rng: rng.current,
      safeMode: safeMode.current,
      encode: () => current.encodeState(acting),
    });
```

- [ ] **Step 4: Typecheck + commit**

`npm run typecheck` — clean. Commit: "Harness and sandbox supply the neural encoding hook".

---

### Task 7: `eval-nn.ts` — external validation against the heuristic bots

**Files:** `harness/src/eval-nn.ts`

- [ ] **Step 1: The script**

```ts
/**
 * Play trained weights against a registered bot, both seatings, and print
 * the head-to-head report.
 *
 * Usage: npx tsx harness/src/eval-nn.ts <weights.json> <opponent-id> [count]
 *
 * This is the spec's external validation path: the Python evaluator measures
 * genomes against genomes; this one measures a weights file against the
 * heuristic bots on the TS side, through the same harness everyone else uses.
 */

import { readFileSync } from "node:fs";
import { makeJsonWeightsBot, registerBot, type WeightsJson } from "@canastra/bots";
import { loadEngine } from "./load-node";
import { headToHead } from "./series";

const [weightsPath, opponent, count = "40"] = process.argv.slice(2);
if (!weightsPath || !opponent) {
  console.error("usage: npx tsx harness/src/eval-nn.ts <weights.json> <opponent-id> [count]");
  process.exit(2);
}

await loadEngine();
const weights = JSON.parse(readFileSync(weightsPath, "utf8")) as WeightsJson;
registerBot(makeJsonWeightsBot(weights, "nn"));
const report = headToHead("nn", opponent, Number(count));
console.log(JSON.stringify(report, null, 2));
```

- [ ] **Step 2: Smoke it**

```bash
npx tsx harness/src/eval-nn.ts bots/src/fixtures/random-init.json random 4
```

Expected: a head-to-head report JSON, matches finished. (Random weights vs Random: no expected ordering — this smoke proves plumbing, not strength.)

- [ ] **Step 3: Commit**

Commit: "Harness: eval-nn plays a weights file against any registered bot".

---

### Task 8: Sanity gates, docs, final sweep

**Files:** `training/README.md`, `CLAUDE.md`

- [ ] **Step 1: Sanity gate 1 — random genomes indistinguishable (Python)**

Run the real gate at a locally-affordable scale and record it:

```bash
cd training && .venv/bin/python -m canastra_train.sanity --pairs 200
```

Expected: `OK: random genomes are indistinguishable` (mean diff inside its 95% CI). Record the verbatim output. Note in the report that the spec's 1000-pair default is the training-machine version of the same command. If it FAILs, stop — an evaluator bias invalidates everything downstream (report BLOCKED with the numbers).

- [ ] **Step 2: Sanity gate 2 — `nn-random` plays legal matches (TS)**

```bash
npx canastra-harness --seed 7 nn-random nn-random nn-random nn-random | head -1
```

Expected: `"type":"result"`, `"unfinished":false`. Restarts may be nonzero — random weights strand turns exactly like the toy bots do; the safe-mode backstop is the documented recovery. Also run the external validation smoke at a meaningful count and record it:

```bash
npx tsx harness/src/eval-nn.ts bots/src/fixtures/random-init.json random-plus 20
```

No ordering expected (untrained weights); record the report.

- [ ] **Step 3: Docs**

`training/README.md`: add an "Evaluation" section — `evaluate.py` (duplicate-deal paired seatings, what the differential means), `sanity.py` (the random-genome gate, `--pairs`), `eval-nn.ts` on the TS side; note the pool now returns per-row `(game, seat)` for per-seat policy routing. Also add the sentence the final M1 review asked for: the pool deliberately re-captures `turn_start` on refusal phases (red-3 replacements replay from after the draw, keeping the red 3 on the table) — a documented divergence from the harness checkpoint rule.

`CLAUDE.md`: repo status component 2 — M2 landed: weights-JSON format pinned, `JSONWeightsBot` plays trained weights in the harness/sandbox, duplicate-deal evaluation on both sides. Commands — add the eval-nn usage line. Architecture — one sentence on the weights format + the lazy `BotContext.encode` hook in the bots bullet.

- [ ] **Step 4: Final gate sweep**

```bash
cargo test --workspace                                        # from engine/
cargo clippy --workspace --all-targets -- -D warnings         # from engine/
cargo fmt --check                                             # from engine/
cargo build -p canastra-wasm --target wasm32-unknown-unknown  # from engine/
npm run typecheck                                             # from worktree root
npx canastra-harness --seed 7 random random-plus random random-plus | head -1
cd training && .venv/bin/maturin develop --release && .venv/bin/pytest -q && .venv/bin/ruff check . && .venv/bin/mypy python/canastra_train tests
```

All green.

- [ ] **Step 5: Commit**

Commit: "M2 sanity gates, evaluation docs, and repo guide updates".

---

## Done criteria for M2

- Weights format pinned and round-tripping (Python ↔ JSON; TS reader validates and plays it).
- `JSONWeightsBot` plays full legal matches through the harness and sandbox wiring (`nn-random` registered from the committed fixture).
- Python duplicate-deal evaluator runs paired seatings over the Pool; the random-genome sanity gate passes at 200 pairs locally (1000-pair command documented for the training machine).
- `eval-nn.ts` measures a weights file against any registered bot.
- All M1 gates still green plus the new Python tests.

Then: write the M3 plan (GA trainer) from the spec's Section G.
