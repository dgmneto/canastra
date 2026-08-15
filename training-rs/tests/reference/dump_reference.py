"""Dump reference data from the Python implementation for Rust equivalence tests.

Run from `training/` (where the .venv is):

    .venv/bin/python ../training-rs/tests/reference/dump_reference.py

Writes to `training-rs/tests/reference/`:
  - genome.json         — a fixed canastra-weights@1 genome (training arch)
  - genome_flat.npy     — the flat f32 vector (Python's load_json ordering)
  - forward.npz         — ~1000 rows of (obs, acts, mask, logits, argmax)
  - replay.json         — one game's ply-by-ply pick sequence + final result
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from canastra_py import Pool
from canastra_train import genome as genome_mod
from canastra_train import policy

# The training arch (must match Rust TRAINING_ARCH).
ARCH = {"obs": 2002, "act": 101, "trunk": [512, 256], "head": [128], "activation": "tanh"}

OUT = Path(__file__).resolve().parent
SEED = 42


def dump_genome() -> np.ndarray:
    """Save a fixed genome as JSON + flat .npy."""
    vec = genome_mod.random_genome(ARCH, SEED).astype(np.float32)
    genome_mod.save_json(str(OUT / "genome.json"), ARCH, vec)
    np.save(OUT / "genome_flat.npy", vec)
    return vec


def dump_forward(vec: np.ndarray) -> None:
    """Encode plies from a small pool and score them with the genome.

    Collects ~1000 rows of (obs, acts, mask, logits, argmax) where logits come
    from Python's `policy.logits` (single-genome path). Each row is padded to
    the ply's global width, so logits[i, j] = -inf where mask[i, j] is False.
    """
    trunk, head = genome_mod.to_modules(vec, ARCH)
    # A handful of seeds gives many plies; we cap total rows at ~1000.
    seeds = list(range(1000, 1008))
    pool = Pool(seeds, max_hands=1)

    all_obs, all_acts, all_mask, all_logits = [], [], [], []
    while pool.has_live() and sum(len(o) for o in all_obs) < 1000:
        obs, acts, mask, rows = pool.encode()
        if len(rows) == 0:
            break
        # Single-genome forward: every row belongs to the same genome.
        obs_t = torch.from_numpy(obs)
        acts_t = torch.from_numpy(acts)
        mask_t = torch.from_numpy(mask)
        logits = policy.logits(trunk, head, obs_t, acts_t, mask_t).numpy()
        all_obs.append(obs)
        all_acts.append(acts)
        all_mask.append(mask)
        all_logits.append(logits)
        picks = logits.argmax(axis=1).tolist()
        pool.apply(picks)

    # Plies can have different widths; pad every ply to the global max width so
    # they concatenate into one [N, width] array. Padded columns get mask=False
    # and logit -inf (the masked_fill sentinel).
    max_w = max(a.shape[1] for a in all_acts)
    act_dim = all_acts[0].shape[2]
    obs_dim = all_obs[0].shape[1]

    def pad_acts(a: np.ndarray) -> np.ndarray:
        n, w, d = a.shape
        if w == max_w:
            return a
        pad = np.zeros((n, max_w - w, d), dtype=a.dtype)
        return np.concatenate([a, pad], axis=1)

    def pad_mask(m: np.ndarray) -> np.ndarray:
        n, w = m.shape
        if w == max_w:
            return m
        pad = np.zeros((n, max_w - w), dtype=m.dtype)
        return np.concatenate([m, pad], axis=1)

    def pad_logits(l: np.ndarray) -> np.ndarray:
        n, w = l.shape
        if w == max_w:
            return l
        pad = np.full((n, max_w - w), float("-inf"), dtype=l.dtype)
        return np.concatenate([l, pad], axis=1)

    obs = np.concatenate(all_obs, axis=0)
    acts = np.concatenate([pad_acts(a) for a in all_acts], axis=0)
    mask = np.concatenate([pad_mask(m) for m in all_mask], axis=0)
    logits = np.concatenate([pad_logits(l) for l in all_logits], axis=0)
    np.savez(
        OUT / "forward.npz",
        obs=obs,
        acts=acts,
        mask=mask,
        logits=logits,
        width=max_w,
    )
    # Also save individual .npy files (trivial to parse in Rust without a zip crate).
    np.save(OUT / "forward_obs.npy", obs)
    np.save(OUT / "forward_acts.npy", acts)
    np.save(OUT / "forward_mask.npy", mask)
    np.save(OUT / "forward_logits.npy", logits)
    (OUT / "forward_meta.json").write_text(json.dumps({"width": max_w, "n_rows": obs.shape[0]}))
    print(f"forward: {obs.shape[0]} rows, width {max_w}")


def dump_replay(vec: np.ndarray) -> None:
    """Drive ONE game to completion with a fixed genome as the only policy,
    recording the menu-index pick at every ply. Inference is deterministic, so
    Rust must reproduce this exact sequence.
    """
    trunk, head = genome_mod.to_modules(vec, ARCH)
    seed = 12345
    pool = Pool([seed], max_hands=1)

    picks_log: list[int] = []
    menu_sizes: list[int] = []
    while pool.has_live():
        obs, acts, mask, rows = pool.encode()
        if len(rows) == 0:
            break
        logits = policy.logits(trunk, head, torch.from_numpy(obs), torch.from_numpy(acts), torch.from_numpy(mask))
        pick = int(logits.argmax(axis=1).tolist()[0])
        picks_log.append(pick)
        menu_sizes.append(int(mask.sum()))
        pool.apply([pick])

    results = pool.results()
    assert len(results) == 1
    _seed, scores, winner, hands, unfinished = results[0]
    record = {
        "seed": seed,
        "picks": picks_log,
        "menu_sizes": menu_sizes,
        "final_scores": list(scores),
        "winner": winner,
        "hands": hands,
        "unfinished": bool(unfinished),
    }
    (OUT / "replay.json").write_text(json.dumps(record, indent=2))
    print(f"replay: {len(picks_log)} plies, scores {scores}, unfinished {unfinished}")


def main() -> None:
    vec = dump_genome()
    dump_forward(vec)
    dump_replay(vec)
    print(f"reference data written to {OUT}")


if __name__ == "__main__":
    main()