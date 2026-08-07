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