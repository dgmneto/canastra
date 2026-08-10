"""Batched policy scoring over pool rows.

Two flavors of the same forward pass:

- `logits` scores one genome's rows (`[N, M]` from a `(trunk, head)` pair).
- `logits_stacked` scores a whole roster at once: the per-genome weights are
  stacked on a batch dimension G and every layer runs as one batched `einsum`
  over the padded group blocks, so a ply costs ~2·layers torch ops regardless
  of how many genomes are in the roster. Same math as `logits`, just tiled —
  this is what makes large populations cheap on both CPU and GPU. (einsum,
  not `bmm`: torch's batched `bmm` switches kernels at small block sizes and
  produces last-ulp score differences, which would flip near-tie argmax picks
  between sharded and single-process evaluation. einsum is bit-stable across
  block shapes, which is required for the shards-are-identical contract.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from canastra_train import genome as genome_mod


def _forward(stack: torch.nn.ModuleList, x: torch.Tensor, final: bool) -> torch.Tensor:
    last = len(stack) - 1
    for index, layer in enumerate(stack):
        x = layer(x)
        if index < last or not final:
            x = torch.tanh(x)
    return x


@torch.inference_mode()
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


@dataclass
class WeightStack:
    """A roster's parameters stacked on a leading `G` (genome) dimension.

    `trunk_w[0]` is `[G, hidden, obs]`, `head_w[-1]` is `[G, 1, hidden]`, each
    bias is `[G, out]`. Built once per generation by `stack_weights` and reused
    every ply — the per-ply forward is pure batched `einsum`.
    """

    trunk_w: list[torch.Tensor]
    trunk_b: list[torch.Tensor]
    head_w: list[torch.Tensor]
    head_b: list[torch.Tensor]


def stack_weights(
    roster: list[np.ndarray],
    arch: genome_mod.Arch,
    device: str,
) -> WeightStack:
    """Stack the flat genomes in `roster` into per-layer `[G, out, in]` tensors.

    Layer order matches `genome.layer_shapes`; the reshape mirrors
    `genome.to_modules` (row-major per-layer weights), so `logits_stacked`
    reproduces per-genome `logits` exactly in math.
    """
    vecs = np.stack(roster)  # [G, size]
    trunk_w: list[torch.Tensor] = []
    trunk_b: list[torch.Tensor] = []
    head_w: list[torch.Tensor] = []
    head_b: list[torch.Tensor] = []
    offset = 0
    for name, out, inn in genome_mod.layer_shapes(arch):
        weight = torch.from_numpy(
            vecs[:, offset : offset + out * inn].reshape(vecs.shape[0], out, inn).copy()
        ).to(device)
        offset += out * inn
        bias = torch.from_numpy(vecs[:, offset : offset + out].copy()).to(device)
        offset += out
        if name.startswith("trunk"):
            trunk_w.append(weight)
            trunk_b.append(bias)
        else:
            head_w.append(weight)
            head_b.append(bias)
    assert offset == vecs.shape[1]
    return WeightStack(trunk_w, trunk_b, head_w, head_b)


@torch.inference_mode()
def logits_stacked(
    stack: WeightStack,
    obs: torch.Tensor,
    acts: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """[G, Nmax, width] action logits, -inf where the mask is false.

    obs: [G, Nmax, OBS] float32; acts: [G, Nmax, width, ACT]; mask: [G, Nmax,
    width] bool. Rows are padded per genome to `Nmax`; padded rows and columns
    are masked out by the caller, and the caller unpads the picks.
    """
    g, n, width = acts.shape[0], acts.shape[1], acts.shape[2]
    x = obs
    for w, b in zip(stack.trunk_w, stack.trunk_b):
        x = torch.tanh(torch.einsum("gni,goi->gno", x, w) + b.unsqueeze(1))  # [G, N, E]
    emb = x
    # The first head layer is linear over cat(emb, acts), so split its weight
    # matrix and fold the two pieces independently — this avoids materializing
    # [G, N, width, E+ACT] (the cat) entirely, the dominant cost of a wide ply.
    emb_w, act_w = torch.split(stack.head_w[0], [emb.shape[2], acts.shape[3]], dim=2)
    emb_in = torch.einsum("gni,goi->gno", emb, emb_w).unsqueeze(2)           # [G, N, 1, H]
    act_in = torch.einsum("gnwi,goi->gnwo", acts, act_w)                     # [G, N, W, H]
    x = emb_in + act_in + stack.head_b[0].unsqueeze(1).unsqueeze(1)
    x = torch.tanh(x).reshape(g, n * width, x.shape[3])
    for index, (w, b) in enumerate(list(zip(stack.head_w, stack.head_b))[1:]):
        x = torch.einsum("gmi,goi->gmo", x, w) + b.unsqueeze(1)
        if index < len(stack.head_w) - 2:
            x = torch.tanh(x)
    scores = x.reshape(g, n, width)                                        # [G, N, width]
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