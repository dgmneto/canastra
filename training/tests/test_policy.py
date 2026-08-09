"""Masked scoring: padding never wins, determinism holds."""

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