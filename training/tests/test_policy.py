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


def test_stacked_bmm_preserves_fixed_batch_picks() -> None:
    roster = [genome.random_genome(ARCH, seed) for seed in (11, 23, 37)]
    stack = policy.stack_weights(roster, ARCH, "cpu")
    generator = torch.Generator().manual_seed(91)
    obs = torch.randn(3, 5, 2002, generator=generator)
    acts = torch.randn(3, 5, 7, 101, generator=generator)
    mask = torch.tensor(
        [
            [[True, True, False, False, False, False, False]] * 5,
            [[True, True, True, False, False, False, False]] * 5,
            [[True, True, True, True, True, False, False]] * 5,
        ]
    )

    default = policy.logits_stacked(stack, obs, acts, mask)
    einsum = policy.logits_stacked(stack, obs, acts, mask, kernel="einsum")
    bmm = policy.logits_stacked(stack, obs, acts, mask, kernel="bmm")
    valid = mask
    max_abs = float((einsum[valid] - bmm[valid]).abs().max())
    disagreements = int(
        (einsum.argmax(dim=2) != bmm.argmax(dim=2)).sum()
    )
    print(f"\nstacked bmm max_abs={max_abs:.3e}, argmax_disagreements={disagreements}")

    assert torch.equal(default, einsum)
    assert disagreements == 0
