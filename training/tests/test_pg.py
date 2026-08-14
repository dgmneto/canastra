"""The PG trainer: forward parity, grad flow, reward signs, smoke, resume."""

import json
from pathlib import Path

import numpy as np
import torch
from canastra_train import genome, model, pg, train_pg

ARCH = {"obs": 2002, "act": 101, "trunk": [16], "head": [], "activation": "tanh"}


def test_forward_parity_with_policy_logits() -> None:
    """CanastraNet.forward (inference) == policy.logits to 1e-6 — deployment-path equivalence."""
    vec = genome.random_genome(ARCH, seed=42)
    net = model.CanastraNet.from_genome_vec(vec, ARCH)
    obs = torch.randn(3, 2002)
    acts = torch.randn(3, 5, 101)
    mask = torch.tensor([[True, False, False, False, False],
                         [True, True, True, False, False],
                         [True, True, True, True, True]])
    max_diff = model.parity_check(net, obs, acts, mask)
    assert max_diff < 1e-6, f"forward diverges from policy.logits by {max_diff}"


def test_logp_grad_populates_all_parameters() -> None:
    """logp_and_pick returns log_prob whose backward() populates .grad on every parameter."""
    vec = genome.random_genome(ARCH, seed=7)
    net = model.CanastraNet.from_genome_vec(vec, ARCH)
    obs = torch.randn(2, 2002)
    acts = torch.randn(2, 4, 101)
    mask = torch.ones(2, 4, dtype=torch.bool)
    _picks, logp, _ent = net.logp_and_pick(obs, acts, mask)
    logp.sum().backward()  # type: ignore[no-untyped-call]
    for name, param in net.named_parameters():
        assert param.grad is not None, f"{name} has no grad"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite grad"


def test_game_rewards_signs_match_team_ownership() -> None:
    """Per-game reward = learner_team_score - opponent_team_score, sign-flipped for seating 2."""
    count = 2
    results: list[tuple[int, tuple[int, int], int | None, int, bool]] = [
        (1, (100, 50), 0, 10, False),   # game 0: learner=team0 → +50
        (2, (50, 100), 1, 10, False),   # game 1: learner=team0 → -50
        (1, (100, 50), 0, 10, False),   # game 2: learner=team1 → 50-100 = -50
        (2, (50, 100), 1, 10, False),   # game 3: learner=team1 → 100-50 = +50
    ]
    rewards = pg._game_rewards(results, count)
    assert rewards.tolist() == [50, -50, -50, 50]


def test_one_update_changes_parameters() -> None:
    """One Adam step from random init doesn't raise and changes the parameters."""
    cfg = pg.PGConfig(
        games_per_update=4, mini_batch=4, lr=1e-3,
        cap=6000, device="cpu", amp=False, compile=False,
    )
    trainer = pg.Trainer(ARCH, cfg, run_seed=99)
    before = trainer.net.to_genome_vec().copy()
    trainer.step()
    after = trainer.net.to_genome_vec()
    assert not np.allclose(before, after), "parameters did not change"


def test_train_smoke_produces_artifacts(tmp_path: Path) -> None:
    train_pg.run(
        arch=ARCH, run_dir=tmp_path, episodes=2,
        games_per_update=4, mini_batch=4, cap=6000,
        run_seed=9, device="cpu", amp=False, compile=False,
        log_interval=1, ckpt_interval=1,
    )
    assert (tmp_path / "config.json").exists()
    lines = (tmp_path / "updates.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert "mean_reward" in first and "update" in first and "baseline" in first
    champions = sorted(tmp_path.glob("champion-*.json"))
    assert champions, "champion weights exported"
    _arch, _vec = genome.load_json(str(champions[-1]))
    assert _arch == ARCH
    assert sorted(tmp_path.glob("model-*.pt")), "checkpoint written"


def test_resume_continues_from_checkpoint(tmp_path: Path) -> None:
    def run_once(episodes: int, resume: bool = False) -> None:
        train_pg.run(
            arch=ARCH, run_dir=tmp_path, episodes=episodes,
            games_per_update=4, mini_batch=4, cap=6000,
            run_seed=9, device="cpu", amp=False, compile=False,
            log_interval=1, ckpt_interval=1, resume=resume,
        )

    run_once(1)
    run_once(2, resume=True)
    lines = (tmp_path / "updates.jsonl").read_text().strip().splitlines()
    assert [json.loads(line)["update"] for line in lines] == [0, 1]


def test_resume_is_bit_identical_to_uninterrupted(tmp_path: Path) -> None:
    def run_into(directory: Path, episodes: int, resume: bool = False) -> None:
        train_pg.run(
            arch=ARCH, run_dir=directory, episodes=episodes,
            games_per_update=4, mini_batch=4, cap=6000,
            run_seed=9, device="cpu", amp=False, compile=False,
            log_interval=1, ckpt_interval=1, resume=resume,
        )

    continuous = tmp_path / "continuous"
    resumed = tmp_path / "resumed"
    run_into(continuous, episodes=2)
    run_into(resumed, episodes=1)
    run_into(resumed, episodes=2, resume=True)

    cont_vec = _load_final(continuous)
    res_vec = _load_final(resumed)
    assert np.array_equal(cont_vec, res_vec), "resumed weights diverge from continuous"


def test_json_roundtrip_survives_precision_cut(tmp_path: Path) -> None:
    vec = genome.random_genome(ARCH, seed=3)
    net = model.CanastraNet.from_genome_vec(vec, ARCH)
    path = tmp_path / "weights.json"
    net.save_json(str(path))
    loaded = model.CanastraNet.load_json(str(path))
    assert np.allclose(loaded.to_genome_vec(), vec, atol=1e-6)


def _load_final(directory: Path) -> np.ndarray:
    path = directory / "champion-final.json"
    _arch, vec = genome.load_json(str(path))
    return vec
