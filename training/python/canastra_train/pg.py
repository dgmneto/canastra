"""REINFORCE policy-gradient trainer.

One network is trained by policy gradient against a fixed (or periodically
frozen) opponent, with the match score differential as the reward — the same
tabula rasa reward the GA uses (spec §G). The GA trainer (`ga.py` / `train.py`)
is untouched; this is an opt-in alternative.

Algorithm: REINFORCE with an EMA baseline. Each gradient step:

1. Play `games_per_update` games via duplicate deal (each seed in both seatings,
   so the deal cancels and what remains is policy — same layout as
   `evaluate.evaluate_pair`). The learner samples actions stochastically; the
   opponent plays argmax from a frozen weights file.
2. Per-game reward = `learner_team_score - opponent_team_score`, sign-flipped for
   the seating where the learner is team 1. The reward is terminal — every ply
   in a game's trajectory gets that game's reward (reward-to-go = full reward).
3. Loss = `-(log_prob * (reward - baseline)).mean()` summed over all learner
   plies, plus an optional entropy bonus. Gradient accumulation splits the
   batch into mini-batches so the autograd graph fits in GPU memory.
4. EMA baseline = the running mean of batch rewards; subtracting it lowers
   variance without biasing the gradient (the baseline is independent of the
   action).

GPU adaptations (16GB 5060 Ti target):
- **bf16 autocast** on by default (`--no-amp` for fp32). Half the activation
  memory, ~2x forward/backward throughput on Blackwell. Master weights stay
  fp32 (standard AMP practice); resume reproduces parameters exactly, but
  per-episode rewards drift ~1e-3 (documented).
- **`torch.compile`** the learner net for kernel fusion across per-ply forwards.
- **Gradient accumulation**: `--games-per-update 512 --mini-batch 64` runs eight
  64-game rollouts, backward each, step once. The autograd graph is bounded to
  one mini-batch's worth.
- **Sharded rollouts** (`shards_pg.py`): multiple Pools across processes, each
  driving a games mini-batch, grads reduced in the parent. Parallelizes the
  CPU-bound per-ply glue across cores while GPU work pipelines behind them.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from canastra_py import Pool

from canastra_train import genome as genome_mod
from canastra_train import model as model_mod
from canastra_train import policy, seedstream

Arch = model_mod.Arch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PGConfig:
    """The PG trainer's static configuration (one run)."""

    games_per_update: int = 512
    mini_batch: int = 64
    lr: float = 1e-3
    baseline_decay: float = 0.95
    entropy_coef: float = 0.0
    grad_clip: float = 0.0  # 0 = no clipping
    cap: int = 200_000
    device: str = "cpu"
    amp: bool = True
    compile: bool = True
    sample_opponent: bool = False  # off by default — sampling the opp doesn't help the learner's grad

    @property
    def num_mini_batches(self) -> int:
        return max(1, math.ceil(self.games_per_update / self.mini_batch))


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


@dataclass
class RolloutResult:
    """One mini-batch's worth of play, ready for a backward pass.

    `logps` and `entropies` are per-ply tensors (with grad) on the device;
    `game_ids` is the per-ply numpy array mapping each logp row to its game
    index in the pool. `rewards` is the per-game terminal reward. The loss is
    assembled in `Trainer._loss_from_rollout`.
    """

    logps: list[torch.Tensor]
    entropies: list[torch.Tensor]
    game_ids: list[np.ndarray]
    rewards: np.ndarray  # [n_games] signed team-score differential for the learner
    plies: int
    unfinished: int
    wins: int
    losses: int
    mean_actions: float


ProgressFn = Any  # Callable[[int, int, int, int], None] — (mini_batch, total_mb, plies, games_done)


def _game_rewards(results: list[tuple[int, tuple[int, int], int | None, int, bool]], count: int) -> np.ndarray:
    """Per-game reward = learner_team_score - opponent_team_score.

    `results` is `pool.results()`: `(seed, scores, winner, hands, unfinished)`.
    `count` = len(seeds) — the first half of the pool has the learner in team 0,
    the second half in team 1 (the duplicate-deal layout from
    `evaluate.evaluate_pair`).
    """
    rewards = np.zeros(len(results), dtype=np.float32)
    for index, (_seed, scores, _winner, _hands, is_unfinished) in enumerate(results):
        learner_is_team_zero = index < count
        if learner_is_team_zero:
            rewards[index] = scores[0] - scores[1]
        else:
            rewards[index] = scores[1] - scores[0]
    return rewards


def rollout(
    learner: model_mod.CanastraNet,
    opp_vec: np.ndarray,
    arch: Arch,
    seeds: list[int],
    cfg: PGConfig,
    generator: torch.Generator | None = None,
    progress: Any = None,
) -> RolloutResult:
    """Play one mini-batch of games; return the trajectory data for a backward pass.

    The pool layout is the duplicate-deal layout from `evaluate.evaluate_pair`:
    `pool_seeds = seeds + seeds`, so the first `len(seeds)` games have the learner
    in seats 0/2 (team 0) and the second half has it in seats 1/3 (team 1). The
    deal cancels across the pair; the reward is the per-game differential.

    The learner samples stochastically (REINFORCE) with its forward in autograd
    mode; the opponent plays argmax under `inference_mode` from a frozen weights
    vector. Per ply, the learner rows' `log_prob` and `entropy` tensors are
    retained (with grad) for the loss; the picks are recombined with the
    opponent's and fed to `pool.apply` in the original row order.
    """
    count = len(seeds)
    pool_seeds = seeds + seeds  # duplicate-deal: learner=team0 first, team1 second
    pool = Pool(pool_seeds, max_actions_per_game=cfg.cap)
    device = cfg.device

    opp_trunk, opp_head = genome_mod.to_modules(opp_vec, arch)
    opp_trunk = opp_trunk.to(device)
    opp_head = opp_head.to(device)

    logps: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    game_ids: list[np.ndarray] = []
    plies = 0
    action_counts: list[int] = []

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if cfg.amp and device == "cuda"
        else torch.amp.autocast(device_type="cpu", enabled=False)  # type: ignore[attr-defined]
    )

    while pool.has_live():
        obs_np, acts_np, mask_np, rows = pool.encode()
        games = rows[:, 0]
        seats = rows[:, 1]
        learner_owns = (seats % 2 == 0) == (games < count)  # boolean per row

        picks = np.empty(len(rows), dtype=np.int64)
        if learner_owns.any():
            lsel = np.flatnonzero(learner_owns)
            obs_l = torch.from_numpy(obs_np[lsel]).to(device)
            acts_l = torch.from_numpy(acts_np[lsel]).to(device)
            mask_l = torch.from_numpy(mask_np[lsel]).to(device)
            with autocast_ctx:
                l_picks, l_logp, l_ent = learner.logp_and_pick(obs_l, acts_l, mask_l, generator)
            picks[lsel] = l_picks
            logps.append(l_logp)
            entropies.append(l_ent)
            game_ids.append(games[lsel].astype(np.int64))
        if (~learner_owns).any():
            osel = np.flatnonzero(~learner_owns)
            obs_o = torch.from_numpy(obs_np[osel]).to(device)
            acts_o = torch.from_numpy(acts_np[osel]).to(device)
            mask_o = torch.from_numpy(mask_np[osel]).to(device)
            with torch.inference_mode():
                o_scores = policy.logits(opp_trunk, opp_head, obs_o, acts_o, mask_o)
            picks[osel] = policy.pick_argmax(o_scores)

        pool.apply(picks.tolist())
        plies += 1
        action_counts.append(len(rows))
        if progress is not None and plies % 8 == 0:
            progress(plies, len(pool.results()))

    results = pool.results()
    rewards = _game_rewards(results, count)
    unfinished = sum(r[4] for r in results)
    wins = sum(1 for i, r in enumerate(results) if r[2] is not None and ((r[2] == 0) == (i < count)))
    losses = sum(1 for i, r in enumerate(results) if r[2] is not None and ((r[2] == 1) == (i < count)))

    return RolloutResult(
        logps=logps,
        entropies=entropies,
        game_ids=game_ids,
        rewards=rewards,
        plies=plies,
        unfinished=unfinished,
        wins=wins,
        losses=losses,
        mean_actions=float(np.mean(action_counts)) if action_counts else 0.0,
    )


# ---------------------------------------------------------------------------
# Trainer (single-process; sharded path in shards_pg.py)
# ---------------------------------------------------------------------------


def compute_loss(
    result: RolloutResult,
    baseline: float,
    entropy_coef: float,
    device: str,
) -> torch.Tensor:
    """`-(logp * (reward - baseline)).mean() + entropy_coef * H` over all plies.

    Standalone so both the single-process `Trainer` and the sharded workers
    (`shards_pg.py`) build the same loss from a `RolloutResult`. The baseline
    is a scalar passed in (the parent's EMA baseline), so the advantage is
    independent of the action — subtracting it lowers variance without biasing
    the gradient.
    """
    if not result.logps:
        return torch.zeros((), device=device, requires_grad=True)
    adv = torch.from_numpy(result.rewards).to(device) - baseline     # [n_games]
    policy_loss = torch.zeros((), device=device)
    entropy_loss = torch.zeros((), device=device)
    for logp, ent, gids in zip(result.logps, result.entropies, result.game_ids):
        per_row_adv = adv[gids]                                       # [n_ply_rows]
        policy_loss = policy_loss + (logp * per_row_adv).sum()
        if entropy_coef > 0:
            entropy_loss = entropy_loss + ent.sum()
    n = sum(t.numel() for t in result.logps)
    policy_loss = policy_loss / max(n, 1)
    if entropy_coef > 0:
        entropy_loss = entropy_coef * entropy_loss / max(n, 1)
        return -policy_loss + entropy_loss
    return -policy_loss


def unwrap(net: model_mod.CanastraNet) -> model_mod.CanastraNet:
    """Return the underlying `CanastraNet` whether `torch.compile`d or not."""
    if hasattr(net, "_orig_mod"):
        return cast("model_mod.CanastraNet", net._orig_mod)
    return net


class Trainer:
    """Holds the learner, optimizer, baseline, and opponent state.

    One `step` = one gradient update: `games_per_update` games split into
    `num_mini_batches` rollouts, each backward'd with gradient accumulation,
    then one optimizer step. The EMA baseline tracks the running mean reward.
    """

    def __init__(self, arch: Arch, cfg: PGConfig, run_seed: int, shards: int = 1) -> None:
        self.arch = arch
        self.cfg = cfg
        self.run_seed = run_seed
        self.shards = shards
        self.update_step = 0
        self.baseline = 0.0
        self.best_ever = float("-inf")

        # Deterministic init: seed the net's weights from run_seed so two Trainer
        # calls with the same seed start from the same weights. Without this,
        # nn.Linear's default init uses the global PyTorch RNG, which advances
        # between calls and breaks resume bit-identity.
        init_vec = genome_mod.random_genome(arch, seed=run_seed)
        net = model_mod.CanastraNet.from_genome_vec(init_vec, arch).to(cfg.device)
        if cfg.compile:
            self.net = model_mod.compile_net(net, cfg.device)
        else:
            self.net = net
        self.optimizer = torch.optim.Adam(unwrap(self.net).parameters(), lr=cfg.lr)

        self.opp_vec = genome_mod.random_genome(arch, seed=run_seed ^ _OPPONENT_SEED)
        self._frozen_opp_vec: np.ndarray | None = None  # set by `--opponent self --opponent-refresh`

    def load_opponent(self, path: str) -> None:
        """Load a fixed opponent from a `canastra-weights@1` JSON file."""
        _arch, vec = genome_mod.load_json(path)
        self.opp_vec = vec.astype(np.float32)

    def freeze_self_as_opponent(self) -> None:
        """Snapshot the current learner as the opponent (frozen-self-play)."""
        self._frozen_opp_vec = unwrap(self.net).to_genome_vec().copy()

    def _opp_vec(self) -> np.ndarray:
        return self._frozen_opp_vec if self._frozen_opp_vec is not None else self.opp_vec

    def step(self, progress: Any = None) -> dict[str, Any]:
        """One gradient update: rollouts → loss → backward → optimizer step.

        `progress(mini_batch, total_mb, plies, games_done)` is called during
        the rollout so a dashboard can show intra-update progress.
        """
        began = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        n_deals = self.cfg.games_per_update // 2
        all_seeds = seedstream.generation_seeds(self.run_seed, self.update_step, n_deals)

        if self.shards > 1:
            from canastra_train import shards_pg
            metrics = shards_pg.run_sharded_step(
                trainer=self,
                all_seeds=all_seeds,
                progress=progress,
            )
        else:
            metrics = self._local_rollouts(all_seeds, progress=progress)

        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(unwrap(self.net).parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        mean_reward = metrics["mean_reward"]
        self.baseline = (1.0 - self.cfg.baseline_decay) * self.baseline + self.cfg.baseline_decay * mean_reward
        self.update_step += 1
        improved = mean_reward > self.best_ever
        if improved:
            self.best_ever = mean_reward

        elapsed = time.perf_counter() - began
        return {
            "update": self.update_step - 1,
            "mean_reward": mean_reward,
            "baseline": self.baseline,
            "best_ever": self.best_ever,
            "plies": metrics["plies"],
            "unfinished": metrics["unfinished"],
            "wins": metrics["wins"],
            "losses": metrics["losses"],
            "games": metrics["games"],
            "mean_actions": metrics["mean_actions"],
            "wall_seconds": round(elapsed, 2),
            "improved": improved,
        }

    def _local_rollouts(self, all_seeds: list[int], progress: Any = None) -> dict[str, Any]:
        """Single-process: run all mini-batches locally, accumulate grads."""
        batch_rewards: list[float] = []
        total_plies = 0
        total_unfinished = 0
        total_wins = 0
        total_losses = 0
        total_games = 0
        total_actions = 0

        gen = self._make_generator()
        for mb_idx in range(self.cfg.num_mini_batches):
            mb_start = mb_idx * (self.cfg.mini_batch // 2)
            mb_end = min(mb_start + self.cfg.mini_batch // 2, len(all_seeds))
            mb_seeds = all_seeds[mb_start:mb_end]
            if not mb_seeds:
                break

            _games_so_far = total_games

            def _ply_progress(plies: int, games_done: int, _mb: int = mb_idx, _base: int = _games_so_far) -> None:
                if progress is not None:
                    progress(_mb, self.cfg.num_mini_batches, plies, _base + games_done)

            result = rollout(self.net, self._opp_vec(), self.arch, mb_seeds, self.cfg, generator=gen, progress=_ply_progress)
            loss = compute_loss(result, self.baseline, self.cfg.entropy_coef, self.cfg.device)
            (loss / self.cfg.num_mini_batches).backward()  # type: ignore[no-untyped-call]

            batch_rewards.extend(result.rewards.tolist())
            total_plies += result.plies
            total_unfinished += result.unfinished
            total_wins += result.wins
            total_losses += result.losses
            total_games += len(result.rewards)
            total_actions += int(result.mean_actions * len(result.rewards))

        return {
            "mean_reward": float(np.mean(batch_rewards)) if batch_rewards else 0.0,
            "plies": total_plies,
            "unfinished": total_unfinished,
            "wins": total_wins,
            "losses": total_losses,
            "games": total_games,
            "mean_actions": total_actions / max(total_games, 1),
        }

    def _load_flat_grad(self, flat_grad: np.ndarray) -> None:
        """Load a flat gradient vector into the optimizer's parameter `.grad` slots."""
        offset = 0
        for param in unwrap(self.net).parameters():
            n = param.numel()
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            param.grad.copy_(torch.from_numpy(flat_grad[offset : offset + n].reshape(param.shape)).to(param.device))
            offset += n
        assert offset == flat_grad.size

    def _make_generator(self) -> torch.Generator:
        g = torch.Generator(device=self.cfg.device)
        g.manual_seed(self._generator_seed())
        return g

    def _generator_seed(self) -> int:
        return int(seedstream.splitmix64(self.run_seed + self.update_step + 0xC1)) & 0x7FFFFFFF

    # -- checkpoints -----------------------------------------------------

    def save_checkpoint(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"model-{self.update_step:06d}.pt"
        torch.save(
            {
                "net_state": unwrap(self.net).state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "baseline": self.baseline,
                "update_step": self.update_step,
                "best_ever": self.best_ever,
                "opp_vec": torch.from_numpy(self._opp_vec()),
                "config": asdict(self.cfg),
                "arch": self.arch,
                "run_seed": self.run_seed,
            },
            path,
        )
        _prune_checkpoints(directory, keep=10)
        return path

    def load_checkpoint(self, directory: Path) -> None:
        path = max(directory.glob("model-*.pt"))
        state = torch.load(path, map_location=self.cfg.device, weights_only=True)
        unwrap(self.net).load_state_dict(state["net_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.baseline = float(state["baseline"])
        self.update_step = int(state["update_step"])
        self.best_ever = float(state["best_ever"])
        self.opp_vec = state["opp_vec"].numpy().astype(np.float32)

    def export_champion(self, path: str) -> None:
        """Export the current learner weights to a `canastra-weights@1` JSON."""
        unwrap(self.net).save_json(path)


_OPPONENT_SEED = 0xC0FFEE


def _prune_checkpoints(directory: Path, keep: int) -> None:
    checkpoints = sorted(directory.glob("model-*.pt"))
    for stale in checkpoints[:-keep]:
        stale.unlink()
