"""The policy network as an `nn.Module` — the gradient-trainer's view of the MLP.

The GA path (`policy.py`) treats the network as flat numpy vectors and runs every
forward under `@torch.inference_mode()`; gradients are explicitly off there. This
module is the autograd-friendly twin: same math, same `canastra-weights@1` layout,
but the parameters live in `nn.Linear` layers with `requires_grad=True` and the
forward runs in normal (autograd-tracking) mode by default.

The deployment path is unchanged. `to_genome_vec` / `from_genome_vec` round-trip
through the exact `genome.layer_shapes` layout, so a trained `CanastraNet` exports
to a JSON file that `bots/` and `harness/src/eval-nn.ts` play verbatim — no wiring
changes on the TS side. The parity test pins that `CanastraNet.forward` under
`inference_mode` agrees with `policy.logits` to 1e-6, guaranteeing the trained file
behaves identically through the existing deployment path.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from canastra_train import genome as genome_mod
from canastra_train import policy

Arch = genome_mod.Arch


class CanastraNet(nn.Module):
    """The trunk+head MLP, training-shaped.

    Trunk: `Linear(obs→trunk[0]) → tanh → … → Linear(trunk[-2]→trunk[-1]) → tanh`,
    producing the embedding `E = trunk[-1]`. Head: `Linear(E+ACT→head[0]) → tanh →
    … → Linear(head[-1]→1)` (final layer linear). Identical to `policy._forward` /
    `policy.logits` in math; differs only in that the parameters are `nn.Parameter`s
    and the forward tracks autograd by default.
    """

    def __init__(self, arch: Arch) -> None:
        super().__init__()
        self.arch = arch
        self.trunk = nn.ModuleList()
        prev = int(arch["obs"])
        for width in arch["trunk"]:
            self.trunk.append(nn.Linear(prev, int(width)))
            prev = int(width)
        act_dim = int(arch["act"])
        prev_head_in = prev + act_dim
        self.head = nn.ModuleList()
        for width in arch["head"]:
            self.head.append(nn.Linear(prev_head_in, int(width)))
            prev_head_in = int(width)
        self.head.append(nn.Linear(prev_head_in, 1))

    # -- forward --------------------------------------------------------

    def forward(self, obs: torch.Tensor, acts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """`[N, M]` action logits, `-inf` on padded columns.

        Same math as `policy.logits`. `obs: [N, OBS]`, `acts: [N, M, ACT]`,
        `mask: [N, M]` bool. Runs in autograd mode — gradients flow to the
        parameters; the inputs never carry `requires_grad`.
        """
        emb = _forward_tanh(self.trunk, obs)                       # [N, E]
        width = acts.shape[1]
        emb = emb.unsqueeze(1).expand(-1, width, -1)               # [N, M, E]
        x = torch.cat([emb, acts], dim=2)                          # [N, M, E+ACT]
        scores = _forward_final_linear(self.head, x).squeeze(-1)   # [N, M]
        return scores.masked_fill(~mask, float("-inf"))

    # -- sampling -------------------------------------------------------

    def logp_and_pick(
        self,
        obs: torch.Tensor,
        acts: torch.Tensor,
        mask: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        """Sample one action per row from the masked softmax.

        Returns `(picks, log_prob, entropy)`:
        - `picks`: plain `int`s (for `Pool.apply`), no grad.
        - `log_prob`: `[N]` tensor with grad back to the parameters.
        - `entropy`: `[N]` tensor with grad — the policy entropy over valid
          actions, for the optional entropy bonus. Zero when only one action is
          legal (a deterministic ply contributes no entropy).

        `generator`, when given, makes the sample deterministic — used by the
        rollout so a resumed run replays the same trajectory.
        """
        logits = self.forward(obs, acts, mask)                    # [N, M]
        # Numerically stable masked softmax: -inf columns go to zero prob.
        probs = torch.softmax(logits, dim=1)                       # [N, M]
        # Categorical requires finite logits and non-negative probs that sum to 1.
        # Replace any nan (from all-inf rows, which shouldn't happen given the
        # engine always offers at least one legal action) and zero out padded cols.
        probs = torch.nan_to_num(probs, nan=0.0)
        row_sums = probs.sum(dim=1, keepdim=True).clamp(min=1e-12)
        probs = probs / row_sums
        dist = torch.distributions.Categorical(probs)
        if generator is not None:
            picks_t = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)  # [N]
        else:
            picks_t = dist.sample()  # type: ignore[no-untyped-call]
        logp = dist.log_prob(picks_t)  # type: ignore[no-untyped-call]  # [N] with grad
        entropy = dist.entropy()  # type: ignore[no-untyped-call]  # [N] with grad
        return picks_t.tolist(), logp, entropy

    # -- serialization --------------------------------------------------

    def to_genome_vec(self) -> np.ndarray:
        """Flatten the parameters into the `genome.layer_shapes` layout.

        The inverse of `from_genome_vec`; matches `genome.from_modules` exactly.
        Gradients are detached — this is for export/checkpoint, not for grad.
        """
        parts: list[np.ndarray] = []
        for layer in [*self.trunk, *self.head]:
            parts.append(layer.weight.detach().reshape(-1).to(torch.float32).cpu().numpy())
            parts.append(layer.bias.detach().reshape(-1).to(torch.float32).cpu().numpy())
        return np.concatenate(parts).astype(np.float32)

    @classmethod
    def from_genome_vec(cls, vec: np.ndarray, arch: Arch) -> CanastraNet:
        """Build a `CanastraNet` from a flat genome vector.

        Mirrors `genome.to_modules` — same offset walk over `layer_shapes` — but
        keeps the result as a live `nn.Module` with `requires_grad=True` params
        rather than freezing into an `inference_mode` module list.
        """
        net = cls(arch)
        offset = 0
        for layer, (name, out, inn) in zip(
            [*net.trunk, *net.head], genome_mod.layer_shapes(arch)
        ):
            w = torch.from_numpy(vec[offset : offset + out * inn].reshape(out, inn).copy())
            offset += out * inn
            b = torch.from_numpy(vec[offset : offset + out].copy())
            offset += out
            assert layer.weight.shape == (out, inn), f"{name}: {layer.weight.shape} != {(out, inn)}"
            assert layer.bias.shape == (out,), f"{name}: {layer.bias.shape} != {(out,)}"
            with torch.no_grad():
                layer.weight.copy_(w.to(layer.weight.dtype))
                layer.bias.copy_(b.to(layer.bias.dtype))
        assert offset == vec.size
        return net

    def save_json(self, path: str) -> None:
        """Export to `canastra-weights@1` JSON — byte-identical format to the GA's champions."""
        genome_mod.save_json(path, self.arch, self.to_genome_vec())

    @classmethod
    def load_json(cls, path: str) -> CanastraNet:
        """Load a `canastra-weights@1` JSON file into a live `CanastraNet`."""
        arch, vec = genome_mod.load_json(path)
        return cls.from_genome_vec(vec, arch)


def _forward_tanh(stack: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
    """All layers with tanh — the trunk."""
    for layer in stack:
        x = torch.tanh(layer(x))
    return x


def _forward_final_linear(stack: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
    """All hidden layers tanh, the final layer linear — the head."""
    last = len(stack) - 1
    for index, layer in enumerate(stack):
        x = layer(x)
        if index < last:
            x = torch.tanh(x)
    return x


def compile_net(net: CanastraNet, device: str) -> CanastraNet:
    """`torch.compile` the learner net for kernel fusion across per-ply forwards.

    A no-op fallback when compile is unavailable (older torch, or `--no-compile`).
    **Do not enable this without triton** — on Windows (no triton) the per-ply
    input shapes change (variable legal-action count `M`), and without triton's
    JIT `torch.compile` recompiles on every shape change, which is ~2000x slower
    than eager. Returns the net un-compiled if triton is not available.
    """
    net = net.to(device)
    try:
        import torch.triton  # type: ignore[import-not-found]
    except ImportError:
        import warnings
        warnings.warn(
            "torch.compile requested but triton is not available — "
            "falling back to eager (pass --no-compile to silence this). "
            "On Windows, triton is typically unavailable and compile is harmful.",
            stacklevel=2,
        )
        return net
    try:
        compiled = torch.compile(net)
        return compiled  # type: ignore[return-value]
    except Exception:  # noqa: BLE001 — compile is an optimization, never a hard dependency
        return net


def forward_inference(
    net: CanastraNet, obs: torch.Tensor, acts: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Run `net.forward` under `inference_mode` — for the parity test and the
    opponent path (which never needs grad)."""
    with torch.inference_mode():
        return net.forward(obs, acts, mask)


def parity_check(net: CanastraNet, obs: torch.Tensor, acts: torch.Tensor, mask: torch.Tensor) -> float:
    """Max abs diff between `CanastraNet.forward` (inference) and `policy.logits`.

    Used by the test suite to pin the math equivalence. Returns the max abs
    difference over masked (valid) entries.
    """
    vec = net.to_genome_vec()
    trunk, head = genome_mod.to_modules(vec, net.arch)
    ref = policy.logits(trunk, head, obs, acts, mask)
    ours = forward_inference(net, obs, acts, mask)
    diff = (ours - ref).abs()
    # Only compare valid (non -inf) entries — padded columns are -inf in both.
    valid = mask
    return float(diff[valid].max()) if valid.any() else 0.0


__all__ = [
    "Arch",
    "CanastraNet",
    "compile_net",
    "forward_inference",
    "parity_check",
]
