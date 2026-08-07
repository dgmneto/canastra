"""Pin the Python↔TypeScript forward-pass parity with a committed test.

Both sides of the weights pipeline (training/python/canastra_train/policy.py and
bots/src/forward.ts) implement the same arch: trunk layers all tanh, head hidden
layers all tanh, the final `head.out` layer linear. Parity has been verified by
hand but never pinned; this test locks it so a drift on either side fails loudly.

It builds ONE small deterministic weights file, writes it to JSON, and loads that
SAME rounded file back into both implementations — the TS side reads it directly
and the Python side reloads it via `genome.load_json` — then scores the same
observation and action row through each and asserts they agree. A constant
obs/feat row avoids any rng dependence in the inputs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch
from canastra_train import genome, policy

ROOT = Path(__file__).resolve().parents[2]

ARCH = {"obs": 2002, "act": 101, "trunk": [32], "head": [16], "activation": "tanh"}

OBS_DIM: int = 2002
ACT_DIM: int = 101

_TS_SCRIPT = """\
import {{ compileWeights, embed, scoreAction }} from {forward_import};
import {{ readFileSync }} from "node:fs";

const weights = JSON.parse(readFileSync(process.argv[2], "utf8"));
const cw = compileWeights(weights);
const obs = new Array({obs}).fill({obs_val});
const feats = new Array({act}).fill({act_val});
const emb = embed(cw, obs);
const score = scoreAction(cw, emb, feats);
console.log(score.toFixed(10));
"""


def _ts_score(weights_path: Path, tmp_path: Path) -> float:
    """Run the real TypeScript forward pass and return its single score."""
    script = tmp_path / "parity.ts"
    script.write_text(
        _TS_SCRIPT.format(
            forward_import=f'"{ROOT / "bots/src/forward.ts"}"',
            obs=OBS_DIM,
            act=ACT_DIM,
            obs_val=0.01,
            act_val=0.02,
        )
    )
    result = subprocess.run(
        ["npx", "tsx", str(script), str(weights_path)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"tsx failed:\n{result.stderr}")
    return float(result.stdout.strip())


def _py_score(vec: np.ndarray) -> float:
    obs = np.full(OBS_DIM, 0.01, dtype=np.float32)
    feats = np.full(ACT_DIM, 0.02, dtype=np.float32)
    trunk, head = genome.to_modules(vec, ARCH)
    obs_t = torch.from_numpy(obs).unsqueeze(0)  # [1, obs]
    acts_t = torch.from_numpy(feats).unsqueeze(0).unsqueeze(0)  # [1, 1, act]
    mask = torch.tensor([[True]])
    logit = policy.logits(trunk, head, obs_t, acts_t, mask)
    return float(logit[0, 0].detach())


def test_python_ts_forward_pass_parity(tmp_path: Path) -> None:
    vec = genome.random_genome(ARCH, seed=99)
    weights_path = tmp_path / "weights.json"
    genome.save_json(str(weights_path), ARCH, vec)

    # Both sides must consume the SAME rounded JSON bytes: reload the file
    # here so the Python vector matches exactly what TypeScript reads.
    _, loaded = genome.load_json(str(weights_path))
    py_score = _py_score(loaded)
    ts_score = _ts_score(weights_path, tmp_path)

    print(f"\npy score = {py_score:.10f}")
    print(f"ts score = {ts_score:.10f}")
    assert abs(py_score - ts_score) < 1e-4, (
        f"forward-pass parity broken: Python {py_score} vs TypeScript {ts_score}"
    )
