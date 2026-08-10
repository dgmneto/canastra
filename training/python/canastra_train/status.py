"""Durable run status: a throttled, atomically written status.json per run.

Pure IO — writes never touch the RNG, so determinism and the checkpoint
format are unaffected. The file is for external consumers (a separate
terminal tailing a run, a future web view); the TUI renders from the same
data in memory.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class RunStatus:
    run_dir: str
    generation: int = 0
    total_generations: int = 0
    phase: str = "starting"  # starting | evaluating | evolving | done
    plies: int = 0
    games_finished: int = 0
    games_total: int = 0
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    sigma: float = 0.0
    best_ever: float | None = None
    last_best: float | None = None
    last_mean: float | None = None
    device: str = "cpu"


class StatusWriter:
    """Writes status.json at most once per second; flush forces a write."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self._last = 0.0

    def write(self, status: RunStatus) -> None:
        now = time.monotonic()
        if now - self._last < 1.0:
            return
        self._last = now
        self._put(status)

    def flush(self, status: RunStatus) -> None:
        self._put(status)

    def _put(self, status: RunStatus) -> None:
        path = self.run_dir / "status.json"
        tmp = self.run_dir / "status.json.tmp"
        tmp.write_text(json.dumps(asdict(status)))
        os.replace(tmp, path)