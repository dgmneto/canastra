"""ELO rating tracker for the GA.

Replaces per-generation score differentials with a persistent rating that
carries across generations. Children inherit their parent's rating, so a
small mutation starts with a meaningful prior rather than from scratch.

Standard ELO: expected = 1 / (1 + 10^((opp - self) / 400)), and
rating += K * (result - expected). Results are 1.0 (win), 0.0 (loss), 0.5
(draw). Updates are in-place and processed in deterministic game order
(fixed by ``batch_layout``), so a resumed run reproduces ratings exactly.
"""

from __future__ import annotations

import numpy as np


class EloTracker:
    """Persistent ELO ratings for a roster (population + hall of fame)."""

    def __init__(
        self,
        size: int,
        k_factor: float = 32.0,
        base: float = 1200.0,
    ) -> None:
        self.ratings = np.full(size, base, dtype=np.float64)
        self.k_factor = k_factor
        self.base = base

    def expected(self, a: int, b: int) -> float:
        return float(1.0 / (1.0 + 10.0 ** ((self.ratings[b] - self.ratings[a]) / 400.0)))

    def update(self, a: int, b: int, result: float) -> None:
        exp_a = self.expected(a, b)
        self.ratings[a] += self.k_factor * (result - exp_a)
        self.ratings[b] += self.k_factor * (1.0 - result - (1.0 - exp_a))

    def batch_update(self, results: list[tuple[int, int, float]]) -> None:
        for a, b, result in results:
            self.update(a, b, result)

    def grow(self, n_new: int, parent_indices: list[int] | None = None) -> None:
        """Extend the ratings array. If ``parent_indices`` is given, each new
        entry inherits the corresponding parent's rating (for HOF champions
        that are children of a known parent); otherwise starts at ``base``."""
        old = self.ratings
        if parent_indices is not None:
            inherited = np.array(
                [old[p] if p < len(old) else self.base for p in parent_indices],
                dtype=np.float64,
            )
            self.ratings = np.concatenate([old, inherited])
        else:
            self.ratings = np.concatenate([old, np.full(n_new, self.base, dtype=np.float64)])

    def __len__(self) -> int:
        return len(self.ratings)

    def copy(self) -> EloTracker:
        tracker = EloTracker(len(self), self.k_factor, self.base)
        tracker.ratings = self.ratings.copy()
        return tracker