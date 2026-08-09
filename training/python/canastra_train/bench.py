"""Plies-per-second benchmark: a random legal policy over a 64-game pool.

Env stepping is the training loop's ceiling (the forward pass is batched
elsewhere), so this number is the one to watch when touching the pool or the
encoder. Run: `.venv/bin/python -m canastra_train.bench`
"""

import time

import numpy as np
from canastra_py import Pool


def main() -> None:
    pool = Pool(list(range(64)))
    rng = np.random.default_rng(7)
    plies = 0
    start = time.perf_counter()
    while pool.has_live():
        _, _, mask, _rows = pool.encode()
        pool.apply([int(rng.integers(0, int(menu.sum()))) for menu in mask])
        plies += 1
    elapsed = time.perf_counter() - start
    print(f"{plies} plies in {elapsed:.2f}s = {plies / elapsed:.0f} plies/s")


if __name__ == "__main__":
    main()
