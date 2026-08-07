"""The pool drives whole batches of games with one FFI crossing per ply."""

import numpy as np
from canastra_py import ACT_DIM, OBS_DIM, Pool


def test_pool_plays_full_matches_to_completion() -> None:
    pool = Pool([1, 2, 3, 4])
    rng = np.random.default_rng(0)
    plies = 0
    while pool.has_live():
        obs, acts, mask = pool.encode()
        assert obs.shape == (mask.shape[0], OBS_DIM)
        assert acts.shape == (mask.shape[0], mask.shape[1], ACT_DIM)
        assert obs.dtype == np.float32
        assert mask.dtype == bool
        picks = [int(rng.integers(0, int(menu.sum()))) for menu in mask]
        pool.apply(picks)
        plies += 1
        assert plies < 200_000, "matches should end long before the cap"
    results = pool.results()
    assert len(results) == 4
    for _seed, scores, _winner, hands in results:
        assert hands >= 1
        assert max(scores) >= 5000
