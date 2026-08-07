"""The pool drives whole batches of games with one FFI crossing per ply."""

import numpy as np
from canastra_py import ACT_DIM, OBS_DIM, Pool


def test_pool_plays_full_matches_to_completion() -> None:
    pool = Pool([1, 2, 3, 4], max_actions_per_game=200_000)
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
    for _seed, scores, _winner, hands, unfinished in results:
        assert hands >= 1
        assert max(scores) >= 5000
        assert not unfinished


def test_pool_caps_runaway_matches() -> None:
    pool = Pool([5], max_actions_per_game=50)
    rng = np.random.default_rng(1)
    plies = 0
    while pool.has_live():
        _, _, mask = pool.encode()
        pool.apply([int(rng.integers(0, int(menu.sum()))) for menu in mask])
        plies += 1
        assert plies < 1000, "the action cap should stop the match on its own"
    results = pool.results()
    assert len(results) == 1
    _seed, _scores, winner, _hands, unfinished = results[0]
    assert unfinished is True
    assert winner is None
