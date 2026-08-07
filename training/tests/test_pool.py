"""The pool drives whole batches of games, two batched crossings per ply."""

import numpy as np
from canastra_py import ACT_DIM, OBS_DIM, Pool


def test_pool_plays_full_matches_to_completion() -> None:
    pool = Pool([1, 2, 3, 4], max_actions_per_game=200_000)
    rng = np.random.default_rng(0)
    plies = 0
    while pool.has_live():
        obs, acts, mask, rows = pool.encode()
        assert obs.shape == (mask.shape[0], OBS_DIM)
        assert acts.shape == (mask.shape[0], mask.shape[1], ACT_DIM)
        assert obs.dtype == np.float32
        assert mask.dtype == bool
        assert rows.shape == (mask.shape[0], 2)
        assert rows.dtype == np.int64
        game_ids = set(rows[:, 0].tolist())
        seat_ids = set(rows[:, 1].tolist())
        assert game_ids <= set(range(4))
        assert seat_ids <= set(range(4))
        pairs = [tuple(row) for row in rows.tolist()]
        assert len(set(pairs)) == len(pairs), "no (game, seat) pair repeats"
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
        _, _, mask, rows = pool.encode()
        assert rows.shape[1] == 2
        pool.apply([int(rng.integers(0, int(menu.sum()))) for menu in mask])
        plies += 1
        assert plies < 1000, "the action cap should stop the match on its own"
    results = pool.results()
    assert len(results) == 1
    _seed, _scores, winner, _hands, unfinished = results[0]
    assert unfinished is True
    assert winner is None


# A meld-greedy policy: lay a new meld if one is on the menu, else extend an
# existing one, else discard, else whatever is offered. Under the pooled bug,
# greedy play on seed 0 dead-ends its first turn and loops — safe mode was
# cleared on the retry's own draw, so the full meld menu came straight back.
_GREEDY = ("LayMeld", "AddToMeld", "Discard")


def _greedy_pick(kinds: list[str]) -> int:
    for want in _GREEDY:
        if want in kinds:
            return kinds.index(want)
    return 0


def test_safe_mode_terminates_dead_ended_turns() -> None:
    pool = Pool([0], max_actions_per_game=100_000)
    plies = 0
    while pool.has_live():
        _, _, _mask, _rows = pool.encode()
        pool.apply([_greedy_pick(kinds) for kinds in pool.menu_kinds()])
        plies += 1
        assert plies < 50_000, "safe mode should terminate the dead-ended turn"
    results = pool.results()
    assert len(results) == 1
    _seed, _scores, _winner, _hands, unfinished = results[0]
    assert not unfinished
