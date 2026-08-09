import numpy as np

OBS_DIM: int
ACT_DIM: int


class Pool:
    def __init__(self, seeds: list[int], max_actions_per_game: int | None = None) -> None: ...

    def has_live(self) -> bool: ...

    def encode(self) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]: ...

    def apply(self, picks: list[int]) -> None: ...

    def results(self) -> list[tuple[int, tuple[int, int], int | None, int, bool]]: ...

    def menu_kinds(self) -> list[list[str]]: ...
