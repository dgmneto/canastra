"""The live dashboard: a watch-only rich TUI over a running training run.

Renders in-process — the trainer itself owns the screen, no second process,
no polling. Intra-generation progress (games finished, batch rounds/s, ETA) comes
from the pool loop's progress callback; generation records and promotion
events (champion exports, HOF archivals, new bests) land as they happen.

On a non-TTY (piped/filed output) — or with `--no-tui` — it degrades to the
plain per-generation console lines the trainer always printed, so scripts and
logs keep working unchanged.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from canastra_train.status import RunStatus, StatusWriter

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    _RICH = True
except ImportError:  # pragma: no cover - exercised only when rich is absent
    _RICH = False
    Console = None  # type: ignore[assignment,misc]
    Live = None  # type: ignore[assignment,misc]
    Panel = None  # type: ignore[assignment,misc]
    Group = None  # type: ignore[assignment,misc]
    Table = None  # type: ignore[assignment,misc]
    Text = None  # type: ignore[assignment,misc]

RenderableType = Any

_BLOCKS = "▁▂▃▄▅▆▇█"
_SPARK_WIDTH = 42


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _bar(finished: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "░" * width
    filled = round(width * finished / total)
    return "█" * filled + "░" * (width - filled)


def _spark(values: list[float], width: int = _SPARK_WIDTH) -> str:
    """Unicode-block sparkline over [min, max]; flat series render mid-line."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if hi <= lo:
        return _BLOCKS[len(_BLOCKS) // 2] * width
    buckets = [len(_BLOCKS) - 1] * width
    for i, value in enumerate(values):
        if i < width:
            buckets[i] = int((value - lo) / (hi - lo) * (len(_BLOCKS) - 1))
    return "".join(_BLOCKS[b] for b in buckets)


class Dashboard:
    """Watch-only view of a run. Owns the Live display and status.json."""

    def __init__(
        self,
        run_dir: Path,
        total_generations: int,
        start_generation: int,
        *,
        no_tui: bool,
        device: str,
        sigma: float,
        games_total: int,
    ) -> None:
        self.run_dir = run_dir
        self.total = total_generations
        self.no_tui = no_tui or not _RICH or not sys.stdout.isatty()
        self.status = RunStatus(
            run_dir=str(run_dir),
            generation=start_generation,
            total_generations=total_generations,
            sigma=sigma,
            games_total=games_total,
            device=device,
        )
        self.writer = StatusWriter(run_dir)
        self.history: list[dict[str, Any]] = []
        self.events: list[str] = []
        self._started = time.monotonic()
        self._last_render = 0.0
        self._live: Any = None

    def start(self) -> None:
        self.status.phase = "starting"
        if not self.no_tui:
            self._live = Live(self._render(), console=Console(), refresh_per_second=8)
            self._live.start()
        self._tick()

    def stop(self) -> None:
        self.status.phase = "done"
        self._tick(force=True)
        if self._live is not None:
            self._live.stop()
            self._live = None

    def on_progress(self, plies: int, games_finished: int) -> None:
        self.status.plies = plies
        self.status.games_finished = games_finished
        self._tick()

    def set_phase(self, phase: str) -> None:
        self.status.phase = phase
        self._tick()

    def on_generation(self, record: dict[str, Any]) -> None:
        self.history.append(record)
        self.status.generation = int(record["generation"]) + 1
        self.status.last_best = float(record["fitness_best"])
        self.status.last_mean = float(record["fitness_mean"])
        self.status.sigma = float(record["sigma"])
        if self.no_tui:
            print(
                f"gen {record['generation']}: best {record['fitness_best']:+.1f} "
                f"mean {record['fitness_mean']:+.1f} "
                f"sigma {record['sigma']:.4f} ({record['wall_seconds']}s)"
            )
        self._tick()

    def on_event(self, kind: str, detail: str) -> None:
        self.events.append(f"{kind}: {detail}")
        self.events = self.events[-6:]
        if self.no_tui and kind in ("best", "export", "hof"):
            print(f"  {detail}")
        self._tick()

    def _tick(self, force: bool = False) -> None:
        self.status.elapsed_seconds = time.monotonic() - self._started
        self.status.eta_seconds = self._eta_run()
        if self._live is not None and (force or time.monotonic() - self._last_render >= 0.25):
            self._last_render = time.monotonic()
            self._live.update(self._render())
        if force:
            self.writer.flush(self.status)
        else:
            self.writer.write(self.status)

    # -- rendering -------------------------------------------------------

    def _render(self) -> RenderableType:
        s = self.status
        games_total = max(s.games_total, 1)
        pct = 100.0 * s.games_finished / games_total
        pace = s.plies / s.elapsed_seconds if s.elapsed_seconds > 0 else 0.0
        eta_gen = self._eta_generation()
        eta_run = self._eta_run()

        header = Text(
            f"run {s.run_dir} · gen {s.generation}/{s.total_generations} · phase {s.phase} · device {s.device}",
            style="bold cyan",
        )
        progress = Text(
            f"{_bar(s.games_finished, games_total)} {pct:5.1f}%  "
            f"games {s.games_finished}/{s.games_total}  batch rounds {s.plies}  "
            f"{pace:,.0f} batch rounds/s",
        )
        timing = Text(
            f"elapsed {_fmt_duration(s.elapsed_seconds)}"
            + (f"  ·  ETA this gen {_fmt_duration(eta_gen)}" if eta_gen is not None else "")
            + (f"  ·  ETA run {_fmt_duration(eta_run)}" if eta_run is not None else "")
            + f"  ·  sigma {s.sigma:.4f}"
        )
        fitness: RenderableType
        if s.last_best is not None:
            fitness = Text(
                f"fitness  best {s.last_best:+.1f}  mean {s.last_mean:+.1f}"
                + (f"  ·  best ever {s.best_ever:+.1f}" if s.best_ever is not None else "")
            )
        else:
            fitness = Text("fitness: waiting for the first generation to finish", style="dim")
        spark = self._sparklines()

        blocks: list[RenderableType] = [header, progress, timing, fitness]
        if spark is not None:
            blocks.append(spark)
        if self.history:
            blocks.append(self._history_table())
        if self.events:
            lines = [Text("events:", style="bold yellow")]
            for event in self.events:
                lines.append(Text("  " + event, style="yellow"))
            blocks.append(Group(*lines))
        return Panel(Group(*blocks), title="canastra-train", border_style="cyan")

    def _history_table(self) -> RenderableType:
        table = Table(title="generations", show_lines=False, header_style="bold")
        table.add_column("gen", justify="right")
        table.add_column("best", justify="right")
        table.add_column("mean", justify="right")
        table.add_column("sigma", justify="right")
        table.add_column("wall", justify="right")
        for record in self.history[-8:]:
            table.add_row(
                str(record["generation"]),
                f"{record['fitness_best']:+.1f}",
                f"{record['fitness_mean']:+.1f}",
                f"{record['sigma']:.4f}",
                _fmt_duration(float(record["wall_seconds"])),
            )
        return table

    def _sparklines(self) -> RenderableType | None:
        if len(self.history) < 2:
            return None
        bests = [float(r["fitness_best"]) for r in self.history]
        means = [float(r["fitness_mean"]) for r in self.history]
        return Group(
            Text("best " + _spark(bests)),
            Text("mean " + _spark(means)),
        )

    def _eta_generation(self) -> float | None:
        if self.status.games_finished <= 0 or self.status.games_total <= 0:
            return None
        elapsed = self.status.elapsed_seconds
        per_game = elapsed / self.status.games_finished
        return per_game * (self.status.games_total - self.status.games_finished)

    def _eta_run(self) -> float | None:
        if self.status.games_finished <= 0 or self.status.games_total <= 0:
            return None
        per_game = self.status.elapsed_seconds / self.status.games_finished
        completed = len(self.history)
        remaining_gens = max(self.total - completed - 1, 0)
        remaining_games = remaining_gens * self.status.games_total + (
            self.status.games_total - self.status.games_finished
        )
        return per_game * remaining_games
