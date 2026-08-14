"""The live dashboard for the PG trainer: a watch-only rich TUI over a run.

Mirrors `tui.py` (the GA dashboard) but for the PG trainer's update-shaped
records: mean reward, baseline, best-ever, wins/losses, plies, ETA. Reuses the
sparkline/bar helpers from `tui.py` and the `status.json` writer from
`status.py`. On a non-TTY (or `--no-tui`) it degrades to plain per-update
console lines, so logs and scripts keep working.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from canastra_train.status import RunStatus, StatusWriter
from canastra_train.tui import _bar, _fmt_duration, _spark

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

_SPARK_WIDTH = 42


class PGDashboard:
    """Watch-only view of a PG training run. Owns the Live display and status.json."""

    def __init__(
        self,
        run_dir: Path,
        total_episodes: int,
        start_episode: int,
        *,
        no_tui: bool,
        device: str,
        lr: float,
        games_per_update: int,
    ) -> None:
        self.run_dir = run_dir
        self.total = total_episodes
        self.no_tui = no_tui or not _RICH or not sys.stdout.isatty()
        self.status = RunStatus(
            run_dir=str(run_dir),
            generation=start_episode,  # reuse the field as "update"
            total_generations=total_episodes,
            sigma=lr,  # reuse as lr display
            games_total=games_per_update,
            device=device,
        )
        self.writer = StatusWriter(run_dir)
        self.history: list[dict[str, Any]] = []
        self.evals: list[dict[str, Any]] = []
        self.events: list[str] = []
        self._rollout_mb = 0
        self._rollout_total_mb = 0
        self._rollout_plies = 0
        self._rollout_games_done = 0
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

    def on_update(self, record: dict[str, Any]) -> None:
        """Called after each gradient update with the step's record."""
        self.history.append(record)
        self.status.generation = int(record["update"]) + 1
        self.status.last_best = float(record["mean_reward"])
        # Reuse sigma slot to display baseline; best_ever for the running best.
        self.status.sigma = float(record["baseline"])
        if record.get("improved"):
            self.status.best_ever = float(record["best_ever"])
        if self.no_tui:
            print(
                f"update {record['update']:5d}: reward {record['mean_reward']:+.1f} "
                f"baseline {record['baseline']:+.1f} best {record['best_ever']:+.1f} "
                f"plies {record['plies']} wins {record['wins']}/{record['games']} "
                f"({record['wall_seconds']}s)"
            )
        self._tick()

    def on_event(self, kind: str, detail: str) -> None:
        self.events.append(f"{kind}: {detail}")
        self.events = self.events[-6:]
        if self.no_tui and kind in ("best", "export", "opp-refresh", "eval"):
            print(f"  {detail}")
        self._tick()

    def on_rollout_progress(self, mini_batch: int, total_mb: int, plies: int, games_done: int) -> None:
        """Intra-update progress: which mini-batch, how many plies/games done."""
        self._rollout_mb = mini_batch
        self._rollout_total_mb = total_mb
        self._rollout_plies = plies
        self._rollout_games_done = games_done
        self.status.phase = "rolling out"
        self.status.plies = plies
        self.status.games_finished = games_done
        self.status.games_total = max(total_mb * (self.status.games_total // max(total_mb, 1)), games_done, 1)
        self._tick()

    def on_eval(self, eval_record: dict[str, Any]) -> None:
        """Called after each periodic growth eval against the fixed baseline."""
        self.evals.append(eval_record)
        diff = float(eval_record["mean_diff"])
        detail = (
            f"eval upd {eval_record['update']}: diff {diff:+.1f} "
            f"(±{eval_record['ci95']:.0f}) wins {eval_record['wins']}/{eval_record['pairs']}"
        )
        self.on_event("eval", detail)

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
        done = len(self.history)
        pct = 100.0 * done / max(self.total, 1)
        eta = self._eta_run()

        header = Text(
            f"run {s.run_dir} · update {done}/{s.total_generations} · phase {s.phase} · device {s.device}",
            style="bold cyan",
        )
        progress = Text(
            f"{_bar(done, s.total_generations)} {pct:5.1f}%  "
            f"elapsed {_fmt_duration(s.elapsed_seconds)}"
            + (f"  ·  ETA {_fmt_duration(eta)}" if eta is not None else "")
        )
        # Intra-update rollout progress (shows activity during the first long step).
        rollout_line: RenderableType | None = None
        if self._rollout_total_mb > 0 and done < self.total:
            mb_pct = 100.0 * self._rollout_mb / max(self._rollout_total_mb, 1)
            rollout_line = Text(
                f"  rollout: mini-batch {self._rollout_mb}/{self._rollout_total_mb} "
                f"({mb_pct:.0f}%)  plies {self._rollout_plies}  "
                f"games done {self._rollout_games_done}",
                style="dim",
            )
        reward: RenderableType
        if s.last_best is not None:
            reward = Text(
                f"reward  last {s.last_best:+.1f}  baseline {s.sigma:+.1f}"
                + (f"  ·  best ever {s.best_ever:+.1f}" if s.best_ever is not None else "")
            )
        else:
            reward = Text("reward: waiting for the first update to finish", style="dim")
        spark = self._sparklines()

        blocks: list[RenderableType] = [header, progress]
        if rollout_line is not None:
            blocks.append(rollout_line)
        blocks.append(reward)
        if spark is not None:
            blocks.append(spark)
        if self.evals:
            blocks.append(self._eval_sparkline())
            blocks.append(self._eval_table())
        if self.history:
            blocks.append(self._history_table())
        if self.events:
            lines = [Text("events:", style="bold yellow")]
            for event in self.events:
                lines.append(Text("  " + event, style="yellow"))
            blocks.append(Group(*lines))
        return Panel(Group(*blocks), title="canastra-train (REINFORCE)", border_style="cyan")

    def _history_table(self) -> RenderableType:
        table = Table(title="updates", show_lines=False, header_style="bold")
        table.add_column("upd", justify="right")
        table.add_column("reward", justify="right")
        table.add_column("baseline", justify="right")
        table.add_column("best", justify="right")
        table.add_column("wins", justify="right")
        table.add_column("wall", justify="right")
        for record in self.history[-8:]:
            table.add_row(
                str(record["update"]),
                f"{record['mean_reward']:+.1f}",
                f"{record['baseline']:+.1f}",
                f"{record['best_ever']:+.1f}",
                f"{record['wins']}/{record['games']}",
                _fmt_duration(float(record["wall_seconds"])),
            )
        return table

    def _sparklines(self) -> RenderableType | None:
        if len(self.history) < 2:
            return None
        rewards = [float(r["mean_reward"]) for r in self.history]
        bests = [float(r["best_ever"]) for r in self.history]
        return Group(
            Text("reward " + _spark(rewards, width=_SPARK_WIDTH)),
            Text("best   " + _spark(bests, width=_SPARK_WIDTH)),
        )

    def _eval_sparkline(self) -> RenderableType:
        """Sparkline of the learner's strength vs the fixed baseline over time."""
        diffs = [float(e["mean_diff"]) for e in self.evals]
        wins = [float(e["wins"]) / max(float(e["pairs"]), 1) for e in self.evals]
        return Group(
            Text("vs base " + _spark(diffs, width=_SPARK_WIDTH)),
            Text("winrate " + _spark(wins, width=_SPARK_WIDTH)),
        )

    def _eval_table(self) -> RenderableType:
        """The last few eval points — the growth curve in numbers."""
        table = Table(title="growth (vs fixed baseline)", show_lines=False, header_style="bold green")
        table.add_column("upd", justify="right")
        table.add_column("diff", justify="right")
        table.add_column("ci", justify="right")
        table.add_column("wins", justify="right")
        table.add_column("winrate", justify="right")
        for record in self.evals[-6:]:
            pairs = max(int(record["pairs"]), 1)
            table.add_row(
                str(record["update"]),
                f"{record['mean_diff']:+.1f}",
                f"+/-{record['ci95']:.0f}",
                f"{record['wins']}/{record['pairs']}",
                f"{100.0 * record['wins'] / pairs:.0f}%",
            )
        return table

    def _eta_run(self) -> float | None:
        done = len(self.history)
        if done <= 0 or self.total <= 0:
            return None
        elapsed = self.status.elapsed_seconds
        per_update = elapsed / done
        return per_update * (self.total - done)
