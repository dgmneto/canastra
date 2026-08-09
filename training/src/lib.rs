//! Python bindings for driving batches of Canastra games.
//!
//! The pool owns one engine per seed and answers three questions: which
//! seats are waiting on a decision, what those decisions look like as tensors
//! (two batched crossings per ply, buffers filled in Rust), and what happened when
//! a match ended. The engine remains the only referee — nothing here knows a
//! rule.

use canastra_encode::{encode_actions, encode_observation, ACT_DIM, OBS_DIM};
use canastra_engine::{apply, enumerate, new_game, observe, settle_hand, Action, GameState, Phase};
use numpy::{PyArray2, PyArray3, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// A match that ended: either reached §14, or was cut short by the action cap
/// and left `unfinished`.
type MatchResult = (u64, [i32; 2], Option<u8>, u32, bool);

/// A short, stable label for an action's kind — the menu columns a greedy
/// policy keys on (they never carry seat or card data, so they are safe to
/// expose over the Python boundary).
fn action_kind(action: &Action) -> &'static str {
    match action {
        Action::Draw => "Draw",
        Action::KeepDrawnCard => "KeepDrawnCard",
        Action::RefuseDrawnCard => "RefuseDrawnCard",
        Action::TakeDiscardPile { .. } => "TakeDiscardPile",
        Action::LayMeld { .. } => "LayMeld",
        Action::AddToMeld { .. } => "AddToMeld",
        Action::Discard { .. } => "Discard",
        Action::EndTurnWithoutDiscard => "EndTurnWithoutDiscard",
    }
}

struct Game {
    state: GameState,
    turn_start: GameState,
    /// The previous attempt at this turn dead-ended; menus are restricted to
    /// draw/refuse/discard until the turn ends (the driver's safeMode).
    safe: bool,
    seed: u64,
    hands: u32,
    /// How many actions this match has played (one per `Pool::apply` row).
    actions_played: u64,
    result: Option<MatchResult>,
}

impl Game {
    fn new(seed: u64) -> Game {
        let state = new_game(seed);
        Game {
            turn_start: state.clone(),
            state,
            safe: false,
            seed,
            hands: 1,
            actions_played: 0,
            result: None,
        }
    }

    fn live(&self) -> bool {
        self.result.is_none()
    }

    /// The moves the policy may pick from this ply. In a safe turn, melding
    /// and pile-taking are withheld — they are exactly what dead-ends.
    fn menu(&self) -> Vec<Action> {
        let legal = enumerate(&self.state, self.state.turn);
        if !self.safe {
            return legal;
        }
        legal
            .into_iter()
            .filter(|action| {
                matches!(
                    action,
                    Action::Draw
                        | Action::KeepDrawnCard
                        | Action::RefuseDrawnCard
                        | Action::Discard { .. }
                        | Action::EndTurnWithoutDiscard
                )
            })
            .collect()
    }
}

/// One engine per seed, driven a ply at a time across the whole batch.
#[pyclass]
struct Pool {
    games: Vec<Game>,
    /// The rows `encode` last handed out: which game, and its menu.
    pending: Vec<usize>,
    menus: Vec<Vec<Action>>,
    /// Per-game action ceiling; a live match that reaches it ends unfinished.
    max_actions_per_game: Option<u64>,
}

#[pymethods]
impl Pool {
    #[new]
    #[pyo3(signature = (seeds, max_actions_per_game=None))]
    fn new(seeds: Vec<u64>, max_actions_per_game: Option<u64>) -> Pool {
        Pool {
            games: seeds.into_iter().map(Game::new).collect(),
            pending: Vec::new(),
            menus: Vec::new(),
            max_actions_per_game,
        }
    }

    /// Any match still in progress?
    fn has_live(&self) -> bool {
        self.games.iter().any(Game::live)
    }

    /// `(obs, acts, mask, rows)` for every seat awaiting a decision: obs is
    /// `[N, OBS_DIM] f32`, acts `[N, M, ACT_DIM] f32` zero-padded, mask
    /// `[N, M] bool` marking the real columns, and rows `[N, 2] i64` carries
    /// per-row `(game index, seat)` so callers can route picks to per-seat
    /// policies. Menus are never truncated — padding, never dropping, is how
    /// a turn-ending action can't get lost.
    #[allow(clippy::type_complexity)]
    fn encode<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray2<bool>>,
        Bound<'py, PyArray2<i64>>,
    )> {
        self.pending = (0..self.games.len())
            .filter(|&i| {
                self.games[i].live()
                    && matches!(
                        self.games[i].state.phase,
                        Phase::AwaitingDraw | Phase::AwaitingRefusalChoice | Phase::Melding
                    )
            })
            .collect();

        // Menus first (enumerate is the expensive part — parallel). A game
        // whose menu comes back empty has dead-ended its turn: back out to
        // the turn's start and retry safe, exactly the harness driver's path.
        let mut menus: Vec<Vec<Action>> = self
            .pending
            .par_iter()
            .map(|&i| self.games[i].menu())
            .collect();
        for (row, &game) in self.pending.iter().enumerate() {
            if menus[row].is_empty() {
                let game = &mut self.games[game];
                // §6.1: backing out after laying cards, before the partnership
                // has opened, is a failed opening — the bar steps up one tier.
                // Judge it before the restore (the laid cards vanish in it) and
                // latch it after, exactly like the harness driver's restartTurn.
                let failed = game.state.restart_penalizes_opening(game.state.turn);
                let team = game.state.turn.team();
                game.state = game.turn_start.clone();
                if failed {
                    game.state.penalize_opening(team);
                }
                game.safe = true;
                menus[row] = game.menu();
                debug_assert!(!menus[row].is_empty(), "even the safe retry dead-ended");
            }
        }
        self.menus = menus;

        let rows = self.pending.len();
        let width = self.menus.iter().map(Vec::len).max().unwrap_or(1).max(1);
        let obs = PyArray2::<f32>::zeros(py, [rows, OBS_DIM], false);
        let acts = PyArray3::<f32>::zeros(py, [rows, width, ACT_DIM], false);
        let mask = PyArray2::<bool>::zeros(py, [rows, width], false);
        let grid = PyArray2::<i64>::zeros(py, [rows, 2], false);

        // Encode (views + features) in parallel into per-row scratch, then
        // copy into the numpy buffers — enumerate/apply dominate; the copy is
        // memcpy-cheap and keeps the numpy API single-threaded.
        let encoded: Vec<(Vec<f32>, Vec<f32>)> = self
            .pending
            .par_iter()
            .zip(self.menus.par_iter())
            .map(|(&i, menu)| {
                let view = observe(&self.games[i].state, self.games[i].state.turn);
                let mut obs_row = vec![0.0; OBS_DIM];
                encode_observation(&view, &mut obs_row);
                let mut act_rows = vec![0.0; menu.len() * ACT_DIM];
                encode_actions(&view, menu, &mut act_rows);
                (obs_row, act_rows)
            })
            .collect();

        {
            let mut obs_view = unsafe { obs.as_array_mut() };
            let mut acts_view = unsafe { acts.as_array_mut() };
            let mut mask_view = unsafe { mask.as_array_mut() };
            let mut rows_view = unsafe { grid.as_array_mut() };
            for (row, (obs_row, act_rows)) in encoded.iter().enumerate() {
                let game = self.pending[row];
                rows_view[[row, 0]] = game as i64;
                rows_view[[row, 1]] = self.games[game].state.turn.index() as i64;
                obs_view
                    .row_mut(row)
                    .assign(&ndarray::ArrayView1::from(obs_row));
                for (col, chunk) in act_rows.chunks(ACT_DIM).enumerate() {
                    let mut acts_row = acts_view.index_axis_mut(ndarray::Axis(0), row);
                    acts_row
                        .index_axis_mut(ndarray::Axis(0), col)
                        .assign(&ndarray::ArrayView1::from(chunk));
                    mask_view[[row, col]] = true;
                }
            }
        }

        Ok((obs, acts, mask, grid))
    }

    /// Play the picked menu index on every pending row.
    fn apply(&mut self, picks: Vec<usize>) -> PyResult<()> {
        if picks.len() != self.pending.len() {
            return Err(PyValueError::new_err(format!(
                "expected {} picks, got {}",
                self.pending.len(),
                picks.len()
            )));
        }
        for (row, pick) in picks.into_iter().enumerate() {
            let game = &mut self.games[self.pending[row]];
            let action = self.menus[row]
                .get(pick)
                .ok_or_else(|| PyValueError::new_err("menu index out of range"))?
                .clone();
            // Refresh the turn checkpoint on entry to a fresh turn. Note the
            // deliberate divergence from the harness driver here: we also
            // re-capture on `AwaitingRefusalChoice`, so a red-3 replacement
            // turn replays from the post-draw position with the red 3 already
            // on the table, rather than replaying the draw. Safe-clearing is
            // independent of this and happens on turn end below.
            if matches!(
                game.state.phase,
                Phase::AwaitingDraw | Phase::AwaitingRefusalChoice
            ) {
                game.turn_start = game.state.clone();
            }
            // The safe flag lasts the whole retried turn — the harness driver
            // clears safeMode the same way, on the turn-ending action.
            // Clearing it earlier (e.g. on the retry's own Draw) would hand
            // the full meld menu straight back and reproduce the dead-end.
            if matches!(
                action,
                Action::Discard { .. } | Action::EndTurnWithoutDiscard
            ) {
                game.safe = false;
            }
            game.state = apply(&game.state, game.state.turn, &action)
                .map_err(|violation| PyValueError::new_err(violation.to_string()))?;
            if game.state.phase == Phase::HandOver {
                game.state = settle_hand(&game.state)
                    .map_err(|violation| PyValueError::new_err(violation.to_string()))?;
                if game.state.phase == Phase::MatchOver {
                    game.result = Some((
                        game.seed,
                        game.state.scores,
                        game.state.winner().map(|team| team.index() as u8),
                        game.hands,
                        false,
                    ));
                } else {
                    game.hands += 1;
                }
            }
            game.actions_played += 1;
            // A live game that blows past its action ceiling is ended as
            // unfinished rather than left to straggle — one pathological
            // genome pair must not hang a generation.
            if game.result.is_none() {
                if let Some(cap) = self.max_actions_per_game {
                    if game.actions_played >= cap {
                        game.result = Some((game.seed, game.state.scores, None, game.hands, true));
                    }
                }
            }
        }
        Ok(())
    }

    /// The matches that ended: `(seed, scores, winner, hands, unfinished)`.
    fn results(&self) -> Vec<MatchResult> {
        self.games.iter().filter_map(|game| game.result).collect()
    }

    /// The action kinds of the current `encode` menu, one list per pending row,
    /// so a policy can tell what a mask column means. Aligns with the last
    /// `encode()` call (and with `apply`'s picks for the same rows).
    fn menu_kinds(&self) -> Vec<Vec<String>> {
        self.menus
            .iter()
            .map(|menu| menu.iter().map(action_kind).map(str::to_owned).collect())
            .collect()
    }
}

#[pymodule]
fn canastra_py(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Pool>()?;
    module.add("OBS_DIM", OBS_DIM)?;
    module.add("ACT_DIM", ACT_DIM)?;
    Ok(())
}
