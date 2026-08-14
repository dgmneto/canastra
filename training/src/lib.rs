//! Python bindings for driving batches of Canastra games.
//!
//! The pool owns one engine per seed and answers three questions: which
//! seats are waiting on a decision, what those decisions look like as tensors
//! (two batched crossings per ply, buffers filled in Rust), and what happened when
//! a match ended. The engine remains the only referee — nothing here knows a
//! rule.

use canastra_encode::{encode_actions, encode_observation, ACT_DIM, OBS_DIM};
use canastra_engine::{
    apply, enumerate, new_game, observe, settle_hand, Action, GameState, Phase, RuleViolation,
};
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
    /// Stop after this many hands have settled (scores banked). `None` plays
    /// full matches to MatchOver. A hand is guaranteed to finish in finite
    /// time (the stock depletes), so no action cap is needed when this is set.
    max_hands: Option<u32>,
}

impl Game {
    fn new(seed: u64, max_hands: Option<u32>) -> Game {
        let state = new_game(seed);
        Game {
            turn_start: state.clone(),
            state,
            safe: false,
            seed,
            hands: 1,
            actions_played: 0,
            result: None,
            max_hands,
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

    fn apply_selected(
        &mut self,
        action: &Action,
        max_actions_per_game: Option<u64>,
    ) -> Result<(), RuleViolation> {
        let next_state = apply(&self.state, self.state.turn, action);
        self.commit_selected(action, next_state, max_actions_per_game)
    }

    fn commit_selected(
        &mut self,
        action: &Action,
        next_state: Result<GameState, RuleViolation>,
        max_actions_per_game: Option<u64>,
    ) -> Result<(), RuleViolation> {
        // Keep the whole transition together: checkpointing, safe-mode, settlement,
        // and the cap all belong to this game's action, not to the batch.
        if matches!(
            self.state.phase,
            Phase::AwaitingDraw | Phase::AwaitingRefusalChoice
        ) {
            self.turn_start = self.state.clone();
        }
        if matches!(
            action,
            Action::Discard { .. } | Action::EndTurnWithoutDiscard
        ) {
            self.safe = false;
        }
        self.state = next_state?;
        if self.state.phase == Phase::HandOver {
            self.state = settle_hand(&self.state)?;
            if self.state.phase == Phase::MatchOver {
                self.result = Some((
                    self.seed,
                    self.state.scores,
                    self.state.winner().map(|team| team.index() as u8),
                    self.hands,
                    false,
                ));
            } else if let Some(max) = self.max_hands {
                if self.hands >= max {
                    // The hand just settled and the scores are banked in
                    // `self.state.scores`. Stop here — the next hand would
                    // start, but we only wanted `max` hands.
                    self.result =
                        Some((self.seed, self.state.scores, None, self.hands, false));
                    return Ok(());
                }
                self.hands += 1;
            } else {
                self.hands += 1;
            }
        }
        self.actions_played += 1;
        // A live game that blows past its action ceiling is ended as unfinished
        // rather than left to straggle — one pathological genome pair must not
        // hang a generation.
        if self.result.is_none() {
            if let Some(cap) = max_actions_per_game {
                if self.actions_played >= cap {
                    self.result = Some((self.seed, self.state.scores, None, self.hands, true));
                }
            }
        }
        Ok(())
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
    /// Stop after this many hands have settled. `None` plays full matches.
    max_hands: Option<u32>,
}

#[pymethods]
impl Pool {
    #[new]
    #[pyo3(signature = (seeds, max_actions_per_game=None, max_hands=None))]
    fn new(
        seeds: Vec<u64>,
        max_actions_per_game: Option<u64>,
        max_hands: Option<u32>,
    ) -> Pool {
        Pool {
            games: seeds.into_iter().map(|s| Game::new(s, max_hands)).collect(),
            pending: Vec::new(),
            menus: Vec::new(),
            max_actions_per_game,
            max_hands,
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

        // Menus first (enumerate is the expensive part — parallel, GIL-free).
        // We split the computation: the menu enumeration is read-only on
        // self.games, so it runs inside py.allow_threads, releasing the GIL
        // so a GPU forward thread can run concurrently. Safe-mode backout
        // (rare, needs &mut self.games) runs after, GIL held.
        let pending = self.pending.clone();
        let games = &self.games[..];

        let mut menus: Vec<Vec<Action>> = py.allow_threads(move || {
            pending
                .par_iter()
                .map(|&i| games[i].menu())
                .collect()
        });

        // Safe-mode backout (GIL held, rare — only when a turn dead-ended).
        for (row, &game) in self.pending.iter().enumerate() {
            if menus[row].is_empty() {
                let game = &mut self.games[game];
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

        // Encode directly into the disjoint NumPy rows. Keeping the arrays
        // NumPy-owned avoids per-row scratch vectors and serial row copies.
        // Released GIL: the array pointers are valid (arrays live until the
        // function returns), each row's range is unique, and the GIL prevents
        // Python from accessing the arrays during the writes.
        let obs_ptr = obs.data() as usize;
        let acts_ptr = acts.data() as usize;
        let mask_ptr = mask.data() as usize;
        let grid_ptr = grid.data() as usize;
        let pending = self.pending.clone();
        let menus = self.menus.clone();
        let games = &self.games[..];

        py.allow_threads(move || {
            pending
                .par_iter()
                .enumerate()
                .zip(menus.par_iter())
                .for_each(|((row, &game), menu)| {
                    // SAFETY: each row range is unique to this closure, the arrays
                    // stay alive until the parallel loop completes, and the GIL
                    // prevents Python from accessing them during the writes.
                    unsafe {
                        let obs_row = std::slice::from_raw_parts_mut(
                            (obs_ptr as *mut f32).add(row * OBS_DIM),
                            OBS_DIM,
                        );
                        let acts_row = std::slice::from_raw_parts_mut(
                            (acts_ptr as *mut f32).add(row * width * ACT_DIM),
                            width * ACT_DIM,
                        );
                        let mask_row = std::slice::from_raw_parts_mut(
                            (mask_ptr as *mut bool).add(row * width),
                            width,
                        );
                        let grid_row =
                            std::slice::from_raw_parts_mut((grid_ptr as *mut i64).add(row * 2), 2);
                        let view = observe(&games[game].state, games[game].state.turn);
                        encode_observation(&view, obs_row);
                        encode_actions(&view, menu, &mut acts_row[..menu.len() * ACT_DIM]);
                        mask_row[..menu.len()].fill(true);
                        grid_row[0] = game as i64;
                        grid_row[1] = games[game].state.turn.index() as i64;
                    }
                });
        });

        Ok((obs, acts, mask, grid))
    }

    /// Play the picked menu index on every pending row.
    fn apply(&mut self, py: Python<'_>, picks: Vec<usize>) -> PyResult<()> {
        if picks.len() != self.pending.len() {
            return Err(PyValueError::new_err(format!(
                "expected {} picks, got {}",
                self.pending.len(),
                picks.len()
            )));
        }
        let max_actions_per_game = self.max_actions_per_game;
        let mut selected = vec![None; self.games.len()];
        for (row, pick) in picks.into_iter().enumerate() {
            let game_index = self.pending[row];
            let action = match self.menus[row].get(pick) {
                Some(action) => action.clone(),
                None => {
                    // Preserve the old error path: rows before the bad pick
                    // have already been applied when the error is reported.
                    for &previous_game in self.pending.iter().take(row) {
                        let action = selected[previous_game].take().expect("selected action");
                        self.games[previous_game]
                            .apply_selected(&action, max_actions_per_game)
                            .map_err(|violation| PyValueError::new_err(violation.to_string()))?;
                    }
                    return Err(PyValueError::new_err("menu index out of range"));
                }
            };
            selected[game_index] = Some(action);
        }

        // The parallel apply is CPU-heavy and GIL-free. Releasing the GIL
        // here lets a GPU forward thread run concurrently during pipelined
        // single-process training.
        let games = &self.games[..];
        let selected_ref = &selected[..];
        let mut applied: Vec<(usize, Result<GameState, RuleViolation>)> =
            py.allow_threads(move || {
                games
                    .par_iter()
                    .enumerate()
                    .filter_map(|(game_index, game)| {
                        let action = selected_ref[game_index].as_ref()?;
                        Some((game_index, apply(&game.state, game.state.turn, action)))
                    })
                    .collect()
            });
        applied.sort_unstable_by_key(|(game, _)| *game);
        for (game_index, next_state) in applied {
            let action = selected[game_index].as_ref().expect("selected action");
            if let Err(violation) =
                self.games[game_index].commit_selected(action, next_state, max_actions_per_game)
            {
                return Err(PyValueError::new_err(violation.to_string()));
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

    /// Limit the number of threads rayon uses for parallel encode/apply.
    /// Call once per process, before creating a Pool. With N shard workers
    /// each spawning rayon threads, the default (one per core) oversubscribes
    /// the CPU; this lets the caller cap it (e.g. 2 threads per shard).
    #[pyfn(module)]
    fn set_rayon_threads(n: usize) -> PyResult<()> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    Ok(())
}
