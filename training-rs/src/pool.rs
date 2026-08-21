//! Batched game stepping — owns N engines, drives them a ply at a time.
//!
//! Ported from `training/src/lib.rs` (the PyO3 extension), but as a native
//! Rust library returning owned arrays instead of numpy PyArrays. The GIL
//! no longer exists — all encode/apply runs in native threads with rayon.

use canastra_encode::{encode_actions, encode_observation, ACT_DIM, OBS_DIM};
use canastra_engine::{
    apply, enumerate, new_game, observe, settle_hand, Action, GameState, Phase, RuleViolation,
};
use rayon::prelude::*;

/// A match that ended: (seed, scores, winner, hands, unfinished).
pub type MatchResult = (u64, [i32; 2], Option<u8>, u32, bool);

struct Game {
    state: GameState,
    turn_start: GameState,
    safe: bool,
    seed: u64,
    hands: u32,
    actions_played: u64,
    result: Option<MatchResult>,
    max_hands: Option<u32>,
}

impl Game {
    fn new(seed: u64, max_hands: Option<u32>) -> Self {
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

    fn menu(&self) -> Vec<Action> {
        let legal = enumerate(&self.state, self.state.turn);
        if !self.safe {
            return legal;
        }
        legal
            .into_iter()
            .filter(|a| {
                matches!(
                    a,
                    Action::Draw
                        | Action::KeepDrawnCard
                        | Action::RefuseDrawnCard
                        | Action::Discard { .. }
                        | Action::EndTurnWithoutDiscard
                )
            })
            .collect()
    }

    /// Commit the picked action. Returns `Ok(true)` when this call finished
    /// the game (set its result), so the caller can maintain its finished
    /// counter without re-scanning every game.
    fn commit_selected(
        &mut self,
        action: &Action,
        next_state: Result<GameState, RuleViolation>,
        max_actions_per_game: Option<u64>,
    ) -> Result<bool, RuleViolation> {
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
        let mut finished = false;
        if self.state.phase == Phase::HandOver {
            self.state = settle_hand(&self.state)?;
            if self.state.phase == Phase::MatchOver {
                self.result = Some((
                    self.seed,
                    self.state.scores,
                    self.state.winner().map(|t| t.index() as u8),
                    self.hands,
                    false,
                ));
                finished = true;
            } else if let Some(max) = self.max_hands {
                if self.hands >= max {
                    self.result = Some((self.seed, self.state.scores, None, self.hands, false));
                    return Ok(true);
                }
                self.hands += 1;
            } else {
                self.hands += 1;
            }
        }
        self.actions_played += 1;
        if !finished {
            if let Some(cap) = max_actions_per_game {
                if self.actions_played >= cap {
                    self.result = Some((self.seed, self.state.scores, None, self.hands, true));
                    finished = true;
                }
            }
        }
        Ok(finished)
    }
}

/// One batched ply's encode output: observation/action/mask/row arrays.
/// The legal actions per row are held by `Pool::menus` (read by `apply`),
/// not duplicated here — cloning them every ply was pure waste (at pop=500
/// the menus are ~11 MB/ply, ~2.3 GB of allocation churn per generation).
///
/// Both `obs` and `acts` are **100% binary** (one-hot/thermometer/bits —
/// verified in `task3a-sparsity.md`). Storing as u8 cuts H2D transfer volume
/// 4x with zero precision loss; the forward path casts device-side via
/// `to_dtype` (u8→bf16 on CUDA, u8→f32 on CPU).
///
/// In addition to the full `obs` (kept for the dense test path), the sparse
/// split (`obs_sparse_idx` + `obs_dense`) is produced for the embedding-bag
/// training path — see `src/sparse.rs`.
pub struct EncodedPly {
    pub obs: Vec<u8>,              // [N, OBS_DIM] row-major, binary (dense path)
    pub obs_sparse_idx: Vec<u32>,  // [N, MAX_NNZ] sparse local indices, padded with 1925
    pub obs_dense: Vec<u8>,        // [N, DENSE_WIDTH] dense feature values, binary
    pub acts: Vec<u8>,             // [N, width, ACT_DIM] row-major, binary
    pub mask: Vec<bool>,           // [N, width]
    pub rows: Vec<(usize, usize)>, // (game_index, seat) per row
    pub width: usize,
}

/// The batched game pool — owns N engines, drives them a ply at a time.
pub struct Pool {
    games: Vec<Game>,
    pending: Vec<usize>,
    menus: Vec<Vec<Action>>,
    max_actions_per_game: Option<u64>,
    max_width: usize,
    /// Games that have finished so far (result set). Maintained incrementally
    /// so the live dashboard can read progress in O(1) per ply.
    n_finished: usize,
}

impl Pool {
    pub fn new(seeds: Vec<u64>, max_actions_per_game: Option<u64>, max_hands: Option<u32>) -> Self {
        Self::with_max_width(seeds, max_actions_per_game, max_hands, usize::MAX)
    }

    /// Create a pool with a cap on the menu width (max legal actions per row).
    /// Menus longer than `max_width` are truncated — the bot picks from the
    /// first `max_width` actions. This cuts the `acts` tensor transfer volume
    /// at peak plies (where width can reach 250 but mean is ~15).
    pub fn with_max_width(
        seeds: Vec<u64>,
        max_actions_per_game: Option<u64>,
        max_hands: Option<u32>,
        max_width: usize,
    ) -> Self {
        Pool {
            games: seeds.into_iter().map(|s| Game::new(s, max_hands)).collect(),
            pending: Vec::new(),
            menus: Vec::new(),
            max_actions_per_game,
            max_width,
            n_finished: 0,
        }
    }

    pub fn has_live(&self) -> bool {
        self.games.iter().any(Game::live)
    }

    pub fn n_games(&self) -> usize {
        self.games.len()
    }

    /// Encode every seat awaiting a decision into flat arrays.
    pub fn encode(&mut self) -> EncodedPly {
        self.pending = (0..self.games.len())
            .filter(|&i| {
                self.games[i].live()
                    && matches!(
                        self.games[i].state.phase,
                        Phase::AwaitingDraw | Phase::AwaitingRefusalChoice | Phase::Melding,
                    )
            })
            .collect();

        // Menu enumeration — parallel via rayon (no GIL!).
        let pending = self.pending.clone();
        let games: &[Game] = &self.games;
        let mut menus: Vec<Vec<Action>> = pending.par_iter().map(|&i| games[i].menu()).collect();

        // Safe-mode backout (rare).
        for (row, &game) in self.pending.iter().enumerate() {
            if menus[row].is_empty() {
                let g = &mut self.games[game];
                let failed = g.state.restart_penalizes_opening(g.state.turn);
                let team = g.state.turn.team();
                g.state = g.turn_start.clone();
                if failed {
                    g.state.penalize_opening(team);
                }
                g.safe = true;
                menus[row] = g.menu();
                debug_assert!(!menus[row].is_empty(), "even the safe retry dead-ended");
            }
        }

        // Cap menu width: truncate menus longer than max_width. The bot picks
        // from the first max_width actions. Cuts acts tensor volume at peak
        // plies (width can reach 250 but mean is ~15).
        if self.max_width < usize::MAX {
            for menu in &mut menus {
                if menu.len() > self.max_width {
                    menu.truncate(self.max_width);
                }
            }
        }

        let n_rows = self.pending.len();
        let width = self.menus_iter_max(&menus).max().unwrap_or(1).max(1);

        let mut obs = vec![0u8; n_rows * OBS_DIM];
        let mut acts = vec![0u8; n_rows * width * ACT_DIM];
        let mut mask = vec![false; n_rows * width];

        // Encode in parallel (no GIL!). Each thread reuses an f32 scratch
        // buffer (the encoder API requires &mut [f32]); the binary result is
        // cast to u8 into the big buffers, keeping the H2D transfer at 1 byte
        // per feature instead of 4.
        let pending = self.pending.clone();
        let menus_ref = &menus;
        let games: &[Game] = &self.games;
        let obs_ptr = obs.as_mut_ptr() as usize;
        let acts_ptr = acts.as_mut_ptr() as usize;
        let mask_ptr = mask.as_mut_ptr() as usize;

        let row_data: Vec<(usize, usize)> = pending
            .par_iter()
            .enumerate()
            .zip(menus_ref.par_iter())
            .map_with(
                (
                    Vec::<f32>::with_capacity(OBS_DIM),
                    Vec::<f32>::with_capacity(64 * ACT_DIM),
                ),
                |(obs_f32, acts_f32), ((row, &game), menu)| {
                    let view = observe(&games[game].state, games[game].state.turn);
                    obs_f32.clear();
                    obs_f32.resize(OBS_DIM, 0.0);
                    acts_f32.clear();
                    acts_f32.resize(menu.len() * ACT_DIM, 0.0);
                    encode_observation(&view, obs_f32);
                    encode_actions(&view, menu, acts_f32);
                    // SAFETY: each row's range is unique to this closure.
                    unsafe {
                        let obs_row = std::slice::from_raw_parts_mut(
                            (obs_ptr as *mut u8).add(row * OBS_DIM),
                            OBS_DIM,
                        );
                        for (i, &v) in obs_f32.iter().enumerate() {
                            obs_row[i] = v as u8;
                        }
                        let acts_row = std::slice::from_raw_parts_mut(
                            (acts_ptr as *mut u8).add(row * width * ACT_DIM),
                            width * ACT_DIM,
                        );
                        for (i, &v) in acts_f32.iter().enumerate() {
                            acts_row[i] = v as u8;
                        }
                        let mask_row = std::slice::from_raw_parts_mut(
                            (mask_ptr as *mut bool).add(row * width),
                            width,
                        );
                        mask_row[..menu.len()].fill(true);
                    }
                    (game, games[game].state.turn.index())
                },
            )
            .collect();

        self.menus = menus;
        let rows = row_data;

        EncodedPly {
            obs,
            obs_sparse_idx: Vec::new(),
            obs_dense: Vec::new(),
            acts,
            mask,
            rows,
            width,
        }
    }

    /// Apply the picked menu index on every pending row.
    pub fn apply(&mut self, picks: &[usize]) -> Result<(), RuleViolation> {
        if picks.len() != self.pending.len() {
            return Err(RuleViolation::HandIsOver); // closest error
        }
        let max_actions = self.max_actions_per_game;
        let mut selected: Vec<Option<Action>> = vec![None; self.games.len()];
        for (row, &pick) in picks.iter().enumerate() {
            let game_index = self.pending[row];
            let action = self.menus[row]
                .get(pick)
                .cloned()
                .ok_or(RuleViolation::HandIsOver)?;
            selected[game_index] = Some(action);
        }

        // Parallel apply (no GIL!).
        let selected_ref = &selected;
        let mut applied: Vec<(usize, Result<GameState, RuleViolation>)> = self
            .games
            .par_iter()
            .enumerate()
            .filter_map(|(i, g)| {
                let action = selected_ref[i].as_ref()?;
                Some((i, apply(&g.state, g.state.turn, action)))
            })
            .collect();
        applied.sort_unstable_by_key(|(g, _)| *g);

        for (i, next_state) in applied {
            let action = selected[i].as_ref().expect("selected action");
            let finished = self.games[i].commit_selected(action, next_state, max_actions)?;
            if finished {
                self.n_finished += 1;
            }
        }
        Ok(())
    }

    pub fn results(&self) -> Vec<MatchResult> {
        self.games.iter().filter_map(|g| g.result).collect()
    }

    /// Number of games that have finished so far (O(1)).
    pub fn finished_count(&self) -> usize {
        self.n_finished
    }

    /// Every game's result **by game index** — `None` for games still in
    /// play. Unlike [`Pool::results`], this preserves the index alignment the
    /// league's `meta` uses, which is what the live dashboard needs to
    /// attribute partial results to individuals mid-generation.
    pub fn snapshot_results(&self) -> Vec<Option<MatchResult>> {
        self.games.iter().map(|g| g.result).collect()
    }

    /// The legal actions per row from the last `encode()` call. Read-only
    /// accessor for tests that need to inspect the menu (e.g. width-cap
    /// diversity checks). `apply()` consumes these by index.
    pub fn menus(&self) -> &[Vec<Action>] {
        &self.menus
    }

    fn menus_iter_max<'a>(&self, menus: &'a [Vec<Action>]) -> impl Iterator<Item = usize> + 'a {
        menus.iter().map(Vec::len)
    }
}
