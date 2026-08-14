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
                    self.state.winner().map(|t| t.index() as u8),
                    self.hands,
                    false,
                ));
            } else if let Some(max) = self.max_hands {
                if self.hands >= max {
                    self.result = Some((self.seed, self.state.scores, None, self.hands, false));
                    return Ok(());
                }
                self.hands += 1;
            } else {
                self.hands += 1;
            }
        }
        self.actions_played += 1;
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

/// One batched ply's encode output: observation/action/mask/row arrays.
pub struct EncodedPly {
    pub obs: Vec<f32>,             // [N, OBS_DIM] row-major
    pub acts: Vec<f32>,            // [N, width, ACT_DIM] row-major
    pub mask: Vec<bool>,           // [N, width]
    pub rows: Vec<(usize, usize)>, // (game_index, seat) per row
    pub width: usize,
    pub menus: Vec<Vec<Action>>, // the legal actions per row (for apply)
}

/// The batched game pool — owns N engines, drives them a ply at a time.
pub struct Pool {
    games: Vec<Game>,
    pending: Vec<usize>,
    menus: Vec<Vec<Action>>,
    max_actions_per_game: Option<u64>,
    max_hands: Option<u32>,
}

impl Pool {
    pub fn new(seeds: Vec<u64>, max_actions_per_game: Option<u64>, max_hands: Option<u32>) -> Self {
        Pool {
            games: seeds.into_iter().map(|s| Game::new(s, max_hands)).collect(),
            pending: Vec::new(),
            menus: Vec::new(),
            max_actions_per_game,
            max_hands,
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

        let n_rows = self.pending.len();
        let width = self.menus_iter_max(&menus).max().unwrap_or(1).max(1);

        let mut obs = vec![0.0f32; n_rows * OBS_DIM];
        let mut acts = vec![0.0f32; n_rows * width * ACT_DIM];
        let mut mask = vec![false; n_rows * width];
        let mut rows = Vec::with_capacity(n_rows);

        // Encode in parallel (no GIL!).
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
            .map(|((row, &game), menu)| {
                let view = observe(&games[game].state, games[game].state.turn);
                // SAFETY: each row's range is unique to this closure.
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
                    encode_observation(&view, obs_row);
                    encode_actions(&view, menu, &mut acts_row[..menu.len() * ACT_DIM]);
                    mask_row[..menu.len()].fill(true);
                }
                (game, games[game].state.turn.index())
            })
            .collect();

        self.menus = menus;
        rows = row_data;

        EncodedPly {
            obs,
            acts,
            mask,
            rows,
            width,
            menus: self.menus.clone(),
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
        let games: &[Game] = &self.games;
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
            self.games[i].commit_selected(action, next_state, max_actions)?;
        }
        Ok(())
    }

    pub fn results(&self) -> Vec<MatchResult> {
        self.games.iter().filter_map(|g| g.result).collect()
    }

    fn menus_iter_max<'a>(&self, menus: &'a [Vec<Action>]) -> impl Iterator<Item = usize> + 'a {
        menus.iter().map(Vec::len)
    }
}
