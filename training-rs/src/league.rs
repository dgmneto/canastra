//! Self-play league: batched pairings and the ply driver.
//!
//! Two rollout paths:
//!
//! - **lockstep** (default for pop ≤ 500): a single `Pool` drives all games in
//!   lockstep. One forward per ply, weights read once per present genome, one
//!   sync point per ply. No channels, no mutex, no worker threads. Best when the
//!   per-ply batch fits comfortably in VRAM and the grouped matmul is efficient.
//!
//! - **coalesced** (default for pop > 500): a `GpuServer` thread caches the full
//!   weight stack on the device; N worker threads each drive a slice of games,
//!   submitting forward requests through a channel. Encode/apply overlaps with
//!   GPU computation. Best when the population is large enough that the lockstep
//!   per-ply transfer volume dominates.
//!
//! Both paths share the same `forward_picks`, `schedule_pairings`, `batch_layout`,
//! and ELO computation.

use crate::elo::EloTracker;
use crate::ga::HallOfFame;
use crate::genome::{Arch, Genome};
use crate::policy::{forward_picks, CpuRoster, WeightStack};
use crate::pool::{EncodedPly, MatchResult, Pool};
use candle_core::{DType, Device, Tensor};
use rand::rngs::StdRng;
use rand::Rng;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;

#[cfg(feature = "profile")]
use crate::profile;

type Pairing = (usize, usize);
type GameMeta = (usize, usize, usize);

/// Which rollout path to use.
#[derive(Clone, Copy, Debug)]
pub enum Rollout {
    Lockstep,
    Coalesced,
}

impl Rollout {
    /// Default selection per population size: lockstep for small populations
    /// (where the per-ply batch is small), coalesced for large (where the
    /// lockstep's transfer volume and lost encode/apply overlap dominate).
    /// This heuristic is temporary — once one path dominates everywhere, the
    /// flag goes away.
    pub fn default_for(pop: usize) -> Self {
        if pop <= 500 {
            Rollout::Lockstep
        } else {
            Rollout::Coalesced
        }
    }
}

pub fn schedule_pairings(
    pop_size: usize,
    opponents: usize,
    hof: &HallOfFame,
    rng: &mut StdRng,
) -> Vec<Pairing> {
    let mut pairings = Vec::new();
    for me in 0..pop_size {
        let others: Vec<usize> = (0..pop_size).filter(|&i| i != me).collect();
        let n = opponents.min(others.len());
        let mut indices: Vec<usize> = (0..others.len()).collect();
        for i in (1..indices.len()).rev() {
            let j = (rng.gen::<u32>() as usize) % (i + 1);
            indices.swap(i, j);
        }
        let mut chosen: Vec<usize> = indices[..n].iter().map(|&i| others[i]).collect();
        if !hof.is_empty() && n > 0 {
            chosen[n - 1] = pop_size + (rng.gen::<u32>() as usize) % hof.len();
        }
        for opp in chosen {
            pairings.push((me, opp));
        }
    }
    pairings
}

pub fn batch_layout(
    pop: &[Genome],
    hof: &HallOfFame,
    pairings: &[Pairing],
    seeds: &[u64],
) -> (Vec<Genome>, Vec<u64>, Vec<GameMeta>) {
    let mut roster: Vec<Genome> = pop.to_vec();
    roster.extend(hof.genomes.iter().cloned());
    let mut game_seeds = Vec::new();
    let mut meta = Vec::new();
    for &(a, b) in pairings {
        for &seed in seeds {
            game_seeds.push(seed);
            game_seeds.push(seed);
            meta.push((a, b, 0));
            meta.push((a, b, 1));
        }
    }
    (roster, game_seeds, meta)
}

/// Drive all games in lockstep with a single `Pool`. One forward per ply.
fn rollout_lockstep(
    roster: &[Genome],
    arch: &Arch,
    game_seeds: Vec<u64>,
    meta: &[GameMeta],
    max_hands: Option<u32>,
    device: &Device,
    max_width: usize,
) -> Vec<MatchResult> {
    let dtype = match device {
        Device::Cuda(_) => DType::BF16,
        _ => DType::F32,
    };
    let roster_refs: Vec<&Genome> = roster.iter().collect();
    let stack = WeightStack::from_roster(&roster_refs, arch, device, dtype);
    let obs_dim = arch.obs;
    let act_dim = arch.act;

    let mut pool = Pool::with_max_width(game_seeds, None, max_hands, max_width);

    while pool.has_live() {
        #[cfg(feature = "profile")]
        profile::WKR_PLIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        #[cfg(feature = "profile")]
        let _enc = profile::Span::new(&profile::WKR_ENCODE_NS);
        let encoded = pool.encode();
        #[cfg(feature = "profile")]
        drop(_enc);

        let n_rows = encoded.rows.len();
        if n_rows == 0 {
            break;
        }

        let genome_idx: Vec<usize> = encoded
            .rows
            .iter()
            .map(|&(game_idx, seat)| {
                let (a, b, seating) = meta[game_idx];
                if (seat % 2 == 0) == (seating == 0) {
                    a
                } else {
                    b
                }
            })
            .collect();

        #[cfg(feature = "profile")]
        let _h2d = profile::Span::new(&profile::SRV_H2D_NS);
        let obs = Tensor::from_vec(encoded.obs, (n_rows, obs_dim), device)
            .unwrap_or_else(|e| panic!("obs tensor: {e}"));
        let acts = Tensor::from_vec(encoded.acts, (n_rows, encoded.width, act_dim), device)
            .unwrap_or_else(|e| panic!("acts tensor: {e}"));
        let mask = Tensor::from_vec(
            encoded
                .mask
                .iter()
                .map(|&v| if v { 1u32 } else { 0u32 })
                .collect::<Vec<_>>(),
            (n_rows, encoded.width),
            device,
        )
        .unwrap_or_else(|e| panic!("mask tensor: {e}"));
        #[cfg(feature = "profile")]
        drop(_h2d);

        #[cfg(feature = "profile")]
        {
            profile::SRV_REQS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            profile::SRV_ROWS.fetch_add(n_rows as u64, std::sync::atomic::Ordering::Relaxed);
        }

        #[cfg(feature = "profile")]
        let _gpu = profile::Span::new(&profile::SRV_GPU_NS);
        let picks = forward_picks(&stack, &obs, &acts, &mask, &genome_idx);
        #[cfg(feature = "profile")]
        drop(_gpu);

        #[cfg(feature = "profile")]
        let _app = profile::Span::new(&profile::WKR_APPLY_NS);
        let _ = pool.apply(&picks);
        #[cfg(feature = "profile")]
        drop(_app);
    }

    pool.results()
}

// ── Coalesced rollout: GpuServer + N worker threads ──────────────────────

/// One forward request from a worker to the GPU server.
struct GpuRequest {
    obs: Vec<f32>,
    acts: Vec<f32>,
    mask: Vec<bool>,
    width: usize,
    genome_idx: Vec<usize>,
    response_tx: mpsc::Sender<Vec<usize>>,
}

/// The coalesced GPU server: ONE thread, ONE cached weight stack, shared by
/// all workers via `Arc<GpuServer>`. Workers submit forward requests through
/// a `Mutex<Sender>` and block on a per-request response channel — the GPU
/// processes them serially in one CUDA context.
pub struct GpuServer {
    tx: Arc<Mutex<Option<mpsc::Sender<GpuRequest>>>>,
    _handle: Option<thread::JoinHandle<()>>,
}

impl GpuServer {
    pub fn new(roster: Arc<CpuRoster>, device: Device, obs_dim: usize, act_dim: usize) -> Self {
        let (tx, rx) = mpsc::channel::<GpuRequest>();
        let tx = Arc::new(Mutex::new(Some(tx)));
        let tx_clone = tx.clone();
        let handle = thread::spawn(move || {
            let n_genomes = roster.n_genomes();
            let cached_stack: Option<WeightStack> = if n_genomes <= 2000 {
                let all: Vec<usize> = (0..n_genomes).collect();
                roster.build_chunk(&all, &device).ok()
            } else {
                None
            };

            loop {
                #[cfg(feature = "profile")]
                let _idle = profile::Span::new(&profile::SRV_IDLE_NS);
                let req = match rx.recv() {
                    Ok(r) => r,
                    Err(_) => break,
                };
                #[cfg(feature = "profile")]
                drop(_idle);

                let n_rows = req.obs.len() / obs_dim;
                if n_rows == 0 {
                    let _ = req.response_tx.send(Vec::new());
                    continue;
                }

                #[cfg(feature = "profile")]
                {
                    profile::SRV_REQS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    profile::SRV_ROWS
                        .fetch_add(n_rows as u64, std::sync::atomic::Ordering::Relaxed);
                }

                #[cfg(feature = "profile")]
                let _h2d = profile::Span::new(&profile::SRV_H2D_NS);
                let obs = Tensor::from_vec(req.obs, (n_rows, obs_dim), &device)
                    .unwrap_or_else(|e| panic!("obs tensor: {e}"));
                let acts = Tensor::from_vec(req.acts, (n_rows, req.width, act_dim), &device)
                    .unwrap_or_else(|e| panic!("acts tensor: {e}"));
                let mask = Tensor::from_vec(
                    req.mask
                        .iter()
                        .map(|&v| if v { 1u32 } else { 0u32 })
                        .collect::<Vec<_>>(),
                    (n_rows, req.width),
                    &device,
                )
                .unwrap_or_else(|e| panic!("mask tensor: {e}"));
                #[cfg(feature = "profile")]
                drop(_h2d);

                #[cfg(feature = "profile")]
                let _gpu = profile::Span::new(&profile::SRV_GPU_NS);
                let picks = if let Some(ref stack) = cached_stack {
                    forward_picks(stack, &obs, &acts, &mask, &req.genome_idx)
                } else {
                    // Fallback: build a chunk from the roster for present genomes.
                    let mut present_set = vec![false; n_genomes];
                    for &g in &req.genome_idx {
                        present_set[g] = true;
                    }
                    let present: Vec<usize> = (0..n_genomes).filter(|&g| present_set[g]).collect();
                    let stack = roster.build_chunk(&present, &device).unwrap();
                    forward_picks(&stack, &obs, &acts, &mask, &req.genome_idx)
                };
                #[cfg(feature = "profile")]
                drop(_gpu);
                let _ = req.response_tx.send(picks);
            }
        });
        GpuServer {
            tx: tx_clone,
            _handle: Some(handle),
        }
    }

    pub fn forward(&self, encoded: EncodedPly, genome_idx: Vec<usize>) -> Vec<usize> {
        let (r_tx, r_rx) = mpsc::channel();
        let req = GpuRequest {
            obs: encoded.obs,
            acts: encoded.acts,
            mask: encoded.mask,
            width: encoded.width,
            genome_idx,
            response_tx: r_tx,
        };
        let guard = self.tx.lock().unwrap();
        guard
            .as_ref()
            .expect("forward after shutdown")
            .send(req)
            .unwrap();
        drop(guard);
        r_rx.recv().unwrap()
    }

    pub fn shutdown(&self) {
        self.tx.lock().unwrap().take();
    }
}

impl Drop for GpuServer {
    fn drop(&mut self) {
        self.tx.lock().unwrap().take();
        if let Some(h) = self._handle.take() {
            let _ = h.join();
        }
    }
}

/// Drive a pool with a shared GpuServer. Each worker runs: encode → submit to
/// GPU → apply, in a loop. The GpuServer processes forward requests serially
/// in one CUDA context; workers overlap encode/apply (pure CPU) with the GPU
/// forward.
fn drive_worker(
    gpu: Arc<GpuServer>,
    game_seeds: Vec<u64>,
    meta: Vec<GameMeta>,
    max_hands: Option<u32>,
    max_width: usize,
) -> Vec<MatchResult> {
    let mut pool = Pool::with_max_width(game_seeds, None, max_hands, max_width);

    while pool.has_live() {
        #[cfg(feature = "profile")]
        profile::WKR_PLIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        #[cfg(feature = "profile")]
        let _enc = profile::Span::new(&profile::WKR_ENCODE_NS);
        let encoded = pool.encode();
        #[cfg(feature = "profile")]
        drop(_enc);

        let genome_idx: Vec<usize> = encoded
            .rows
            .iter()
            .map(|&(game_idx, seat)| {
                let (a, b, seating) = meta[game_idx];
                if (seat % 2 == 0) == (seating == 0) {
                    a
                } else {
                    b
                }
            })
            .collect();

        #[cfg(feature = "profile")]
        let _fwd = profile::Span::new(&profile::WKR_FWD_NS);
        let picks = gpu.forward(encoded, genome_idx);
        #[cfg(feature = "profile")]
        drop(_fwd);

        #[cfg(feature = "profile")]
        let _app = profile::Span::new(&profile::WKR_APPLY_NS);
        let _ = pool.apply(&picks);
        #[cfg(feature = "profile")]
        drop(_app);
    }

    pool.results()
}

/// The coalesced driver: ONE GpuServer, N worker threads. Results are returned
/// in **global game order** (index 0..n_games).
#[allow(clippy::too_many_arguments)]
fn rollout_coalesced(
    roster: &[Genome],
    arch: &Arch,
    game_seeds: Vec<u64>,
    meta: Vec<GameMeta>,
    max_hands: Option<u32>,
    device: &Device,
    n_workers: usize,
    max_width: usize,
) -> Vec<MatchResult> {
    let n_games = game_seeds.len();
    let n_workers = n_workers.min(n_games);

    let cpu_roster = Arc::new(CpuRoster::new(roster.to_vec(), arch.clone()));
    let gpu = Arc::new(GpuServer::new(
        cpu_roster,
        device.clone(),
        arch.obs,
        arch.act,
    ));

    let mut worker_data: Vec<(Vec<u64>, Vec<GameMeta>, Vec<usize>)> = Vec::new();
    for wid in 0..n_workers {
        let indices: Vec<usize> = (wid..n_games).step_by(n_workers).collect();
        let seeds: Vec<u64> = indices.iter().map(|&i| game_seeds[i]).collect();
        let meta_slice: Vec<GameMeta> = indices.iter().map(|&i| meta[i]).collect();
        worker_data.push((seeds, meta_slice, indices));
    }

    let mut handles = Vec::new();
    for (seeds, meta_slice, indices) in worker_data {
        let gpu = gpu.clone();
        let handle = thread::spawn(move || {
            let results = drive_worker(gpu, seeds, meta_slice, max_hands, max_width);
            (indices, results)
        });
        handles.push(handle);
    }

    let mut by_global: Vec<Option<MatchResult>> = vec![None; n_games];
    for h in handles {
        let (indices, results) = h.join().expect("worker panicked");
        for (local_i, &global_i) in indices.iter().enumerate() {
            by_global[global_i] = Some(results[local_i]);
        }
    }

    by_global
        .into_iter()
        .map(|opt| opt.expect("coalesced driver lost a game"))
        .collect()
}

/// Immutable inputs to one generation's evaluation.
pub struct EvalInputs<'a> {
    pub pop: &'a [Genome],
    pub hof: &'a HallOfFame,
    pub pairings: &'a [Pairing],
    pub arch: &'a Arch,
    pub seeds: &'a [u64],
    pub max_hands: Option<u32>,
    pub device: &'a Device,
    pub n_workers: usize,
    pub rollout: Rollout,
    /// Cap on legal actions per row (menu width). usize::MAX = no cap.
    /// Cuts the acts tensor transfer volume at peak plies.
    pub max_width: usize,
}

/// Evaluate one generation: play all pairings, compute ELO updates.
pub fn evaluate_generation(inputs: &EvalInputs<'_>, elo: &mut EloTracker) {
    #[cfg(feature = "profile")]
    {
        profile::reset();
    }
    let EvalInputs {
        pop,
        hof,
        pairings,
        arch,
        seeds,
        max_hands,
        device,
        n_workers,
        rollout,
        max_width,
    } = inputs;
    let (roster, game_seeds, meta) = batch_layout(pop, hof, pairings, seeds);

    #[cfg(feature = "profile")]
    let _gen_wall = profile::Span::new(&profile::GEN_WALL_NS);

    let results = match rollout {
        Rollout::Lockstep => rollout_lockstep(
            &roster, arch, game_seeds, &meta, *max_hands, device, *max_width,
        ),
        Rollout::Coalesced => rollout_coalesced(
            &roster, arch, game_seeds, meta, *max_hands, device, *n_workers, *max_width,
        ),
    };

    #[cfg(feature = "profile")]
    {
        drop(_gen_wall);
        let games = results.len();
        profile::report(games);
    }

    // Compute ELO updates.
    let per_game = 2 * seeds.len();
    let mut elo_results = Vec::new();
    for (gidx, (_seed, scores, _winner, _hands, _unfinished)) in results.iter().enumerate() {
        let pairing_idx = gidx / per_game;
        let seating = gidx % 2;
        let (ga_idx, gb_idx) = pairings[pairing_idx];
        let result = if scores[0] == scores[1] {
            0.5
        } else if (scores[0] > scores[1]) == (seating == 0) {
            1.0
        } else {
            0.0
        };
        elo_results.push((ga_idx, gb_idx, result));
    }
    elo.batch_update(&elo_results);
}
