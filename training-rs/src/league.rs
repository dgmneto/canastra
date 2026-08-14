//! Self-play league: batched pairings, the ply driver, and the coalesced
//! GPU server. All native Rust — no GIL, no IPC, no process spawning.

use crate::elo::EloTracker;
use crate::ga::{GAConfig, HallOfFame};
use crate::genome::{Arch, Genome};
use crate::policy::{forward_and_pick, WeightStack};
use crate::pool::{EncodedPly, MatchResult, Pool};
use crate::seedstream;
use candle_core::{Device, Tensor};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

type Pairing = (usize, usize);
type GameMeta = (usize, usize, usize);

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
        if hof.len() > 0 && n > 0 {
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

/// One forward request from a worker to the GPU server.
struct GpuRequest {
    obs: Vec<f32>,
    acts: Vec<f32>,
    mask: Vec<bool>,
    width: usize,
    genome_idx: Vec<usize>,
    response_tx: mpsc::Sender<Vec<usize>>,
}

/// The coalesced GPU server: processes forward requests from worker threads.
pub struct GpuServer {
    tx: mpsc::Sender<GpuRequest>,
    handle: Option<thread::JoinHandle<()>>,
}

impl GpuServer {
    pub fn new(stacked: Arc<WeightStack>, obs_dim: usize, act_dim: usize) -> Self {
        let (tx, rx) = mpsc::channel::<GpuRequest>();
        let handle = thread::spawn(move || {
            while let Ok(req) = rx.recv() {
                let n_rows = req.obs.len() / obs_dim;
                if n_rows == 0 {
                    let _ = req.response_tx.send(Vec::new());
                    continue;
                }

                let device = &stacked.device;
                let obs = Tensor::from_vec(req.obs, (n_rows, obs_dim), device)
                    .unwrap_or_else(|e| panic!("obs tensor: {e}"));
                let acts = Tensor::from_vec(req.acts, (n_rows, req.width, act_dim), device)
                    .unwrap_or_else(|e| panic!("acts tensor: {e}"));
                let mask = Tensor::from_vec(
                    req.mask
                        .iter()
                        .map(|&v| if v { 1u32 } else { 0u32 })
                        .collect::<Vec<_>>(),
                    (n_rows, req.width),
                    device,
                )
                .unwrap_or_else(|e| panic!("mask tensor: {e}"));

                let n_groups = stacked.n_groups();
                // Use all genomes as present — avoids index remapping issues.
                let present: Vec<usize> = (0..n_groups).collect();
                let n_max = present
                    .iter()
                    .map(|&g| req.genome_idx.iter().filter(|&&gg| gg == g).count())
                    .max()
                    .unwrap_or(0)
                    .max(4);

                let sub = (*stacked).shallow_clone();

                let picks =
                    forward_and_pick(&sub, &obs, &acts, &mask, &req.genome_idx, &present, n_max);
                let _ = req.response_tx.send(picks);
            }
        });
        GpuServer {
            tx,
            handle: Some(handle),
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
        self.tx.send(req).unwrap();
        r_rx.recv().unwrap()
    }
}

impl Drop for GpuServer {
    fn drop(&mut self) {
        drop(&self.tx);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Drive a pool with the coalesced GPU server. This is what each worker
/// thread runs: encode → submit to GPU → apply, in a loop.
pub fn drive_worker(
    stacked: Arc<WeightStack>,
    game_seeds: Vec<u64>,
    meta: Vec<GameMeta>,
    max_hands: Option<u32>,
    obs_dim: usize,
    act_dim: usize,
) -> Vec<MatchResult> {
    let gpu = GpuServer::new(stacked.clone(), obs_dim, act_dim);
    let mut pool = Pool::new(game_seeds, None, max_hands);

    while pool.has_live() {
        let encoded = pool.encode();
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
        let picks = gpu.forward(encoded, genome_idx);
        let _ = pool.apply(&picks);
    }

    pool.results()
}

/// The top-level coalesced driver: splits games across N worker threads,
/// each with its own Pool, all sharing one GpuServer.
pub fn drive_coalesced(
    stacked: Arc<WeightStack>,
    game_seeds: Vec<u64>,
    meta: Vec<GameMeta>,
    max_hands: Option<u32>,
    n_workers: usize,
    obs_dim: usize,
    act_dim: usize,
) -> Vec<MatchResult> {
    let n_games = game_seeds.len();
    let n_workers = n_workers.min(n_games);

    // Split games across workers (interleaved).
    let mut worker_data: Vec<(Vec<u64>, Vec<GameMeta>)> = Vec::new();
    for wid in 0..n_workers {
        let indices: Vec<usize> = (wid..n_games).step_by(n_workers).collect();
        let seeds: Vec<u64> = indices.iter().map(|&i| game_seeds[i]).collect();
        let meta_slice: Vec<GameMeta> = indices.iter().map(|&i| meta[i]).collect();
        worker_data.push((seeds, meta_slice));
    }

    // Each worker gets its own GpuServer. With native threads (no GIL),
    // multiple GPU servers can coexist — tch-rs handles CUDA context
    // sharing automatically within one process.
    let mut handles = Vec::new();
    for (seeds, meta_slice) in worker_data {
        let stacked = stacked.clone();
        let handle = thread::spawn(move || {
            drive_worker(stacked, seeds, meta_slice, max_hands, obs_dim, act_dim)
        });
        handles.push(handle);
    }

    let mut all_results = Vec::new();
    for h in handles {
        let mut results = h.join().expect("worker panicked");
        all_results.append(&mut results);
    }
    all_results
}

/// Evaluate one generation: play all pairings, compute ELO updates.
pub fn evaluate_generation(
    pop: &[Genome],
    hof: &HallOfFame,
    pairings: &[Pairing],
    arch: &Arch,
    seeds: &[u64],
    max_hands: Option<u32>,
    device: &Device,
    elo: &mut EloTracker,
    n_workers: usize,
) {
    let (roster, game_seeds, meta) = batch_layout(pop, hof, pairings, seeds);

    // Build stacked weights on the device.
    let roster_refs: Vec<&Genome> = roster.iter().collect();
    let stacked = Arc::new(WeightStack::from_roster(&roster_refs, arch, device));

    let results = drive_coalesced(
        stacked, game_seeds, meta, max_hands, n_workers, arch.obs, arch.act,
    );

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
