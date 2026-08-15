//! Self-play league: batched pairings, the ply driver, and the coalesced
//! GPU server. All native Rust — no GIL, no IPC, no process spawning.

use crate::elo::EloTracker;
use crate::ga::HallOfFame;
use crate::genome::{Arch, Genome};
use crate::policy::{forward_picks, forward_scores_roster, CpuRoster, WeightStack};
use crate::pool::{EncodedPly, MatchResult, Pool};
use candle_core::{Device, Tensor};
use rand::rngs::StdRng;
use rand::Rng;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
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
///
/// On startup, the full weight stack is built on the device once (up to ~2000
/// genomes ≈ 9.6 GB — fits in 16 GB VRAM with room for activations). This
/// eliminates per-ply weight uploads entirely. For populations that exceed
/// VRAM, it falls back to lazy per-chunk uploads via `forward_scores_roster`.
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
            // Build the full WeightStack on the device once.
            // ~2000 genomes × 4.8 MB ≈ 9.6 GB — fits in 16 GB with room for
            // activation tensors (~300 MB per ply). ONE copy shared by all
            // workers, unlike the previous design where each worker built
            // its own (4 × G bytes).
            let n_genomes = roster.n_genomes();
            let cached_stack: Option<WeightStack> = if n_genomes <= 2000 {
                let all: Vec<usize> = (0..n_genomes).collect();
                roster.build_chunk(&all, &device).ok()
            } else {
                None
            };

            while let Ok(req) = rx.recv() {
                let n_rows = req.obs.len() / obs_dim;
                if n_rows == 0 {
                    let _ = req.response_tx.send(Vec::new());
                    continue;
                }

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

                let picks = if let Some(ref stack) = cached_stack {
                    forward_picks(stack, &obs, &acts, &mask, &req.genome_idx)
                } else {
                    forward_scores_roster(&roster, &obs, &acts, &mask, &req.genome_idx, &device)
                        .picks
                };
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
pub fn drive_worker(
    gpu: Arc<GpuServer>,
    game_seeds: Vec<u64>,
    meta: Vec<GameMeta>,
    max_hands: Option<u32>,
) -> Vec<MatchResult> {
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

/// The top-level coalesced driver: ONE GpuServer, N worker threads.
///
/// Results are returned in **global game order** (index 0..n_games).
#[allow(clippy::too_many_arguments)]
pub fn drive_coalesced(
    roster: Arc<CpuRoster>,
    device: Device,
    game_seeds: Vec<u64>,
    meta: Vec<GameMeta>,
    max_hands: Option<u32>,
    n_workers: usize,
    obs_dim: usize,
    act_dim: usize,
) -> Vec<MatchResult> {
    let n_games = game_seeds.len();
    let n_workers = n_workers.min(n_games);

    // ONE GpuServer, shared by all workers via Arc.
    let gpu = Arc::new(GpuServer::new(roster, device, obs_dim, act_dim));

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
            let results = drive_worker(gpu, seeds, meta_slice, max_hands);
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
}

/// Evaluate one generation: play all pairings, compute ELO updates.
pub fn evaluate_generation(inputs: &EvalInputs<'_>, elo: &mut EloTracker) {
    let EvalInputs {
        pop,
        hof,
        pairings,
        arch,
        seeds,
        max_hands,
        device,
        n_workers,
    } = *inputs;
    let (roster, game_seeds, meta) = batch_layout(pop, hof, pairings, seeds);

    // Build a CPU roster — genomes stay on CPU, GPU only holds small chunks
    // per forward pass. This bounds GPU memory to ~300 MB regardless of
    // population size.
    let cpu_roster = Arc::new(CpuRoster::new(roster, arch.clone()));

    let results = drive_coalesced(
        cpu_roster,
        device.clone(),
        game_seeds,
        meta,
        max_hands,
        n_workers,
        arch.obs,
        arch.act,
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
