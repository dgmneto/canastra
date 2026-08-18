//! Self-play league: batched pairings and the lockstep ply driver.
//!
//! One rollout path: **lockstep**. A single `Pool` drives all games in
//! lockstep — one forward per ply, weights read once per present genome, one
//! sync point per ply. No channels, no mutex, no worker threads. The coalesced
//! GpuServer+workers path was removed (it lost the benchmark at the production
//! population; see `docs/decision-ga-vs-es.md`).
//!
//! **Scaling note:** lockstep was measured up to pop=1000 (422 games/s,
//! `docs/benchmarks.md`). A transient f32→bf16 cast spike in
//! `WeightStack::from_roster` causes OOM near pop=2000 on 16 GB VRAM
//! (`docs/task4-ksweep.md` Task 4a). The production config stays at pop=1000;
//! a future run targeting pop>2000 would need a coalesced/stream-overlap path
//! restored (or the f32→bf16 upload de-spiked) — see
//! `docs/decision-ga-vs-es.md`.

use crate::elo::EloTracker;
use crate::genome::{Arch, Genome};
use crate::hof::HallOfFame;
use crate::policy::{forward_picks, WeightStack};
use crate::pool::{MatchResult, Pool};
use candle_core::{DType, Device, Tensor};
use rand::rngs::StdRng;
use rand::Rng;

#[cfg(feature = "profile")]
use crate::profile;

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

/// Public wrapper for the lockstep rollout (used by anchored evaluation).
pub fn rollout_lockstep_public(
    roster: &[Genome],
    arch: &Arch,
    game_seeds: Vec<u64>,
    meta: &[GameMeta],
    max_hands: Option<u32>,
    device: &Device,
    max_width: usize,
) -> Vec<MatchResult> {
    rollout_lockstep(roster, arch, game_seeds, meta, max_hands, device, max_width)
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
        Device::Cuda(_) => DType::F16,
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
        // Both obs and acts are 100% binary (verified in `task3a-sparsity.md`).
        // Upload as u8 (1 byte/feature) and cast device-side via `to_dtype` in
        // the forward — 4x transfer-volume cut with zero precision loss.
        let obs = Tensor::from_vec(encoded.obs, (n_rows, obs_dim), device)
            .unwrap_or_else(|e| panic!("obs tensor: {e}"));
        let acts = Tensor::from_vec(encoded.acts, (n_rows, encoded.width, act_dim), device)
            .unwrap_or_else(|e| panic!("acts tensor: {e}"));
        // Mask is also binary — upload as u8, cast device-side.
        let mask = Tensor::from_vec(
            encoded
                .mask
                .iter()
                .map(|&v| if v { 1u8 } else { 0u8 })
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

/// Immutable inputs to one generation's evaluation.
pub struct EvalInputs<'a> {
    pub pop: &'a [Genome],
    pub hof: &'a HallOfFame,
    pub pairings: &'a [Pairing],
    pub arch: &'a Arch,
    pub seeds: &'a [u64],
    pub max_hands: Option<u32>,
    pub device: &'a Device,
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
        max_width,
    } = inputs;
    let (roster, game_seeds, meta) = batch_layout(pop, hof, pairings, seeds);

    #[cfg(feature = "profile")]
    let _gen_wall = profile::Span::new(&profile::GEN_WALL_NS);

    let results = rollout_lockstep(
        &roster, arch, game_seeds, &meta, *max_hands, device, *max_width,
    );

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
