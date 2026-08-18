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
        for opp in draw_opponents(&others, opponents, pop_size, hof, rng) {
            pairings.push((me, opp));
        }
    }
    pairings
}

/// Schedule pairings with **common random numbers across each mirrored pair**.
///
/// ES's gradient estimate is `Σⱼ (f⁺ⱼ − f⁻ⱼ) εⱼ`, so its variance is driven by
/// `Var(f⁺ − f⁻)`. Every condition the twins *share* cancels out of that
/// difference; every condition that differs survives as noise. `es.rs` claims
/// this cancellation ("Mirrored sampling: the difference f⁺ − f⁻ cancels the
/// bias of the base policy's own fitness"), but plain [`schedule_pairings`]
/// draws a fresh opponent list for every genome, so the twins are measured
/// against different opponents and the cancellation never happens — at σ=0.02
/// the twins are near-identical policies and the opponent draw dominates the
/// signal entirely.
///
/// Here genome `2j` and genome `2j+1` get the **same** opponent list. Combined
/// with the shared deal seeds every pairing already uses, the only difference
/// between the twins' conditions is the sign of `εⱼ` — which is exactly the
/// quantity the gradient is trying to measure.
///
/// The opponent list excludes both twins, so neither plays itself and the pair
/// is scheduled symmetrically.
pub fn schedule_pairings_mirrored(
    pop_size: usize,
    opponents: usize,
    hof: &HallOfFame,
    rng: &mut StdRng,
) -> Vec<Pairing> {
    assert!(
        pop_size.is_multiple_of(2),
        "mirrored scheduling needs an even population (got {pop_size})"
    );
    let mut pairings = Vec::new();
    for j in 0..pop_size / 2 {
        let (plus, minus) = (2 * j, 2 * j + 1);
        let others: Vec<usize> = (0..pop_size).filter(|&i| i != plus && i != minus).collect();
        let chosen = draw_opponents(&others, opponents, pop_size, hof, rng);
        for &opp in &chosen {
            pairings.push((plus, opp));
        }
        for &opp in &chosen {
            pairings.push((minus, opp));
        }
    }
    pairings
}

/// Sample `opponents` distinct entries from `others`, reserving the last slot
/// for a random hall-of-fame entry when the HOF is non-empty. HOF entries are
/// addressed at roster indices `pop_size + k` (see [`batch_layout`]).
fn draw_opponents(
    others: &[usize],
    opponents: usize,
    pop_size: usize,
    hof: &HallOfFame,
    rng: &mut StdRng,
) -> Vec<usize> {
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
    chosen
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

/// The weight-stack dtype for a device when the caller has no preference.
///
/// **BF16 on CUDA**, F32 on CPU for exactness.
///
/// BF16 over F16 because the two are within ±2% of each other on this card
/// (measured both ways across pop 96/500/1000 — `docs/benchmarks.md`, "Phase 2
/// re-measured"), while BF16 carries f32's exponent range. The earlier switch
/// to F16 was made for a "~2.5x speedup" that turned out to be the masking bug
/// it introduced: `-1e9` is finite in BF16 but overflows to `-inf` in F16, and
/// F16 was the only dtype where that mattered. Same speed, one fewer way to
/// silently break — and BF16 is what the GPU equivalence test already covers.
///
/// Masking and argmax are done in f32 regardless of this choice — see
/// `policy::mask_illegal_f32` and `docs/decision-ranking-metric.md`.
pub fn default_dtype(device: &Device) -> DType {
    match device {
        Device::Cuda(_) => DType::BF16,
        _ => DType::F32,
    }
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
    rollout_lockstep(
        roster, arch, game_seeds, meta, max_hands, device, max_width, None,
    )
}

/// Drive all games in lockstep with a single `Pool`. One forward per ply.
///
/// `dtype` overrides the weight-stack precision; `None` takes
/// [`default_dtype`].
#[allow(clippy::too_many_arguments)]
fn rollout_lockstep(
    roster: &[Genome],
    arch: &Arch,
    game_seeds: Vec<u64>,
    meta: &[GameMeta],
    max_hands: Option<u32>,
    device: &Device,
    max_width: usize,
    dtype: Option<DType>,
) -> Vec<MatchResult> {
    let dtype = dtype.unwrap_or_else(|| default_dtype(device));
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
    /// Weight-stack precision. `None` takes [`default_dtype`] for the device.
    /// Exposed so the benchmark can A/B F16 against BF16 — the switch between
    /// them is what introduced the masking bug, and its actual speed benefit
    /// had never been measured on non-degenerate games.
    pub dtype: Option<DType>,
}

/// Play one generation: every pairing, every deal, both seatings.
///
/// Returns results index-aligned with [`batch_layout`]'s game order
/// (pairing-major, then seed, then seating), which is the alignment both
/// [`crate::fitness::score_generation`] and [`elo_updates`] rely on.
pub fn play_generation(inputs: &EvalInputs<'_>) -> Vec<MatchResult> {
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
        dtype,
    } = inputs;
    let (roster, game_seeds, meta) = batch_layout(pop, hof, pairings, seeds);
    let scheduled = meta.len();

    #[cfg(feature = "profile")]
    let _gen_wall = profile::Span::new(&profile::GEN_WALL_NS);

    let results = rollout_lockstep(
        &roster, arch, game_seeds, &meta, *max_hands, device, *max_width, *dtype,
    );

    #[cfg(feature = "profile")]
    {
        drop(_gen_wall);
        let games = results.len();
        profile::report(games);
    }

    // `Pool::results` drops games that never finished, which would shift every
    // downstream index by one and silently misattribute the rest of the
    // generation. Every game is meant to terminate (match over, `max_hands`
    // reached, or the action cap), so a short vector is a bug, not a case to
    // absorb.
    assert_eq!(
        results.len(),
        scheduled,
        "rollout returned {} results for {scheduled} scheduled games; game→pairing \
         attribution would be silently wrong",
        results.len()
    );

    results
}

/// Fold one generation's results into ELO updates.
///
/// Retained for anchored evaluation and for the Python-equivalence tests, which
/// pin this exact arithmetic and update order. **Not** the selection signal for
/// ES — see [`crate::fitness`] for why, and for what replaced it.
pub fn elo_updates(
    results: &[MatchResult],
    pairings: &[Pairing],
    seeds_per_pairing: usize,
) -> Vec<(usize, usize, f64)> {
    let per_pairing = 2 * seeds_per_pairing;
    results
        .iter()
        .enumerate()
        .map(|(gidx, (_seed, scores, _winner, _hands, _unfinished))| {
            let (ga_idx, gb_idx) = pairings[gidx / per_pairing];
            let seating = gidx % 2;
            let result = if scores[0] == scores[1] {
                0.5
            } else if (scores[0] > scores[1]) == (seating == 0) {
                1.0
            } else {
                0.0
            };
            (ga_idx, gb_idx, result)
        })
        .collect()
}

/// Evaluate one generation and apply ELO updates.
///
/// Kept as the Python-equivalence entry point (`tests/equivalence.rs` pins its
/// output against `gen_elo_after.json`). Training uses [`play_generation`] plus
/// [`crate::fitness::score_generation`].
pub fn evaluate_generation(inputs: &EvalInputs<'_>, elo: &mut EloTracker) {
    let results = play_generation(inputs);
    elo.batch_update(&elo_updates(&results, inputs.pairings, inputs.seeds.len()));
}
