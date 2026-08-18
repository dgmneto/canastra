//! Fitness from duplicate-deal score differentials.
//!
//! This replaces ELO as the *selection* signal for ES. ELO remains the right
//! tool for anchored evaluation (`anchors.rs`), where a rating is accumulated
//! against fixed opponents across generations. It is the wrong tool here, for
//! four reasons:
//!
//! 1. **The tracker is reset every generation.** ELO earns its keep by
//!    accumulating over a long history against opponents of known strength.
//!    Restarted from a flat 1200 baseline and given ~K games, it is a noisy,
//!    path-dependent re-encoding of the win count and nothing more. Being
//!    zero-sum, its population mean is pinned to exactly 1200 by construction —
//!    which is why `elo_mean` in `runs/es-smoke/generations.jsonl` reads
//!    1200.0 for every generation of the run.
//!
//! 2. **It is order-dependent.** `batch_update` folds games in sequentially, so
//!    two genomes with identical win/loss records get different ratings
//!    depending on the order their games happened to be scheduled. That is pure
//!    variance injected into the fitness, with no compensating information.
//!
//! 3. **It discards the margin.** A hand is scored win/draw/loss, throwing away
//!    the point differential — by far the richest, lowest-variance signal
//!    Canastra offers. Losing by 20 and losing by 800 are the same number to
//!    ELO.
//!
//! 4. **ES rank-normalises anyway.** `es::centred_ranks` reduces fitness to its
//!    ordering, so the one thing ELO adds over a plain win rate — calibrating
//!    the *size* of a result by opponent strength — is discarded downstream.
//!    All that survives is ELO's noise.
//!
//! # What replaces it
//!
//! The **paired (duplicate-deal) score differential**. `batch_layout` already
//! plays every deal twice with the seats swapped; until now the two seatings
//! were fed to ELO as two independent games, which throws away the entire point
//! of dealing them twice. Pairing them instead:
//!
//! ```text
//! diff(a, b, seed) = [ (score_a − score_b)│seating 0 + (score_a − score_b)│seating 1 ] / 2
//! ```
//!
//! The deal's luck enters the two seatings with opposite sign and cancels to
//! first order, leaving the part attributable to play. Halving makes the result
//! "mean points per game", the same unit as [`crate::evaluate::PairReport`]'s
//! `mean_diff`, so a training log and an A/B evaluation can be read against
//! each other.
//!
//! This is the same duplicate-deal comparison the project already uses for
//! evaluation on the Python and TypeScript sides (`sanity.py`, `eval-nn.ts`),
//! and `evaluate.rs` has implemented it in Rust all along — with no callers.
//! The training loop was the one place that needed it and reached for ELO
//! instead.
//!
//! `fitness[i]` is then the mean of `diff` over every game genome `i` played.
//! It is antisymmetric (`diff(a,b) = −diff(b,a)`), so the population mean sits
//! near zero and a positive fitness means "beats its opponents on shared
//! deals". Heavy tails (§13.3's flat −300, a 1000-point ace canastra) need no
//! special handling because the ES gradient consumes ranks, not magnitudes.

use crate::pool::MatchResult;

/// Per-genome fitness plus the diagnostics worth logging.
#[derive(Debug, Clone)]
pub struct FitnessReport {
    /// Mean paired score differential per genome. Index-aligned with the
    /// population (entries past `pop_size` are HOF opponents, kept so callers
    /// can inspect them, but ES only reads `[..pop_size]`).
    pub fitness: Vec<f64>,
    /// Paired comparisons contributing to each genome's fitness.
    pub games: Vec<usize>,
    /// Paired win rate per genome (win = 1, tie = 0.5), for logging only. Not
    /// used for selection — the differential carries strictly more information.
    pub win_rate: Vec<f64>,
    /// Mean |differential| across all pairs — the scale of the signal, useful
    /// for spotting a population that has collapsed to identical play.
    pub mean_abs_diff: f64,
}

/// Score one generation's results into per-genome fitness.
///
/// `results` must be index-aligned with the game layout produced by
/// [`crate::league::batch_layout`]: pairing-major, then seed, then the two
/// seatings. `seeds_per_pairing` is the number of deals per pairing.
pub fn score_generation(
    results: &[MatchResult],
    pairings: &[(usize, usize)],
    seeds_per_pairing: usize,
    roster_size: usize,
) -> FitnessReport {
    let per_pairing = 2 * seeds_per_pairing;
    assert_eq!(
        results.len(),
        pairings.len() * per_pairing,
        "results must cover every scheduled game; a short result vector would \
         silently misattribute every game after the gap"
    );

    let mut sum = vec![0.0f64; roster_size];
    let mut wins = vec![0.0f64; roster_size];
    let mut count = vec![0usize; roster_size];
    let mut abs_total = 0.0f64;
    let mut pairs = 0usize;

    for (pairing_idx, &(a, b)) in pairings.iter().enumerate() {
        for s in 0..seeds_per_pairing {
            let base = pairing_idx * per_pairing + s * 2;
            // Seating 0: `a` is team 0. Seating 1: `a` is team 1. Halved so the
            // result reads as mean points per game (see the module docs).
            let (_, s0, ..) = results[base];
            let (_, s1, ..) = results[base + 1];
            let diff = ((s0[0] - s0[1]) as f64 + (s1[1] - s1[0]) as f64) / 2.0;

            sum[a] += diff;
            sum[b] -= diff;
            count[a] += 1;
            count[b] += 1;

            let (win_a, win_b) = match diff.partial_cmp(&0.0) {
                Some(std::cmp::Ordering::Greater) => (1.0, 0.0),
                Some(std::cmp::Ordering::Less) => (0.0, 1.0),
                _ => (0.5, 0.5),
            };
            wins[a] += win_a;
            wins[b] += win_b;

            abs_total += diff.abs();
            pairs += 1;
        }
    }

    let fitness = sum
        .iter()
        .zip(count.iter())
        .map(|(&s, &c)| if c == 0 { 0.0 } else { s / c as f64 })
        .collect();
    let win_rate = wins
        .iter()
        .zip(count.iter())
        .map(|(&w, &c)| if c == 0 { 0.5 } else { w / c as f64 })
        .collect();

    FitnessReport {
        fitness,
        games: count,
        win_rate,
        mean_abs_diff: if pairs == 0 {
            0.0
        } else {
            abs_total / pairs as f64
        },
    }
}

/// Distribution summary of a fitness vector, for `generations.jsonl`.
pub fn fitness_stats(values: &[f64]) -> serde_json::Value {
    if values.is_empty() {
        return serde_json::json!(null);
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let pct = |p: usize| sorted[(p * n / 100).min(n - 1)];
    let mean = sorted.iter().sum::<f64>() / n as f64;
    let var = sorted.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    serde_json::json!({
        "min": sorted[0],
        "p25": pct(25),
        "median": pct(50),
        "p75": pct(75),
        "p95": pct(95),
        "max": sorted[n - 1],
        "mean": mean,
        "std": var.sqrt(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `MatchResult` with the given team scores.
    fn game(a: i32, b: i32) -> MatchResult {
        (0, [a, b], None, 1, false)
    }

    #[test]
    fn differential_is_antisymmetric() {
        // One pairing (0, 1), one deal, two seatings.
        // Seating 0: genome 0 is team 0 and scores 500 vs 200 → +300.
        // Seating 1: genome 0 is team 1 and scores 400 vs 100 → +300.
        let results = vec![game(500, 200), game(100, 400)];
        let report = score_generation(&results, &[(0, 1)], 1, 2);
        assert_eq!(report.fitness[0], 300.0);
        assert_eq!(report.fitness[1], -300.0);
        assert_eq!(report.win_rate[0], 1.0);
        assert_eq!(report.win_rate[1], 0.0);
    }

    #[test]
    fn deal_luck_cancels_across_seatings() {
        // A deal worth +1000 to whoever holds team 0's cards, with both
        // genomes playing it identically: seating 0 gives it to genome 0,
        // seating 1 gives it to genome 1. The paired differential is zero,
        // where scoring the two seatings independently would read as one
        // decisive win each.
        let results = vec![game(1000, 0), game(1000, 0)];
        let report = score_generation(&results, &[(0, 1)], 1, 2);
        assert_eq!(report.fitness[0], 0.0);
        assert_eq!(report.fitness[1], 0.0);
        assert_eq!(report.win_rate[0], 0.5);
    }

    #[test]
    fn fitness_averages_over_deals_and_opponents() {
        // Genome 0 plays genome 1 (+200 paired) and genome 2 (−100 paired).
        let results = vec![
            // pairing 0, seed 0. Genome 0 is team 0 then team 1.
            game(200, 0), // seating 0 → +200
            game(0, 0),   // seating 1 →    0
            // pairing 1, seed 0. Genome 0 is team 0 then team 1.
            game(0, 50), // seating 0 → −50
            game(50, 0), // seating 1 → −50
        ];
        // Halved per seating pair: pairing 0 → +100, pairing 1 → −50.
        let report = score_generation(&results, &[(0, 1), (0, 2)], 1, 3);
        assert_eq!(report.fitness[0], (100.0 - 50.0) / 2.0);
        assert_eq!(report.fitness[1], -100.0);
        assert_eq!(report.fitness[2], 50.0);
        assert_eq!(report.games[0], 2);
        assert_eq!(report.mean_abs_diff, 75.0);
    }

    #[test]
    #[should_panic(expected = "results must cover every scheduled game")]
    fn short_results_are_rejected() {
        score_generation(&[game(1, 0)], &[(0, 1)], 1, 2);
    }
}
