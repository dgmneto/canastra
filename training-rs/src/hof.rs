//! Hall of Fame: archived champion genomes used as fixed-reference opponents
//! in self-play pairings and as frozen anchors for progress tracking.
//!
//! Shared infrastructure — used by the ES optimiser (`es.rs`), the self-play
//! league (`league.rs`), and the anchor evaluator (`anchors.rs`). Lives here
//! rather than in an optimizer-specific module so neither depends on the other.

use crate::genome::Genome;
use rand::rngs::StdRng;
use rand::Rng;

/// Default cap on archived genomes. At ~4.8 MB per genome this is ~96 MB,
/// negligible against a pop=1000 roster.
///
/// The cap is not optional in practice. Every archived genome is cloned into
/// the roster and re-uploaded through `WeightStack::from_roster` *every
/// generation*, and serialized into *every checkpoint*. Unbounded, a 1000-
/// generation run at `--hof-interval 5` accumulates 200 entries — ~960 MB
/// re-uploaded per generation, ~960 MB per checkpoint file, and a roster 20%
/// larger than the population it is meant to be a small reference set for.
/// That walks straight into the `from_roster` VRAM spike documented in
/// `docs/task4-ksweep.md` Task 4a.
pub const DEFAULT_HOF_CAPACITY: usize = 20;

pub struct HallOfFame {
    pub genomes: Vec<Genome>,
    pub elo_ratings: Vec<f64>,
    pub generations: Vec<u32>,
    /// Maximum archived genomes. Exceeding it evicts by thinning (see
    /// [`HallOfFame::archive`]).
    pub capacity: usize,
}

impl Default for HallOfFame {
    fn default() -> Self {
        Self::new()
    }
}

impl HallOfFame {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_HOF_CAPACITY)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            genomes: Vec::new(),
            elo_ratings: Vec::new(),
            generations: Vec::new(),
            capacity: capacity.max(1),
        }
    }

    /// Archive a genome, evicting by **thinning** if that exceeds `capacity`.
    ///
    /// Thinning drops the interior entry whose two neighbours are closest
    /// together in generation — the most redundant point in the archive. The
    /// first and last entries are never evicted, so the set keeps the run's
    /// earliest reference (a genuinely weak opponent, which is what makes it
    /// informative) and its most recent, and stays roughly evenly spread over
    /// training history in between.
    ///
    /// Evicting the *oldest* instead would be wrong: the archive would collapse
    /// onto recent generations, which under ES are all near-identical to the
    /// current θ and so provide almost no signal as opponents.
    pub fn archive(&mut self, genome: &Genome, elo: f64, generation: u32) {
        self.genomes.push(genome.clone());
        self.elo_ratings.push(elo);
        self.generations.push(generation);

        while self.genomes.len() > self.capacity {
            let victim = self.thinning_victim();
            self.genomes.remove(victim);
            self.elo_ratings.remove(victim);
            self.generations.remove(victim);
        }
    }

    /// Index of the most redundant interior entry: the one whose neighbours
    /// span the smallest generation gap. Ties go to the lower index.
    fn thinning_victim(&self) -> usize {
        let n = self.generations.len();
        debug_assert!(n >= 3, "thinning needs an interior to choose from");
        let mut best = 1usize;
        let mut best_gap = u32::MAX;
        for i in 1..n - 1 {
            let gap = self.generations[i + 1].saturating_sub(self.generations[i - 1]);
            if gap < best_gap {
                best_gap = gap;
                best = i;
            }
        }
        best
    }

    pub fn len(&self) -> usize {
        self.genomes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.genomes.is_empty()
    }

    pub fn sample(&self, rng: &mut StdRng) -> &Genome {
        let idx = (rng.gen::<u32>() as usize) % self.genomes.len();
        &self.genomes[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn archive_n(cap: usize, gens: &[u32]) -> HallOfFame {
        let mut hof = HallOfFame::with_capacity(cap);
        for &g in gens {
            hof.archive(&vec![g as f32; 4], g as f64, g);
        }
        hof
    }

    #[test]
    fn archive_is_capped() {
        let gens: Vec<u32> = (0..200).map(|i| i * 5).collect();
        let hof = archive_n(20, &gens);
        assert_eq!(hof.len(), 20, "archive must not exceed its capacity");
    }

    #[test]
    fn thinning_keeps_the_endpoints() {
        let gens: Vec<u32> = (0..100).map(|i| i * 5).collect();
        let hof = archive_n(10, &gens);
        assert_eq!(
            hof.generations[0], 0,
            "the earliest reference is the most informative opponent and must survive"
        );
        assert_eq!(
            *hof.generations.last().unwrap(),
            495,
            "the most recent archive must survive"
        );
    }

    #[test]
    fn thinning_keeps_history_spread_not_just_recent() {
        // 500 generations archived every 5; capacity 10. A naive "drop the
        // oldest" policy would leave only generations 455..495, which under ES
        // are near-identical to current θ and useless as opponents.
        let gens: Vec<u32> = (0..100).map(|i| i * 5).collect();
        let hof = archive_n(10, &gens);

        let span = hof.generations.last().unwrap() - hof.generations[0];
        assert!(
            span >= 450,
            "archive collapsed onto {:?} (span {span}) instead of spanning the run",
            hof.generations
        );
        // No half of the archive should be crammed into a tenth of the run.
        let midpoint = hof.generations[hof.len() / 2];
        assert!(
            (100..400).contains(&midpoint),
            "median archived generation {midpoint} is not near the middle of the run: {:?}",
            hof.generations
        );
    }

    #[test]
    fn under_capacity_archives_everything() {
        let hof = archive_n(20, &[0, 5, 10]);
        assert_eq!(hof.generations, vec![0, 5, 10]);
        assert_eq!(hof.elo_ratings, vec![0.0, 5.0, 10.0]);
    }
}
