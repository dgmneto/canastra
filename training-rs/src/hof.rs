//! Hall of Fame: archived champion genomes used as fixed-reference opponents
//! in self-play pairings and as frozen anchors for progress tracking.
//!
//! Shared infrastructure — used by the ES optimiser (`es.rs`), the self-play
//! league (`league.rs`), and the anchor evaluator (`anchors.rs`). Lives here
//! rather than in an optimizer-specific module so neither depends on the other.

use crate::genome::Genome;
use rand::rngs::StdRng;
use rand::Rng;

pub struct HallOfFame {
    pub genomes: Vec<Genome>,
    pub elo_ratings: Vec<f64>,
    pub generations: Vec<u32>,
}

impl Default for HallOfFame {
    fn default() -> Self {
        Self::new()
    }
}

impl HallOfFame {
    pub fn new() -> Self {
        Self {
            genomes: Vec::new(),
            elo_ratings: Vec::new(),
            generations: Vec::new(),
        }
    }

    pub fn archive(&mut self, genome: &Genome, elo: f64, generation: u32) {
        self.genomes.push(genome.clone());
        self.elo_ratings.push(elo);
        self.generations.push(generation);
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
