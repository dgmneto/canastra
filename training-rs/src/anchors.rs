//! Anchored evaluation: play the current champion against fixed-reference
//! genomes with frozen ratings, producing an absolute progress metric.
//!
//! ELO inside a closed self-play population is relative — it rises
//! monotonically whether or not absolute strength improves. Fixed anchors
//! with frozen ratings break this: if the champion's anchor ELO rises over
//! generations, the run is working. If it plateaus or cycles, the run is
//! stuck.
//!
//! Anchors:
//! - **Random bot**: a zero-weight genome (all zeros → argmax picks the first
//!   legal action, deterministic but unskilled). Frozen at 1000 ELO.
//! - **Frozen champions**: the generation-0 champion and every Nth-generation
//!   champion, saved as anchors with their internal ELO at archival time.
//!
//! Every generation, the current champion plays a small number of games
//! against each anchor. The anchor's rating is frozen; only the champion's
//! rating updates. The result is logged to `generations.jsonl`.

use crate::genome::{Arch, Genome};
use crate::hof::HallOfFame;
use crate::league::batch_layout;
use candle_core::Device;

/// A fixed-reference anchor with a frozen ELO rating.
pub struct Anchor {
    pub name: String,
    pub genome: Genome,
    pub rating: f64,
    pub generation: u32,
}

/// Serializable snapshot of one Anchor for checkpointing.
pub struct AnchorRecord {
    pub name: String,
    pub genome: Genome,
    pub rating: f64,
    pub generation: u32,
}

/// Serializable snapshot of an AnchorSet for checkpointing.
///
/// The anchor set (the champion's accumulated rating against fixed references
/// plus the frozen champions used as references) was not previously part of
/// the ES checkpoint, so a resumed run rebuilt it from scratch: champion_rating
/// reset to 1200 and every --anchor-freeze-interval champion added before the
/// resume was lost. This snapshot makes both durable across resume.
pub struct AnchorSnapshot {
    pub champion_rating: f64,
    pub anchors: Vec<AnchorRecord>,
}

/// The anchor set: random bot + frozen champions.
pub struct AnchorSet {
    pub anchors: Vec<Anchor>,
    /// ELO tracker for the champion's rating against anchors (separate from
    /// the self-play ELO). Starts at 1200.
    pub champion_rating: f64,
}

impl AnchorSet {
    /// Create a new anchor set with a random-bot anchor (zero genome).
    pub fn new(arch: &Arch) -> Self {
        let random_bot = Anchor {
            name: "random".to_string(),
            genome: vec![0.0f32; crate::genome::genome_size(arch)],
            rating: 1000.0,
            generation: 0,
        };
        AnchorSet {
            anchors: vec![random_bot],
            champion_rating: 1200.0,
        }
    }

    /// Add a frozen champion as an anchor.
    pub fn add_champion(&mut self, genome: &Genome, internal_elo: f64, generation: u32) {
        self.anchors.push(Anchor {
            name: format!("gen-{}", generation),
            genome: genome.clone(),
            rating: internal_elo,
            generation,
        });
    }

    /// Snapshot the anchor set for checkpointing. Clones every anchor genome,
    /// so this is O(anchors x genome_size); call it only at checkpoint time.
    pub fn snapshot(&self) -> AnchorSnapshot {
        AnchorSnapshot {
            champion_rating: self.champion_rating,
            anchors: self
                .anchors
                .iter()
                .map(|a| AnchorRecord {
                    name: a.name.clone(),
                    genome: a.genome.clone(),
                    rating: a.rating,
                    generation: a.generation,
                })
                .collect(),
        }
    }

    /// Rebuild an anchor set from a saved snapshot (resume path). The snapshot
    /// carries the random-bot anchor too, so this does not call AnchorSet::new
    /// -- the restored set is exactly what was saved.
    pub fn from_snapshot(snap: AnchorSnapshot) -> Self {
        AnchorSet {
            anchors: snap
                .anchors
                .into_iter()
                .map(|r| Anchor {
                    name: r.name,
                    genome: r.genome,
                    rating: r.rating,
                    generation: r.generation,
                })
                .collect(),
            champion_rating: snap.champion_rating,
        }
    }

    /// Evaluate the champion against all anchors. Updates `champion_rating`
    /// using frozen-anchor ELO. Returns the champion's rating and per-anchor
    /// results for logging.
    pub fn evaluate(
        &mut self,
        champion: &Genome,
        arch: &Arch,
        seeds: &[u64],
        device: &Device,
        max_width: usize,
        max_hands: Option<u32>,
    ) -> AnchorReport {
        let n_anchors = self.anchors.len();
        let n_seeds = seeds.len();
        let _ = n_anchors;

        let mut all_results = Vec::new();

        for anchor in self.anchors.iter() {
            let mini_pop: Vec<Genome> = vec![champion.clone(), anchor.genome.clone()];
            let mini_hof = HallOfFame::new();
            let mini_pairings: Vec<(usize, usize)> = vec![(0, 1)];
            let (_roster, game_seeds, meta) =
                batch_layout(&mini_pop, &mini_hof, &mini_pairings, seeds);

            // Run the lockstep rollout directly.
            let results = crate::league::rollout_lockstep_public(
                &mini_pop, arch, game_seeds, &meta, max_hands, device, max_width,
            );

            // Score: champion is index 0, anchor is index 1.
            // Seating 0: champion is team 0. Seating 1: champion is team 1.
            let mut wins = 0;
            let mut losses = 0;
            let mut draws = 0;
            let mut champ_score_sum = 0i32;
            let mut anchor_score_sum = 0i32;

            for (gidx, (_seed, scores, _winner, _hands, _unfinished)) in results.iter().enumerate()
            {
                let seating = gidx % 2;
                let champ_score = if seating == 0 { scores[0] } else { scores[1] };
                let anchor_score = if seating == 0 { scores[1] } else { scores[0] };
                champ_score_sum += champ_score;
                anchor_score_sum += anchor_score;
                if champ_score > anchor_score {
                    wins += 1;
                } else if champ_score < anchor_score {
                    losses += 1;
                } else {
                    draws += 1;
                }
            }

            // ELO update: champion vs anchor (frozen rating).
            let anchor_rating = anchor.rating;
            let expected =
                1.0 / (1.0 + 10.0_f64.powf((anchor_rating - self.champion_rating) / 400.0));
            let actual = (wins as f64 + 0.5 * draws as f64) / (n_seeds * 2) as f64;
            let k = 32.0;
            self.champion_rating += k * (actual - expected);

            all_results.push(AnchorResult {
                name: anchor.name.clone(),
                generation: anchor.generation,
                wins,
                losses,
                draws,
                champ_score_mean: champ_score_sum as f64 / (n_seeds * 2) as f64,
                anchor_score_mean: anchor_score_sum as f64 / (n_seeds * 2) as f64,
            });
        }

        AnchorReport {
            champion_rating: self.champion_rating,
            results: all_results,
        }
    }
}

/// Per-anchor result from one evaluation.
#[derive(Debug)]
pub struct AnchorResult {
    pub name: String,
    pub generation: u32,
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
    pub champ_score_mean: f64,
    pub anchor_score_mean: f64,
}

/// The full report from one anchored evaluation.
pub struct AnchorReport {
    pub champion_rating: f64,
    pub results: Vec<AnchorResult>,
}

impl AnchorReport {
    /// Serialize to JSON for the generations.jsonl log.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "champion_rating": self.champion_rating,
            "opponents": self.results.iter().map(|r| {
                serde_json::json!({
                    "name": r.name,
                    "gen": r.generation,
                    "w": r.wins,
                    "l": r.losses,
                    "d": r.draws,
                    "champ_score": r.champ_score_mean,
                    "anchor_score": r.anchor_score_mean,
                })
            }).collect::<Vec<_>>(),
        })
    }
}
