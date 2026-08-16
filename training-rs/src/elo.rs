//! ELO rating tracker for the GA.
//!
//! Standard ELO: expected = 1 / (1 + 10^((opp - self) / 400)), and
//! rating += K * (result - expected). Children inherit their parent's rating.

pub struct EloTracker {
    pub ratings: Vec<f64>,
    k_factor: f64,
    base: f64,
}

impl EloTracker {
    pub fn new(size: usize) -> Self {
        Self::with_params(size, 32.0, 1200.0)
    }

    /// Reconstruct an EloTracker from saved ratings (for --resume).
    pub fn from_ratings(ratings: Vec<f64>) -> Self {
        Self {
            ratings,
            k_factor: 32.0,
            base: 1200.0,
        }
    }

    pub fn with_params(size: usize, k_factor: f64, base: f64) -> Self {
        Self {
            ratings: vec![base; size],
            k_factor,
            base,
        }
    }

    pub fn expected(&self, a: usize, b: usize) -> f64 {
        1.0 / (1.0 + 10.0_f64.powf((self.ratings[b] - self.ratings[a]) / 400.0))
    }

    pub fn update(&mut self, a: usize, b: usize, result: f64) {
        let exp_a = self.expected(a, b);
        self.ratings[a] += self.k_factor * (result - exp_a);
        self.ratings[b] += self.k_factor * (1.0 - result - (1.0 - exp_a));
    }

    pub fn batch_update(&mut self, results: &[(usize, usize, f64)]) {
        for &(a, b, result) in results {
            self.update(a, b, result);
        }
    }

    pub fn grow(&mut self, n_new: usize, parent_indices: Option<&[usize]>) {
        match parent_indices {
            Some(parents) => {
                let inherited: Vec<f64> = parents
                    .iter()
                    .map(|&p| {
                        if p < self.ratings.len() {
                            self.ratings[p]
                        } else {
                            self.base
                        }
                    })
                    .collect();
                self.ratings.extend(inherited);
            }
            None => self.ratings.extend(std::iter::repeat_n(self.base, n_new)),
        }
    }

    pub fn len(&self) -> usize {
        self.ratings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ratings.is_empty()
    }
}
