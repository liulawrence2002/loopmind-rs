//! Loop-count sampler. Training draws one count per sequence.

/// Tiny xorshift generator so training does not take a random-number dependency.
#[derive(Debug, Clone)]
pub struct TrainRng {
    state: u64,
}

impl TrainRng {
    /// Seed the generator. A zero seed is replaced so the state does not stick at zero.
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    /// Poisson(`mean`) clamped to `1..=max`.
    ///
    /// Clamping 0 up to 1 lifts the mean by about `exp(-mean)`. For mean 4 that is ~0.018.
    pub fn sample_loops(&mut self, mean: f64, max: usize) -> usize {
        assert!(mean.is_finite() && mean > 0.0, "loop mean must be positive");
        assert!(max >= 1, "loop max must be at least 1");
        let limit = (-mean).exp();
        let mut p = 1.0;
        let mut k = 0usize;
        while p > limit {
            k += 1;
            p *= self.next_unit();
            if k > max.saturating_add(32) {
                break;
            }
        }
        k.saturating_sub(1).clamp(1, max)
    }

    fn next_unit(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        let bits = x >> 11;
        (bits as f64 + 0.5) / ((1u64 << 53) as f64)
    }
}
