//! Warmup followed by cosine decay.
//!
//! Step 0 is the first optimizer update. During warmup the rate climbs from
//! `peak / warmup_steps` to `peak`. The cosine half then moves from `peak`
//! toward `min_learning_rate`, reaching it when `step == total_steps`.

pub fn learning_rate(
    step: u64,
    warmup_steps: u64,
    total_steps: u64,
    peak: f64,
    min_learning_rate: f64,
) -> f64 {
    if warmup_steps > 0 && step < warmup_steps {
        return peak * (step + 1) as f64 / warmup_steps as f64;
    }
    let span = total_steps.saturating_sub(warmup_steps).max(1);
    let into = step.saturating_sub(warmup_steps);
    let progress = (into as f64 / span as f64).clamp(0.0, 1.0);
    let cosine = (progress * std::f64::consts::PI).cos();
    min_learning_rate + 0.5 * (peak - min_learning_rate) * (1.0 + cosine)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warmup_hits_the_peak_and_cosine_ends_at_the_floor() {
        let peak = 1.0e-3;
        let floor = 1.0e-4;
        assert!((learning_rate(0, 10, 100, peak, floor) - peak / 10.0).abs() < 1e-15);
        assert!((learning_rate(9, 10, 100, peak, floor) - peak).abs() < 1e-15);
        assert!((learning_rate(10, 10, 100, peak, floor) - peak).abs() < 1e-12);
        let mid = learning_rate(55, 10, 100, peak, floor);
        assert!(mid < peak && mid > floor);
        assert!((learning_rate(100, 10, 100, peak, floor) - floor).abs() < 1e-12);
        assert!(learning_rate(20, 10, 100, peak, floor) < learning_rate(15, 10, 100, peak, floor));
    }
}
