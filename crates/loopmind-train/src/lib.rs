//! Pretrain loop. SFT stays a stub until M5.
//!
//! Burn stays in `loopmind-model`. This crate owns the epoch cursor, the
//! learning-rate schedule, the logs, and the checkpoints.

mod config;
mod error;
mod run;
mod schedule;

pub use config::PretrainConfig;
pub use error::TrainError;
pub use run::{generate, pretrain, RunOverrides, RunSummary};
pub use schedule::learning_rate;

/// Milestone that owns the pretrain loop.
pub const PRETRAIN_MILESTONE: &str = "M4";

/// Milestone that owns supervised finetuning.
pub const SFT_MILESTONE: &str = "M5";

/// What `loopmind pretrain` is.
pub fn pretrain_status() -> &'static str {
    "M4 pretrain loop"
}

/// What `loopmind sft` reports until M5.
pub fn sft_status() -> &'static str {
    "sft is not implemented yet (planned for M5)"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statuses_name_their_milestones() {
        assert!(pretrain_status().contains(PRETRAIN_MILESTONE));
        assert!(sft_status().contains(SFT_MILESTONE));
    }
}
