//! Pretrain (M4) and SFT (M5) loops.
//!
//! Burn stays in `loopmind-model`. This crate owns the epoch cursor, the
//! learning-rate schedule, the logs, and the checkpoints.

mod config;
mod error;
mod run;
mod schedule;
mod sft;

pub use config::{PretrainConfig, SftConfig};
pub use error::TrainError;
pub use run::{generate, load_inference, pretrain, resolve_checkpoint, RunOverrides, RunSummary};
pub use schedule::learning_rate;
pub use sft::sft;

/// Milestone that owns the pretrain loop.
pub const PRETRAIN_MILESTONE: &str = "M4";

/// Milestone that owns supervised finetuning.
pub const SFT_MILESTONE: &str = "M5";

/// What `loopmind pretrain` is.
pub fn pretrain_status() -> &'static str {
    "M4 pretrain loop"
}

/// What `loopmind sft` is.
pub fn sft_status() -> &'static str {
    "M5 SFT loop"
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
