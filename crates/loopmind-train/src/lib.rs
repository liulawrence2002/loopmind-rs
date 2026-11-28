//! Pretrain (M4) and SFT (M5) loops.
//!
//! Burn stays in `loopmind-model`. This crate owns the epoch cursor, the
//! learning-rate schedule, the logs, and the checkpoints.

mod bench;
mod config;
mod dpo;
mod error;
mod eval;
mod run;
mod schedule;
mod score;
mod sft;

pub use bench::{
    bench, bench_markdown, choice_rows, headline_metric, read_task, summarize, BenchItem,
    BenchOptions, BenchReport, TaskScore,
};
pub use config::{DpoConfig, PretrainConfig, SftConfig};
pub use dpo::{dpo, make_pairs, read_pairs, PairOptions, PairReport, PreferencePair};
pub use error::TrainError;
pub use eval::{
    evaluate, read_report, report_markdown, sampling_named, write_report, BestModel, EvalOptions,
    EvalReport, Sampling, SMOKE_PROMPTS,
};
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
