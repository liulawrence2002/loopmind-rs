//! Pretrain and SFT drivers.
//!
//! M4 owns the pretrain loop (random loop count per sequence, truncated
//! backprop). M5 owns SFT. This crate will depend on `loopmind-model` and
//! `loopmind-data`. It does not depend on Burn; the tensor framework stays
//! inside `loopmind-model`.

/// Milestone that owns the pretrain loop.
pub const PRETRAIN_MILESTONE: &str = "M4";

/// Milestone that owns supervised finetuning.
pub const SFT_MILESTONE: &str = "M5";

/// What `loopmind pretrain` reports until M4.
pub fn pretrain_status() -> &'static str {
    "pretrain is not implemented yet (planned for M4)"
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
