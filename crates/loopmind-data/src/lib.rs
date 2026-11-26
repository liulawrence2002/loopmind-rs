//! Pretrain mix and SFT selection rules.
//!
//! M3 turns these budgets into loaders. Token counts are the contract.

/// One slice of the English-first pretrain mix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CorpusSlice {
    /// Short name used in logs.
    pub name: &'static str,
    /// Dataset id, plus the split or column the loader has to read.
    pub dataset: &'static str,
    /// Tokens to take from this slice.
    pub tokens: u64,
}

/// Pretrain sources, in mix order. The token budgets sum to [`PRETRAIN_TOKEN_TARGET`].
pub const PRETRAIN_SLICES: &[CorpusSlice] = &[
    CorpusSlice {
        name: "FineWeb-Edu",
        dataset: "FineWeb-Edu sample-10BT",
        tokens: 25_000_000,
    },
    CorpusSlice {
        name: "Cosmopedia-v2",
        dataset: "Cosmopedia-v2 (SmolLM corpus)",
        tokens: 10_000_000,
    },
    CorpusSlice {
        name: "TinyStories",
        dataset: "TinyStories",
        tokens: 10_000_000,
    },
    CorpusSlice {
        name: "English Wikipedia",
        dataset: "KeremTT/FABLE-5.2-MAX-DATABASE raw_text_corpus",
        tokens: 5_000_000,
    },
];

/// Pretrain length in tokens.
pub const PRETRAIN_TOKEN_TARGET: u64 = 50_000_000;

/// How English chat SFT rows are chosen from FABLE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SftPolicy {
    /// Source dataset.
    pub dataset: &'static str,
    /// Language kept for training and eval.
    pub keep_language: &'static str,
    /// Languages removed before the length filter.
    pub drop_languages: &'static [&'static str],
    /// Drop rows longer than this, matching the model context.
    pub max_tokens: usize,
    /// Checkable eval rows held out of training.
    pub eval_holdout: usize,
}

/// SFT selection rules. DPO is out of scope.
pub const SFT_POLICY: SftPolicy = SftPolicy {
    dataset: "KeremTT/FABLE-5.2-MAX-DATABASE",
    keep_language: "english",
    drop_languages: &["turkish"],
    max_tokens: 1_024,
    eval_holdout: 2_000,
};

/// Tokens in the pretrain mix.
pub fn pretrain_tokens() -> u64 {
    PRETRAIN_SLICES.iter().map(|slice| slice.tokens).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pretrain_mix_sums_to_the_target() {
        assert_eq!(pretrain_tokens(), PRETRAIN_TOKEN_TARGET);
        assert_eq!(PRETRAIN_TOKEN_TARGET, 50_000_000);
        assert_eq!(PRETRAIN_SLICES.len(), 4);
    }

    #[test]
    fn sft_holds_out_two_thousand_english_rows_at_the_context_cap() {
        assert_eq!(SFT_POLICY.max_tokens, 1_024);
        assert_eq!(SFT_POLICY.eval_holdout, 2_000);
        assert_eq!(SFT_POLICY.keep_language, "english");
        assert_eq!(SFT_POLICY.drop_languages, &["turkish"]);
    }
}
