//! Pretrain mix and SFT selection rules.
//!
//! Token counts are the contract the shard builder has to hit.

/// One slice of the English-first pretrain mix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CorpusSlice {
    /// Short name used in logs.
    pub name: &'static str,
    /// Dataset id, plus the split or column the loader has to read.
    pub dataset: &'static str,
    /// Tokens to take from this slice at the default 50M budget.
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

/// Scale the 50/20/20/10 mix to `total_tokens`.
///
/// Integer division keeps 50_000_000 at 25M/10M/10M/5M. The Wikipedia slice
/// absorbs the remainder so the four budgets sum to `total_tokens`.
pub fn scaled_budgets(total_tokens: u64) -> [(&'static str, u64); 4] {
    let fineweb = total_tokens / 2;
    let cosmopedia = total_tokens / 5;
    let tinystories = total_tokens / 5;
    let wiki = total_tokens - fineweb - cosmopedia - tinystories;
    [
        ("fineweb", fineweb),
        ("cosmopedia", cosmopedia),
        ("tinystories", tinystories),
        ("fable-wiki", wiki),
    ]
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

    #[test]
    fn default_budget_matches_the_named_slices() {
        let budgets = scaled_budgets(PRETRAIN_TOKEN_TARGET);
        let expected = [25_000_000, 10_000_000, 10_000_000, 5_000_000];
        for (budget, expected) in budgets.iter().zip(expected) {
            assert_eq!(budget.1, expected);
        }
    }

    #[test]
    fn larger_budget_keeps_the_same_ratios() {
        let budgets = scaled_budgets(200_000_000);
        assert_eq!(
            budgets.map(|(_, tokens)| tokens),
            [100_000_000, 40_000_000, 40_000_000, 20_000_000]
        );
    }
}
