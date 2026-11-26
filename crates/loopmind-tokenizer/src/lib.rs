//! Byte-pair encoder for LoopMind.
//!
//! M2 trains an English-first BPE of [`TARGET_VOCAB`] tokens with the
//! `tokenizers` crate. The vocabulary size matches `LoopMindConfig::vocab_size`.

/// Vocabulary size the M2 trainer has to produce.
pub const TARGET_VOCAB: usize = 6_400;

/// Milestone that owns tokenizer training.
pub const TRAIN_MILESTONE: &str = "M2";

/// What `loopmind tokenize-train` reports until M2.
pub fn train_status() -> &'static str {
    "tokenize-train is not implemented yet (planned for M2)"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_names_the_milestone() {
        assert_eq!(TARGET_VOCAB, 6_400);
        assert!(train_status().contains(TRAIN_MILESTONE));
    }
}
