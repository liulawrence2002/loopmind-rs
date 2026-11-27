//! `configs/data.toml`.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::DataError;
use crate::plan::scaled_budgets;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataConfig {
    pub seed: u64,
    pub tokenizer: PathBuf,
    pub output_dir: PathBuf,
    pub python: String,
    pub revisions: Revisions,
    pub pretrain: PretrainConfig,
    pub sft: SftConfig,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Revisions {
    pub fineweb: String,
    pub cosmopedia: String,
    pub tinystories: String,
    pub fable: String,
    /// Commit of `jingyaogong/minimind_dataset`, for `sft.minimind_file`. Left out of
    /// the build fingerprint when unset, so configs without it keep their builds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimind: Option<String>,
    /// Commit of `HuggingFaceTB/smol-smoltalk`, for `sft.smoltalk`. Left out of the build
    /// fingerprint when unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub smoltalk: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PretrainConfig {
    pub total_tokens: u64,
    pub context_length: usize,
    pub val_ratio: f64,
    pub shard_sequences: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SftConfig {
    pub max_tokens: usize,
    pub eval_rows: usize,
    pub exclude_cc_by_sa: bool,
    /// Local path of a MiniMind SFT file (`sft_t2t_mini.jsonl` or `sft_t2t.jsonl`).
    /// Downloaded from `jingyaogong/minimind_dataset` at `revisions.minimind` when it is
    /// missing. Its English rows are added to the SFT train split after FABLE's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimind_file: Option<PathBuf>,
    /// Add the English train split of `HuggingFaceTB/smol-smoltalk` at
    /// `revisions.smoltalk`, after MiniMind's rows. Long chats keep their first turns.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub smoltalk: bool,
}

impl DataConfig {
    pub fn from_path(path: &Path) -> Result<Self, DataError> {
        let text = fs::read_to_string(path)?;
        let config: Self =
            toml::from_str(&text).map_err(|error| DataError::new(error.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), DataError> {
        if self.pretrain.total_tokens == 0 {
            return Err(DataError::new(
                "pretrain.total_tokens must be greater than 0",
            ));
        }
        if self.pretrain.context_length == 0 {
            return Err(DataError::new(
                "pretrain.context_length must be greater than 0",
            ));
        }
        if !(self.pretrain.val_ratio > 0.0 && self.pretrain.val_ratio < 0.5) {
            return Err(DataError::new(
                "pretrain.val_ratio must be between 0 and 0.5",
            ));
        }
        if self.pretrain.shard_sequences == 0 {
            return Err(DataError::new(
                "pretrain.shard_sequences must be greater than 0",
            ));
        }
        if self.sft.max_tokens == 0 || self.sft.eval_rows == 0 {
            return Err(DataError::new("sft limits must be greater than 0"));
        }
        if self.sft.smoltalk && self.revisions.smoltalk.is_none() {
            return Err(DataError::new(
                "sft.smoltalk needs revisions.smoltalk (a dataset commit)",
            ));
        }
        if self.sft.minimind_file.is_some() && self.revisions.minimind.is_none() {
            return Err(DataError::new(
                "sft.minimind_file needs revisions.minimind (a dataset commit)",
            ));
        }
        Ok(())
    }

    pub fn pretrain_budgets(&self) -> [(&'static str, u64); 4] {
        scaled_budgets(self.pretrain.total_tokens)
    }

    pub fn packed_tokens(budget: u64, seq_len: usize) -> u64 {
        let seq = seq_len as u64;
        budget / seq * seq
    }

    pub fn revision(&self, source: &str) -> Result<&str, DataError> {
        Ok(match source {
            "fineweb" => self.revisions.fineweb.as_str(),
            "cosmopedia" => self.revisions.cosmopedia.as_str(),
            "tinystories" => self.revisions.tinystories.as_str(),
            "fable-wiki" => self.revisions.fable.as_str(),
            "fable" => self.revisions.fable.as_str(),
            "minimind" => self
                .revisions
                .minimind
                .as_deref()
                .ok_or_else(|| DataError::new("revisions.minimind is not set"))?,
            "smoltalk" => self
                .revisions
                .smoltalk
                .as_deref()
                .ok_or_else(|| DataError::new("revisions.smoltalk is not set"))?,
            _ => return Err(DataError::new(format!("unknown source {source}"))),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn checked_in_config_is_the_50m_mix() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/data.toml");
        let config = DataConfig::from_path(&path).unwrap();
        assert_eq!(config.seed, 42);
        assert_eq!(config.pretrain.total_tokens, 50_000_000);
        assert_eq!(config.pretrain.context_length, 1_024);
        assert!((config.pretrain.val_ratio - 0.01).abs() < 1e-12);
        assert_eq!(config.sft.max_tokens, 1_024);
        assert_eq!(config.sft.eval_rows, 2_000);
        assert!(!config.sft.exclude_cc_by_sa);
        assert_eq!(config.revision("minimind").unwrap().len(), 40);
        assert!(config
            .sft
            .minimind_file
            .as_ref()
            .is_some_and(|file| file.ends_with("sft_t2t_mini.jsonl")));
        assert!(config.sft.smoltalk);
        assert_eq!(config.revision("smoltalk").unwrap().len(), 40);
        assert_eq!(
            config.pretrain_budgets().map(|(_, tokens)| tokens),
            [25_000_000, 10_000_000, 10_000_000, 5_000_000]
        );
        assert_eq!(config.revision("fineweb").unwrap().len(), 40);
    }

    #[test]
    fn minimind_is_optional_and_left_out_of_the_fingerprint_when_unset() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/data.toml");
        let mut config = DataConfig::from_path(&path).unwrap();
        config.sft.minimind_file = None;
        config.revisions.minimind = None;
        config.sft.smoltalk = false;
        config.revisions.smoltalk = None;
        let view = serde_json::to_string(&config).unwrap();
        assert!(!view.contains("minimind") && !view.contains("smoltalk"));
        config.sft.minimind_file = Some("x.jsonl".into());
        assert!(config.validate().is_err());
        config.sft.minimind_file = None;
        config.sft.smoltalk = true;
        assert!(config.validate().is_err());
    }
}
