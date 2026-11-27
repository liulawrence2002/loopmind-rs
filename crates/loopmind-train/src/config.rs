//! `configs/pretrain.toml`.

use std::fs;
use std::path::{Path, PathBuf};

use loopmind_model::LoopMindConfig;
use serde::{Deserialize, Serialize};

use crate::TrainError;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PretrainConfig {
    pub model: PathBuf,
    pub data_dir: PathBuf,
    pub run_dir: PathBuf,
    pub seed: u64,
    pub epochs: u64,
    pub max_steps: u64,
    pub device: String,
    pub peak_tflops: f64,
    pub optim: OptimConfig,
    pub batch: BatchConfig,
    pub log: LogConfig,
    pub checkpoint: CheckpointConfig,
    #[serde(default)]
    pub val_loops: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimConfig {
    pub learning_rate: f64,
    pub min_learning_rate: f64,
    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub grad_clip: f64,
    pub warmup_steps: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchConfig {
    pub micro_batch: usize,
    pub tokens_per_step: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogConfig {
    pub every: u64,
    pub val_every: u64,
    pub val_sequences: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckpointConfig {
    pub every: u64,
    pub keep: usize,
}

impl PretrainConfig {
    pub fn from_path(path: &Path) -> Result<Self, TrainError> {
        let text = fs::read_to_string(path)?;
        let mut config: Self = toml::from_str(&text)?;
        if config.val_loops.is_empty() {
            config.val_loops = vec![1, 2, 4, 8];
        }
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), TrainError> {
        if self.epochs == 0 {
            return Err(TrainError::new("epochs must be greater than 0"));
        }
        if self.batch.micro_batch == 0 || self.batch.tokens_per_step == 0 {
            return Err(TrainError::new("batch sizes must be greater than 0"));
        }
        if !(self.optim.learning_rate.is_finite() && self.optim.learning_rate > 0.0) {
            return Err(TrainError::new("learning_rate must be greater than 0"));
        }
        if !(self.optim.min_learning_rate.is_finite()
            && self.optim.min_learning_rate >= 0.0
            && self.optim.min_learning_rate <= self.optim.learning_rate)
        {
            return Err(TrainError::new(
                "min_learning_rate must sit between 0 and learning_rate",
            ));
        }
        if self.optim.grad_clip < 0.0 {
            return Err(TrainError::new("grad_clip must be at least 0"));
        }
        if self.checkpoint.keep == 0 {
            return Err(TrainError::new("checkpoint.keep must be at least 1"));
        }
        if self.peak_tflops <= 0.0 {
            return Err(TrainError::new("peak_tflops must be greater than 0"));
        }
        for loops in &self.val_loops {
            if *loops == 0 {
                return Err(TrainError::new("val loop counts must be at least 1"));
            }
        }
        Ok(())
    }

    pub fn load_model(&self) -> Result<LoopMindConfig, TrainError> {
        Ok(LoopMindConfig::from_path(&self.model)?)
    }
}

impl From<std::io::Error> for TrainError {
    fn from(error: std::io::Error) -> Self {
        Self::new(error.to_string())
    }
}

impl From<toml::de::Error> for TrainError {
    fn from(error: toml::de::Error) -> Self {
        Self::new(error.to_string())
    }
}

impl From<loopmind_model::ConfigError> for TrainError {
    fn from(error: loopmind_model::ConfigError) -> Self {
        Self::new(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_in_pretrain_config_is_one_4070_epoch() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/pretrain.toml");
        let config = PretrainConfig::from_path(&path).unwrap();
        assert_eq!(config.epochs, 1);
        assert_eq!(config.max_steps, 0);
        assert_eq!(config.batch.micro_batch, 1);
        assert_eq!(config.batch.tokens_per_step, 8192);
        assert_eq!(config.optim.warmup_steps, 100);
        assert_eq!(config.val_loops, vec![1, 2, 4, 8]);
        assert!((config.optim.learning_rate - 3.0e-4).abs() < 1e-12);
        assert_eq!(config.checkpoint.keep, 2);
    }
}
