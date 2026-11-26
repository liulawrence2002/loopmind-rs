//! Architecture config. The checked-in file is `configs/default.toml`.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Recoverable config failure: I/O, TOML, or a broken architecture invariant.
#[derive(Debug)]
pub enum ConfigError {
    /// The config file could not be read.
    Io(std::io::Error),
    /// The file was not the expected TOML object.
    Parse(toml::de::Error),
    /// A field broke an architecture invariant.
    Invalid(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Io(err) => write!(f, "failed to read config: {err}"),
            ConfigError::Parse(err) => write!(f, "invalid config toml: {err}"),
            ConfigError::Invalid(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ConfigError::Io(err) => Some(err),
            ConfigError::Parse(err) => Some(err),
            ConfigError::Invalid(_) => None,
        }
    }
}

impl From<std::io::Error> for ConfigError {
    fn from(err: std::io::Error) -> Self {
        ConfigError::Io(err)
    }
}

impl From<toml::de::Error> for ConfigError {
    fn from(err: toml::de::Error) -> Self {
        ConfigError::Parse(err)
    }
}

/// Huginn-style recurrent depth with a fine-grained MoE core.
///
/// Layout: one dense prelude layer, a core of [`Self::n_core_layers`] MoE
/// layers applied `r` times with shared weights, then one dense coda layer.
/// Each block is pre-norm RMSNorm, bias-free grouped-query attention with
/// RoPE, and SwiGLU. Prelude and coda use a dense SwiGLU. Each core layer
/// routes top-[`Self::experts_per_token`] of [`Self::n_experts`] SwiGLU
/// experts and is re-routed on every loop pass.
///
/// Embeddings are tied with the output head when [`Self::tie_embeddings`] is
/// set. Loop count scales compute, not the stored parameter count.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoopMindConfig {
    /// Residual width.
    pub d_model: usize,
    /// Query heads. Head dim is `d_model / n_q_heads`.
    pub n_q_heads: usize,
    /// Key and value heads (grouped-query attention).
    pub n_kv_heads: usize,
    /// RoPE base frequency. Not a parameter.
    pub rope_theta: f64,
    /// RMSNorm epsilon. Not a parameter.
    pub rms_norm_eps: f64,
    /// BPE vocabulary size. The M2 tokenizer target matches this.
    pub vocab_size: usize,
    /// Share the token embedding with the output projection.
    pub tie_embeddings: bool,
    /// Maximum sequence length. RoPE has no learned position table.
    pub context_length: usize,
    /// Dense transformer layers before the loop.
    pub n_prelude_layers: usize,
    /// MoE transformer layers inside the shared loop.
    pub n_core_layers: usize,
    /// Dense transformer layers after the loop.
    pub n_coda_layers: usize,
    /// SwiGLU intermediate width of each dense (prelude and coda) layer.
    pub dense_intermediate_size: usize,
    /// Experts in each core layer.
    pub n_experts: usize,
    /// SwiGLU intermediate width of one expert.
    pub expert_intermediate_size: usize,
    /// Experts selected per token per core layer (top-k).
    pub experts_per_token: usize,
    /// Routers run again on every loop pass. This adds no parameters.
    pub reroute_every_pass: bool,
    /// Mean of the per-sequence training loop count.
    pub loop_count_mean: f64,
    /// Maximum training or inference loop count.
    pub loop_count_max: usize,
    /// Truncated backprop depth, in loop passes.
    pub bptt_passes: usize,
}

impl Default for LoopMindConfig {
    fn default() -> Self {
        Self {
            d_model: 256,
            n_q_heads: 4,
            n_kv_heads: 2,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            vocab_size: 6_400,
            tie_embeddings: true,
            context_length: 1_024,
            n_prelude_layers: 1,
            n_core_layers: 2,
            n_coda_layers: 1,
            // Two-thirds of 4 * d_model (682), rounded up to a multiple of 256.
            dense_intermediate_size: 768,
            n_experts: 32,
            expert_intermediate_size: 128,
            experts_per_token: 2,
            reroute_every_pass: true,
            loop_count_mean: 4.0,
            loop_count_max: 16,
            bptt_passes: 4,
        }
    }
}

impl LoopMindConfig {
    /// Query/key/value head dimension. Valid after [`Self::validate`].
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_q_heads
    }

    /// Check architecture invariants used by the census and the future forward pass.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.d_model == 0 {
            return Err(invalid("d_model must be greater than 0"));
        }
        if self.n_q_heads == 0 {
            return Err(invalid("n_q_heads must be greater than 0"));
        }
        if self.n_kv_heads == 0 {
            return Err(invalid("n_kv_heads must be greater than 0"));
        }
        if !self.d_model.is_multiple_of(self.n_q_heads) {
            return Err(invalid("d_model must be divisible by n_q_heads"));
        }
        if !self.n_q_heads.is_multiple_of(self.n_kv_heads) {
            return Err(invalid("n_q_heads must be divisible by n_kv_heads"));
        }
        if !(self.rope_theta.is_finite() && self.rope_theta > 0.0) {
            return Err(invalid("rope_theta must be finite and greater than 0"));
        }
        if !(self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0) {
            return Err(invalid("rms_norm_eps must be finite and greater than 0"));
        }
        if self.vocab_size == 0 {
            return Err(invalid("vocab_size must be greater than 0"));
        }
        if self.context_length == 0 {
            return Err(invalid("context_length must be greater than 0"));
        }
        if self.dense_intermediate_size == 0 {
            return Err(invalid("dense_intermediate_size must be greater than 0"));
        }
        if self.n_experts == 0 {
            return Err(invalid("n_experts must be greater than 0"));
        }
        if self.expert_intermediate_size == 0 {
            return Err(invalid("expert_intermediate_size must be greater than 0"));
        }
        if self.experts_per_token == 0 || self.experts_per_token > self.n_experts {
            return Err(invalid("experts_per_token must be between 1 and n_experts"));
        }
        if self.loop_count_max == 0 {
            return Err(invalid("loop_count_max must be greater than 0"));
        }
        if !(self.loop_count_mean.is_finite()
            && self.loop_count_mean > 0.0
            && self.loop_count_mean <= self.loop_count_max as f64)
        {
            return Err(invalid(
                "loop_count_mean must be greater than 0 and at most loop_count_max",
            ));
        }
        if self.bptt_passes == 0 || self.bptt_passes > self.loop_count_max {
            return Err(invalid("bptt_passes must be between 1 and loop_count_max"));
        }
        Ok(())
    }

    /// Parse a complete config document and validate it.
    pub fn from_toml_str(text: &str) -> Result<Self, ConfigError> {
        let config: Self = toml::from_str(text)?;
        config.validate()?;
        Ok(config)
    }

    /// Read a TOML config from `path` and validate it.
    pub fn from_path(path: &Path) -> Result<Self, ConfigError> {
        let text = fs::read_to_string(path)?;
        Self::from_toml_str(&text)
    }
}

fn invalid(msg: &str) -> ConfigError {
    ConfigError::Invalid(msg.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_matches_checked_in_toml() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/default.toml");
        let loaded = LoopMindConfig::from_path(&path).unwrap();
        assert_eq!(loaded, LoopMindConfig::default());
        assert_eq!(loaded.head_dim(), 64);
    }

    #[test]
    fn default_config_roundtrips_through_toml() {
        let text = toml::to_string(&LoopMindConfig::default()).unwrap();
        let loaded = LoopMindConfig::from_toml_str(&text).unwrap();
        assert_eq!(loaded, LoopMindConfig::default());
    }

    #[test]
    fn unknown_field_is_rejected() {
        let mut text = toml::to_string(&LoopMindConfig::default()).unwrap();
        text.push_str("\nextra = 1\n");
        assert!(LoopMindConfig::from_toml_str(&text).is_err());
    }

    #[test]
    fn rejects_heads_that_do_not_divide_width() {
        let config = LoopMindConfig {
            n_q_heads: 3,
            ..LoopMindConfig::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("divisible by n_q_heads"));
    }

    #[test]
    fn rejects_topk_above_expert_count() {
        let config = LoopMindConfig {
            experts_per_token: 64,
            ..LoopMindConfig::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("experts_per_token"));
    }
}
