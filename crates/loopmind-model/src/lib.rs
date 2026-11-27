//! LoopMind architecture and parameter census.
//!
//! Burn is a dependency of this crate only. The data crate returns plain
//! token batches; [`pretrain_batch`] turns them into tensors. An M1 Candle bakeoff
//! stays behind the same boundary.
//!
//! # Examples
//!
//! ```
//! use loopmind_model::LoopMindConfig;
//!
//! let census = LoopMindConfig::default().param_breakdown().unwrap();
//! assert_eq!(census.total, 9_914_624);
//! assert_eq!(census.active, 2_377_984);
//! ```

mod batches;
mod blocks;
mod config;
mod engine;
mod framework;
mod loopmind;
mod loss;
mod params;
mod rng;

#[cfg(test)]
mod model_tests;

pub use batches::{pretrain_batch, sft_batch};
pub use blocks::Mixture;
pub use config::{ConfigError, LoopMindConfig};
pub use engine::{
    ApplyStats, EngineDevice, MicroStats, OptimSettings, Precision, PretrainEngine, SftStats,
};
pub use framework::{compiled_backends, framework_name};
pub use loopmind::{
    export_parity_fixture, greedy_continue, sample_continue, ForwardOutput, GenerateOptions,
    LoopMind,
};
pub use loss::{masked_shifted_nll_sum, shifted_token_cross_entropy, token_cross_entropy};
pub use params::ParamBreakdown;
pub use rng::TrainRng;
