//! LoopMind architecture and parameter census.
//!
//! Burn is a dependency of this crate only. Tokenizer, data, and train code
//! depend on the types exported here, not on Burn. An M1 Candle bakeoff stays
//! behind the same boundary.
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

mod blocks;
mod config;
mod framework;
mod loopmind;
mod loss;
mod params;
mod rng;

#[cfg(test)]
mod model_tests;

pub use blocks::Mixture;
pub use config::{ConfigError, LoopMindConfig};
pub use framework::{compiled_backends, framework_name};
pub use loopmind::{export_parity_fixture, ForwardOutput, LoopMind};
pub use loss::token_cross_entropy;
pub use params::ParamBreakdown;
pub use rng::TrainRng;
