//! Pretrain and SFT shards, plus the loaders that sample them.
//!
//! Shard bytes and batches are plain integers. Burn tensors are built in
//! `loopmind-model` so this crate stays free of the framework.

mod config;
mod error;
mod loader;
mod plan;
mod pretrain;
mod sft;
mod shard;

pub use config::DataConfig;
pub use error::DataError;
pub use loader::{EpochCursor, PretrainBatch, PretrainLoader, SftBatch, SftLoader};
pub use plan::{
    pretrain_tokens, scaled_budgets, CorpusSlice, SftPolicy, PRETRAIN_SLICES,
    PRETRAIN_TOKEN_TARGET, SFT_POLICY,
};
pub use pretrain::{prepare_pretrain, PretrainManifest, SourceReport};
pub use sft::{prepare_sft, ChatRow, DropCounts, SftManifest};
pub use shard::{
    read_pretrain_sequence, read_pretrain_shard, write_pretrain_shard, write_sft_shard,
    PretrainShard, SftRow,
};
