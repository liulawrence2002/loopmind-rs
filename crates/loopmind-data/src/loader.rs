//! Stream packed pretrain sequences and length-bucketed SFT rows.
//!
//! These loaders stay on plain `u16` ids. `loopmind-model` turns a batch into
//! Burn tensors so this crate does not depend on Burn.

use std::fs;
use std::path::Path;

use crate::error::DataError;
use crate::shard::{self, SftRow};

const SFT_BUCKETS: [usize; 5] = [64, 128, 256, 512, 1024];

#[derive(Debug, Clone)]
pub struct PretrainBatch {
    pub tokens: Vec<u16>,
    pub batch: usize,
    pub seq_len: usize,
}

#[derive(Debug, Clone)]
pub struct SftBatch {
    pub tokens: Vec<u16>,
    pub mask: Vec<u8>,
    pub batch: usize,
    pub seq_len: usize,
}

struct ShardLoc {
    path: std::path::PathBuf,
    sequences: u32,
    seq_len: usize,
}

/// Position in a shuffled epoch. The permutation is `seed` mixed with `epoch`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EpochCursor {
    pub epoch: u64,
    pub index: u64,
    pub seed: u64,
}

impl EpochCursor {
    pub fn new(seed: u64) -> Self {
        Self {
            epoch: 0,
            index: 0,
            seed,
        }
    }
}

pub struct PretrainLoader {
    shards: Vec<ShardLoc>,
    total: u64,
    rng: Rng,
    order_key: Option<(u64, u64)>,
    order: Vec<u32>,
}

impl PretrainLoader {
    pub fn from_dir(dir: &Path, seed: u64) -> Result<Self, DataError> {
        let mut shards = Vec::new();
        let mut names = Vec::new();
        for entry in fs::read_dir(dir)? {
            let path = entry?.path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("bin") {
                names.push(path);
            }
        }
        names.sort();
        let mut total = 0u64;
        for path in names {
            let (seq_len, sequences) = shard::pretrain_shard_meta(&path)?;
            total += u64::from(sequences);
            shards.push(ShardLoc {
                path,
                sequences,
                seq_len,
            });
        }
        if shards.is_empty() || total == 0 {
            return Err(DataError::new(format!(
                "no pretrain sequences in {}",
                dir.display()
            )));
        }
        Ok(Self {
            shards,
            total,
            rng: Rng::new(seed),
            order_key: None,
            order: Vec::new(),
        })
    }

    pub fn len(&self) -> u64 {
        self.total
    }

    pub fn seq_len(&self) -> usize {
        self.shards[0].seq_len
    }

    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// Next `batch` packed sequences in epoch order.
    ///
    /// The batch is always full when the corpus is non-empty. Crossing the end of an
    /// epoch continues into the next shuffle inside the same batch.
    pub fn next_batch(
        &mut self,
        batch: usize,
        cursor: &mut EpochCursor,
    ) -> Result<PretrainBatch, DataError> {
        if batch == 0 {
            return Err(DataError::new("batch size must be greater than 0"));
        }
        if self.total == 0 {
            return Err(DataError::new("pretrain loader is empty"));
        }
        let seq_len = self.shards[0].seq_len;
        let mut tokens = Vec::with_capacity(batch * seq_len);
        let mut filled = 0usize;
        while filled < batch {
            self.ensure_order(cursor);
            if cursor.index >= self.order.len() as u64 {
                cursor.epoch += 1;
                cursor.index = 0;
                self.order_key = None;
                continue;
            }
            let sequence = self.read_sequence(u64::from(self.order[cursor.index as usize]))?;
            if sequence.len() != seq_len {
                return Err(DataError::new(
                    "pretrain shards disagree on sequence length",
                ));
            }
            tokens.extend(sequence);
            cursor.index += 1;
            filled += 1;
            if cursor.index >= self.order.len() as u64 && filled < batch {
                cursor.epoch += 1;
                cursor.index = 0;
                self.order_key = None;
            }
        }
        Ok(PretrainBatch {
            tokens,
            batch: filled,
            seq_len,
        })
    }

    fn ensure_order(&mut self, cursor: &EpochCursor) {
        let key = (cursor.seed, cursor.epoch);
        if self.order_key == Some(key) {
            return;
        }
        let n = u32::try_from(self.total).unwrap_or(u32::MAX);
        let n = n.min(self.total as u32);
        self.order = permutation(n, mix_epoch(cursor.seed, cursor.epoch));
        self.order_key = Some(key);
    }

    pub fn sample(&mut self, batch: usize) -> Result<PretrainBatch, DataError> {
        if batch == 0 {
            return Err(DataError::new("batch size must be greater than 0"));
        }
        let seq_len = self.shards[0].seq_len;
        let mut tokens = Vec::with_capacity(batch * seq_len);
        for _ in 0..batch {
            let index = self.rng.next() % self.total;
            let sequence = self.read_sequence(index)?;
            if sequence.len() != seq_len {
                return Err(DataError::new(
                    "pretrain shards disagree on sequence length",
                ));
            }
            tokens.extend(sequence);
        }
        Ok(PretrainBatch {
            tokens,
            batch,
            seq_len,
        })
    }

    fn read_sequence(&self, mut index: u64) -> Result<Vec<u16>, DataError> {
        for shard in &self.shards {
            if index < u64::from(shard.sequences) {
                return shard::read_pretrain_sequence(&shard.path, index as u32);
            }
            index -= u64::from(shard.sequences);
        }
        Err(DataError::new("pretrain sequence index was out of range"))
    }
}

pub struct SftLoader {
    rows: Vec<SftRow>,
    buckets: [Vec<usize>; 5],
    pad_id: u16,
    rng: Rng,
}

impl SftLoader {
    pub fn from_dir(dir: &Path, pad_id: u16, seed: u64) -> Result<Self, DataError> {
        let mut names = Vec::new();
        for entry in fs::read_dir(dir)? {
            let path = entry?.path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("bin") {
                names.push(path);
            }
        }
        names.sort();
        let mut rows = Vec::new();
        for path in names {
            rows.extend(shard::read_sft_shard(&path)?);
        }
        if rows.is_empty() {
            return Err(DataError::new(format!("no sft rows in {}", dir.display())));
        }
        let mut buckets: [Vec<usize>; 5] =
            [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for (index, row) in rows.iter().enumerate() {
            buckets[bucket_index(row.tokens.len())].push(index);
        }
        Ok(Self {
            rows,
            buckets,
            pad_id,
            rng: Rng::new(seed),
        })
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn sample(&mut self, batch: usize) -> Result<SftBatch, DataError> {
        if batch == 0 {
            return Err(DataError::new("batch size must be greater than 0"));
        }
        let occupied: Vec<usize> = (0..self.buckets.len())
            .filter(|bucket| !self.buckets[*bucket].is_empty())
            .collect();
        let bucket = occupied[self.rng.gen_index(occupied.len())];
        let width = bucket_width(
            self.buckets[bucket]
                .iter()
                .map(|index| self.rows[*index].tokens.len())
                .max()
                .unwrap_or(1),
        );
        let mut tokens = vec![self.pad_id; batch * width];
        let mut mask = vec![0u8; batch * width];
        for slot in 0..batch {
            let choice = self.buckets[bucket][self.rng.gen_index(self.buckets[bucket].len())];
            let row = &self.rows[choice];
            let start = slot * width;
            tokens[start..start + row.tokens.len()].copy_from_slice(&row.tokens);
            mask[start..start + row.mask.len()].copy_from_slice(&row.mask);
        }
        Ok(SftBatch {
            tokens,
            mask,
            batch,
            seq_len: width,
        })
    }
}

fn bucket_index(len: usize) -> usize {
    SFT_BUCKETS
        .iter()
        .position(|width| len <= *width)
        .unwrap_or(SFT_BUCKETS.len() - 1)
}

fn bucket_width(len: usize) -> usize {
    SFT_BUCKETS
        .iter()
        .copied()
        .find(|width| len <= *width)
        .unwrap_or(len)
}

fn mix_epoch(seed: u64, epoch: u64) -> u64 {
    seed ^ epoch.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA5A5_A5A5_A5A5_A5A5
}

fn permutation(n: u32, seed: u64) -> Vec<u32> {
    let mut order: Vec<u32> = (0..n).collect();
    if n <= 1 {
        return order;
    }
    let mut rng = Rng::new(seed);
    for i in (1..n).rev() {
        let j = (rng.next() % u64::from(i + 1)) as usize;
        order.swap(i as usize, j);
    }
    order
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn gen_index(&mut self, len: usize) -> usize {
        (self.next() as usize) % len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shard::{write_pretrain_shard, write_sft_shard, SftRow};

    #[test]
    fn pretrain_sampling_is_seeded_and_reads_one_sequence() {
        let dir = std::env::temp_dir().join(format!("loopmind-load-pre-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let first: Vec<Vec<u16>> = (0..3).map(|row| vec![row, 1, 2, 3]).collect();
        let second: Vec<Vec<u16>> = (0..2).map(|row| vec![10 + row, 4, 5, 6]).collect();
        write_pretrain_shard(&dir.join("a.bin"), 4, &first).unwrap();
        write_pretrain_shard(&dir.join("b.bin"), 4, &second).unwrap();
        let mut left = PretrainLoader::from_dir(&dir, 11).unwrap();
        let mut right = PretrainLoader::from_dir(&dir, 11).unwrap();
        let batch = left.sample(4).unwrap();
        assert_eq!(batch.batch, 4);
        assert_eq!(batch.seq_len, 4);
        assert_eq!(batch.tokens.len(), 16);
        assert_eq!(batch.tokens, right.sample(4).unwrap().tokens);
        assert_eq!(left.len(), 5);
    }

    #[test]
    fn epoch_cursor_is_deterministic_and_wraps() {
        let dir = std::env::temp_dir().join(format!("loopmind-epoch-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let sequences: Vec<Vec<u16>> = (0..5).map(|row| vec![row, 1, 2, 3]).collect();
        write_pretrain_shard(&dir.join("a.bin"), 4, &sequences).unwrap();
        let mut loader = PretrainLoader::from_dir(&dir, 1).unwrap();
        let mut cursor = EpochCursor::new(9);
        let first = loader.next_batch(2, &mut cursor).unwrap();
        assert_eq!(cursor.epoch, 0);
        assert_eq!(cursor.index, 2);
        let mut again = PretrainLoader::from_dir(&dir, 1).unwrap();
        let mut restarted = EpochCursor::new(9);
        assert_eq!(
            again.next_batch(2, &mut restarted).unwrap().tokens,
            first.tokens
        );
        loader.next_batch(2, &mut cursor).unwrap();
        loader.next_batch(2, &mut cursor).unwrap();
        assert_eq!(cursor.epoch, 1);
        assert_eq!(cursor.index, 1);
    }

    #[test]
    fn sft_batches_pad_inside_a_length_bucket() {
        let dir = std::env::temp_dir().join(format!("loopmind-load-sft-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let rows = vec![
            SftRow {
                tokens: vec![1, 2, 3, 4],
                mask: vec![0, 1, 1, 1],
            },
            SftRow {
                tokens: vec![5, 6, 7],
                mask: vec![0, 1, 1],
            },
        ];
        write_sft_shard(&dir.join("sft-00000.bin"), &rows).unwrap();
        let mut loader = SftLoader::from_dir(&dir, 0, 5).unwrap();
        let batch = loader.sample(2).unwrap();
        assert_eq!(batch.seq_len, 64);
        assert_eq!(batch.tokens.len(), 128);
        assert!(batch.mask.contains(&0));
        assert!(batch.tokens.contains(&0));
        let row_end = batch.mask[..64].iter().rposition(|bit| *bit == 1).unwrap();
        assert!(batch.mask[row_end + 1..64].iter().all(|bit| *bit == 0));
    }
}
