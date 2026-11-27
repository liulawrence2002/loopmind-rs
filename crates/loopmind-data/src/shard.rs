//! Compact little-endian shard files.
//!
//! Pretrain shards hold full sequences of `u16` token ids. SFT shards hold
//! variable-length token rows plus a 0/1 loss mask.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::DataError;

pub const PRETRAIN_MAGIC: &[u8; 4] = b"LMDP";
pub const SFT_MAGIC: &[u8; 4] = b"LMDS";
const VERSION: u16 = 1;

#[derive(Debug, Clone)]
pub struct PretrainShard {
    pub seq_len: usize,
    pub sequences: Vec<Vec<u16>>,
}

pub fn write_pretrain_shard(
    path: &Path,
    seq_len: usize,
    sequences: &[Vec<u16>],
) -> Result<String, DataError> {
    let mut body = Vec::new();
    body.extend_from_slice(PRETRAIN_MAGIC);
    body.extend_from_slice(&VERSION.to_le_bytes());
    body.extend_from_slice(&(seq_len as u16).to_le_bytes());
    body.extend_from_slice(&(sequences.len() as u32).to_le_bytes());
    for sequence in sequences {
        if sequence.len() != seq_len {
            return Err(DataError::new(format!(
                "sequence length {} does not match {seq_len}",
                sequence.len()
            )));
        }
        for token in sequence {
            body.extend_from_slice(&token.to_le_bytes());
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("bin.partial");
    {
        let mut file = File::create(&tmp)?;
        file.write_all(&body)?;
        file.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(sha256_hex(&body))
}

pub fn read_pretrain_shard(path: &Path) -> Result<PretrainShard, DataError> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 12 || &bytes[0..4] != PRETRAIN_MAGIC {
        return Err(DataError::new(format!(
            "not a pretrain shard: {}",
            path.display()
        )));
    }
    let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
    if version != VERSION {
        return Err(DataError::new(format!(
            "unsupported shard version {version}"
        )));
    }
    let seq_len = u16::from_le_bytes(bytes[6..8].try_into().unwrap()) as usize;
    let count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let mut sequences = Vec::with_capacity(count);
    let mut cursor = 12;
    for _ in 0..count {
        let mut sequence = Vec::with_capacity(seq_len);
        for _ in 0..seq_len {
            let token = u16::from_le_bytes(
                bytes
                    .get(cursor..cursor + 2)
                    .ok_or_else(|| DataError::new("truncated pretrain shard"))?
                    .try_into()
                    .unwrap(),
            );
            sequence.push(token);
            cursor += 2;
        }
        sequences.push(sequence);
    }
    Ok(PretrainShard { seq_len, sequences })
}

#[derive(Debug, Clone)]
pub struct SftRow {
    pub tokens: Vec<u16>,
    pub mask: Vec<u8>,
}

pub fn write_sft_shard(path: &Path, rows: &[SftRow]) -> Result<String, DataError> {
    let mut body = Vec::new();
    body.extend_from_slice(SFT_MAGIC);
    body.extend_from_slice(&VERSION.to_le_bytes());
    body.extend_from_slice(&(rows.len() as u32).to_le_bytes());
    for row in rows {
        if row.tokens.len() != row.mask.len() || row.tokens.len() > u16::MAX as usize {
            return Err(DataError::new("sft row mask does not match tokens"));
        }
        body.extend_from_slice(&(row.tokens.len() as u16).to_le_bytes());
        for token in &row.tokens {
            body.extend_from_slice(&token.to_le_bytes());
        }
        body.extend_from_slice(&row.mask);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("bin.partial");
    {
        let mut file = File::create(&tmp)?;
        file.write_all(&body)?;
        file.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(sha256_hex(&body))
}

pub fn read_sft_shard(path: &Path) -> Result<Vec<SftRow>, DataError> {
    let mut file = File::open(path)?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != SFT_MAGIC {
        return Err(DataError::new(format!(
            "not an sft shard: {}",
            path.display()
        )));
    }
    let mut version = [0u8; 2];
    file.read_exact(&mut version)?;
    if u16::from_le_bytes(version) != VERSION {
        return Err(DataError::new("unsupported sft shard version"));
    }
    let mut count = [0u8; 4];
    file.read_exact(&mut count)?;
    let count = u32::from_le_bytes(count);
    let mut rows = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let mut len_buf = [0u8; 2];
        file.read_exact(&mut len_buf)?;
        let len = u16::from_le_bytes(len_buf) as usize;
        let mut tokens = vec![0u16; len];
        for token in &mut tokens {
            let mut buf = [0u8; 2];
            file.read_exact(&mut buf)?;
            *token = u16::from_le_bytes(buf);
        }
        let mut mask = vec![0u8; len];
        file.read_exact(&mut mask)?;
        rows.push(SftRow { tokens, mask });
    }
    Ok(rows)
}

pub fn pretrain_shard_meta(path: &Path) -> Result<(usize, u32), DataError> {
    let mut file = File::open(path)?;
    let mut header = [0u8; 12];
    file.read_exact(&mut header)?;
    if &header[0..4] != PRETRAIN_MAGIC {
        return Err(DataError::new(format!(
            "not a pretrain shard: {}",
            path.display()
        )));
    }
    let seq_len = u16::from_le_bytes(header[6..8].try_into().unwrap()) as usize;
    let count = u32::from_le_bytes(header[8..12].try_into().unwrap());
    Ok((seq_len, count))
}

pub fn read_pretrain_sequence(path: &Path, seq_index: u32) -> Result<Vec<u16>, DataError> {
    let mut file = File::open(path)?;
    let mut header = [0u8; 12];
    file.read_exact(&mut header)?;
    if &header[0..4] != PRETRAIN_MAGIC {
        return Err(DataError::new("not a pretrain shard"));
    }
    let seq_len = u16::from_le_bytes(header[6..8].try_into().unwrap()) as usize;
    let count = u32::from_le_bytes(header[8..12].try_into().unwrap());
    if seq_index >= count {
        return Err(DataError::new("sequence index past the end of the shard"));
    }
    let offset = 12 + seq_index as u64 * seq_len as u64 * 2;
    use std::io::{Seek, SeekFrom};
    file.seek(SeekFrom::Start(offset))?;
    let mut sequence = Vec::with_capacity(seq_len);
    for _ in 0..seq_len {
        let mut buf = [0u8; 2];
        file.read_exact(&mut buf)?;
        sequence.push(u16::from_le_bytes(buf));
    }
    Ok(sequence)
}

pub fn sha256_file(path: &Path) -> Result<String, DataError> {
    Ok(sha256_bytes(&std::fs::read(path)?))
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    sha256_hex(bytes)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}
