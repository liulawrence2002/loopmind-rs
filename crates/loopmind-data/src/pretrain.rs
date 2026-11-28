//! Pack pretrain documents into fixed-length `u16` sequences.
//!
//! Each document is tokenized and followed by `<|eos|>`. Sequences are exactly
//! `context_length` tokens. A source stops at the largest multiple of the
//! context that does not exceed its budget, so 50_000_000 tokens with context
//! 1024 packs to 49_997_824 across the four sources.

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use loopmind_tokenizer::{LoopTokenizer, EOS};
use serde::{Deserialize, Serialize};

use crate::config::DataConfig;
use crate::error::DataError;
use crate::shard;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardRecord {
    pub file: String,
    pub sequences: u32,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceReport {
    pub name: String,
    pub budget_tokens: u64,
    pub train_tokens: u64,
    pub val_tokens: u64,
    pub train_sequences: u64,
    pub val_sequences: u64,
    pub docs_seen: u64,
    pub train_shards: Vec<ShardRecord>,
    pub val_shards: Vec<ShardRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SourceState {
    fingerprint: String,
    docs_seen: u64,
    train_tokens: u64,
    val_tokens: u64,
    train_sequences: u64,
    val_sequences: u64,
    remainder_train: Vec<u16>,
    remainder_val: Vec<u16>,
    pending_train: Vec<Vec<u16>>,
    pending_val: Vec<Vec<u16>>,
    next_train_shard: u32,
    next_val_shard: u32,
    train_shards: Vec<ShardRecord>,
    val_shards: Vec<ShardRecord>,
    complete: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PretrainManifest {
    pub fingerprint: String,
    pub total_tokens_budget: u64,
    pub context_length: usize,
    pub val_ratio: f64,
    pub seed: u64,
    pub sources: Vec<SourceReport>,
}

pub fn prepare_pretrain(
    config: &DataConfig,
    tokenizer: &LoopTokenizer,
) -> Result<PretrainManifest, DataError> {
    let tokenizer_sha = shard::sha256_file(&config.tokenizer)?;
    let fingerprint = fingerprint(config, &tokenizer_sha)?;
    let eos = tokenizer
        .token_id(EOS)
        .ok_or_else(|| DataError::new("tokenizer is missing <|eos|>"))?;
    if eos > u16::MAX as u32 {
        return Err(DataError::new("eos id does not fit in u16"));
    }
    let root = config.output_dir.join("pretrain");
    fs::create_dir_all(root.join("state"))?;
    let manifest_path = root.join("manifest.json");
    if let Some(existing) = load_manifest(&manifest_path)? {
        if existing.fingerprint == fingerprint {
            return Ok(existing);
        }
        return Err(DataError::new(format!(
            "pretrain output at {} was built with a different config; choose another output_dir",
            config.output_dir.display()
        )));
    }
    let script = find_script()?;
    let mut sources = Vec::new();
    for (name, budget) in config.pretrain_budgets() {
        let revision = config.revision(&name)?.to_string();
        let report = ingest_hf_source(
            config,
            tokenizer,
            &script,
            &name,
            &revision,
            budget,
            eos as u16,
            &fingerprint,
        )?;
        let expected = DataConfig::packed_tokens(budget, config.pretrain.context_length);
        if report.train_tokens < expected {
            return Err(DataError::new(format!(
                "{name} packed {} tokens; the context floor of the budget is {expected}",
                report.train_tokens
            )));
        }
        eprintln!(
            "{name}: train {} tokens ({} seq), val {} tokens ({} seq), {} docs",
            report.train_tokens,
            report.train_sequences,
            report.val_tokens,
            report.val_sequences,
            report.docs_seen
        );
        sources.push(report);
    }
    let manifest = PretrainManifest {
        fingerprint,
        total_tokens_budget: config.pretrain.total_tokens,
        context_length: config.pretrain.context_length,
        val_ratio: config.pretrain.val_ratio,
        seed: config.seed,
        sources,
    };
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
    Ok(manifest)
}

/// Pack an in-memory document list. Used by tests and by the HF ingester.
#[allow(clippy::too_many_arguments)]
pub fn ingest_documents(
    out_dir: &Path,
    source: &str,
    budget: u64,
    val_ratio: f64,
    seq_len: usize,
    shard_sequences: usize,
    seed: u64,
    eos: u16,
    fingerprint: &str,
    tokenizer: &LoopTokenizer,
    docs: impl Iterator<Item = String>,
) -> Result<SourceReport, DataError> {
    let state_path = out_dir.join("state").join(format!("{source}.json"));
    let state = load_state(&state_path)?;
    if let Some(existing) = &state {
        if existing.fingerprint != fingerprint {
            return Err(DataError::new(format!(
                "source {source} progress does not match this config"
            )));
        }
        if existing.complete {
            return Ok(report_from_state(source, budget, existing));
        }
    }
    let mut state = state.unwrap_or(SourceState {
        fingerprint: fingerprint.to_string(),
        docs_seen: 0,
        train_tokens: 0,
        val_tokens: 0,
        train_sequences: 0,
        val_sequences: 0,
        remainder_train: Vec::new(),
        remainder_val: Vec::new(),
        pending_train: Vec::new(),
        pending_val: Vec::new(),
        next_train_shard: 0,
        next_val_shard: 0,
        train_shards: Vec::new(),
        val_shards: Vec::new(),
        complete: false,
    });
    let train_limit = DataConfig::packed_tokens(budget, seq_len);
    let val_limit = DataConfig::packed_tokens((budget as f64 * val_ratio) as u64, seq_len);
    let mut pending_train = std::mem::take(&mut state.pending_train);
    let mut pending_val = std::mem::take(&mut state.pending_val);
    let mut index = state.docs_seen;
    for text in docs {
        index += 1;
        state.docs_seen = index;
        let tokens = encode_doc(tokenizer, &text, eos)?;
        let to_val = is_val(seed, source, index, val_ratio);
        if to_val && state.val_tokens < val_limit {
            absorb(
                &mut state.remainder_val,
                &mut pending_val,
                &tokens,
                seq_len,
                val_limit,
                &mut state.val_tokens,
            );
        } else if !to_val && state.train_tokens < train_limit {
            absorb(
                &mut state.remainder_train,
                &mut pending_train,
                &tokens,
                seq_len,
                train_limit,
                &mut state.train_tokens,
            );
        }
        flush_ready(
            out_dir,
            source,
            "train",
            seq_len,
            shard_sequences,
            &mut pending_train,
            &mut state.next_train_shard,
            &mut state.train_shards,
            &mut state.train_sequences,
        )?;
        flush_ready(
            out_dir,
            source,
            "val",
            seq_len,
            shard_sequences,
            &mut pending_val,
            &mut state.next_val_shard,
            &mut state.val_shards,
            &mut state.val_sequences,
        )?;
        if state.train_tokens >= train_limit && state.val_tokens >= val_limit {
            break;
        }
        if index.is_multiple_of(2_000) {
            checkpoint(&state_path, &mut state, &pending_train, &pending_val)?;
            eprintln!(
                "{source}: {index} docs, train {} / {train_limit} tokens, val {} / {val_limit}",
                state.train_tokens, state.val_tokens
            );
        }
    }
    flush_tail(
        out_dir,
        source,
        "train",
        seq_len,
        &mut pending_train,
        &mut state.next_train_shard,
        &mut state.train_shards,
        &mut state.train_sequences,
    )?;
    flush_tail(
        out_dir,
        source,
        "val",
        seq_len,
        &mut pending_val,
        &mut state.next_val_shard,
        &mut state.val_shards,
        &mut state.val_sequences,
    )?;
    if state.train_tokens >= train_limit {
        state.remainder_train.clear();
        state.remainder_val.clear();
    }
    state.complete = state.train_tokens >= train_limit && state.val_tokens >= val_limit;
    checkpoint(&state_path, &mut state, &pending_train, &pending_val)?;
    Ok(report_from_state(source, budget, &state))
}

#[allow(clippy::too_many_arguments)]
fn ingest_hf_source(
    config: &DataConfig,
    tokenizer: &LoopTokenizer,
    script: &Path,
    source: &str,
    revision: &str,
    budget: u64,
    eos: u16,
    fingerprint: &str,
) -> Result<SourceReport, DataError> {
    let out_dir = config.output_dir.join("pretrain");
    let state_path = out_dir.join("state").join(format!("{source}.json"));
    let skip = load_state(&state_path)?
        .filter(|state| state.fingerprint == fingerprint && !state.complete)
        .map(|state| state.docs_seen)
        .unwrap_or(0);
    if let Some(state) = load_state(&state_path)? {
        if state.fingerprint == fingerprint && state.complete {
            return Ok(report_from_state(source, budget, &state));
        }
    }
    let mut child = spawn_stream(
        &config.python,
        script,
        &[
            "--kind",
            "pretrain",
            "--source",
            source,
            "--revision",
            revision,
            "--skip",
            &skip.to_string(),
        ],
    )?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| DataError::new("pretrain stream has no stdout"))?;
    let docs = BufReader::new(stdout).lines().map(|line| -> String {
        let line = line.unwrap_or_default();
        serde_json::from_str::<serde_json::Value>(&line)
            .ok()
            .and_then(|value| {
                value
                    .get("text")
                    .and_then(|text| text.as_str())
                    .map(str::to_owned)
            })
            .unwrap_or_default()
    });
    let report = ingest_documents(
        &out_dir,
        source,
        budget,
        config.pretrain.val_ratio,
        config.pretrain.context_length,
        config.pretrain.shard_sequences,
        config.seed,
        eos,
        fingerprint,
        tokenizer,
        docs.filter(|text| !text.is_empty()),
    );
    let _ = child.kill();
    let _ = child.wait();
    report
}

fn absorb(
    remainder: &mut Vec<u16>,
    pending: &mut Vec<Vec<u16>>,
    incoming: &[u16],
    seq_len: usize,
    limit: u64,
    packed: &mut u64,
) {
    if *packed >= limit || seq_len == 0 {
        return;
    }
    remainder.extend_from_slice(incoming);
    while remainder.len() >= seq_len && *packed + seq_len as u64 <= limit {
        let sequence: Vec<u16> = remainder.drain(..seq_len).collect();
        pending.push(sequence);
        *packed += seq_len as u64;
    }
    if *packed >= limit {
        remainder.clear();
    }
}

#[allow(clippy::too_many_arguments)]
fn flush_ready(
    out_dir: &Path,
    source: &str,
    split: &str,
    seq_len: usize,
    shard_sequences: usize,
    pending: &mut Vec<Vec<u16>>,
    next_shard: &mut u32,
    records: &mut Vec<ShardRecord>,
    sequences: &mut u64,
) -> Result<(), DataError> {
    while pending.len() >= shard_sequences {
        let batch: Vec<Vec<u16>> = pending.drain(..shard_sequences).collect();
        let path = out_dir
            .join(split)
            .join(format!("{source}-{next_shard:05}.bin"));
        let sha = shard::write_pretrain_shard(&path, seq_len, &batch)?;
        records.push(ShardRecord {
            file: path.to_string_lossy().into_owned(),
            sequences: batch.len() as u32,
            sha256: sha,
        });
        *sequences += batch.len() as u64;
        *next_shard += 1;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn flush_tail(
    out_dir: &Path,
    source: &str,
    split: &str,
    seq_len: usize,
    pending: &mut Vec<Vec<u16>>,
    next_shard: &mut u32,
    records: &mut Vec<ShardRecord>,
    sequences: &mut u64,
) -> Result<(), DataError> {
    if pending.is_empty() {
        return Ok(());
    }
    let batch = std::mem::take(pending);
    let path = out_dir
        .join(split)
        .join(format!("{source}-{next_shard:05}.bin"));
    let sha = shard::write_pretrain_shard(&path, seq_len, &batch)?;
    records.push(ShardRecord {
        file: path.to_string_lossy().into_owned(),
        sequences: batch.len() as u32,
        sha256: sha,
    });
    *sequences += batch.len() as u64;
    *next_shard += 1;
    Ok(())
}

fn checkpoint(
    path: &Path,
    state: &mut SourceState,
    pending_train: &[Vec<u16>],
    pending_val: &[Vec<u16>],
) -> Result<(), DataError> {
    state.pending_train = pending_train.to_vec();
    state.pending_val = pending_val.to_vec();
    save_state(path, state)
}

fn encode_doc(tokenizer: &LoopTokenizer, text: &str, eos: u16) -> Result<Vec<u16>, DataError> {
    let mut ids = tokenizer.encode(text)?;
    ids.push(eos as u32);
    ids.into_iter()
        .map(|id| u16::try_from(id).map_err(|_| DataError::new("token id does not fit in u16")))
        .collect()
}

fn is_val(seed: u64, source: &str, index: u64, val_ratio: f64) -> bool {
    let threshold = (val_ratio * 1_000_000.0).round() as u64;
    stable_hash(seed, &[source], index) % 1_000_000 < threshold
}

pub(crate) fn stable_hash(seed: u64, parts: &[&str], index: u64) -> u64 {
    let mut hash = seed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for part in parts {
        for byte in part.bytes() {
            hash = hash
                .wrapping_mul(0x100_0000_01b3)
                .wrapping_add(u64::from(byte));
        }
        hash = hash.wrapping_mul(0x100_0000_01b3).wrapping_add(0xff);
    }
    hash
}

pub(crate) fn fingerprint(config: &DataConfig, tokenizer_sha: &str) -> Result<String, DataError> {
    let view = serde_json::json!({
        "seed": config.seed,
        "revisions": &config.revisions,
        "pretrain": &config.pretrain,
        "sft": &config.sft,
        "tokenizer_sha256": tokenizer_sha,
    });
    Ok(shard::sha256_bytes(view.to_string().as_bytes()))
}

fn load_state(path: &Path) -> Result<Option<SourceState>, DataError> {
    if !path.exists() {
        return Ok(None);
    }
    let text = fs_read(path)?;
    Ok(Some(serde_json::from_str(&text)?))
}

fn save_state(path: &Path, state: &SourceState) -> Result<(), DataError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(state)?)?;
    Ok(())
}

fn load_manifest(path: &Path) -> Result<Option<PretrainManifest>, DataError> {
    if !path.exists() {
        return Ok(None);
    }
    Ok(Some(serde_json::from_str(&fs_read(path)?)?))
}

fn fs_read(path: &Path) -> Result<String, DataError> {
    Ok(fs::read_to_string(path)?)
}

fn report_from_state(source: &str, budget: u64, state: &SourceState) -> SourceReport {
    SourceReport {
        name: source.to_string(),
        budget_tokens: budget,
        train_tokens: state.train_tokens,
        val_tokens: state.val_tokens,
        train_sequences: state.train_sequences,
        val_sequences: state.val_sequences,
        docs_seen: state.docs_seen,
        train_shards: state.train_shards.clone(),
        val_shards: state.val_shards.clone(),
    }
}

pub(crate) fn find_script() -> Result<PathBuf, DataError> {
    let mut dir = std::env::current_dir()?;
    loop {
        let candidate = dir.join("python/stream_corpus.py");
        if candidate.is_file() {
            return Ok(candidate);
        }
        if !dir.pop() {
            break;
        }
    }
    Err(DataError::new("python/stream_corpus.py was not found"))
}

pub(crate) fn spawn_stream(python: &str, script: &Path, args: &[&str]) -> Result<Child, DataError> {
    let mut parts = python.split_whitespace();
    let program = parts
        .next()
        .ok_or_else(|| DataError::new("python command is empty"))?;
    let mut command = Command::new(program);
    command
        .args(parts)
        .arg(script)
        .args(args)
        .env("PYTHONUNBUFFERED", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    command
        .spawn()
        .map_err(|error| DataError::new(format!("failed to start {python}: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::scaled_budgets;
    use loopmind_tokenizer::{LoopTokenizer, EOS};
    use std::fs;

    fn tokenizer() -> LoopTokenizer {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tokenizer/tokenizer.json");
        LoopTokenizer::from_file(&path).unwrap()
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("loopmind-pretrain-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn docs(count: usize) -> Vec<String> {
        (0..count)
            .map(|index| format!("document {index} says the quick brown fox jumps."))
            .collect()
    }

    fn packed(dir: &Path, split: &str, seq_len: usize) -> Vec<u16> {
        let mut files: Vec<_> = fs::read_dir(dir.join(split))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("bin"))
            .collect();
        files.sort();
        let mut tokens = Vec::new();
        for path in files {
            let shard = shard::read_pretrain_shard(&path).unwrap();
            assert_eq!(shard.seq_len, seq_len);
            for sequence in shard.sequences {
                tokens.extend(sequence);
            }
        }
        tokens
    }

    #[test]
    fn fifty_million_packs_just_under_the_budget() {
        let mut packed = 0;
        for (_, budget) in scaled_budgets(50_000_000) {
            packed += DataConfig::packed_tokens(budget, 1024);
        }
        assert_eq!(packed, 49_997_824);
        assert_eq!(50_000_000 - packed, 2_176);
    }

    #[test]
    fn packing_is_resumable_and_a_finished_source_is_skipped() {
        let tokenizer = tokenizer();
        let eos = tokenizer.token_id(EOS).unwrap() as u16;
        let docs = docs(80);
        let seq_len = 32;
        let budget = 32 * 6;
        let resumed = temp_dir("resume");
        let once = temp_dir("once");
        let partial = ingest_documents(
            &resumed,
            "fineweb",
            budget,
            0.25,
            seq_len,
            2,
            7,
            eos,
            "fp",
            &tokenizer,
            docs.iter().take(4).cloned(),
        )
        .unwrap();
        assert!(partial.train_tokens < DataConfig::packed_tokens(budget, seq_len));
        let finished = ingest_documents(
            &resumed,
            "fineweb",
            budget,
            0.25,
            seq_len,
            2,
            7,
            eos,
            "fp",
            &tokenizer,
            docs.iter().skip(4).cloned(),
        )
        .unwrap();
        let one_shot = ingest_documents(
            &once,
            "fineweb",
            budget,
            0.25,
            seq_len,
            2,
            7,
            eos,
            "fp",
            &tokenizer,
            docs.into_iter(),
        )
        .unwrap();
        assert_eq!(
            finished.train_tokens,
            DataConfig::packed_tokens(budget, seq_len)
        );
        assert_eq!(finished.train_tokens, one_shot.train_tokens);
        assert_eq!(finished.val_tokens, one_shot.val_tokens);
        assert_eq!(
            packed(&resumed, "train", seq_len),
            packed(&once, "train", seq_len)
        );
        assert!(packed(&resumed, "train", seq_len).contains(&eos));
        let skipped = ingest_documents(
            &resumed,
            "fineweb",
            budget,
            0.25,
            seq_len,
            2,
            7,
            eos,
            "fp",
            &tokenizer,
            std::iter::empty(),
        )
        .unwrap();
        assert_eq!(skipped.train_sequences, finished.train_sequences);
        assert_eq!(skipped.docs_seen, finished.docs_seen);
    }
}
