//! English chat SFT rows from FABLE, rendered with the M2 chat template.
//!
//! Turkish rows are dropped. `raw_text_corpus` is not part of this stream.
//! Conversations longer than the context cap are dropped. A fixed number of
//! checkable rows (math, logic, science, JSON) are held out of the train shards.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;

use loopmind_tokenizer::{ChatTurn, LoopTokenizer, Role};
use serde::{Deserialize, Serialize};

use crate::config::DataConfig;
use crate::error::DataError;
use crate::pretrain::{self, fingerprint, stable_hash, ShardRecord};
use crate::shard::{self, SftRow};

const EVAL_CONFIGS: [&str; 4] = [
    "math_reasoning",
    "logic_reasoning",
    "science_knowledge",
    "structured_output",
];
const SFT_SHARD_ROWS: usize = 256;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DropCounts {
    pub non_english: u64,
    pub cc_by_sa: u64,
    pub too_long: u64,
    pub empty: u64,
    pub no_assistant: u64,
    pub bad_role: u64,
}

impl DropCounts {
    fn new() -> Self {
        Self {
            non_english: 0,
            cc_by_sa: 0,
            too_long: 0,
            empty: 0,
            no_assistant: 0,
            bad_role: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChatRow {
    pub id: String,
    pub config_name: String,
    pub license: String,
    pub language: String,
    pub messages: Vec<(String, String)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SftManifest {
    pub fingerprint: String,
    pub seen: u64,
    pub kept_train: u64,
    pub kept_eval: u64,
    pub dropped: DropCounts,
    pub english_license_counts: BTreeMap<String, u64>,
    pub train_shards: Vec<ShardRecord>,
    pub eval_file: String,
    pub eval_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StageHeader {
    fingerprint: String,
    seen: u64,
    rows_len: u64,
    index_len: u64,
    dropped: DropCounts,
    english_license_counts: BTreeMap<String, u64>,
    kept: u64,
    complete: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexRow {
    hash: u64,
    config_name: String,
    license: String,
    id: String,
    reference: String,
    offset: u64,
    len: u32,
}

pub fn prepare_sft(
    config: &DataConfig,
    tokenizer: &LoopTokenizer,
) -> Result<SftManifest, DataError> {
    let tokenizer_sha = shard::sha256_file(&config.tokenizer)?;
    let fingerprint = fingerprint(config, &tokenizer_sha)?;
    let script = pretrain::find_script()?;
    let out_dir = config.output_dir.join("sft");
    let header_path = out_dir.join("staging").join("header.json");
    let seen = read_header(&header_path)?
        .filter(|header| header.fingerprint == fingerprint && !header.complete)
        .map(|header| header.seen)
        .unwrap_or(0);
    if manifest_matches(&out_dir.join("manifest.json"), &fingerprint)? {
        return read_manifest(&out_dir.join("manifest.json"));
    }
    let revision = config.revision("fable")?.to_string();
    let skip = seen.to_string();
    let mut child = pretrain::spawn_stream(
        &config.python,
        &script,
        &["--kind", "sft", "--revision", &revision, "--skip", &skip],
    )?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| DataError::new("sft stream has no stdout"))?;
    let rows = BufReader::new(stdout).lines().map(|line| match line {
        Ok(line) => parse_row(&line).unwrap_or_else(empty_row),
        Err(_) => empty_row(),
    });
    let manifest = build_sft(
        &out_dir,
        tokenizer,
        config.sft.max_tokens,
        config.sft.eval_rows,
        config.sft.exclude_cc_by_sa,
        config.seed,
        &fingerprint,
        rows,
    );
    let _ = child.kill();
    let _ = child.wait();
    manifest
}

#[allow(clippy::too_many_arguments)]
pub fn build_sft(
    out_dir: &Path,
    tokenizer: &LoopTokenizer,
    max_tokens: usize,
    eval_rows: usize,
    exclude_cc_by_sa: bool,
    seed: u64,
    fingerprint: &str,
    rows: impl Iterator<Item = ChatRow>,
) -> Result<SftManifest, DataError> {
    let manifest_path = out_dir.join("manifest.json");
    if manifest_matches(&manifest_path, fingerprint)? {
        return read_manifest(&manifest_path);
    }
    let staging = out_dir.join("staging");
    fs::create_dir_all(&staging)?;
    let header_path = staging.join("header.json");
    let rows_path = staging.join("rows.bin");
    let index_path = staging.join("index.jsonl");
    let mut header = match read_header(&header_path)? {
        Some(header) if header.fingerprint == fingerprint && header.complete => header,
        Some(header) if header.fingerprint == fingerprint => {
            truncate(&rows_path, header.rows_len)?;
            truncate(&index_path, header.index_len)?;
            header
        }
        _ => {
            let _ = fs::remove_file(&rows_path);
            let _ = fs::remove_file(&index_path);
            StageHeader {
                fingerprint: fingerprint.to_string(),
                seen: 0,
                rows_len: 0,
                index_len: 0,
                dropped: DropCounts::new(),
                english_license_counts: BTreeMap::new(),
                kept: 0,
                complete: false,
            }
        }
    };
    if !header.complete {
        let mut rows_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&rows_path)?;
        let mut index_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&index_path)?;
        for row in rows {
            header.seen += 1;
            if let Some(staged) =
                consider_row(tokenizer, &row, max_tokens, exclude_cc_by_sa, &mut header)?
            {
                let offset = header.rows_len;
                let nbytes = write_staged(&mut rows_file, &staged)?;
                header.rows_len += nbytes;
                let hash = stable_hash(seed, &[&row.id, &row.config_name], header.seen);
                let reference = last_assistant(&row.messages);
                let index = IndexRow {
                    hash,
                    config_name: row.config_name,
                    license: row.license,
                    id: row.id,
                    reference,
                    offset,
                    len: staged.tokens.len() as u32,
                };
                let line = serde_json::to_string(&index)?;
                index_file.write_all(line.as_bytes())?;
                index_file.write_all(b"\n")?;
                header.index_len += line.len() as u64 + 1;
                header.kept += 1;
            }
            if header.seen.is_multiple_of(2_000) {
                rows_file.flush()?;
                index_file.flush()?;
                write_header(&header_path, &header)?;
                eprintln!(
                    "sft: seen {}, kept {}, dropped {}",
                    header.seen,
                    header.kept,
                    dropped_total(&header.dropped)
                );
            }
        }
        rows_file.flush()?;
        index_file.flush()?;
        header.complete = true;
        write_header(&header_path, &header)?;
    }
    split_stage(
        out_dir,
        &rows_path,
        &index_path,
        &header,
        eval_rows,
        fingerprint,
    )
}

fn consider_row(
    tokenizer: &LoopTokenizer,
    row: &ChatRow,
    max_tokens: usize,
    exclude_cc_by_sa: bool,
    header: &mut StageHeader,
) -> Result<Option<SftRow>, DataError> {
    if row.messages.is_empty()
        || row
            .messages
            .iter()
            .all(|(_, content)| content.trim().is_empty())
    {
        header.dropped.empty += 1;
        return Ok(None);
    }
    if !is_english(&row.language) {
        header.dropped.non_english += 1;
        return Ok(None);
    }
    *header
        .english_license_counts
        .entry(row.license.clone())
        .or_insert(0) += 1;
    let mut turns = Vec::with_capacity(row.messages.len());
    for (role, content) in &row.messages {
        let Some(role) = role_of(role) else {
            header.dropped.bad_role += 1;
            return Ok(None);
        };
        turns.push(ChatTurn::new(role, content.clone()));
    }
    if exclude_cc_by_sa && is_cc_by_sa(&row.license) {
        header.dropped.cc_by_sa += 1;
        return Ok(None);
    }
    let rendered = tokenizer.render_chat(&turns, false)?;
    if rendered.ids.len() != rendered.loss_mask.len() {
        return Err(DataError::new("chat mask length does not match token ids"));
    }
    if rendered.loss_mask.iter().all(|bit| !bit) {
        header.dropped.no_assistant += 1;
        return Ok(None);
    }
    if rendered.ids.len() > max_tokens {
        header.dropped.too_long += 1;
        return Ok(None);
    }
    let tokens = rendered
        .ids
        .into_iter()
        .map(|id| u16::try_from(id).map_err(|_| DataError::new("token id does not fit in u16")))
        .collect::<Result<Vec<_>, _>>()?;
    let mask = rendered.loss_mask.into_iter().map(u8::from).collect();
    Ok(Some(SftRow { tokens, mask }))
}

fn split_stage(
    out_dir: &Path,
    rows_path: &Path,
    index_path: &Path,
    header: &StageHeader,
    eval_rows: usize,
    fingerprint: &str,
) -> Result<SftManifest, DataError> {
    let index = read_index(index_path)?;
    let chosen = choose_eval(&index, eval_rows);
    let mut chosen_set = vec![false; index.len()];
    for slot in &chosen {
        chosen_set[*slot] = true;
    }
    let train_dir = out_dir.join("train");
    if train_dir.exists() {
        fs::remove_dir_all(&train_dir)?;
    }
    fs::create_dir_all(&train_dir)?;
    let mut rows_file = File::open(rows_path)?;
    let mut batch = Vec::new();
    let mut shards = Vec::new();
    let mut shard_index = 0u32;
    let mut kept_train = 0u64;
    for (slot, row) in index.iter().enumerate() {
        if chosen_set[slot] {
            continue;
        }
        batch.push(read_staged(&mut rows_file, row.offset, row.len)?);
        kept_train += 1;
        if batch.len() == SFT_SHARD_ROWS {
            shards.push(flush_sft_shard(&train_dir, shard_index, &batch)?);
            shard_index += 1;
            batch.clear();
        }
    }
    if !batch.is_empty() {
        shards.push(flush_sft_shard(&train_dir, shard_index, &batch)?);
    }
    let eval_path = out_dir.join("eval.jsonl");
    write_eval(&eval_path, &index, &chosen, &mut rows_file)?;
    let manifest = SftManifest {
        fingerprint: fingerprint.to_string(),
        seen: header.seen,
        kept_train,
        kept_eval: chosen.len() as u64,
        dropped: header.dropped.clone(),
        english_license_counts: header.english_license_counts.clone(),
        train_shards: shards,
        eval_file: eval_path.to_string_lossy().into_owned(),
        eval_sha256: shard::sha256_file(&eval_path)?,
    };
    fs::write(
        out_dir.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(manifest)
}

fn choose_eval(index: &[IndexRow], eval_rows: usize) -> Vec<usize> {
    if eval_rows == 0 || index.is_empty() {
        return Vec::new();
    }
    let quota = eval_rows / EVAL_CONFIGS.len();
    let mut chosen = Vec::new();
    let mut rest = Vec::new();
    for config in EVAL_CONFIGS {
        let mut slots: Vec<usize> = index
            .iter()
            .enumerate()
            .filter(|(_, row)| row.config_name == config)
            .map(|(slot, _)| slot)
            .collect();
        slots.sort_by_key(|&slot| index[slot].hash);
        let take = quota.min(slots.len());
        chosen.extend(slots.iter().take(take).copied());
        rest.extend(slots.iter().skip(take).copied());
    }
    if chosen.len() < eval_rows {
        rest.sort_by_key(|&slot| index[slot].hash);
        let need = eval_rows - chosen.len();
        chosen.extend(rest.into_iter().take(need));
    }
    chosen
}

fn write_eval(
    path: &Path,
    index: &[IndexRow],
    chosen: &[usize],
    rows: &mut File,
) -> Result<(), DataError> {
    let mut file = File::create(path)?;
    for slot in chosen {
        let row = &index[*slot];
        let staged = read_staged(rows, row.offset, row.len)?;
        let record = serde_json::json!({
            "id": row.id,
            "config": row.config_name,
            "license": row.license,
            "reference": row.reference,
            "tokens": staged.tokens,
            "loss_mask": staged.mask,
        });
        serde_json::to_writer(&mut file, &record)?;
        file.write_all(b"\n")?;
    }
    Ok(())
}

fn flush_sft_shard(dir: &Path, index: u32, rows: &[SftRow]) -> Result<ShardRecord, DataError> {
    let path = dir.join(format!("sft-{index:05}.bin"));
    let sha = shard::write_sft_shard(&path, rows)?;
    Ok(ShardRecord {
        file: path.to_string_lossy().into_owned(),
        sequences: rows.len() as u32,
        sha256: sha,
    })
}

fn write_staged(file: &mut File, row: &SftRow) -> Result<u64, DataError> {
    let len = u32::try_from(row.tokens.len()).map_err(|_| DataError::new("sft row is too long"))?;
    file.write_all(&len.to_le_bytes())?;
    for token in &row.tokens {
        file.write_all(&token.to_le_bytes())?;
    }
    file.write_all(&row.mask)?;
    Ok(4 + row.tokens.len() as u64 * 2 + row.mask.len() as u64)
}

fn read_staged(file: &mut File, offset: u64, len: u32) -> Result<SftRow, DataError> {
    file.seek(SeekFrom::Start(offset))?;
    let mut len_buf = [0u8; 4];
    file.read_exact(&mut len_buf)?;
    let stored = u32::from_le_bytes(len_buf);
    if stored != len {
        return Err(DataError::new(
            "staged sft row length does not match the index",
        ));
    }
    let mut tokens = Vec::with_capacity(len as usize);
    for _ in 0..len {
        let mut buf = [0u8; 2];
        file.read_exact(&mut buf)?;
        tokens.push(u16::from_le_bytes(buf));
    }
    let mut mask = vec![0u8; len as usize];
    file.read_exact(&mut mask)?;
    Ok(SftRow { tokens, mask })
}

fn read_index(path: &Path) -> Result<Vec<IndexRow>, DataError> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = File::open(path)?;
    let mut rows = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        rows.push(serde_json::from_str(&line)?);
    }
    Ok(rows)
}

fn empty_row() -> ChatRow {
    ChatRow {
        id: String::new(),
        config_name: String::new(),
        license: String::new(),
        language: String::new(),
        messages: Vec::new(),
    }
}

fn parse_row(line: &str) -> Option<ChatRow> {
    let value: serde_json::Value = serde_json::from_str(line).ok()?;
    let messages = value
        .get("messages")?
        .as_array()?
        .iter()
        .filter_map(|message| {
            Some((
                message.get("role")?.as_str()?.to_string(),
                message.get("content")?.as_str().unwrap_or("").to_string(),
            ))
        });
    Some(ChatRow {
        id: value
            .get("id")
            .and_then(|id| id.as_str())
            .unwrap_or("")
            .to_string(),
        config_name: value
            .get("config")
            .and_then(|config| config.as_str())
            .unwrap_or("")
            .to_string(),
        license: value
            .get("license")
            .and_then(|license| license.as_str())
            .unwrap_or("")
            .to_string(),
        language: value
            .get("language")
            .and_then(|language| language.as_str())
            .unwrap_or("")
            .to_string(),
        messages: messages.collect(),
    })
}

fn last_assistant(messages: &[(String, String)]) -> String {
    messages
        .iter()
        .rev()
        .find(|(role, _)| role_of(role) == Some(Role::Assistant))
        .map(|(_, content)| content.clone())
        .unwrap_or_default()
}

fn role_of(name: &str) -> Option<Role> {
    loopmind_tokenizer::role_from_name(name.trim())
}

fn is_english(language: &str) -> bool {
    matches!(
        language.trim().to_ascii_lowercase().as_str(),
        "en" | "english"
    )
}

fn is_cc_by_sa(license: &str) -> bool {
    license.to_ascii_lowercase().contains("cc-by-sa")
}

fn dropped_total(counts: &DropCounts) -> u64 {
    counts.non_english
        + counts.cc_by_sa
        + counts.too_long
        + counts.empty
        + counts.no_assistant
        + counts.bad_role
}

fn write_header(path: &Path, header: &StageHeader) -> Result<(), DataError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(header)?)?;
    Ok(())
}

fn read_header(path: &Path) -> Result<Option<StageHeader>, DataError> {
    if !path.exists() {
        return Ok(None);
    }
    Ok(Some(serde_json::from_str(&fs::read_to_string(path)?)?))
}

fn read_manifest(path: &Path) -> Result<SftManifest, DataError> {
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

fn manifest_matches(path: &Path, fingerprint: &str) -> Result<bool, DataError> {
    if !path.exists() {
        return Ok(false);
    }
    let manifest = read_manifest(path)?;
    if manifest.fingerprint != fingerprint {
        return Err(DataError::new(
            "sft output was built with a different config; choose another output_dir",
        ));
    }
    Ok(true)
}

fn truncate(path: &Path, len: u64) -> Result<(), DataError> {
    if !path.exists() {
        return Ok(());
    }
    let file = OpenOptions::new().write(true).open(path)?;
    file.set_len(len)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use loopmind_tokenizer::LoopTokenizer;

    fn tokenizer() -> LoopTokenizer {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tokenizer/tokenizer.json");
        LoopTokenizer::from_file(&path).unwrap()
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("loopmind-sft-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn row(
        id: &str,
        config: &str,
        license: &str,
        language: &str,
        messages: &[(&str, &str)],
    ) -> ChatRow {
        ChatRow {
            id: id.to_string(),
            config_name: config.to_string(),
            license: license.to_string(),
            language: language.to_string(),
            messages: messages
                .iter()
                .map(|(role, content)| ((*role).to_string(), (*content).to_string()))
                .collect(),
        }
    }

    fn sample_rows() -> Vec<ChatRow> {
        let mut rows = Vec::new();
        for config in [
            "math_reasoning",
            "logic_reasoning",
            "science_knowledge",
            "structured_output",
        ] {
            for copy in 0..3 {
                rows.push(row(
                    &format!("{config}-{copy}"),
                    config,
                    if copy == 0 { "CC-BY-SA-4.0" } else { "MIT" },
                    "en",
                    &[
                        ("user", &format!("question {config} {copy}")),
                        ("assistant", &format!("answer {config} {copy}")),
                    ],
                ));
            }
        }
        rows.push(row(
            "tr",
            "math_reasoning",
            "MIT",
            "tr",
            &[("user", "merhaba"), ("assistant", "selam")],
        ));
        rows.push(row("empty", "math_reasoning", "MIT", "en", &[]));
        rows.push(row(
            "role",
            "math_reasoning",
            "MIT",
            "en",
            &[("narrator", "hello"), ("assistant", "hi")],
        ));
        rows.push(row(
            "no-asst",
            "general_instruction",
            "MIT",
            "en",
            &[("user", "only a question")],
        ));
        rows.push(row(
            "long",
            "general_instruction",
            "Apache-2.0",
            "english",
            &[
                ("user", &"token ".repeat(400)),
                ("assistant", &"token ".repeat(400)),
            ],
        ));
        rows.push(row(
            "keep",
            "general_instruction",
            "Apache-2.0",
            "en",
            &[("user", "say hi"), ("assistant", "hi")],
        ));
        rows
    }

    #[test]
    fn holds_out_checkable_rows_and_can_drop_cc_by_sa() {
        let tokenizer = tokenizer();
        let dir = temp_dir("keep");
        let manifest = build_sft(
            &dir,
            &tokenizer,
            128,
            4,
            false,
            3,
            "fp-keep",
            sample_rows().into_iter(),
        )
        .unwrap();
        assert_eq!(manifest.kept_eval, 4);
        assert_eq!(manifest.dropped.non_english, 1);
        assert_eq!(manifest.dropped.empty, 1);
        assert_eq!(manifest.dropped.bad_role, 1);
        assert_eq!(manifest.dropped.no_assistant, 1);
        assert_eq!(manifest.dropped.too_long, 1);
        assert_eq!(manifest.dropped.cc_by_sa, 0);
        assert!(
            manifest
                .english_license_counts
                .get("CC-BY-SA-4.0")
                .copied()
                .unwrap_or(0)
                >= 4
        );
        let eval: Vec<serde_json::Value> = fs::read_to_string(dir.join("eval.jsonl"))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(eval.len(), 4);
        let train = shard::read_sft_shard(&dir.join("train").join("sft-00000.bin")).unwrap();
        let eval_tokens: Vec<Vec<u16>> = eval
            .iter()
            .map(|row| serde_json::from_value(row["tokens"].clone()).unwrap())
            .collect();
        for row in &train {
            assert!(!eval_tokens.contains(&row.tokens));
            assert!(row.mask.contains(&1));
        }
        assert!(eval
            .iter()
            .all(|row| !row["reference"].as_str().unwrap().is_empty()));
        let again = build_sft(
            &dir,
            &tokenizer,
            128,
            4,
            false,
            3,
            "fp-keep",
            std::iter::empty(),
        )
        .unwrap();
        assert_eq!(again.eval_sha256, manifest.eval_sha256);
        assert_eq!(again.kept_train, manifest.kept_train);

        let excluded = temp_dir("drop-cc");
        let dropped = build_sft(
            &excluded,
            &tokenizer,
            128,
            4,
            true,
            3,
            "fp-drop",
            sample_rows().into_iter(),
        )
        .unwrap();
        assert!(dropped.dropped.cc_by_sa >= 4);
        assert!(dropped.kept_train + dropped.kept_eval < manifest.kept_train + manifest.kept_eval);
    }
}
