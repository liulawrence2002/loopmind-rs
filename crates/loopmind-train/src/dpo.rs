//! Self-generated DPO: preference pairs from the model's own answers, then a DPO run.
//!
//! A pair is a training prompt from the SFT shards with two answers. The chosen one is
//! the dataset's reference answer. The rejected one is what the model being refined
//! samples for that prompt. Training raises the policy's log-probability ratio of
//! chosen over rejected against a frozen copy of the starting weights
//! (Rafailov et al., 2023), plus a small NLL term on the chosen answer so the policy
//! does not drift from text it already writes.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use loopmind_data::{
    epoch_order, read_sft_shard, split_final_answer, EpochCursor, SftBatch, SftRow,
};
use loopmind_model::{EngineDevice, Precision, PretrainEngine};
use loopmind_tokenizer::{LoopTokenizer, EOS, IM_END, PAD};
use serde::{Deserialize, Serialize};

use crate::config::DpoConfig;
use crate::eval::Sampling;
use crate::run::{
    append_log, checkpoint_dir, checkpoint_model, load_inference, loop_hist, optim_settings,
    read_latest, resolve_checkpoint, save_checkpoint, RUN_LOCK,
};
use crate::schedule::learning_rate;
use crate::{RunOverrides, RunSummary, TrainError};

/// One line of the pairs file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreferencePair {
    pub prompt: Vec<u16>,
    pub chosen: Vec<u16>,
    pub chosen_mask: Vec<u8>,
    pub rejected: Vec<u16>,
    pub rejected_mask: Vec<u8>,
    /// Whether the rejected answer stopped on its own before the length cap.
    pub finished: bool,
}

impl PreferencePair {
    fn rows(&self) -> [SftRow; 2] {
        let row = |answer: &[u16], mask: &[u8]| {
            let mut tokens = self.prompt.clone();
            tokens.extend_from_slice(answer);
            let mut full = vec![0u8; self.prompt.len()];
            full.extend_from_slice(mask);
            SftRow { tokens, mask: full }
        };
        [
            row(&self.chosen, &self.chosen_mask),
            row(&self.rejected, &self.rejected_mask),
        ]
    }
}

#[derive(Debug, Clone)]
pub struct PairOptions {
    /// The model whose answers become the rejected side.
    pub checkpoint: PathBuf,
    /// `data/prepared/sft/train`.
    pub train_dir: PathBuf,
    pub tokenizer: PathBuf,
    pub device: String,
    pub out: PathBuf,
    pub count: usize,
    /// Longest prompt kept, in tokens.
    pub max_prompt: usize,
    /// Answer length cap. Rows whose reference answer is longer are skipped, so both
    /// sides of a pair had the same budget.
    pub max_new: usize,
    pub loops: usize,
    pub sampling: Sampling,
    /// Prompts per generation batch. Prompts in a batch share a length.
    pub batch: usize,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PairReport {
    pub pairs: usize,
    pub candidates: usize,
    pub finished: usize,
    pub identical_skipped: usize,
    pub mean_prompt: f64,
    pub mean_chosen: f64,
    pub mean_rejected: f64,
    pub seconds: f64,
}

/// Sample rejected answers for training prompts and write the pairs as JSON lines.
pub fn make_pairs(options: &PairOptions) -> Result<PairReport, TrainError> {
    let started = Instant::now();
    let tokenizer = LoopTokenizer::from_file(&options.tokenizer)
        .map_err(|error| TrainError::new(error.to_string()))?;
    let id = |name: &str| -> Result<u32, TrainError> {
        tokenizer
            .token_id(name)
            .ok_or_else(|| TrainError::new(format!("tokenizer has no {name}")))
    };
    let im_end = id(IM_END)?;
    let eos = id(EOS)?;
    let newline: Vec<u32> = tokenizer
        .encode("\n")
        .map_err(|error| TrainError::new(error.to_string()))?;
    let stop = [im_end, eos];
    let mut rows = Vec::new();
    let mut shards: Vec<PathBuf> = fs::read_dir(&options.train_dir)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("bin"))
        .collect();
    shards.sort();
    for shard in &shards {
        rows.extend(read_sft_shard(shard)?);
    }
    // Rows that end on an assistant answer short enough for the generation budget.
    let closing = 2 + newline.len();
    let mut by_prompt: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (index, row) in rows.iter().enumerate() {
        if row.tokens.last().map(|id| u32::from(*id)) != Some(eos) || row.mask.last() != Some(&1) {
            continue;
        }
        let split = split_final_answer(&row.mask);
        let answer = row.tokens.len() - split;
        if split == 0 || split > options.max_prompt || answer > options.max_new + closing {
            continue;
        }
        by_prompt.entry(split).or_default().push(index);
    }
    let candidates = by_prompt.values().map(Vec::len).sum();
    // Fill whole batches first: lengths with the most prompts lead.
    let mut groups: Vec<(usize, Vec<usize>)> = by_prompt
        .into_iter()
        .map(|(length, members)| {
            let order = epoch_order(members.len() as u32, options.seed, length as u64);
            (
                length,
                order.iter().map(|slot| members[*slot as usize]).collect(),
            )
        })
        .collect();
    groups.sort_by(|left, right| right.1.len().cmp(&left.1.len()).then(left.0.cmp(&right.0)));
    let engine = load_inference(&options.checkpoint, &options.device)?;
    if let Some(parent) = options.out.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut out = BufWriter::new(File::create(&options.out)?);
    let batch = options.batch.max(1);
    let mut report = PairReport {
        pairs: 0,
        candidates,
        finished: 0,
        identical_skipped: 0,
        mean_prompt: 0.0,
        mean_chosen: 0.0,
        mean_rejected: 0.0,
        seconds: 0.0,
    };
    let mut chunk_index = 0u64;
    // At most two batches per prompt length, so one templated prompt family cannot fill the set.
    'groups: for (_, members) in &groups {
        for chunk in members.chunks(batch).take(2) {
            if report.pairs >= options.count {
                break 'groups;
            }
            let prompts: Vec<Vec<u32>> = chunk
                .iter()
                .map(|index| {
                    let row = &rows[*index];
                    row.tokens[..split_final_answer(&row.mask)]
                        .iter()
                        .map(|id| u32::from(*id))
                        .collect()
                })
                .collect();
            let generate = options.sampling.options(
                options.loops,
                options.max_new,
                options
                    .seed
                    .wrapping_add(chunk_index.wrapping_mul(batch as u64)),
            );
            chunk_index += 1;
            let answers = engine.generate_batch(&prompts, &generate, &stop);
            for (index, answer) in chunk.iter().zip(answers) {
                if report.pairs >= options.count {
                    break;
                }
                let row = &rows[*index];
                let split = split_final_answer(&row.mask);
                let finished = answer.len() < options.max_new;
                let mut rejected: Vec<u16> = answer.iter().map(|id| *id as u16).collect();
                let mut rejected_mask = vec![1u8; rejected.len()];
                if finished {
                    rejected.push(im_end as u16);
                    rejected_mask.push(1);
                    rejected.extend(newline.iter().map(|id| *id as u16));
                    rejected_mask.extend(std::iter::repeat_n(0u8, newline.len()));
                    rejected.push(eos as u16);
                    rejected_mask.push(1);
                }
                let pair = PreferencePair {
                    prompt: row.tokens[..split].to_vec(),
                    chosen: row.tokens[split..].to_vec(),
                    chosen_mask: row.mask[split..].to_vec(),
                    rejected,
                    rejected_mask,
                    finished,
                };
                if pair.rejected == pair.chosen {
                    report.identical_skipped += 1;
                    continue;
                }
                report.pairs += 1;
                report.finished += usize::from(finished);
                report.mean_prompt += pair.prompt.len() as f64;
                report.mean_chosen += pair.chosen.len() as f64;
                report.mean_rejected += pair.rejected.len() as f64;
                serde_json::to_writer(&mut out, &pair)?;
                out.write_all(b"\n")?;
            }
            out.flush()?;
            if chunk_index.is_multiple_of(20) {
                println!(
                    "dpo pairs: {} / {} after {:.0}s",
                    report.pairs,
                    options.count,
                    started.elapsed().as_secs_f64()
                );
            }
        }
    }
    let n = report.pairs.max(1) as f64;
    report.mean_prompt /= n;
    report.mean_chosen /= n;
    report.mean_rejected /= n;
    report.seconds = started.elapsed().as_secs_f64();
    Ok(report)
}

pub fn read_pairs(path: &Path) -> Result<Vec<PreferencePair>, TrainError> {
    fs::read_to_string(path)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| Ok(serde_json::from_str(line)?))
        .collect()
}

#[derive(Debug, Clone, Serialize)]
struct DpoLogRow {
    step: u64,
    epoch: u64,
    loss: f32,
    margin: f32,
    /// Fraction of the step's pairs with a positive margin.
    accuracy: f32,
    chosen_reward: f32,
    rejected_reward: f32,
    chosen_nll: f32,
    aux: f32,
    lr: f64,
    grad_norm: f32,
    tokens: u64,
    tokens_per_sec: f64,
    mean_loops: f64,
    loop_counts: BTreeMap<String, u64>,
    val: BTreeMap<String, f32>,
}

pub fn dpo(config: &DpoConfig, overrides: &RunOverrides) -> Result<RunSummary, TrainError> {
    let _run = RUN_LOCK.lock().expect("train lock");
    let device_name = overrides
        .device
        .clone()
        .unwrap_or_else(|| config.device.clone());
    let kind = EngineDevice::parse(&device_name)?;
    let max_steps = overrides.max_steps.unwrap_or(config.max_steps);
    let tokenizer = LoopTokenizer::from_file(&config.tokenizer)
        .map_err(|error| TrainError::new(error.to_string()))?;
    let pad = tokenizer
        .token_id(PAD)
        .and_then(|id| u16::try_from(id).ok())
        .ok_or_else(|| TrainError::new("tokenizer has no u16 <|pad|>"))?;
    let pairs = read_pairs(&config.pairs)?;
    if pairs.len() <= config.val_pairs {
        return Err(TrainError::new(format!(
            "{} pairs leave none for training after {} validation pairs",
            pairs.len(),
            config.val_pairs
        )));
    }
    let (val, train) = pairs.split_at(config.val_pairs);
    let batches = |pairs: &[PreferencePair]| -> Vec<SftBatch> {
        pairs
            .iter()
            .map(|pair| SftBatch::from_rows(&pair.rows(), pad, config.pad_multiple))
            .collect()
    };
    let val = batches(val);
    let train = batches(train);
    let per_step = config.batch.pairs_per_step.max(1);
    let steps_per_epoch = (train.len() as u64).div_ceil(per_step as u64).max(1);
    let settings = optim_settings(&config.optim);
    let init = resolve_checkpoint(&config.init)?;
    let (_, model_config) = checkpoint_model(&init)?;
    fs::create_dir_all(&config.run_dir)?;
    let (mut engine, mut cursor, mut step, mut tokens_seen, schedule_steps) = if overrides.resume {
        let meta = read_latest(&config.run_dir)?;
        let dir = checkpoint_dir(&config.run_dir, meta.step);
        let engine = PretrainEngine::load(
            &model_config,
            config.seed,
            kind,
            settings,
            &dir.join("model.bpk"),
            &dir.join("optim.bpk"),
            meta.rng_state,
            meta.precision.clone(),
        )?;
        (
            engine,
            meta.cursor,
            meta.step,
            meta.tokens_seen,
            meta.schedule_steps,
        )
    } else {
        let engine = PretrainEngine::from_weights(
            &model_config,
            config.seed,
            kind,
            settings,
            &init.join("model.bpk"),
            Precision::parse(&config.precision)?,
        )?;
        (
            engine,
            EpochCursor::new(config.seed),
            0,
            0,
            steps_per_epoch.saturating_mul(config.epochs).max(1),
        )
    };
    engine.load_reference(&init.join("model.bpk"))?;
    println!(
        "dpo init {} pairs {} val {} pairs/step {per_step} schedule {schedule_steps} steps beta {} nll_weight {}",
        init.display(),
        train.len(),
        val.len(),
        config.beta,
        config.nll_weight
    );
    let beta = config.beta as f32;
    let nll_weight = config.nll_weight as f32;
    let mut order_key = None;
    let mut order = Vec::new();
    let mut last_loss = 0.0f32;
    while cursor.epoch < config.epochs && (max_steps == 0 || step < max_steps) {
        let started = Instant::now();
        let mut picked = Vec::with_capacity(per_step);
        while picked.len() < per_step && cursor.epoch < config.epochs {
            if order_key != Some(cursor.epoch) {
                order = epoch_order(train.len() as u32, cursor.seed, cursor.epoch);
                order_key = Some(cursor.epoch);
            }
            picked.push(order[cursor.index as usize] as usize);
            cursor.index += 1;
            if cursor.index >= order.len() as u64 {
                cursor.epoch += 1;
                cursor.index = 0;
            }
        }
        if picked.is_empty() {
            break;
        }
        let scale = 1.0 / picked.len() as f32;
        let mut stats = Vec::with_capacity(picked.len());
        for index in &picked {
            stats.push(engine.accumulate_dpo(&train[*index], beta, nll_weight, scale, scale)?);
        }
        let lr = learning_rate(
            step,
            config.optim.warmup_steps,
            schedule_steps,
            config.optim.learning_rate,
            config.optim.min_learning_rate,
        );
        let applied = engine.apply_summed(lr)?;
        step += 1;
        let step_tokens: u64 = stats.iter().map(|stat| stat.tokens).sum();
        tokens_seen += step_tokens;
        let mean = |pick: fn(&loopmind_model::DpoStats) -> f32| {
            stats.iter().map(pick).sum::<f32>() / stats.len() as f32
        };
        let loops: Vec<usize> = stats.iter().map(|stat| stat.loops).collect();
        let loss = mean(|stat| stat.loss);
        last_loss = loss;
        if config.checkpoint.every > 0 && step.is_multiple_of(config.checkpoint.every) {
            save_checkpoint(
                &config.run_dir,
                config.checkpoint.keep,
                &engine,
                &model_config,
                &cursor,
                step,
                tokens_seen,
                schedule_steps,
            )?;
        }
        let mut val_stats = BTreeMap::new();
        if config.log.val_every > 0 && step.is_multiple_of(config.log.val_every) {
            val_stats = validate(&engine, &val, beta, config.val_loops)?;
        }
        let row = DpoLogRow {
            step,
            epoch: cursor.epoch,
            loss,
            margin: mean(|stat| stat.margin),
            accuracy: stats.iter().filter(|stat| stat.margin > 0.0).count() as f32
                / stats.len() as f32,
            chosen_reward: mean(|stat| stat.chosen_reward),
            rejected_reward: mean(|stat| stat.rejected_reward),
            chosen_nll: mean(|stat| stat.chosen_nll),
            aux: mean(|stat| stat.aux),
            lr,
            grad_norm: applied.grad_norm,
            tokens: step_tokens,
            tokens_per_sec: step_tokens as f64 / started.elapsed().as_secs_f64().max(1.0e-9),
            mean_loops: loops.iter().sum::<usize>() as f64 / loops.len() as f64,
            loop_counts: loop_hist(&loops),
            val: val_stats,
        };
        if config.log.every == 0 || step.is_multiple_of(config.log.every) {
            append_log(&config.run_dir.join("metrics.jsonl"), &row)?;
            let val = if row.val.is_empty() {
                String::new()
            } else {
                format!(" val {:+.4?}", row.val)
            };
            println!(
                "dpo {:>5}  ep {}  loss {:.4}  margin {:+.3}  acc {:.2}  chosen {:+.3}  rejected {:+.3}  nll {:.3}  lr {:.2e}  grad {:.3}  {:.0} tok/s{val}",
                row.step,
                row.epoch,
                row.loss,
                row.margin,
                row.accuracy,
                row.chosen_reward,
                row.rejected_reward,
                row.chosen_nll,
                row.lr,
                row.grad_norm,
                row.tokens_per_sec
            );
        }
    }
    save_checkpoint(
        &config.run_dir,
        config.checkpoint.keep,
        &engine,
        &model_config,
        &cursor,
        step,
        tokens_seen,
        schedule_steps,
    )?;
    println!(
        "dpo final val {:+.4?}",
        validate(&engine, &val, beta, config.val_loops)?
    );
    Ok(RunSummary {
        steps: step,
        tokens_seen,
        final_loss: last_loss,
        precision: engine.precision().to_string(),
        device: engine.device_name().to_string(),
    })
}

/// Mean DPO loss, margin, and pair accuracy over the held-out pairs at a fixed `r`.
fn validate(
    engine: &PretrainEngine,
    pairs: &[SftBatch],
    beta: f32,
    loops: usize,
) -> Result<BTreeMap<String, f32>, TrainError> {
    let mut loss = 0.0f32;
    let mut margin = 0.0f32;
    let mut wins = 0usize;
    for pair in pairs {
        let (pair_loss, pair_margin) = engine.validate_dpo(pair, beta, loops)?;
        loss += pair_loss;
        margin += pair_margin;
        wins += usize::from(pair_margin > 0.0);
    }
    let n = pairs.len().max(1) as f32;
    Ok(BTreeMap::from([
        ("loss".to_string(), loss / n),
        ("margin".to_string(), margin / n),
        ("accuracy".to_string(), wins as f32 / n),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CheckpointConfig, DpoBatchConfig, DpoLogConfig};
    use crate::sft::tests::{optim, pretrained};
    use loopmind_data::write_sft_shard;

    #[test]
    fn self_generated_pairs_train_a_positive_margin_and_resume() {
        let root = std::env::temp_dir().join(format!("loopmind-dpo-run-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let init = pretrained(&root);
        let tokenizer =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tokenizer/tokenizer.json");
        let train = root.join("sft/train");
        fs::create_dir_all(&train).unwrap();
        // bos, user turn, generation prompt, answer, im_end, newline, eos: prompts of one length.
        let rows: Vec<SftRow> = (0..8u16)
            .map(|row| SftRow {
                tokens: vec![0, 3, 5, 20 + row, 4, 9, 3, 7, 9, 40 + row, 41, 4, 9, 1],
                mask: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
            })
            .collect();
        write_sft_shard(&train.join("sft-00000.bin"), &rows).unwrap();
        let pairs_path = root.join("dpo/pairs.jsonl");
        let report = make_pairs(&PairOptions {
            checkpoint: init.clone(),
            train_dir: train,
            tokenizer: tokenizer.clone(),
            device: "cpu".into(),
            out: pairs_path.clone(),
            count: 6,
            max_prompt: 64,
            max_new: 6,
            loops: 2,
            sampling: Sampling::presets()[2].clone(),
            batch: 4,
            seed: 5,
        })
        .unwrap();
        assert_eq!(report.candidates, 8);
        assert_eq!(report.pairs + report.identical_skipped, 6);
        let pairs = read_pairs(&pairs_path).unwrap();
        assert_eq!(pairs.len(), report.pairs);
        for pair in &pairs {
            assert_eq!(pair.prompt.len(), 9);
            assert_eq!(pair.chosen_mask, vec![1, 1, 1, 0, 1]);
            assert_eq!(pair.rejected.len(), pair.rejected_mask.len());
        }
        let config = DpoConfig {
            init,
            pairs: pairs_path,
            run_dir: root.join("dpo/run"),
            tokenizer,
            seed: 2,
            epochs: 50,
            max_steps: 15,
            device: "cpu".into(),
            precision: "f32".into(),
            pad_multiple: 4,
            beta: 0.5,
            nll_weight: 0.0,
            val_pairs: 2,
            val_loops: 2,
            optim: optim(1.0e-2),
            batch: DpoBatchConfig { pairs_per_step: 2 },
            log: DpoLogConfig {
                every: 1,
                val_every: 5,
            },
            checkpoint: CheckpointConfig { every: 5, keep: 2 },
        };
        let overrides = |resume, max_steps| RunOverrides {
            resume,
            max_steps: Some(max_steps),
            device: None,
        };
        dpo(&config, &overrides(false, 15)).unwrap();
        let metrics = config.run_dir.join("metrics.jsonl");
        let rows: Vec<serde_json::Value> = fs::read_to_string(&metrics)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let first = rows[0]["loss"].as_f64().unwrap();
        assert!((first - std::f64::consts::LN_2).abs() < 1e-3, "{first}");
        let margin = |row: &serde_json::Value| row["margin"].as_f64().unwrap();
        let late: f64 = rows[10..].iter().map(margin).sum::<f64>() / 5.0;
        assert!(late > 0.1, "margin did not grow: {late}");
        assert!(rows[4]["val"]["accuracy"].as_f64().is_some());
        let resumed = dpo(&config, &overrides(true, 18)).unwrap();
        assert_eq!(resumed.steps, 18);
    }
}
