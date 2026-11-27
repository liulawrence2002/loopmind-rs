//! Pretrain driver: epoch cursor, accumulation, logs, checkpoints.

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

/// One training run at a time. Burn's CPU RNG is process-global, so two
/// overlapping `pretrain` calls in one process would mix their initializers.
static RUN_LOCK: Mutex<()> = Mutex::new(());

use loopmind_data::{EpochCursor, PretrainLoader};
use loopmind_model::{EngineDevice, LoopMindConfig, OptimSettings, Precision, PretrainEngine};
use serde::{Deserialize, Serialize};

use crate::config::PretrainConfig;
use crate::schedule::learning_rate;
use crate::TrainError;

#[derive(Debug, Clone)]
pub struct RunOverrides {
    pub resume: bool,
    pub max_steps: Option<u64>,
    pub device: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointMeta {
    step: u64,
    cursor: EpochCursor,
    rng_state: u64,
    tokens_seen: u64,
    precision: String,
    schedule_steps: u64,
    model_toml: String,
}

#[derive(Debug, Clone, Serialize)]
struct LogRow {
    step: u64,
    loss: f32,
    aux: f32,
    lr: f64,
    grad_norm: f32,
    tokens_per_sec: f64,
    mfu: f64,
    tokens_seen: u64,
    mean_loops: f64,
    loop_counts: BTreeMap<String, u64>,
    expert_slots: Vec<Vec<f32>>,
    val_loss: BTreeMap<String, f32>,
    precision: String,
    device: String,
}

#[derive(Debug, Clone)]
pub struct RunSummary {
    pub steps: u64,
    pub tokens_seen: u64,
    pub final_loss: f32,
    pub precision: String,
    pub device: String,
}

pub fn pretrain(
    config: &PretrainConfig,
    overrides: &RunOverrides,
) -> Result<RunSummary, TrainError> {
    let _run = RUN_LOCK.lock().expect("pretrain lock");
    let device_name = overrides
        .device
        .clone()
        .unwrap_or_else(|| config.device.clone());
    let kind = EngineDevice::parse(&device_name)?;
    let max_steps = overrides.max_steps.unwrap_or(config.max_steps);
    let model_config = config.load_model()?;
    let train_dir = config.data_dir.join("train");
    let val_dir = config.data_dir.join("val");
    let mut loader = PretrainLoader::from_dir(&train_dir, config.seed)?;
    let _val_loader = PretrainLoader::from_dir(&val_dir, config.seed ^ 0x5A17)?;
    let seq_len = loader.seq_len() as u64;
    let micro = config.batch.micro_batch as u64;
    let span = micro * seq_len;
    if span == 0 || !config.batch.tokens_per_step.is_multiple_of(span) {
        return Err(TrainError::new(format!(
            "tokens_per_step {} must be a multiple of micro_batch * seq_len ({span})",
            config.batch.tokens_per_step
        )));
    }
    let accum = (config.batch.tokens_per_step / span) as usize;
    let steps_per_epoch = loader.len().div_ceil(micro * accum as u64).max(1);
    let schedule_steps = steps_per_epoch.saturating_mul(config.epochs).max(1);
    fs::create_dir_all(&config.run_dir)?;
    let settings = optim_settings(config);
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
            meta.precision,
        )?;
        (
            engine,
            meta.cursor,
            meta.step,
            meta.tokens_seen,
            meta.schedule_steps,
        )
    } else {
        let requested = Precision::parse(&config.precision)?;
        let engine = PretrainEngine::new(&model_config, config.seed, kind, settings, requested)?;
        (engine, EpochCursor::new(config.seed), 0, 0, schedule_steps)
    };
    println!(
        "pretrain device {} precision {} seq {seq_len} micro {} accum {accum} schedule {schedule_steps} steps",
        engine.device_name(),
        engine.precision(),
        config.batch.micro_batch
    );
    let mut last_loss = 0.0f32;
    while cursor.epoch < config.epochs && (max_steps == 0 || step < max_steps) {
        let started = Instant::now();
        let mut losses = Vec::with_capacity(accum);
        let mut auxes = Vec::with_capacity(accum);
        let mut loops = Vec::new();
        let mut slots: Vec<Vec<f32>> = Vec::new();
        let mut slot_n = 0.0f32;
        let mut step_tokens = 0u64;
        for _ in 0..accum {
            if cursor.epoch >= config.epochs {
                break;
            }
            let batch = loader.next_batch(config.batch.micro_batch, &mut cursor)?;
            let stats = engine.accumulate(&batch)?;
            losses.push(stats.loss);
            auxes.push(stats.aux);
            loops.extend(stats.loops);
            if slots.is_empty() {
                slots = stats.expert_slots;
            } else {
                add_slots(&mut slots, &stats.expert_slots);
            }
            slot_n += 1.0;
            step_tokens += stats.tokens;
        }
        if losses.is_empty() {
            break;
        }
        scale_slots(&mut slots, 1.0 / slot_n);
        let applied = engine.apply(learning_rate(
            step,
            config.optim.warmup_steps,
            schedule_steps,
            config.optim.learning_rate,
            config.optim.min_learning_rate,
        ))?;
        step += 1;
        tokens_seen += step_tokens;
        let elapsed = started.elapsed().as_secs_f64().max(1.0e-9);
        let loss = mean(&losses);
        let aux = mean(&auxes);
        last_loss = loss;
        let mean_loops = if loops.is_empty() {
            0.0
        } else {
            loops.iter().sum::<usize>() as f64 / loops.len() as f64
        };
        let flops = model_config.train_flops_per_token(mean_loops.max(1.0))?;
        let tokens_per_sec = step_tokens as f64 / elapsed;
        let mfu = (flops * step_tokens as f64) / elapsed / (config.peak_tflops * 1.0e12);
        let mut val_loss = BTreeMap::new();
        if config.log.val_every > 0 && step.is_multiple_of(config.log.val_every) {
            val_loss = validate(&engine, &val_dir, config)?;
        }
        let row = LogRow {
            step,
            loss,
            aux,
            lr: learning_rate(
                step - 1,
                config.optim.warmup_steps,
                schedule_steps,
                config.optim.learning_rate,
                config.optim.min_learning_rate,
            ),
            grad_norm: applied.grad_norm,
            tokens_per_sec,
            mfu,
            tokens_seen,
            mean_loops,
            loop_counts: loop_hist(&loops),
            expert_slots: slots,
            val_loss,
            precision: engine.precision().to_string(),
            device: engine.device_name().to_string(),
        };
        if config.log.every == 0 || step.is_multiple_of(config.log.every) {
            append_log(&config.run_dir.join("metrics.jsonl"), &row)?;
            print_row(&row);
        }
        if config.checkpoint.every > 0 && step.is_multiple_of(config.checkpoint.every) {
            save_checkpoint(
                config,
                &engine,
                &model_config,
                &cursor,
                step,
                tokens_seen,
                schedule_steps,
            )?;
        }
    }
    save_checkpoint(
        config,
        &engine,
        &model_config,
        &cursor,
        step,
        tokens_seen,
        schedule_steps,
    )?;
    Ok(RunSummary {
        steps: step,
        tokens_seen,
        final_loss: last_loss,
        precision: engine.precision().to_string(),
        device: engine.device_name().to_string(),
    })
}

pub fn generate(
    checkpoint: &Path,
    prompt_ids: &[u32],
    loops: usize,
    max_new: usize,
    eos: u32,
    device: &str,
) -> Result<Vec<u32>, TrainError> {
    let meta: CheckpointMeta =
        serde_json::from_str(&fs::read_to_string(checkpoint.join("state.json"))?)?;
    let model_config = LoopMindConfig::from_toml_str(&meta.model_toml)?;
    let kind = EngineDevice::parse(device)?;
    let settings = OptimSettings {
        weight_decay: 0.1,
        beta1: 0.9,
        beta2: 0.95,
        epsilon: 1e-8,
        grad_clip: 1.0,
    };
    let engine = PretrainEngine::load(
        &model_config,
        1,
        kind,
        settings,
        &checkpoint.join("model.bpk"),
        &checkpoint.join("optim.bpk"),
        meta.rng_state,
        meta.precision,
    )?;
    Ok(engine.greedy(prompt_ids.to_vec(), loops, max_new, eos))
}

fn validate(
    engine: &PretrainEngine,
    val_dir: &Path,
    config: &PretrainConfig,
) -> Result<BTreeMap<String, f32>, TrainError> {
    let mut loader = PretrainLoader::from_dir(val_dir, config.seed)?;
    let mut cursor = EpochCursor::new(config.seed ^ 0x5A17);
    let mut batches = Vec::new();
    let mut taken = 0usize;
    let want = config.log.val_sequences.max(1);
    while taken < want {
        let batch = loader.next_batch(1, &mut cursor)?;
        taken += batch.batch;
        batches.push(batch);
        if cursor.epoch > 0 {
            break;
        }
    }
    let mut losses = BTreeMap::new();
    for loops in &config.val_loops {
        let mut values = Vec::new();
        for batch in &batches {
            values.push(engine.validate(batch, *loops)?);
        }
        losses.insert(format!("r{loops}"), mean(&values));
    }
    Ok(losses)
}

fn save_checkpoint(
    config: &PretrainConfig,
    engine: &PretrainEngine,
    model_config: &LoopMindConfig,
    cursor: &EpochCursor,
    step: u64,
    tokens_seen: u64,
    schedule_steps: u64,
) -> Result<(), TrainError> {
    let dir = checkpoint_dir(&config.run_dir, step);
    fs::create_dir_all(&dir)?;
    engine.save(&dir.join("model.bpk"), &dir.join("optim.bpk"))?;
    let meta = CheckpointMeta {
        step,
        cursor: cursor.clone(),
        rng_state: engine.rng_state(),
        tokens_seen,
        precision: engine.precision().to_string(),
        schedule_steps,
        model_toml: toml::to_string(model_config)
            .map_err(|error| TrainError::new(error.to_string()))?,
    };
    fs::write(dir.join("state.json"), serde_json::to_vec_pretty(&meta)?)?;
    fs::write(
        config.run_dir.join("latest.json"),
        serde_json::to_vec_pretty(&meta)?,
    )?;
    prune(&config.run_dir, config.checkpoint.keep)?;
    Ok(())
}

fn prune(run_dir: &Path, keep: usize) -> Result<(), TrainError> {
    let root = run_dir.join("checkpoints");
    if !root.is_dir() {
        return Ok(());
    }
    let mut steps = Vec::new();
    for entry in fs::read_dir(&root)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if let Some(step) = name
            .strip_prefix("step-")
            .and_then(|text| text.parse::<u64>().ok())
        {
            steps.push(step);
        }
    }
    steps.sort_unstable();
    if steps.len() > keep {
        for step in &steps[..steps.len() - keep] {
            fs::remove_dir_all(checkpoint_dir(run_dir, *step))?;
        }
    }
    Ok(())
}

fn read_latest(run_dir: &Path) -> Result<CheckpointMeta, TrainError> {
    let path = run_dir.join("latest.json");
    if !path.is_file() {
        return Err(TrainError::new(format!(
            "no checkpoint to resume at {}",
            path.display()
        )));
    }
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

fn checkpoint_dir(run_dir: &Path, step: u64) -> PathBuf {
    run_dir.join("checkpoints").join(format!("step-{step:08}"))
}

fn append_log(path: &Path, row: &LogRow) -> Result<(), TrainError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut file, row)?;
    file.write_all(b"\n")?;
    Ok(())
}

fn print_row(row: &LogRow) {
    let experts = row
        .expert_slots
        .iter()
        .enumerate()
        .map(|(layer, slots)| {
            let (index, value) = slots
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(index, value)| (index, *value))
                .unwrap_or((0, 0.0));
            format!("c{layer}:{value:.3}@{index}")
        })
        .collect::<Vec<_>>()
        .join(" ");
    let val = if row.val_loss.is_empty() {
        String::new()
    } else {
        format!(" val {:+.4?}", row.val_loss)
    };
    println!(
        "step {:>6}  loss {:.4}  aux {:.4}  lr {:.3e}  grad {:.3}  {:.0} tok/s  mfu {:.2}%  r {:.2}  {experts}{val}",
        row.step,
        row.loss,
        row.aux,
        row.lr,
        row.grad_norm,
        row.tokens_per_sec,
        row.mfu * 100.0,
        row.mean_loops,
    );
}

fn optim_settings(config: &PretrainConfig) -> OptimSettings {
    OptimSettings {
        weight_decay: config.optim.weight_decay as f32,
        beta1: config.optim.beta1 as f32,
        beta2: config.optim.beta2 as f32,
        epsilon: config.optim.epsilon as f32,
        grad_clip: config.optim.grad_clip as f32,
    }
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

fn loop_hist(loops: &[usize]) -> BTreeMap<String, u64> {
    let mut counts = BTreeMap::new();
    for value in loops {
        *counts.entry(value.to_string()).or_insert(0) += 1;
    }
    counts
}

fn add_slots(acc: &mut [Vec<f32>], extra: &[Vec<f32>]) {
    for (layer, next) in acc.iter_mut().zip(extra) {
        for (value, add) in layer.iter_mut().zip(next) {
            *value += *add;
        }
    }
}

fn scale_slots(slots: &mut [Vec<f32>], scale: f32) {
    for layer in slots {
        for value in layer {
            *value *= scale;
        }
    }
}

impl From<serde_json::Error> for TrainError {
    fn from(error: serde_json::Error) -> Self {
        Self::new(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loopmind_data::write_pretrain_shard;
    use loopmind_model::LoopMindConfig;

    fn fixture(root: &Path) -> PretrainConfig {
        let data = root.join("data");
        fs::create_dir_all(data.join("train")).unwrap();
        fs::create_dir_all(data.join("val")).unwrap();
        let sequences: Vec<Vec<u16>> = (0..4)
            .map(|copy| vec![1, 2, 3, 4, 5, 6, 7, 8 + copy])
            .collect();
        write_pretrain_shard(&data.join("train/a.bin"), 8, &sequences).unwrap();
        write_pretrain_shard(&data.join("val/a.bin"), 8, &sequences[..1]).unwrap();
        let mut model = LoopMindConfig::tiny();
        model.context_length = 16;
        model.aux_loss_coef = 0.0;
        fs::write(root.join("model.toml"), toml::to_string(&model).unwrap()).unwrap();
        PretrainConfig {
            model: root.join("model.toml"),
            data_dir: data,
            run_dir: root.join("run"),
            seed: 7,
            epochs: 30,
            max_steps: 40,
            device: "cpu".into(),
            precision: "f32".into(),
            peak_tflops: 1.0,
            optim: crate::config::OptimConfig {
                learning_rate: 1.0e-2,
                min_learning_rate: 1.0e-3,
                weight_decay: 0.0,
                beta1: 0.9,
                beta2: 0.95,
                epsilon: 1.0e-8,
                grad_clip: 1.0,
                warmup_steps: 4,
            },
            batch: crate::config::BatchConfig {
                micro_batch: 2,
                tokens_per_step: 16,
            },
            log: crate::config::LogConfig {
                every: 1,
                val_every: 20,
                val_sequences: 1,
            },
            checkpoint: crate::config::CheckpointConfig { every: 10, keep: 2 },
            val_loops: vec![1, 2, 4, 8],
        }
    }

    fn losses(path: &Path) -> Vec<f32> {
        fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| {
                let row: serde_json::Value = serde_json::from_str(line).unwrap();
                row["loss"].as_f64().unwrap() as f32
            })
            .collect()
    }

    #[test]
    fn a_few_steps_reduce_loss_on_fixture_shards() {
        let root = std::env::temp_dir().join(format!("loopmind-fit-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let config = fixture(&root);
        let summary = pretrain(
            &config,
            &RunOverrides {
                resume: false,
                max_steps: Some(40),
                device: None,
            },
        )
        .unwrap();
        assert_eq!(summary.steps, 40);
        assert_eq!(summary.precision, "f32");
        let curve = losses(&config.run_dir.join("metrics.jsonl"));
        assert!(
            curve.last().unwrap() < curve.first().unwrap(),
            "loss did not fall: {curve:?}"
        );
        let row: serde_json::Value = serde_json::from_str(
            fs::read_to_string(config.run_dir.join("metrics.jsonl"))
                .unwrap()
                .lines()
                .nth(3)
                .unwrap(),
        )
        .unwrap();
        assert!(row["lr"].as_f64().unwrap() > 0.0);
        assert!(row["val_loss"].get("r1").is_none());
        let later: serde_json::Value = serde_json::from_str(
            fs::read_to_string(config.run_dir.join("metrics.jsonl"))
                .unwrap()
                .lines()
                .nth(19)
                .unwrap(),
        )
        .unwrap();
        assert!(later["val_loss"]["r1"].as_f64().unwrap().is_finite());
        assert!(later["val_loss"]["r8"].as_f64().unwrap().is_finite());
    }

    #[test]
    fn resume_matches_an_uninterrupted_step() {
        let root = std::env::temp_dir().join(format!("loopmind-resume-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let mut full = fixture(&root.join("full"));
        full.run_dir = root.join("full/run");
        full.data_dir = root.join("data");
        full.model = root.join("model.toml");
        full.max_steps = 4;
        full.checkpoint.every = 2;
        full.log.val_every = 100;
        full.epochs = 20;
        let data = root.join("data");
        fs::create_dir_all(data.join("train")).unwrap();
        fs::create_dir_all(data.join("val")).unwrap();
        let sequences: Vec<Vec<u16>> = (0..4)
            .map(|copy| vec![1, 2, 3, 4, 5, 6, 7, 8 + copy])
            .collect();
        write_pretrain_shard(&data.join("train/a.bin"), 8, &sequences).unwrap();
        write_pretrain_shard(&data.join("val/a.bin"), 8, &sequences[..1]).unwrap();
        let mut model = LoopMindConfig::tiny();
        model.context_length = 16;
        model.aux_loss_coef = 0.0;
        fs::write(&full.model, toml::to_string(&model).unwrap()).unwrap();
        pretrain(
            &full,
            &RunOverrides {
                resume: false,
                max_steps: Some(4),
                device: None,
            },
        )
        .unwrap();
        let mut stopped = full.clone();
        stopped.run_dir = root.join("stop/run");
        pretrain(
            &stopped,
            &RunOverrides {
                resume: false,
                max_steps: Some(2),
                device: None,
            },
        )
        .unwrap();
        pretrain(
            &stopped,
            &RunOverrides {
                resume: true,
                max_steps: Some(4),
                device: None,
            },
        )
        .unwrap();
        let uninterrupted = losses(&full.run_dir.join("metrics.jsonl"));
        let resumed = losses(&stopped.run_dir.join("metrics.jsonl"));
        assert_eq!(uninterrupted.len(), 4);
        assert_eq!(resumed.len(), 4);
        assert_eq!(resumed[2], uninterrupted[2]);
        assert_eq!(resumed[3], uninterrupted[3]);
    }
}
