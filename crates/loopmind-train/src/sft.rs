//! SFT driver: finetune a pretrain checkpoint on masked chat rows.
//!
//! Each row runs alone with its own sampled loop count, like a pretrain sequence.
//! A step is the token-mean next-token loss over every supervised target in its
//! `rows_per_step` rows (assistant content, its `<|im_end|>`, and the final
//! `<|eos|>`), plus the load-balancing term averaged over rows. Checkpoints use the
//! pretrain layout, so `--resume`, `sample`, and `chat` read them the same way.

use std::collections::BTreeMap;
use std::fs;
use std::time::Instant;

use loopmind_data::{read_sft_eval, EpochCursor, SftBatch, SftEvalRow, SftLoader};
use loopmind_model::{EngineDevice, GenerateOptions, Precision, PretrainEngine};
use loopmind_tokenizer::{LoopTokenizer, EOS, IM_END, PAD};
use serde::Serialize;

use crate::config::SftConfig;
use crate::run::{
    add_slots, append_log, checkpoint_dir, checkpoint_model, loop_hist, mean, optim_settings,
    read_latest, resolve_checkpoint, save_checkpoint, scale_slots, RUN_LOCK,
};
use crate::schedule::learning_rate;
use crate::{RunOverrides, RunSummary, TrainError};

#[derive(Debug, Clone, Serialize)]
struct SftLogRow {
    step: u64,
    epoch: u64,
    /// Token-mean NLL over the step's supervised targets.
    loss: f32,
    aux: f32,
    lr: f64,
    grad_norm: f32,
    rows: usize,
    targets: u64,
    /// Padded tokens fed to the model.
    tokens: u64,
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

pub fn sft(config: &SftConfig, overrides: &RunOverrides) -> Result<RunSummary, TrainError> {
    let _run = RUN_LOCK.lock().expect("train lock");
    let device_name = overrides
        .device
        .clone()
        .unwrap_or_else(|| config.device.clone());
    let kind = EngineDevice::parse(&device_name)?;
    let max_steps = overrides.max_steps.unwrap_or(config.max_steps);
    let tokenizer = LoopTokenizer::from_file(&config.tokenizer)
        .map_err(|error| TrainError::new(error.to_string()))?;
    let special = |name: &str| -> Result<u32, TrainError> {
        tokenizer
            .token_id(name)
            .ok_or_else(|| TrainError::new(format!("tokenizer has no {name}")))
    };
    let pad_id =
        u16::try_from(special(PAD)?).map_err(|_| TrainError::new("pad id does not fit u16"))?;
    let stop = [special(IM_END)?, special(EOS)?];
    let mut loader = SftLoader::from_dir(&config.data_dir.join("train"), pad_id, config.seed)?;
    let eval_path = config.data_dir.join("eval.jsonl");
    let val_rows: Vec<SftBatch> = read_sft_eval(&eval_path, config.log.val_rows)?
        .iter()
        .map(|row| SftBatch::from_row(&row.row(), pad_id, config.pad_multiple))
        .collect();
    let rows_per_step = config.batch.rows_per_step;
    let steps_per_epoch = (loader.len() as u64).div_ceil(rows_per_step as u64).max(1);
    let settings = optim_settings(&config.optim);
    fs::create_dir_all(&config.run_dir)?;
    let (mut engine, model_config, mut cursor, mut step, mut tokens_seen, schedule_steps) =
        if overrides.resume {
            let meta = read_latest(&config.run_dir)?;
            let dir = checkpoint_dir(&config.run_dir, meta.step);
            let (_, model_config) = checkpoint_model(&dir)?;
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
                model_config,
                meta.cursor,
                meta.step,
                meta.tokens_seen,
                meta.schedule_steps,
            )
        } else {
            let init = resolve_checkpoint(&config.init)?;
            let (_, model_config) = checkpoint_model(&init)?;
            let engine = PretrainEngine::from_weights(
                &model_config,
                config.seed,
                kind,
                settings,
                &init.join("model.bpk"),
                Precision::parse(&config.precision)?,
            )?;
            println!("sft init {}", init.display());
            (
                engine,
                model_config,
                EpochCursor::new(config.seed),
                0,
                0,
                steps_per_epoch.saturating_mul(config.epochs).max(1),
            )
        };
    println!(
        "sft device {} precision {} rows {} val rows {} rows/step {} schedule {schedule_steps} steps",
        engine.device_name(),
        engine.precision(),
        loader.len(),
        val_rows.len(),
        rows_per_step,
    );
    let mut last_loss = 0.0f32;
    while cursor.epoch < config.epochs && (max_steps == 0 || step < max_steps) {
        let started = Instant::now();
        let mut rows = Vec::with_capacity(rows_per_step);
        while rows.len() < rows_per_step && cursor.epoch < config.epochs {
            rows.push(loader.next_row(&mut cursor, config.pad_multiple)?);
        }
        if rows.is_empty() {
            break;
        }
        let targets: u64 = rows.iter().map(SftBatch::supervised_targets).sum();
        let token_scale = 1.0 / targets.max(1) as f32;
        let aux_scale = 1.0 / rows.len() as f32;
        let mut nll_sum = 0.0f64;
        let mut auxes = Vec::with_capacity(rows.len());
        let mut loops = Vec::with_capacity(rows.len());
        let mut slots: Vec<Vec<f32>> = Vec::new();
        let mut step_tokens = 0u64;
        for row in &rows {
            let stats = engine.accumulate_sft(row, token_scale, aux_scale)?;
            nll_sum += f64::from(stats.nll_sum);
            auxes.push(stats.aux);
            loops.extend(stats.loops);
            if slots.is_empty() {
                slots = stats.expert_slots;
            } else {
                add_slots(&mut slots, &stats.expert_slots);
            }
            step_tokens += stats.tokens;
        }
        scale_slots(&mut slots, 1.0 / rows.len() as f32);
        let lr = learning_rate(
            step,
            config.optim.warmup_steps,
            schedule_steps,
            config.optim.learning_rate,
            config.optim.min_learning_rate,
        );
        let applied = engine.apply_summed(lr)?;
        step += 1;
        tokens_seen += step_tokens;
        let elapsed = started.elapsed().as_secs_f64().max(1.0e-9);
        let loss = (nll_sum / targets.max(1) as f64) as f32;
        last_loss = loss;
        let mean_loops = if loops.is_empty() {
            0.0
        } else {
            loops.iter().sum::<usize>() as f64 / loops.len() as f64
        };
        let flops = model_config.train_flops_per_token(mean_loops.max(1.0))?;
        let mfu = (flops * step_tokens as f64) / elapsed / (config.peak_tflops * 1.0e12);
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
        let mut val_loss = BTreeMap::new();
        if config.log.val_every > 0 && step.is_multiple_of(config.log.val_every) {
            val_loss = validate(&engine, &val_rows, &config.val_loops)?;
        }
        let row = SftLogRow {
            step,
            epoch: cursor.epoch,
            loss,
            aux: mean(&auxes),
            lr,
            grad_norm: applied.grad_norm,
            rows: rows.len(),
            targets,
            tokens: step_tokens,
            tokens_per_sec: step_tokens as f64 / elapsed,
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
    let final_val = validate(&engine, &val_rows, &config.val_loops)?;
    println!("sft final val {final_val:+.4?}");
    if config.log.samples > 0 {
        let rows = read_sft_eval(&eval_path, config.log.samples)?;
        let text = samples(&engine, &tokenizer, &rows, &stop)?;
        print!("{text}");
        fs::write(config.run_dir.join("samples.txt"), text)?;
    }
    Ok(RunSummary {
        steps: step,
        tokens_seen,
        final_loss: last_loss,
        precision: engine.precision().to_string(),
        device: engine.device_name().to_string(),
    })
}

/// Token-mean masked loss over the validation rows at each fixed `r`.
fn validate(
    engine: &PretrainEngine,
    rows: &[SftBatch],
    val_loops: &[usize],
) -> Result<BTreeMap<String, f32>, TrainError> {
    let mut losses = BTreeMap::new();
    if rows.is_empty() {
        return Ok(losses);
    }
    for loops in val_loops {
        let mut nll = 0.0f64;
        let mut targets = 0u64;
        for row in rows {
            let (sum, count) = engine.validate_sft(row, *loops)?;
            nll += f64::from(sum);
            targets += count;
        }
        losses.insert(format!("r{loops}"), (nll / targets.max(1) as f64) as f32);
    }
    Ok(losses)
}

/// Greedy answers at `r = 4` to the first eval prompts, next to the references.
fn samples(
    engine: &PretrainEngine,
    tokenizer: &LoopTokenizer,
    rows: &[SftEvalRow],
    stop: &[u32],
) -> Result<String, TrainError> {
    let options = GenerateOptions {
        loops: 4,
        max_new: 160,
        temperature: 0.0,
        top_k: 0,
        repetition_penalty: 1.1,
        seed: 1,
    };
    let decode = |ids: &[u32]| {
        tokenizer
            .decode(ids)
            .map_err(|error| TrainError::new(error.to_string()))
    };
    let mut text = String::new();
    for row in rows {
        let prompt: Vec<u32> = row.prompt().iter().map(|id| u32::from(*id)).collect();
        let answer = engine.generate(prompt.clone(), &options, stop);
        let shown = decode(&prompt)?;
        let tail: String = shown
            .chars()
            .rev()
            .take(400)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        text.push_str(&format!(
            "=== {} ({})\n--- prompt (last 400 chars)\n{tail}\n--- model, r = 4, greedy\n{}\n--- reference\n{}\n\n",
            row.id,
            row.config,
            decode(&answer)?,
            row.reference
        ));
    }
    Ok(text)
}

fn print_row(row: &SftLogRow) {
    let val = if row.val_loss.is_empty() {
        String::new()
    } else {
        format!(" val {:+.4?}", row.val_loss)
    };
    println!(
        "sft {:>6}  ep {}  loss {:.4}  aux {:.4}  lr {:.3e}  grad {:.3}  targets {:>5}  {:.0} tok/s  mfu {:.2}%  r {:.2}{val}",
        row.step,
        row.epoch,
        row.loss,
        row.aux,
        row.lr,
        row.grad_norm,
        row.targets,
        row.tokens_per_sec,
        row.mfu * 100.0,
        row.mean_loops,
    );
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::*;
    use crate::config::{
        BatchConfig, CheckpointConfig, LogConfig, OptimConfig, PretrainConfig, SftBatchConfig,
        SftLogConfig,
    };
    use crate::pretrain;
    use loopmind_data::{write_pretrain_shard, write_sft_shard, SftRow};
    use loopmind_model::LoopMindConfig;

    fn optim(learning_rate: f64) -> OptimConfig {
        OptimConfig {
            learning_rate,
            min_learning_rate: learning_rate / 10.0,
            weight_decay: 0.0,
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 1.0e-8,
            grad_clip: 1.0,
            warmup_steps: 2,
        }
    }

    /// A tiny pretrain run, so SFT has a real checkpoint to start from.
    fn pretrained(root: &Path) -> PathBuf {
        let data = root.join("pre");
        fs::create_dir_all(data.join("train")).unwrap();
        fs::create_dir_all(data.join("val")).unwrap();
        let sequences: Vec<Vec<u16>> = (0..4)
            .map(|copy| vec![10, 11, 12, 13, 14, 15, 16, 17 + copy])
            .collect();
        write_pretrain_shard(&data.join("train/a.bin"), 8, &sequences).unwrap();
        write_pretrain_shard(&data.join("val/a.bin"), 8, &sequences[..1]).unwrap();
        let mut model = LoopMindConfig::tiny();
        model.context_length = 16;
        fs::write(root.join("model.toml"), toml::to_string(&model).unwrap()).unwrap();
        let config = PretrainConfig {
            model: root.join("model.toml"),
            data_dir: data,
            run_dir: root.join("pretrain"),
            seed: 3,
            epochs: 10,
            max_steps: 2,
            device: "cpu".into(),
            precision: "f32".into(),
            peak_tflops: 1.0,
            optim: optim(1.0e-3),
            batch: BatchConfig {
                micro_batch: 2,
                tokens_per_step: 16,
            },
            log: LogConfig {
                every: 1,
                val_every: 0,
                val_sequences: 1,
            },
            checkpoint: CheckpointConfig { every: 0, keep: 1 },
            val_loops: vec![1],
        };
        pretrain(
            &config,
            &RunOverrides {
                resume: false,
                max_steps: None,
                device: None,
            },
        )
        .unwrap();
        config.run_dir
    }

    fn sft_fixture(root: &Path) -> SftConfig {
        let _ = fs::remove_dir_all(root);
        let init = pretrained(root);
        let data = root.join("sft");
        fs::create_dir_all(data.join("train")).unwrap();
        // bos, prompt, then a supervised answer that depends on the prompt.
        let rows: Vec<SftRow> = (0..6u16)
            .map(|row| SftRow {
                tokens: vec![0, 3, 20 + row, 4, 3, 7, 40 + row, 41 + row, 4, 1],
                mask: vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            })
            .collect();
        write_sft_shard(&data.join("train/sft-00000.bin"), &rows).unwrap();
        let eval: String = (0..2u16)
            .map(|row| {
                format!(
                    "{}\n",
                    serde_json::json!({
                        "id": format!("e{row}"),
                        "config": "math_reasoning",
                        "license": "MIT",
                        "reference": "x",
                        "tokens": [0, 3, 20 + row, 4, 3, 7, 40 + row, 41 + row, 4, 9, 1],
                        "loss_mask": [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
                    })
                )
            })
            .collect();
        fs::write(data.join("eval.jsonl"), eval).unwrap();
        SftConfig {
            init,
            data_dir: data,
            run_dir: root.join("run"),
            tokenizer: Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tokenizer/tokenizer.json"),
            seed: 9,
            epochs: 50,
            max_steps: 30,
            device: "cpu".into(),
            precision: "f32".into(),
            peak_tflops: 1.0,
            pad_multiple: 4,
            val_loops: vec![1, 4],
            optim: optim(1.0e-2),
            batch: SftBatchConfig { rows_per_step: 3 },
            log: SftLogConfig {
                every: 1,
                val_every: 10,
                val_rows: 2,
                samples: 1,
            },
            checkpoint: CheckpointConfig { every: 10, keep: 2 },
        }
    }

    fn column(path: &Path, key: &str) -> Vec<serde_json::Value> {
        fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap()[key].clone())
            .collect()
    }

    #[test]
    fn sft_from_a_pretrain_checkpoint_lowers_the_masked_loss_and_resumes() {
        let root = std::env::temp_dir().join(format!("loopmind-sft-{}", std::process::id()));
        let config = sft_fixture(&root);
        let overrides = |resume, max_steps| RunOverrides {
            resume,
            max_steps: Some(max_steps),
            device: None,
        };
        let summary = sft(&config, &overrides(false, 20)).unwrap();
        assert_eq!(summary.steps, 20);
        let metrics = config.run_dir.join("metrics.jsonl");
        let losses: Vec<f64> = column(&metrics, "loss")
            .iter()
            .map(|value| value.as_f64().unwrap())
            .collect();
        assert!(
            losses[losses.len() - 1] < losses[0] * 0.8,
            "masked loss did not fall: {losses:?}"
        );
        // 3 rows of 4 supervised targets each (position 0 is never a target).
        assert_eq!(column(&metrics, "targets")[0].as_u64().unwrap(), 12);
        let val = &column(&metrics, "val_loss")[9];
        assert!(val["r1"].as_f64().unwrap().is_finite());
        assert!(val["r4"].as_f64().unwrap().is_finite());
        assert!(fs::read_to_string(config.run_dir.join("samples.txt"))
            .unwrap()
            .contains("=== e0 (math_reasoning)"));
        let resumed = sft(&config, &overrides(true, 25)).unwrap();
        assert_eq!(resumed.steps, 25);
        assert_eq!(column(&metrics, "step").len(), 25);
    }
}
