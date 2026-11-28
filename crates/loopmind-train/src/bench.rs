//! Zero-shot benchmarks scored by log-likelihood, as lm-evaluation-harness does.
//!
//! `python/fetch_benchmarks.py` writes `data/bench/<task>.jsonl`. Every choice is scored
//! as a continuation of its context: the summed log-probability of the choice's tokens.
//! `acc` picks the choice with the highest sum, and `acc_norm` the highest sum per byte
//! of choice text. A task with one choice (LAMBADA) counts an item as right when greedy
//! decoding reproduces the choice; its `acc_norm` is the same number. Each item runs at
//! every requested loop count.
//!
//! Sequences start with `<|eos|>`, which is what precedes every document in the packed
//! pretrain shards. A context that would push a sequence past the model's context
//! is cut from the left.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use loopmind_data::{SftBatch, SftRow};
use loopmind_model::PretrainEngine;
use loopmind_tokenizer::{LoopTokenizer, EOS, PAD};
use serde::{Deserialize, Serialize};

use crate::run::{load_inference, resolve_checkpoint};
use crate::TrainError;

/// One benchmark question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchItem {
    pub id: String,
    pub task: String,
    pub context: String,
    pub choices: Vec<String>,
    pub label: usize,
}

#[derive(Debug, Clone)]
pub struct BenchOptions {
    pub checkpoint: PathBuf,
    pub bench_dir: PathBuf,
    /// Task names (file stems). Empty runs every `*.jsonl` in `bench_dir`.
    pub tasks: Vec<String>,
    pub loops: Vec<usize>,
    /// Items per task, 0 for all. The first items in file order.
    pub limit: usize,
    pub device: String,
    pub tokenizer: PathBuf,
    /// Padded tokens per forward pass.
    pub token_budget: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskScore {
    pub items: usize,
    /// Right by summed log-probability, or by greedy match for one-choice tasks.
    pub acc: f64,
    /// Right by log-probability per byte of choice text.
    pub acc_norm: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchReport {
    pub checkpoint: String,
    pub loops: Vec<usize>,
    /// Task, then `r{loops}`, then the score.
    pub tasks: BTreeMap<String, BTreeMap<String, TaskScore>>,
    /// Expected accuracy of a uniform guess, per task.
    pub chance: BTreeMap<String, f64>,
    /// The metric each task is summarized by: `acc` or `acc_norm`.
    pub headline: BTreeMap<String, String>,
    /// Mean of the headline metric over tasks, per `r{loops}`.
    pub mean: BTreeMap<String, f64>,
    pub seconds: f64,
}

/// BLiMP compares two whole sentences and LAMBADA has one choice, so plain `acc` is their
/// standard number. The other tasks' choices differ in length, so `acc_norm`.
pub fn headline_metric(task: &str) -> &'static str {
    match task {
        "blimp" | "lambada" => "acc",
        _ => "acc_norm",
    }
}

/// Every `*.jsonl` item for `task` in `dir`, the first `limit` when it is not 0.
pub fn read_task(dir: &Path, task: &str, limit: usize) -> Result<Vec<BenchItem>, TrainError> {
    let path = dir.join(format!("{task}.jsonl"));
    let text = fs::read_to_string(&path)
        .map_err(|error| TrainError::new(format!("{}: {error}", path.display())))?;
    let mut items = Vec::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        if limit > 0 && items.len() >= limit {
            break;
        }
        let item: BenchItem = serde_json::from_str(line)
            .map_err(|error| TrainError::new(format!("{}: {error}", path.display())))?;
        if item.choices.is_empty() || item.label >= item.choices.len() {
            return Err(TrainError::new(format!(
                "{}: item {} has label {} and {} choices",
                path.display(),
                item.id,
                item.label,
                item.choices.len()
            )));
        }
        items.push(item);
    }
    Ok(items)
}

/// One scored sequence: item `item`'s choice `choice`, with the choice's tokens masked in.
#[derive(Debug, Clone)]
pub struct ChoiceRow {
    pub item: usize,
    pub choice: usize,
    pub row: SftRow,
}

/// `<|eos|>`, the context, then the choice, with mask 1 on the choice tokens only.
pub fn choice_rows(
    tokenizer: &LoopTokenizer,
    items: &[BenchItem],
    context_length: usize,
) -> Result<Vec<ChoiceRow>, TrainError> {
    let eos = tokenizer
        .token_id(EOS)
        .ok_or_else(|| TrainError::new("tokenizer has no <|eos|>"))?;
    let encode = |text: &str| -> Result<Vec<u32>, TrainError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        tokenizer
            .encode(text)
            .map_err(|error| TrainError::new(error.to_string()))
    };
    let mut rows = Vec::new();
    for (index, item) in items.iter().enumerate() {
        let context = encode(&item.context)?;
        for (choice_index, choice) in item.choices.iter().enumerate() {
            let choice_ids = encode(choice)?;
            if choice_ids.is_empty() || choice_ids.len() + 1 > context_length {
                return Err(TrainError::new(format!(
                    "item {} choice {choice_index} has {} tokens",
                    item.id,
                    choice_ids.len()
                )));
            }
            let room = context_length - 1 - choice_ids.len();
            let kept = &context[context.len().saturating_sub(room)..];
            let mut tokens = Vec::with_capacity(1 + kept.len() + choice_ids.len());
            tokens.push(eos);
            tokens.extend_from_slice(kept);
            tokens.extend_from_slice(&choice_ids);
            let mut mask = vec![0u8; 1 + kept.len()];
            mask.resize(tokens.len(), 1);
            let tokens = tokens
                .into_iter()
                .map(|id| u16::try_from(id).map_err(|_| TrainError::new("token id over u16")))
                .collect::<Result<Vec<_>, _>>()?;
            rows.push(ChoiceRow {
                item: index,
                choice: choice_index,
                row: SftRow { tokens, mask },
            });
        }
    }
    Ok(rows)
}

/// Accuracy from each choice's (log-probability sum, greedy match).
pub fn summarize(items: &[BenchItem], scores: &[Vec<(f32, bool)>]) -> TaskScore {
    let mut right = 0usize;
    let mut right_norm = 0usize;
    for (item, scores) in items.iter().zip(scores) {
        if item.choices.len() == 1 {
            let hit = scores.first().is_some_and(|(_, greedy)| *greedy);
            right += usize::from(hit);
            right_norm += usize::from(hit);
            continue;
        }
        let best = |key: &dyn Fn(usize, f32) -> f64| {
            (0..scores.len())
                .max_by(|left, right| {
                    key(*left, scores[*left].0).total_cmp(&key(*right, scores[*right].0))
                })
                .unwrap_or(0)
        };
        right += usize::from(best(&|_, logprob| f64::from(logprob)) == item.label);
        let per_byte = |choice: usize, logprob: f32| {
            f64::from(logprob) / item.choices[choice].len().max(1) as f64
        };
        right_norm += usize::from(best(&per_byte) == item.label);
    }
    let total = items.len().max(1) as f64;
    TaskScore {
        items: items.len(),
        acc: right as f64 / total,
        acc_norm: right_norm as f64 / total,
    }
}

/// Expected accuracy of a uniform guess. 0 for one-choice tasks.
pub fn chance(items: &[BenchItem]) -> f64 {
    let total: f64 = items
        .iter()
        .map(|item| {
            if item.choices.len() == 1 {
                0.0
            } else {
                1.0 / item.choices.len() as f64
            }
        })
        .sum();
    total / items.len().max(1) as f64
}

/// Score every row at `loops`, batched by padded width. Each batch is filled to the same
/// row count with masked-out copies, so a width compiles one kernel shape.
fn score(
    engine: &PretrainEngine,
    rows: &[ChoiceRow],
    items: usize,
    loops: usize,
    pad: u16,
    token_budget: usize,
) -> Result<Vec<Vec<(f32, bool)>>, TrainError> {
    const MULTIPLE: usize = 32;
    let mut by_width: BTreeMap<usize, Vec<&ChoiceRow>> = BTreeMap::new();
    for row in rows {
        let width = row.row.tokens.len().div_ceil(MULTIPLE).max(1) * MULTIPLE;
        by_width.entry(width).or_default().push(row);
    }
    let mut scores: Vec<Vec<(f32, bool)>> = vec![Vec::new(); items];
    for row in rows {
        let slots = &mut scores[row.item];
        if slots.len() <= row.choice {
            slots.resize(row.choice + 1, (f32::NEG_INFINITY, false));
        }
    }
    for (width, group) in by_width {
        let per_batch = (token_budget / width).clamp(1, 512);
        for chunk in group.chunks(per_batch) {
            let mut packed: Vec<SftRow> = chunk.iter().map(|row| row.row.clone()).collect();
            while packed.len() < per_batch.min(group.len()) {
                let mut filler = packed[0].clone();
                filler.mask.iter_mut().for_each(|bit| *bit = 0);
                packed.push(filler);
            }
            let batch = SftBatch::from_rows(&packed, pad, MULTIPLE);
            let results = engine.score_rows(&batch, loops)?;
            for (row, result) in chunk.iter().zip(results) {
                scores[row.item][row.choice] = result;
            }
        }
    }
    Ok(scores)
}

/// Score `options.tasks` at every loop count in `options.loops`.
pub fn bench(options: &BenchOptions) -> Result<BenchReport, TrainError> {
    let started = Instant::now();
    let tokenizer = LoopTokenizer::from_file(&options.tokenizer)
        .map_err(|error| TrainError::new(error.to_string()))?;
    let pad = tokenizer
        .token_id(PAD)
        .and_then(|id| u16::try_from(id).ok())
        .ok_or_else(|| TrainError::new("tokenizer has no <|pad|> under u16"))?;
    let checkpoint = resolve_checkpoint(&options.checkpoint)?;
    let engine = load_inference(&checkpoint, &options.device)?;
    let context_length = engine.config().context_length;
    let tasks = if options.tasks.is_empty() {
        let mut names: Vec<String> = fs::read_dir(&options.bench_dir)
            .map_err(|error| TrainError::new(format!("{}: {error}", options.bench_dir.display())))?
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| {
                let path = entry.path();
                (path.extension()? == "jsonl")
                    .then(|| path.file_stem()?.to_str().map(str::to_string))?
            })
            .collect();
        names.sort();
        names
    } else {
        options.tasks.clone()
    };
    let mut report = BenchReport {
        checkpoint: checkpoint.display().to_string(),
        loops: options.loops.clone(),
        tasks: BTreeMap::new(),
        chance: BTreeMap::new(),
        headline: BTreeMap::new(),
        mean: BTreeMap::new(),
        seconds: 0.0,
    };
    for task in &tasks {
        let items = read_task(&options.bench_dir, task, options.limit)?;
        let rows = choice_rows(&tokenizer, &items, context_length)?;
        report.chance.insert(task.clone(), chance(&items));
        report
            .headline
            .insert(task.clone(), headline_metric(task).to_string());
        let mut by_loops = BTreeMap::new();
        for loops in &options.loops {
            let task_started = Instant::now();
            let scores = score(
                &engine,
                &rows,
                items.len(),
                *loops,
                pad,
                options.token_budget,
            )?;
            let result = summarize(&items, &scores);
            println!(
                "bench {task} r{loops}: acc {:.4} acc_norm {:.4} over {} items, {} rows, {:.0}s",
                result.acc,
                result.acc_norm,
                result.items,
                rows.len(),
                task_started.elapsed().as_secs_f64()
            );
            by_loops.insert(format!("r{loops}"), result);
        }
        report.tasks.insert(task.clone(), by_loops);
    }
    for loops in &options.loops {
        let key = format!("r{loops}");
        let values: Vec<f64> = report
            .tasks
            .iter()
            .map(|(task, scores)| headline_value(&scores[&key], headline_metric(task)))
            .collect();
        let mean = values.iter().sum::<f64>() / values.len().max(1) as f64;
        report.mean.insert(key, mean);
    }
    report.seconds = started.elapsed().as_secs_f64();
    Ok(report)
}

fn headline_value(score: &TaskScore, metric: &str) -> f64 {
    if metric == "acc" {
        score.acc
    } else {
        score.acc_norm
    }
}

/// Tasks as rows, loop counts as columns, headline metric in percent.
pub fn bench_markdown(report: &BenchReport) -> String {
    let mut out = format!(
        "# Benchmarks of {}\n\nZero-shot, log-likelihood scoring. Percent correct by each task's headline metric.\n\n| Task | Metric | Items | Chance |",
        report.checkpoint
    );
    for loops in &report.loops {
        out.push_str(&format!(" r = {loops} |"));
    }
    out.push_str("\n| --- | --- | ---: | ---: |");
    out.push_str(&" ---: |".repeat(report.loops.len()));
    out.push('\n');
    for (task, scores) in &report.tasks {
        let metric = headline_metric(task);
        let items = scores.values().next().map_or(0, |score| score.items);
        out.push_str(&format!(
            "| {task} | {metric} | {items} | {:.1} |",
            100.0 * report.chance.get(task).copied().unwrap_or(0.0)
        ));
        for loops in &report.loops {
            let value = headline_value(&scores[&format!("r{loops}")], metric);
            out.push_str(&format!(" {:.1} |", 100.0 * value));
        }
        out.push('\n');
    }
    out.push_str("| **Mean** | | | |");
    for loops in &report.loops {
        out.push_str(&format!(
            " **{:.1}** |",
            100.0 * report.mean[&format!("r{loops}")]
        ));
    }
    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(choices: &[&str], label: usize) -> BenchItem {
        BenchItem {
            id: "x".into(),
            task: "t".into(),
            context: "Question: Is it?\nAnswer:".into(),
            choices: choices.iter().map(|choice| choice.to_string()).collect(),
            label,
        }
    }

    #[test]
    fn acc_takes_the_largest_sum_and_acc_norm_the_largest_per_byte() {
        let items = [
            item(&[" a", " bbbbbbbb"], 1),
            item(&[" yes"], 0),
            item(&[" no"], 0),
        ];
        // Choice 0 wins on the sum (-2 > -4) but loses per byte (-1 < -0.44).
        let scores = vec![
            vec![(-2.0, false), (-4.0, false)],
            vec![(-1.0, true)],
            vec![(-1.0, false)],
        ];
        let result = summarize(&items, &scores);
        assert_eq!(result.items, 3);
        assert!((result.acc - 1.0 / 3.0).abs() < 1e-9, "{result:?}");
        assert!((result.acc_norm - 2.0 / 3.0).abs() < 1e-9, "{result:?}");
        assert!((chance(&items) - 0.5 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn rows_mask_only_the_choice_and_cut_long_contexts_from_the_left() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tokenizer/tokenizer.json");
        let tokenizer = LoopTokenizer::from_file(&path).unwrap();
        let eos = tokenizer.token_id(EOS).unwrap() as u16;
        let items = [item(&[" yes", " no"], 0)];
        let rows = choice_rows(&tokenizer, &items, 1024).unwrap();
        assert_eq!(rows.len(), 2);
        let context = tokenizer.encode(&items[0].context).unwrap().len();
        let yes = tokenizer.encode(" yes").unwrap().len();
        assert_eq!(rows[0].row.tokens[0], eos);
        assert_eq!(rows[0].row.tokens.len(), 1 + context + yes);
        assert_eq!(
            rows[0].row.mask.iter().filter(|bit| **bit == 1).count(),
            yes
        );
        assert!(rows[0].row.mask[..1 + context].iter().all(|bit| *bit == 0));
        let short = choice_rows(&tokenizer, &items, 1 + yes + 2).unwrap();
        assert_eq!(short[0].row.tokens.len(), 1 + yes + 2);
        assert_eq!(short[0].row.tokens[0], eos);
        let blimp = BenchItem {
            context: String::new(),
            ..item(&["The cat sleeps.", "The cat sleep."], 0)
        };
        let rows = choice_rows(&tokenizer, &[blimp], 1024).unwrap();
        assert_eq!(rows[0].row.mask[0], 0);
        assert!(rows[0].row.mask[1..].iter().all(|bit| *bit == 1));
    }
}
