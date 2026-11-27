//! Held-out evaluation: masked loss over the loop dial, scored answers, and chat smoke prompts.
//!
//! The rows are `sft/eval.jsonl`, which never enters training. The masked loss runs
//! over every row at each loop count. Answers are generated for the first rows of each
//! config, greedy at each loop count, then with a few sampling settings at the best
//! one, and scored by [`crate::score`]. The best setting is the report's
//! recommendation, which `chat` uses through `runs/best.json`.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use loopmind_data::{read_sft_eval, SftBatch, SftEvalRow};
use loopmind_model::{GenerateOptions, PretrainEngine};
use loopmind_tokenizer::{ChatTurn, LoopTokenizer, Role, EOS, IM_END, PAD};
use serde::{Deserialize, Serialize};

use crate::run::{load_inference, resolve_checkpoint};
use crate::score::{json_valid, score, Verdict};
use crate::TrainError;

/// Fixed prompts that show what the model does with ordinary requests.
pub const SMOKE_PROMPTS: &[&str] = &[
    "Write a short story about a cat who learns to fly.",
    "What is the capital of France?",
    "What is 12 + 7?",
    "Explain photosynthesis in one sentence.",
    "Give me a JSON object with the fields name and age for Ana, who is 30.",
    "List three colors.",
];

/// How answers are sampled.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sampling {
    pub name: String,
    pub temperature: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
}

impl Sampling {
    pub fn greedy() -> Self {
        Self {
            name: "greedy".into(),
            temperature: 0.0,
            top_k: 0,
            repetition_penalty: 1.1,
        }
    }

    /// Greedy, then two sampled settings.
    pub fn presets() -> Vec<Self> {
        vec![
            Self::greedy(),
            Self {
                name: "t0.3-k20".into(),
                temperature: 0.3,
                top_k: 20,
                repetition_penalty: 1.1,
            },
            Self {
                name: "t0.7-k40".into(),
                temperature: 0.7,
                top_k: 40,
                repetition_penalty: 1.1,
            },
        ]
    }

    pub fn options(&self, loops: usize, max_new: usize, seed: u64) -> GenerateOptions {
        GenerateOptions {
            loops,
            max_new,
            temperature: self.temperature,
            top_k: self.top_k,
            repetition_penalty: self.repetition_penalty,
            seed,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EvalOptions {
    pub checkpoint: PathBuf,
    pub eval_file: PathBuf,
    pub tokenizer: PathBuf,
    pub device: String,
    /// Loop counts for the masked loss over every eval row.
    pub nll_loops: Vec<usize>,
    /// Loop counts for the greedy answer sweep.
    pub gen_loops: Vec<usize>,
    /// Sampling settings tried at the best greedy loop count. Greedy is always in the sweep.
    pub sampling: Vec<Sampling>,
    /// Leading eval rows of each config that get generated answers.
    pub gen_rows_per_config: usize,
    pub max_new: usize,
    /// Rows per forward. Masked-loss rows share a padded width; answer rows share a prompt length.
    pub batch: usize,
    pub pad_multiple: usize,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigScore {
    pub rows: usize,
    pub scored: usize,
    pub correct: usize,
    /// `correct / scored`.
    pub accuracy: f64,
    /// Fraction of answers holding a parseable JSON value (`structured_output` only).
    pub json_valid: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnswerSample {
    pub id: String,
    pub config: String,
    pub verdict: String,
    pub answer: String,
    pub reference: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenResult {
    pub loops: usize,
    pub sampling: Sampling,
    /// Mean accuracy over configs that have scored rows.
    pub macro_accuracy: f64,
    pub configs: BTreeMap<String, ConfigScore>,
    pub seconds: f64,
    pub samples: Vec<AnswerSample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommended {
    pub loops: usize,
    pub sampling: Sampling,
    pub macro_accuracy: f64,
    /// Held-out masked loss at `loops`, when it was measured.
    pub nll: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmokeAnswer {
    pub prompt: String,
    pub answer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    pub checkpoint: String,
    /// Token-mean masked NLL over the eval rows, keyed `r{loops}`.
    pub nll: BTreeMap<String, f32>,
    pub nll_rows: usize,
    pub nll_targets: u64,
    pub best_nll_loops: usize,
    pub generation: Vec<GenResult>,
    pub recommended: Recommended,
    pub smoke: Vec<SmokeAnswer>,
    pub seconds: f64,
}

/// `runs/best.json`: the checkpoint and generation settings `chat` uses by default.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestModel {
    pub checkpoint: PathBuf,
    pub loops: usize,
    pub sampling: Sampling,
    #[serde(default)]
    pub reason: String,
}

impl BestModel {
    pub fn read(path: &Path) -> Result<Self, TrainError> {
        Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
    }
}

/// A preset from [`Sampling::presets`] by name.
pub fn sampling_named(name: &str) -> Result<Sampling, TrainError> {
    Sampling::presets()
        .into_iter()
        .find(|preset| preset.name == name)
        .ok_or_else(|| {
            TrainError::new(format!(
                "unknown sampling {name}; use {}",
                Sampling::presets()
                    .iter()
                    .map(|preset| preset.name.clone())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })
}

pub fn evaluate(options: &EvalOptions) -> Result<EvalReport, TrainError> {
    let started = Instant::now();
    let tokenizer = LoopTokenizer::from_file(&options.tokenizer)
        .map_err(|error| TrainError::new(error.to_string()))?;
    let id = |name: &str| -> Result<u32, TrainError> {
        tokenizer
            .token_id(name)
            .ok_or_else(|| TrainError::new(format!("tokenizer has no {name}")))
    };
    let pad = u16::try_from(id(PAD)?).map_err(|_| TrainError::new("pad id does not fit u16"))?;
    let stop = [id(IM_END)?, id(EOS)?];
    let checkpoint = resolve_checkpoint(&options.checkpoint)?;
    let engine = load_inference(&checkpoint, &options.device)?;
    let rows = read_sft_eval(&options.eval_file, 0)?;
    let batch = options.batch.max(1);

    let mut nll = BTreeMap::new();
    let mut nll_targets = 0u64;
    for loops in &options.nll_loops {
        let (sum, targets) = masked_loss(&engine, &rows, *loops, pad, options.pad_multiple, batch)?;
        nll_targets = targets;
        let value = (sum / targets.max(1) as f64) as f32;
        println!("eval nll r{loops}: {value:.4} over {targets} targets");
        nll.insert(format!("r{loops}"), value);
    }
    let best_nll_loops = options
        .nll_loops
        .iter()
        .copied()
        .min_by(|left, right| nll[&format!("r{left}")].total_cmp(&nll[&format!("r{right}")]))
        .unwrap_or(4);

    let subset = generation_rows(&rows, options.gen_rows_per_config);
    let mut generation = Vec::new();
    for loops in &options.gen_loops {
        generation.push(answer_and_score(
            &engine,
            &tokenizer,
            &subset,
            *loops,
            &Sampling::greedy(),
            options,
            &stop,
        )?);
    }
    let best_greedy = best_result(&generation, &nll, best_nll_loops).map(|result| result.loops);
    if let Some(loops) = best_greedy {
        for sampling in options
            .sampling
            .iter()
            .filter(|sampling| **sampling != Sampling::greedy())
        {
            generation.push(answer_and_score(
                &engine, &tokenizer, &subset, loops, sampling, options, &stop,
            )?);
        }
    }
    let recommended = match best_result(&generation, &nll, best_nll_loops) {
        Some(best) => Recommended {
            loops: best.loops,
            sampling: best.sampling.clone(),
            macro_accuracy: best.macro_accuracy,
            nll: nll.get(&format!("r{}", best.loops)).copied(),
        },
        None => Recommended {
            loops: best_nll_loops,
            sampling: Sampling::greedy(),
            macro_accuracy: 0.0,
            nll: nll.get(&format!("r{best_nll_loops}")).copied(),
        },
    };

    let mut smoke = Vec::new();
    let generate = recommended
        .sampling
        .options(recommended.loops, options.max_new, options.seed);
    for prompt in SMOKE_PROMPTS {
        let rendered = tokenizer
            .render_chat(&[ChatTurn::new(Role::User, *prompt)], true)
            .map_err(|error| TrainError::new(error.to_string()))?;
        let ids = engine.generate(rendered.ids, &generate, &stop);
        smoke.push(SmokeAnswer {
            prompt: (*prompt).to_string(),
            answer: decode(&tokenizer, &ids)?,
        });
    }
    Ok(EvalReport {
        checkpoint: checkpoint.display().to_string(),
        nll,
        nll_rows: rows.len(),
        nll_targets,
        best_nll_loops,
        generation,
        recommended,
        smoke,
        seconds: started.elapsed().as_secs_f64(),
    })
}

/// Highest macro accuracy. Ties go to the lower held-out loss at that `r`, then greedy.
fn best_result<'a>(
    results: &'a [GenResult],
    nll: &BTreeMap<String, f32>,
    best_nll_loops: usize,
) -> Option<&'a GenResult> {
    let loss = |result: &GenResult| {
        nll.get(&format!("r{}", result.loops))
            .copied()
            .unwrap_or(f32::INFINITY)
    };
    let best = results.iter().max_by(|left, right| {
        left.macro_accuracy
            .total_cmp(&right.macro_accuracy)
            .then_with(|| loss(right).total_cmp(&loss(left)))
            .then_with(|| (right.sampling.name != "greedy").cmp(&(left.sampling.name != "greedy")))
    })?;
    if best.macro_accuracy > 0.0 {
        Some(best)
    } else {
        results
            .iter()
            .find(|result| result.loops == best_nll_loops && result.sampling.name == "greedy")
            .or(Some(best))
    }
}

/// Sum of masked NLL and supervised targets over every row, batched by padded width.
fn masked_loss(
    engine: &PretrainEngine,
    rows: &[SftEvalRow],
    loops: usize,
    pad: u16,
    pad_multiple: usize,
    batch: usize,
) -> Result<(f64, u64), TrainError> {
    let multiple = pad_multiple.max(1);
    let mut by_width: BTreeMap<usize, Vec<&SftEvalRow>> = BTreeMap::new();
    for row in rows {
        by_width
            .entry(row.tokens.len().div_ceil(multiple) * multiple)
            .or_default()
            .push(row);
    }
    let mut sum = 0.0f64;
    let mut targets = 0u64;
    for group in by_width.values() {
        for chunk in group.chunks(batch) {
            let rows: Vec<_> = chunk.iter().map(|row| row.row()).collect();
            let packed = SftBatch::from_rows(&rows, pad, multiple);
            let sums = engine.nll_rows(&packed, loops)?;
            sum += sums.iter().map(|value| f64::from(*value)).sum::<f64>();
            targets += packed.supervised_targets();
        }
    }
    Ok((sum, targets))
}

/// The first `per_config` rows of each config, in file order.
fn generation_rows(rows: &[SftEvalRow], per_config: usize) -> Vec<&SftEvalRow> {
    let mut taken: BTreeMap<&str, usize> = BTreeMap::new();
    rows.iter()
        .filter(|row| {
            let count = taken.entry(row.config.as_str()).or_insert(0);
            *count += 1;
            *count <= per_config
        })
        .collect()
}

fn answer_and_score(
    engine: &PretrainEngine,
    tokenizer: &LoopTokenizer,
    rows: &[&SftEvalRow],
    loops: usize,
    sampling: &Sampling,
    options: &EvalOptions,
    stop: &[u32],
) -> Result<GenResult, TrainError> {
    let started = Instant::now();
    let generate = sampling.options(loops, options.max_new, options.seed);
    let answers = answer_rows(engine, rows, &generate, stop, options.batch.max(1));
    let mut configs: BTreeMap<String, ConfigScore> = BTreeMap::new();
    let mut valid_json: BTreeMap<String, usize> = BTreeMap::new();
    let mut samples = Vec::new();
    let mut shown: BTreeMap<String, usize> = BTreeMap::new();
    for (row, ids) in rows.iter().zip(&answers) {
        let answer = decode(tokenizer, ids)?;
        let verdict = score(&row.config, &row.reference, &answer);
        let entry = configs.entry(row.config.clone()).or_insert(ConfigScore {
            rows: 0,
            scored: 0,
            correct: 0,
            accuracy: 0.0,
            json_valid: None,
        });
        entry.rows += 1;
        if verdict != Verdict::Unscored {
            entry.scored += 1;
        }
        if verdict == Verdict::Correct {
            entry.correct += 1;
        }
        if row.config == "structured_output" && json_valid(&answer) {
            *valid_json.entry(row.config.clone()).or_insert(0) += 1;
        }
        let count = shown.entry(row.config.clone()).or_insert(0);
        if *count < 2 {
            *count += 1;
            samples.push(AnswerSample {
                id: row.id.clone(),
                config: row.config.clone(),
                verdict: format!("{verdict:?}"),
                answer,
                reference: row.reference.clone(),
            });
        }
    }
    for (name, entry) in configs.iter_mut() {
        entry.accuracy = if entry.scored == 0 {
            0.0
        } else {
            entry.correct as f64 / entry.scored as f64
        };
        if name == "structured_output" {
            entry.json_valid =
                Some(valid_json.get(name).copied().unwrap_or(0) as f64 / entry.rows.max(1) as f64);
        }
    }
    let scored: Vec<f64> = configs
        .values()
        .filter(|entry| entry.scored > 0)
        .map(|entry| entry.accuracy)
        .collect();
    let macro_accuracy = if scored.is_empty() {
        0.0
    } else {
        scored.iter().sum::<f64>() / scored.len() as f64
    };
    let seconds = started.elapsed().as_secs_f64();
    println!(
        "eval answers r{loops} {}: macro {:.3} ({})  {seconds:.0}s",
        sampling.name,
        macro_accuracy,
        configs
            .iter()
            .map(|(name, entry)| format!("{name} {}/{}", entry.correct, entry.scored))
            .collect::<Vec<_>>()
            .join(", ")
    );
    Ok(GenResult {
        loops,
        sampling: sampling.clone(),
        macro_accuracy,
        configs,
        seconds,
        samples,
    })
}

/// Answers in row order. Prompts of one length share a batch.
fn answer_rows(
    engine: &PretrainEngine,
    rows: &[&SftEvalRow],
    options: &GenerateOptions,
    stop: &[u32],
    batch: usize,
) -> Vec<Vec<u32>> {
    let prompts: Vec<Vec<u32>> = rows
        .iter()
        .map(|row| row.prompt().iter().map(|id| u32::from(*id)).collect())
        .collect();
    let mut by_length: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (index, prompt) in prompts.iter().enumerate() {
        by_length.entry(prompt.len()).or_default().push(index);
    }
    let mut answers = vec![Vec::new(); prompts.len()];
    for group in by_length.values() {
        for chunk in group.chunks(batch) {
            let batch_prompts: Vec<Vec<u32>> =
                chunk.iter().map(|index| prompts[*index].clone()).collect();
            for (index, answer) in
                chunk
                    .iter()
                    .zip(engine.generate_batch(&batch_prompts, options, stop))
            {
                answers[*index] = answer;
            }
        }
    }
    answers
}

fn decode(tokenizer: &LoopTokenizer, ids: &[u32]) -> Result<String, TrainError> {
    tokenizer
        .decode(ids)
        .map(|text| text.trim().to_string())
        .map_err(|error| TrainError::new(error.to_string()))
}

/// Write `report` as JSON at `path` and as Markdown next to it.
pub fn write_report(report: &EvalReport, path: &Path) -> Result<(), TrainError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(report)?)?;
    fs::write(path.with_extension("md"), report_markdown(report))?;
    Ok(())
}

pub fn read_report(path: &Path) -> Result<EvalReport, TrainError> {
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

pub fn report_markdown(report: &EvalReport) -> String {
    let mut text = format!(
        "# Eval of {}\n\nHeld-out masked loss over {} eval rows ({} supervised targets):\n\n| r | loss |\n| ---: | ---: |\n",
        report.checkpoint, report.nll_rows, report.nll_targets
    );
    // The keys are strings ("r16" sorts before "r2"), so order the rows by loop count.
    let mut losses: Vec<(usize, f32)> = report
        .nll
        .iter()
        .filter_map(|(key, value)| Some((key.trim_start_matches('r').parse().ok()?, *value)))
        .collect();
    losses.sort_by_key(|(loops, _)| *loops);
    for (loops, value) in losses {
        text.push_str(&format!("| {loops} | {value:.4} |\n"));
    }
    text.push_str(
        "\nScored answers (correct / scored; science is multiple choice only, JSON is exact match):\n\n| r | sampling | macro | math | logic | science | JSON exact | JSON valid | seconds |\n| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    );
    for result in &report.generation {
        let cell = |name: &str| {
            result
                .configs
                .get(name)
                .map(|entry| format!("{}/{}", entry.correct, entry.scored))
                .unwrap_or_default()
        };
        let valid = result
            .configs
            .get("structured_output")
            .and_then(|entry| entry.json_valid)
            .map(|value| format!("{:.0}%", value * 100.0))
            .unwrap_or_default();
        text.push_str(&format!(
            "| {} | {} | {:.1}% | {} | {} | {} | {} | {valid} | {:.0} |\n",
            result.loops,
            result.sampling.name,
            result.macro_accuracy * 100.0,
            cell("math_reasoning"),
            cell("logic_reasoning"),
            cell("science_knowledge"),
            cell("structured_output"),
            result.seconds
        ));
    }
    text.push_str(&format!(
        "\nRecommended: r = {}, {} (macro {:.1}%).\n\n## Chat smoke prompts\n\n",
        report.recommended.loops,
        report.recommended.sampling.name,
        report.recommended.macro_accuracy * 100.0
    ));
    for smoke in &report.smoke {
        text.push_str(&format!(
            "**{}**\n\n```text\n{}\n```\n\n",
            smoke.prompt, smoke.answer
        ));
    }
    if let Some(best) = report.generation.iter().find(|result| {
        result.loops == report.recommended.loops && result.sampling == report.recommended.sampling
    }) {
        text.push_str("## Eval answers at the recommended setting\n\n");
        for sample in &best.samples {
            text.push_str(&format!(
                "**{} ({}) — {}**\n\n```text\n{}\n```\n\nReference:\n\n```text\n{}\n```\n\n",
                sample.id, sample.config, sample.verdict, sample.answer, sample.reference
            ));
        }
    }
    text.push_str(&format!("Eval took {:.0} s.\n", report.seconds));
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every reference must score as correct (or unscored) against itself.
    #[test]
    fn references_score_against_themselves() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/prepared/sft/eval.jsonl");
        if !path.is_file() {
            eprintln!("skipping: {} is not built", path.display());
            return;
        }
        let rows = read_sft_eval(&path, 0).unwrap();
        let mut counts: BTreeMap<(String, String), usize> = BTreeMap::new();
        for row in &rows {
            let verdict = score(&row.config, &row.reference, &row.reference);
            *counts
                .entry((row.config.clone(), format!("{verdict:?}")))
                .or_insert(0) += 1;
            assert_ne!(
                verdict,
                Verdict::Wrong,
                "{} scored wrong against itself",
                row.id
            );
        }
        eprintln!("{counts:?}");
    }

    #[test]
    fn markdown_orders_the_loss_rows_by_loop_count() {
        let report = EvalReport {
            checkpoint: "runs/sft".into(),
            nll: [("r1", 1.2), ("r16", 1.4), ("r2", 1.1), ("r8", 1.0)]
                .into_iter()
                .map(|(key, value)| (key.to_string(), value))
                .collect(),
            nll_rows: 4,
            nll_targets: 40,
            best_nll_loops: 8,
            generation: Vec::new(),
            recommended: Recommended {
                loops: 8,
                sampling: Sampling::greedy(),
                macro_accuracy: 0.0,
                nll: Some(1.0),
            },
            smoke: Vec::new(),
            seconds: 0.0,
        };
        let text = report_markdown(&report);
        let rows: Vec<&str> = text
            .lines()
            .filter(|line| line.ends_with(" |") && line.split('|').count() == 4)
            .skip(2) // the header and its separator
            .collect();
        assert_eq!(
            rows,
            [
                "| 1 | 1.2000 |",
                "| 2 | 1.1000 |",
                "| 8 | 1.0000 |",
                "| 16 | 1.4000 |"
            ]
        );
    }
}
