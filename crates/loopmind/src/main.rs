//! LoopMind command line.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use loopmind_model::{ConfigError, LoopMindConfig};
use loopmind_tokenizer::{LoopTokenizer, TrainConfig};

#[derive(Parser)]
#[command(
    name = "loopmind",
    version,
    about = "Tiny recurrent-depth mixture-of-experts language model",
    arg_required_else_help = true
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train the 6400-vocab BPE.
    TokenizeTrain {
        /// Training config. Corpus paths inside it are relative to the working directory.
        #[arg(long, default_value = "configs/tokenizer.toml")]
        config: PathBuf,
    },
    /// Encode text and print the ids plus the decoded round trip.
    Tokenize {
        /// Text to encode.
        #[arg(long)]
        text: String,
        /// Path to tokenizer.json.
        #[arg(long, default_value = "tokenizer/tokenizer.json")]
        tokenizer: PathBuf,
    },
    /// Bytes per token on one `.txt` file per line of a held-out directory.
    TokenizeStats {
        /// Directory of held-out `.txt` files.
        #[arg(long, default_value = "data/tokenizer/heldout")]
        heldout: PathBuf,
        /// Path to tokenizer.json.
        #[arg(long, default_value = "tokenizer/tokenizer.json")]
        tokenizer: PathBuf,
    },
    /// Pack the pretrain mix into `u16` shards.
    PreparePretrain {
        /// Data config. Paths inside it are relative to the repo root.
        #[arg(long, default_value = "configs/data.toml")]
        config: PathBuf,
    },
    /// Render English FABLE chat into SFT shards and an eval file.
    PrepareSft {
        /// Data config. Paths inside it are relative to the repo root.
        #[arg(long, default_value = "configs/data.toml")]
        config: PathBuf,
    },
    /// Pretrain on the packed mix (M4).
    Pretrain {
        /// Training config. Paths inside it are relative to the repo root.
        #[arg(long, default_value = "configs/pretrain.toml")]
        config: PathBuf,
        /// Continue from the latest checkpoint in the run directory.
        #[arg(long)]
        resume: bool,
        /// Stop after this many optimizer steps. Overrides the config.
        #[arg(long)]
        max_steps: Option<u64>,
        /// `cpu`, `cuda`, or `wgpu`. Overrides the config.
        #[arg(long)]
        device: Option<String>,
    },
    /// Greedy sample from a pretrain checkpoint.
    Sample {
        /// Checkpoint directory (`runs/pretrain/checkpoints/step-00000200`).
        #[arg(long)]
        checkpoint: PathBuf,
        /// Text to continue.
        #[arg(long)]
        prompt: String,
        /// Fixed loop count `r`.
        #[arg(long, default_value_t = 4)]
        loops: usize,
        /// New tokens to generate.
        #[arg(long, default_value_t = 64)]
        max_new: usize,
        /// Path to tokenizer.json.
        #[arg(long, default_value = "tokenizer/tokenizer.json")]
        tokenizer: PathBuf,
        /// `cpu`, `cuda`, or `wgpu`.
        #[arg(long, default_value = "cpu")]
        device: String,
    },
    /// Supervised-finetune on English chat (M5).
    Sft,
    /// Chat with a finetuned checkpoint (M5).
    Chat,
    /// Print the parameter breakdown for a config.
    Params {
        /// Path to a LoopMind TOML config.
        #[arg(long, default_value = "configs/default.toml")]
        config: PathBuf,
    },
}

fn main() -> ExitCode {
    match dispatch(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn dispatch(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Command::TokenizeTrain { config } => {
            let config_path = find_file(config)?;
            let train_config = TrainConfig::from_path(&config_path)?;
            let tokenizer = LoopTokenizer::train(&train_config)?;
            println!(
                "trained {} tokens -> {}",
                tokenizer.vocab_size(),
                train_config.output.display()
            );
        }
        Command::Tokenize { text, tokenizer } => {
            let path = find_file(tokenizer)?;
            let tokenizer = LoopTokenizer::from_file(&path)?;
            let ids = tokenizer.encode(&text)?;
            println!(
                "{}",
                ids.iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            println!("{}", tokenizer.decode(&ids)?);
        }
        Command::TokenizeStats { heldout, tokenizer } => {
            let path = find_file(tokenizer)?;
            let tokenizer = LoopTokenizer::from_file(&path)?;
            let heldout = find_dir(heldout)?;
            let mut files: Vec<_> = std::fs::read_dir(&heldout)?.collect::<Result<Vec<_>, _>>()?;
            files.sort_by_key(|entry| entry.file_name());
            for entry in files {
                let file = entry.path();
                if file.extension().and_then(|ext| ext.to_str()) != Some("txt") {
                    continue;
                }
                let text = std::fs::read_to_string(&file)?;
                let Some(ratio) = tokenizer.bytes_per_token(&text)? else {
                    continue;
                };
                println!(
                    "{}\t{ratio:.4}\t{} bytes\t{} tokens",
                    file.file_stem()
                        .and_then(|stem| stem.to_str())
                        .unwrap_or("?"),
                    text.len(),
                    tokenizer.encode(&text)?.len()
                );
            }
        }
        Command::PreparePretrain { config } => {
            let started = std::time::Instant::now();
            let config = load_data_config(config)?;
            let tokenizer = LoopTokenizer::from_file(&config.tokenizer)?;
            let manifest = loopmind_data::prepare_pretrain(&config, &tokenizer)?;
            let train: u64 = manifest
                .sources
                .iter()
                .map(|source| source.train_tokens)
                .sum();
            let val: u64 = manifest
                .sources
                .iter()
                .map(|source| source.val_tokens)
                .sum();
            println!(
                "pretrain packed {train} train tokens and {val} val tokens across {} sources in {:.1}s",
                manifest.sources.len(),
                started.elapsed().as_secs_f64()
            );
            println!(
                "manifest: {}",
                config.output_dir.join("pretrain/manifest.json").display()
            );
            for source in &manifest.sources {
                println!(
                    "  {} budget {} train {} ({} seq) val {} ({} seq) docs {}",
                    source.name,
                    source.budget_tokens,
                    source.train_tokens,
                    source.train_sequences,
                    source.val_tokens,
                    source.val_sequences,
                    source.docs_seen
                );
            }
        }
        Command::PrepareSft { config } => {
            let started = std::time::Instant::now();
            let config = load_data_config(config)?;
            let tokenizer = LoopTokenizer::from_file(&config.tokenizer)?;
            let manifest = loopmind_data::prepare_sft(&config, &tokenizer)?;
            println!(
                "sft kept {} train rows and {} eval rows, dropped {} in {:.1}s",
                manifest.kept_train,
                manifest.kept_eval,
                manifest.dropped.non_english
                    + manifest.dropped.cc_by_sa
                    + manifest.dropped.too_long
                    + manifest.dropped.empty
                    + manifest.dropped.no_assistant
                    + manifest.dropped.bad_role,
                started.elapsed().as_secs_f64()
            );
            println!(
                "  non_english {}  cc_by_sa {}  too_long {}  empty {}  no_assistant {}  bad_role {}",
                manifest.dropped.non_english,
                manifest.dropped.cc_by_sa,
                manifest.dropped.too_long,
                manifest.dropped.empty,
                manifest.dropped.no_assistant,
                manifest.dropped.bad_role
            );
            println!(
                "manifest: {}",
                config.output_dir.join("sft/manifest.json").display()
            );
        }
        Command::Pretrain {
            config,
            resume,
            max_steps,
            device,
        } => {
            let config = load_pretrain_config(config)?;
            let summary = loopmind_train::pretrain(
                &config,
                &loopmind_train::RunOverrides {
                    resume,
                    max_steps,
                    device,
                },
            )?;
            println!(
                "finished {} steps, {} tokens, loss {:.4}, {} {}",
                summary.steps,
                summary.tokens_seen,
                summary.final_loss,
                summary.device,
                summary.precision
            );
        }
        Command::Sample {
            checkpoint,
            prompt,
            loops,
            max_new,
            tokenizer,
            device,
        } => {
            let checkpoint = find_dir(checkpoint)?;
            let tokenizer_path = find_file(tokenizer)?;
            let tokenizer = LoopTokenizer::from_file(&tokenizer_path)?;
            let eos = tokenizer
                .token_id(loopmind_tokenizer::EOS)
                .ok_or_else(|| ConfigError::Invalid("tokenizer is missing <|eos|>".into()))?;
            let ids = tokenizer.encode(&prompt)?;
            let generated =
                loopmind_train::generate(&checkpoint, &ids, loops, max_new, eos, &device)?;
            println!("{}", tokenizer.decode(&generated)?);
        }
        Command::Sft => {
            let policy = loopmind_data::SFT_POLICY;
            println!("{}", loopmind_train::sft_status());
            println!(
                "sft: {} chat from {}, drop {}, max {} tokens, hold out {}",
                policy.keep_language,
                policy.dataset,
                policy.drop_languages.join(", "),
                policy.max_tokens,
                policy.eval_holdout
            );
        }
        Command::Chat => {
            println!("chat is not implemented yet (planned for M5)");
        }
        Command::Params { config } => {
            let path = find_file(config)?;
            let loaded = LoopMindConfig::from_path(&path)?;
            println!("config: {}", path.display());
            print!("{}", loaded.report()?);
        }
    }
    Ok(())
}

fn load_pretrain_config(
    path: PathBuf,
) -> Result<loopmind_train::PretrainConfig, Box<dyn std::error::Error>> {
    let path = find_file(path)?;
    let mut config = loopmind_train::PretrainConfig::from_path(&path)?;
    let root = path.parent().and_then(|dir| dir.parent()).ok_or_else(|| {
        ConfigError::Invalid("pretrain config is not inside a configs directory".into())
    })?;
    for slot in [&mut config.model, &mut config.data_dir, &mut config.run_dir] {
        if slot.is_relative() {
            *slot = root.join(&*slot);
        }
    }
    Ok(config)
}

fn load_data_config(
    path: PathBuf,
) -> Result<loopmind_data::DataConfig, Box<dyn std::error::Error>> {
    let path = find_file(path)?;
    let mut config = loopmind_data::DataConfig::from_path(&path)?;
    let root = path.parent().and_then(|dir| dir.parent()).ok_or_else(|| {
        ConfigError::Invalid("data config is not inside a configs directory".into())
    })?;
    if config.tokenizer.is_relative() {
        config.tokenizer = root.join(&config.tokenizer);
    }
    if config.output_dir.is_relative() {
        config.output_dir = root.join(&config.output_dir);
    }
    Ok(config)
}

fn find_file(path: PathBuf) -> Result<PathBuf, ConfigError> {
    if path.is_file() {
        return Ok(path);
    }
    if let Some(found) = find_up(&path) {
        if found.is_file() {
            return Ok(found);
        }
    }
    Err(ConfigError::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("file not found at {}", path.display()),
    )))
}

fn find_dir(path: PathBuf) -> Result<PathBuf, ConfigError> {
    if path.is_dir() {
        return Ok(path);
    }
    if let Some(found) = find_up(&path) {
        if found.is_dir() {
            return Ok(found);
        }
    }
    Err(ConfigError::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("directory not found at {}", path.display()),
    )))
}

fn find_up(relative: &Path) -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let candidate = dir.join(relative);
        if candidate.exists() {
            return Some(candidate);
        }
        if !dir.pop() {
            return None;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use loopmind_model::LoopMindConfig;

    #[test]
    fn tokenizer_vocab_matches_the_model() {
        assert_eq!(
            loopmind_tokenizer::TARGET_VOCAB,
            LoopMindConfig::default().vocab_size
        );
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tokenizer/tokenizer.json");
        let tokenizer = loopmind_tokenizer::LoopTokenizer::from_file(&path).unwrap();
        assert_eq!(tokenizer.vocab_size(), LoopMindConfig::default().vocab_size);
    }

    #[test]
    fn sft_length_cap_matches_the_model_context() {
        assert_eq!(
            loopmind_data::SFT_POLICY.max_tokens,
            LoopMindConfig::default().context_length
        );
    }
}
