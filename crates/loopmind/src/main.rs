//! LoopMind command line.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use loopmind_model::{ConfigError, GenerateOptions, LoopMindConfig};
use loopmind_tokenizer::{ChatTurn, LoopTokenizer, Role, TrainConfig};

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
    /// Supervised-finetune a pretrain checkpoint on English chat (M5).
    Sft {
        /// SFT config. Paths inside it are relative to the repo root.
        #[arg(long, default_value = "configs/sft.toml")]
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
        /// Pretrain checkpoint, or a run directory with latest.json. Overrides `init`.
        #[arg(long)]
        init: Option<PathBuf>,
    },
    /// Chat with an SFT checkpoint (M5). Reads turns from stdin unless --prompt is given.
    Chat {
        /// Checkpoint directory, or a run directory whose latest.json names one.
        #[arg(long, default_value = "runs/sft")]
        checkpoint: PathBuf,
        /// One user message. Prints the answer and exits.
        #[arg(long)]
        prompt: Option<String>,
        /// Optional system message at the start of the conversation.
        #[arg(long)]
        system: Option<String>,
        /// Fixed loop count `r`.
        #[arg(long, default_value_t = 4)]
        loops: usize,
        /// New tokens per answer.
        #[arg(long, default_value_t = 256)]
        max_new: usize,
        /// Softmax temperature. 0 is greedy.
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,
        /// Sample among this many most likely tokens. 0 keeps all.
        #[arg(long, default_value_t = 40)]
        top_k: usize,
        /// Penalty on tokens already in the context. 1.0 turns it off.
        #[arg(long, default_value_t = 1.1)]
        repetition_penalty: f32,
        /// Sampling seed.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Path to tokenizer.json.
        #[arg(long, default_value = "tokenizer/tokenizer.json")]
        tokenizer: PathBuf,
        /// `cpu`, `cuda`, or `wgpu`.
        #[arg(long, default_value = "cpu")]
        device: String,
    },
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
        Command::Sft {
            config,
            resume,
            max_steps,
            device,
            init,
        } => {
            let mut config = load_sft_config(config)?;
            if let Some(init) = init {
                config.init = find_dir(init)?;
            }
            let summary = loopmind_train::sft(
                &config,
                &loopmind_train::RunOverrides {
                    resume,
                    max_steps,
                    device,
                },
            )?;
            println!(
                "finished {} sft steps, {} tokens, loss {:.4}, {} {}",
                summary.steps,
                summary.tokens_seen,
                summary.final_loss,
                summary.device,
                summary.precision
            );
        }
        Command::Chat {
            checkpoint,
            prompt,
            system,
            loops,
            max_new,
            temperature,
            top_k,
            repetition_penalty,
            seed,
            tokenizer,
            device,
        } => {
            let checkpoint = find_dir(checkpoint)?;
            let tokenizer = LoopTokenizer::from_file(&find_file(tokenizer)?)?;
            let special = |name: &str| {
                tokenizer
                    .token_id(name)
                    .ok_or_else(|| ConfigError::Invalid(format!("tokenizer is missing {name}")))
            };
            let stop = [
                special(loopmind_tokenizer::IM_END)?,
                special(loopmind_tokenizer::EOS)?,
            ];
            let engine = loopmind_train::load_inference(&checkpoint, &device)?;
            let context = engine.config().context_length;
            let mut chat = Chat {
                engine,
                tokenizer: &tokenizer,
                stop,
                context,
                history: system
                    .map(|text| vec![ChatTurn::new(Role::System, text)])
                    .unwrap_or_default(),
                options: GenerateOptions {
                    loops,
                    max_new,
                    temperature,
                    top_k,
                    repetition_penalty,
                    seed,
                },
            };
            match prompt {
                Some(prompt) => println!("{}", chat.respond(prompt)?),
                None => chat.repl()?,
            }
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

fn load_sft_config(path: PathBuf) -> Result<loopmind_train::SftConfig, Box<dyn std::error::Error>> {
    let path = find_file(path)?;
    let mut config = loopmind_train::SftConfig::from_path(&path)?;
    let root = path.parent().and_then(|dir| dir.parent()).ok_or_else(|| {
        ConfigError::Invalid("sft config is not inside a configs directory".into())
    })?;
    for slot in [
        &mut config.init,
        &mut config.data_dir,
        &mut config.run_dir,
        &mut config.tokenizer,
    ] {
        if slot.is_relative() {
            *slot = root.join(&*slot);
        }
    }
    Ok(config)
}

/// A conversation with one checkpoint. Turns are rendered with the M2 chat template.
struct Chat<'a> {
    engine: loopmind_model::PretrainEngine,
    tokenizer: &'a LoopTokenizer,
    stop: [u32; 2],
    context: usize,
    history: Vec<ChatTurn>,
    options: GenerateOptions,
}

impl Chat<'_> {
    fn respond(&mut self, user: String) -> Result<String, Box<dyn std::error::Error>> {
        self.history.push(ChatTurn::new(Role::User, user));
        let mut prompt = self.tokenizer.render_chat(&self.history, true)?;
        // Drop the oldest non-system turns until the prompt leaves room for the answer.
        while prompt.ids.len() + self.options.max_new > self.context {
            let first = usize::from(
                self.history
                    .first()
                    .is_some_and(|turn| turn.role == Role::System),
            );
            if self.history.len() <= first + 1 {
                break;
            }
            self.history.remove(first);
            prompt = self.tokenizer.render_chat(&self.history, true)?;
        }
        let ids = self.engine.generate(prompt.ids, &self.options, &self.stop);
        self.options.seed = self.options.seed.wrapping_add(1);
        let text = self.tokenizer.decode(&ids)?.trim().to_string();
        self.history
            .push(ChatTurn::new(Role::Assistant, text.clone()));
        Ok(text)
    }

    fn repl(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::{BufRead, Write};
        eprintln!(
            "loopmind chat, r = {}. /reset clears the history, /exit or Ctrl-D quits.",
            self.options.loops
        );
        let stdin = std::io::stdin();
        let keep = usize::from(
            self.history
                .first()
                .is_some_and(|turn| turn.role == Role::System),
        );
        loop {
            eprint!("> ");
            std::io::stderr().flush()?;
            let mut line = String::new();
            if stdin.lock().read_line(&mut line)? == 0 {
                break;
            }
            let line = line.trim();
            match line {
                "" => continue,
                "/exit" | "/quit" => break,
                "/reset" => {
                    self.history.truncate(keep);
                    eprintln!("(history cleared)");
                }
                _ => println!("{}\n", self.respond(line.to_string())?),
            }
        }
        Ok(())
    }
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
