//! LoopMind command line.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use loopmind_model::{ConfigError, LoopMindConfig};

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
    /// Train the 6400-vocab BPE (M2).
    TokenizeTrain,
    /// Pretrain on the 50M-token mix (M4).
    Pretrain,
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

fn dispatch(cli: Cli) -> Result<(), ConfigError> {
    match cli.command {
        Command::TokenizeTrain => {
            println!("{}", loopmind_tokenizer::train_status());
        }
        Command::Pretrain => {
            println!("{}", loopmind_train::pretrain_status());
            println!("pretrain mix: {} tokens", loopmind_data::pretrain_tokens());
            for slice in loopmind_data::PRETRAIN_SLICES {
                println!("  {}  {}  {}", slice.tokens, slice.name, slice.dataset);
            }
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
            let (path, loaded) = load_config(config)?;
            println!("config: {}", path.display());
            print!("{}", loaded.report()?);
        }
    }
    Ok(())
}

fn load_config(path: PathBuf) -> Result<(PathBuf, LoopMindConfig), ConfigError> {
    if path.is_file() {
        let loaded = LoopMindConfig::from_path(&path)?;
        return Ok((path, loaded));
    }
    if path == Path::new("configs/default.toml") {
        if let Some(found) = find_up("configs/default.toml") {
            let loaded = LoopMindConfig::from_path(&found)?;
            return Ok((found, loaded));
        }
    }
    Err(ConfigError::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("config not found at {}", path.display()),
    )))
}

fn find_up(relative: &str) -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let candidate = dir.join(relative);
        if candidate.is_file() {
            return Some(candidate);
        }
        if !dir.pop() {
            return None;
        }
    }
}

#[cfg(test)]
mod tests {
    use loopmind_model::LoopMindConfig;

    #[test]
    fn tokenizer_vocab_matches_the_model() {
        assert_eq!(
            loopmind_tokenizer::TARGET_VOCAB,
            LoopMindConfig::default().vocab_size
        );
    }

    #[test]
    fn sft_length_cap_matches_the_model_context() {
        assert_eq!(
            loopmind_data::SFT_POLICY.max_tokens,
            LoopMindConfig::default().context_length
        );
    }
}
