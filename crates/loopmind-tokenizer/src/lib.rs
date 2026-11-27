//! Byte-level BPE for LoopMind.
//!
//! The vocabulary is exactly [`TARGET_VOCAB`] tokens, including the chat
//! special tokens. Encode then decode is a byte-level round trip: there is
//! no whitespace-stripping normalizer.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::{AddedToken, Tokenizer, TokenizerBuilder};

/// Vocabulary size shared with `LoopMindConfig::vocab_size`.
pub const TARGET_VOCAB: usize = 6_400;

/// Special tokens. Each one is a single vocabulary id.
pub const SPECIAL_TOKENS: &[&str] = &[
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
];

pub const BOS: &str = "<|bos|>";
pub const EOS: &str = "<|eos|>";
pub const PAD: &str = "<|pad|>";
pub const IM_START: &str = "<|im_start|>";
pub const IM_END: &str = "<|im_end|>";

/// Who spoke a chat turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    /// Special token that marks this role.
    pub fn token(self) -> &'static str {
        match self {
            Role::System => "<|system|>",
            Role::User => "<|user|>",
            Role::Assistant => "<|assistant|>",
            Role::Tool => "<|tool|>",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        match name {
            "system" => Some(Role::System),
            "user" => Some(Role::User),
            "assistant" => Some(Role::Assistant),
            "tool" => Some(Role::Tool),
            _ => None,
        }
    }
}

/// One chat turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTurn {
    pub role: Role,
    pub content: String,
}

impl ChatTurn {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

/// Token ids and the SFT loss mask. `loss_mask[i]` is true when id `i` is trained.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatExample {
    pub ids: Vec<u32>,
    pub loss_mask: Vec<bool>,
}

/// Failure while loading, training, or encoding.
#[derive(Debug)]
pub struct TokenizerError(String);

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TokenizerError {}

fn err(message: impl Into<String>) -> TokenizerError {
    TokenizerError(message.into())
}

/// Paths for `tokenize-train`.
#[derive(Debug, Clone, Deserialize)]
pub struct TrainConfig {
    pub vocab_size: usize,
    pub corpus_dir: PathBuf,
    pub output: PathBuf,
}

impl TrainConfig {
    pub fn from_path(path: &Path) -> Result<Self, TokenizerError> {
        let text = fs::read_to_string(path).map_err(|error| err(error.to_string()))?;
        toml::from_str(&text).map_err(|error| err(error.to_string()))
    }
}

/// The trained LoopMind tokenizer.
#[derive(Clone)]
pub struct LoopTokenizer {
    inner: Tokenizer,
}

impl LoopTokenizer {
    /// Load `tokenizer.json`.
    pub fn from_file(path: &Path) -> Result<Self, TokenizerError> {
        let inner = Tokenizer::from_file(path).map_err(|error| err(error.to_string()))?;
        Ok(Self { inner })
    }

    /// Train a byte-level BPE on every `.txt` file under `corpus_dir`.
    pub fn train(config: &TrainConfig) -> Result<Self, TokenizerError> {
        if config.vocab_size != TARGET_VOCAB {
            return Err(err(format!(
                "vocab_size must be {TARGET_VOCAB}, got {}",
                config.vocab_size
            )));
        }
        let files = corpus_files(&config.corpus_dir)?;
        if files.is_empty() {
            return Err(err(format!(
                "no .txt files in {} (run python/fetch_tokenizer_corpus.py)",
                config.corpus_dir.display()
            )));
        }
        let mut trainer = BpeTrainerBuilder::new()
            .vocab_size(config.vocab_size)
            .min_frequency(2)
            .show_progress(true)
            .special_tokens(special_added_tokens())
            .initial_alphabet(ByteLevel::alphabet().into_iter().collect::<HashSet<_>>())
            .build();
        let mut tokenizer = TokenizerBuilder::new()
            .with_model(BPE::default())
            .with_normalizer(None::<NormalizerWrapper>)
            .with_pre_tokenizer(Some(ByteLevel::new(false, false, true)))
            .with_post_processor(None::<PostProcessorWrapper>)
            .with_decoder(Some(ByteLevel::new(false, false, true)))
            .build()
            .map_err(|error| err(error.to_string()))?;
        tokenizer
            .train_from_files(&mut trainer, files)
            .map_err(|error| err(error.to_string()))?;
        if let Some(parent) = config.output.parent() {
            fs::create_dir_all(parent).map_err(|error| err(error.to_string()))?;
        }
        tokenizer
            .save(&config.output, false)
            .map_err(|error| err(error.to_string()))?;
        let trained = Self::from_file(&config.output)?;
        if trained.vocab_size() != TARGET_VOCAB {
            return Err(err(format!(
                "trained vocab is {}, expected {TARGET_VOCAB}",
                trained.vocab_size()
            )));
        }
        Ok(trained)
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Encode text. Special tokens that appear in the string stay whole.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|error| err(error.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode ids, including special tokens.
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, false)
            .map_err(|error| err(error.to_string()))
    }

    /// UTF-8 bytes divided by the number of token ids. Empty text is `None`.
    pub fn bytes_per_token(&self, text: &str) -> Result<Option<f64>, TokenizerError> {
        if text.is_empty() {
            return Ok(None);
        }
        let ids = self.encode(text)?;
        if ids.is_empty() {
            return Ok(None);
        }
        Ok(Some(text.len() as f64 / ids.len() as f64))
    }

    /// Render a chat and the assistant-token loss mask.
    ///
    /// Assistant message text and the `<|im_end|>` that closes it are supervised.
    /// When the conversation ends on an assistant turn and no generation prompt
    /// is requested, the final `<|eos|>` is supervised too.
    pub fn render_chat(
        &self,
        turns: &[ChatTurn],
        add_generation_prompt: bool,
    ) -> Result<ChatExample, TokenizerError> {
        let mut ids = Vec::new();
        let mut loss_mask = Vec::new();
        self.push_special(&mut ids, &mut loss_mask, BOS, false)?;
        for turn in turns {
            let supervise = turn.role == Role::Assistant;
            self.push_special(&mut ids, &mut loss_mask, IM_START, false)?;
            self.push_special(&mut ids, &mut loss_mask, turn.role.token(), false)?;
            self.push_text(&mut ids, &mut loss_mask, "\n", false)?;
            self.push_text(&mut ids, &mut loss_mask, &turn.content, supervise)?;
            self.push_special(&mut ids, &mut loss_mask, IM_END, supervise)?;
            self.push_text(&mut ids, &mut loss_mask, "\n", false)?;
        }
        if add_generation_prompt {
            self.push_special(&mut ids, &mut loss_mask, IM_START, false)?;
            self.push_special(&mut ids, &mut loss_mask, Role::Assistant.token(), false)?;
            self.push_text(&mut ids, &mut loss_mask, "\n", false)?;
        } else if turns
            .last()
            .is_some_and(|turn| turn.role == Role::Assistant)
        {
            self.push_special(&mut ids, &mut loss_mask, EOS, true)?;
        }
        Ok(ChatExample { ids, loss_mask })
    }

    /// The string `render_chat` tokenizes, so tests can check a decode round trip.
    pub fn chat_template(turns: &[ChatTurn], add_generation_prompt: bool) -> String {
        let mut text = String::from(BOS);
        for turn in turns {
            text.push_str(IM_START);
            text.push_str(turn.role.token());
            text.push('\n');
            text.push_str(&turn.content);
            text.push_str(IM_END);
            text.push('\n');
        }
        if add_generation_prompt {
            text.push_str(IM_START);
            text.push_str(Role::Assistant.token());
            text.push('\n');
        } else if turns
            .last()
            .is_some_and(|turn| turn.role == Role::Assistant)
        {
            text.push_str(EOS);
        }
        text
    }

    fn push_special(
        &self,
        ids: &mut Vec<u32>,
        mask: &mut Vec<bool>,
        token: &str,
        supervise: bool,
    ) -> Result<(), TokenizerError> {
        let id = self
            .token_id(token)
            .ok_or_else(|| err(format!("missing special token {token}")))?;
        ids.push(id);
        mask.push(supervise);
        Ok(())
    }

    fn push_text(
        &self,
        ids: &mut Vec<u32>,
        mask: &mut Vec<bool>,
        text: &str,
        supervise: bool,
    ) -> Result<(), TokenizerError> {
        if text.is_empty() {
            return Ok(());
        }
        let encoded = self.encode(text)?;
        mask.extend(std::iter::repeat_n(supervise, encoded.len()));
        ids.extend(encoded);
        Ok(())
    }
}

fn special_added_tokens() -> Vec<AddedToken> {
    SPECIAL_TOKENS
        .iter()
        .map(|token| {
            AddedToken::from(*token, true)
                .normalized(false)
                .single_word(false)
                .lstrip(false)
                .rstrip(false)
        })
        .collect()
}

fn corpus_files(dir: &Path) -> Result<Vec<String>, TokenizerError> {
    let mut files = Vec::new();
    let entries = fs::read_dir(dir).map_err(|error| err(format!("{}: {error}", dir.display())))?;
    for entry in entries {
        let entry = entry.map_err(|error| err(error.to_string()))?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("txt") {
            files.push(path.to_string_lossy().into_owned());
        }
    }
    files.sort();
    Ok(files)
}

/// Parse `system`, `user`, `assistant`, or `tool`.
pub fn role_from_name(name: &str) -> Option<Role> {
    Role::from_name(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer() -> LoopTokenizer {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tokenizer/tokenizer.json");
        LoopTokenizer::from_file(&path).expect("committed tokenizer")
    }

    #[test]
    fn vocab_is_exactly_6400_and_specials_are_single_ids() {
        let tokenizer = tokenizer();
        assert_eq!(tokenizer.vocab_size(), TARGET_VOCAB);
        for token in SPECIAL_TOKENS {
            let id = tokenizer.token_id(token).expect(token);
            assert_eq!(tokenizer.encode(token).unwrap(), vec![id], "{token}");
        }
    }

    #[test]
    fn roundtrip_english_code_json_and_unicode() {
        let tokenizer = tokenizer();
        let samples = [
            "The independent Jane wanted freedom of thought.",
            "fn main() {\n    println!(\"hello\");\n}\n",
            "{\"users\": [{\"id\": 1, \"name\": \"Ada\"}], \"ok\": true}\n",
            "café naïve — Σ x² and 😀 你好",
            "SELECT name FROM users WHERE id = 7;\n",
        ];
        for sample in samples {
            let ids = tokenizer.encode(sample).unwrap();
            assert_eq!(tokenizer.decode(&ids).unwrap(), sample);
        }
    }

    #[test]
    fn chat_template_masks_only_assistant_tokens() {
        let tokenizer = tokenizer();
        let turns = vec![
            ChatTurn::new(Role::System, "You are a careful tutor."),
            ChatTurn::new(Role::User, "What is two plus two?"),
            ChatTurn::new(Role::Assistant, "The answer is four."),
            ChatTurn::new(Role::User, "And three plus three?"),
            ChatTurn::new(Role::Assistant, "The answer is six."),
        ];
        let rendered = tokenizer.render_chat(&turns, false).unwrap();
        assert_eq!(rendered.ids.len(), rendered.loss_mask.len());
        let decoded = tokenizer.decode(&rendered.ids).unwrap();
        assert_eq!(decoded, LoopTokenizer::chat_template(&turns, false));

        let answer_one = tokenizer.encode("The answer is four.").unwrap();
        let answer_two = tokenizer.encode("The answer is six.").unwrap();
        let question = tokenizer.encode("What is two plus two?").unwrap();
        assert_span_mask(&rendered, &answer_one, true);
        assert_span_mask(&rendered, &answer_two, true);
        assert_span_mask(&rendered, &question, false);

        let im_end = tokenizer.token_id(IM_END).unwrap();
        let end_positions: Vec<usize> = rendered
            .ids
            .iter()
            .enumerate()
            .filter_map(|(index, id)| (*id == im_end).then_some(index))
            .collect();
        assert_eq!(end_positions.len(), turns.len());
        assert!(!rendered.loss_mask[end_positions[0]]);
        assert!(!rendered.loss_mask[end_positions[1]]);
        assert!(rendered.loss_mask[end_positions[2]]);
        assert!(!rendered.loss_mask[end_positions[3]]);
        assert!(rendered.loss_mask[end_positions[4]]);
        assert_eq!(
            *rendered.ids.last().unwrap(),
            tokenizer.token_id(EOS).unwrap()
        );
        assert!(*rendered.loss_mask.last().unwrap());

        let prompt = tokenizer.render_chat(&turns[..2], true).unwrap();
        assert!(prompt.loss_mask.iter().all(|bit| !bit));
        assert_eq!(
            tokenizer.decode(&prompt.ids).unwrap(),
            LoopTokenizer::chat_template(&turns[..2], true)
        );
    }

    fn assert_span_mask(rendered: &ChatExample, span: &[u32], expected: bool) {
        let start = rendered
            .ids
            .windows(span.len())
            .position(|window| window == span)
            .expect("span missing");
        assert!(rendered.loss_mask[start..start + span.len()]
            .iter()
            .all(|bit| *bit == expected));
    }
}
