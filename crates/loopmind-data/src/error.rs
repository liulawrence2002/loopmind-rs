use std::fmt;

/// Recoverable data-pipeline failure.
#[derive(Debug)]
pub struct DataError {
    message: String,
}

impl DataError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for DataError {}

impl From<std::io::Error> for DataError {
    fn from(error: std::io::Error) -> Self {
        Self::new(error.to_string())
    }
}

impl From<serde_json::Error> for DataError {
    fn from(error: serde_json::Error) -> Self {
        Self::new(error.to_string())
    }
}

impl From<loopmind_tokenizer::TokenizerError> for DataError {
    fn from(error: loopmind_tokenizer::TokenizerError) -> Self {
        Self::new(error.to_string())
    }
}
