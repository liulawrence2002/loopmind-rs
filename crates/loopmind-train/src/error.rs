use std::fmt;

#[derive(Debug)]
pub struct TrainError {
    message: String,
}

impl TrainError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for TrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for TrainError {}

impl From<String> for TrainError {
    fn from(message: String) -> Self {
        Self { message }
    }
}

impl From<loopmind_data::DataError> for TrainError {
    fn from(error: loopmind_data::DataError) -> Self {
        Self::new(error.to_string())
    }
}
