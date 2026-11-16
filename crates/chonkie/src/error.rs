use std::fmt;

/// Why a chunker could not be built or could not chunk.
#[derive(Debug)]
pub enum Error {
    /// A chunker setting is out of range.
    Config(String),
    /// The tokenizer failed.
    Tokenizer(tokie::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Config(message) => write!(f, "invalid config: {message}"),
            Error::Tokenizer(err) => write!(f, "tokenizer: {err}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<tokie::Error> for Error {
    fn from(err: tokie::Error) -> Self {
        Error::Tokenizer(err)
    }
}
