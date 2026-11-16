//! Chonkie: small, fast, correct text chunking.
//!
//! Every chunk borrows its text from the input. `start` and `end` are byte
//! offsets, so `&text[chunk.start..chunk.end] == chunk.text`.

mod error;
mod recursive;
mod tokenizer;
mod types;

pub use error::Error;
pub use recursive::{Include, Level, RecursiveChunker};
pub use tokenizer::{DEFAULT_TOKENIZER, Tokenizer};
pub use types::Chunk;

pub type Result<T> = std::result::Result<T, Error>;
