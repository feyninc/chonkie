use crate::Result;

/// The Hub repo of the default tokenizer.
pub const DEFAULT_TOKENIZER: &str = "tokiers/gpt2";

/// What a chunker counts.
///
/// The default is gpt2, downloaded from the Hugging Face Hub the first time it
/// counts and cached after that.
pub struct Tokenizer(Kind);

enum Kind {
    Characters,
    Tokie(Box<tokie::Tokenizer>),
    Pretrained(String),
}

impl Tokenizer {
    /// One token per Unicode scalar value. Needs no model file.
    pub fn characters() -> Self {
        Self(Kind::Characters)
    }

    /// Load a tokie tokenizer from a `.tokie` or `tokenizer.json` file.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Ok(tokie::Tokenizer::open(path)?.into())
    }

    /// Parse a tokie tokenizer from the text of a `tokenizer.json`.
    pub fn json(text: &str) -> Result<Self> {
        Ok(tokie::Tokenizer::json(text)?.into())
    }

    /// A tokenizer from a Hub repo such as `tokiers/cl100k`. Nothing is
    /// downloaded until the first count.
    pub fn pretrained(repo: impl Into<String>) -> Self {
        Self(Kind::Pretrained(repo.into()))
    }

    /// Tokens in `text`. Special tokens from a template are not added.
    pub fn count(&mut self, text: &str) -> Result<usize> {
        Ok(match self.load()? {
            Some(tok) => tok.count_tokens(text)?,
            None => text.chars().count(),
        })
    }

    /// Tokens in each of `texts`.
    pub fn count_batch(&mut self, texts: &[&str]) -> Result<Vec<usize>> {
        Ok(match self.load()? {
            Some(tok) => tok.count_tokens_batch(texts)?,
            None => texts.iter().map(|t| t.chars().count()).collect(),
        })
    }

    fn load(&mut self) -> Result<Option<&mut tokie::Tokenizer>> {
        if let Kind::Pretrained(repo) = &self.0 {
            self.0 = Kind::Tokie(Box::new(tokie::Tokenizer::pretrained(repo)?));
        }
        Ok(match &mut self.0 {
            Kind::Tokie(tok) => Some(tok),
            _ => None,
        })
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::pretrained(DEFAULT_TOKENIZER)
    }
}

impl From<tokie::Tokenizer> for Tokenizer {
    fn from(tok: tokie::Tokenizer) -> Self {
        Self(Kind::Tokie(Box::new(tok)))
    }
}
