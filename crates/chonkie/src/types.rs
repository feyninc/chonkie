/// One piece of the input text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chunk<'a> {
    /// The chunk's text, borrowed from the input.
    pub text: &'a str,
    /// Byte offset of the first byte in the input.
    pub start: usize,
    /// Byte offset one past the last byte in the input.
    pub end: usize,
    /// Tokens in `text`, as the chunker's tokenizer counts them.
    pub token_count: usize,
}

impl<'a> Chunk<'a> {
    pub(crate) fn new(source: &'a str, start: usize, end: usize, token_count: usize) -> Self {
        Self {
            text: &source[start..end],
            start,
            end,
            token_count,
        }
    }
}
