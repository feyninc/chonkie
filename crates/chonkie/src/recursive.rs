use chunk::{IncludeDelim, PatternSplitter};

use crate::{Chunk, Error, Result, Tokenizer};

/// Typical bytes per pretoken in English text, until `calibrate` measures it.
const BYTES_PER_PRETOKEN: f64 = 4.0;

/// Which side of a split a delimiter stays on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Include {
    /// End the piece before the delimiter with it: `"Hi." | " Yo."`.
    #[default]
    Prev,
    /// Start the piece after the delimiter with it: `"Hi" | ". Yo"`.
    Next,
}

/// One level of the recursion: the delimiters to split on there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Level {
    pub delimiters: Vec<String>,
    pub include: Include,
}

impl Level {
    pub fn new<S: Into<String>>(delimiters: impl IntoIterator<Item = S>) -> Self {
        Self {
            delimiters: delimiters.into_iter().map(Into::into).collect(),
            include: Include::Prev,
        }
    }

    pub fn include(mut self, include: Include) -> Self {
        self.include = include;
        self
    }

    /// Paragraphs, sentences, pauses, then words.
    pub fn defaults() -> Vec<Level> {
        vec![
            Level::new(["\n\n", "\r\n", "\n", "\r"]),
            Level::new([". ", "! ", "? "]),
            Level::new([
                "{", "}", "\"", "[", "]", "<", ">", "(", ")", ":", ";", ",", "—", "|", "~", "-",
                "...", "`", "'",
            ]),
            Level::new([" "]),
        ]
    }
}

/// Splits on paragraphs, then sentences, then smaller delimiters, until every
/// piece fits `chunk_size`, then merges neighbours back up to `chunk_size`.
/// Text that no level can split is cut at character boundaries.
pub struct RecursiveChunker {
    tokenizer: Tokenizer,
    chunk_size: usize,
    levels: Vec<Level>,
    delims: Vec<PatternSplitter>,
    bytes_per_pretoken: f64,
}

impl Default for RecursiveChunker {
    fn default() -> Self {
        Self::new(Tokenizer::default())
    }
}

impl RecursiveChunker {
    pub fn new(tokenizer: impl Into<Tokenizer>) -> Self {
        Self {
            tokenizer: tokenizer.into(),
            chunk_size: 2048,
            levels: Vec::new(),
            delims: Vec::new(),
            bytes_per_pretoken: BYTES_PER_PRETOKEN,
        }
        .levels(Level::defaults())
    }

    pub fn chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Measure bytes per pretoken on `sample`, text like what will be chunked.
    /// Pieces shorter than that many bytes per allowed token skip the pretoken
    /// bound. This only changes speed: the chunks are the same either way.
    pub fn calibrate(mut self, sample: &str) -> Result<Self> {
        let pretokens = self.tokenizer.lower_bound(sample)?;
        if pretokens > 0 {
            self.bytes_per_pretoken = sample.len() as f64 / pretokens as f64;
        }
        Ok(self)
    }

    pub fn levels(mut self, levels: Vec<Level>) -> Self {
        self.delims = levels
            .iter()
            .map(|l| {
                let patterns: Vec<&[u8]> = l.delimiters.iter().map(String::as_bytes).collect();
                PatternSplitter::new(&patterns)
            })
            .collect();
        self.levels = levels;
        self
    }

    pub fn tokenizer(&mut self) -> &mut Tokenizer {
        &mut self.tokenizer
    }

    pub fn chunk<'a>(&mut self, text: &'a str) -> Result<Vec<Chunk<'a>>> {
        if self.chunk_size == 0 {
            return Err(Error::Config("chunk_size must be at least 1".into()));
        }
        if self
            .levels
            .iter()
            .any(|l| l.delimiters.iter().any(String::is_empty))
        {
            return Err(Error::Config("delimiters cannot be empty".into()));
        }
        let mut chunks = Vec::new();
        if !text.is_empty() {
            let count = match self.over(text)? {
                true => None,
                false => Some(self.tokenizer.count(text)?),
            };
            self.split(text, 0, text.len(), count, 0, &mut chunks)?;
        }
        Ok(chunks)
    }

    /// Whether `text` is sure to exceed `chunk_size`, judged without encoding.
    /// Pieces under `bytes_per_pretoken * chunk_size` bytes rarely pass the
    /// bound, so they skip it and are counted directly.
    fn over(&mut self, text: &str) -> Result<bool> {
        Ok(
            text.len() as f64 > self.bytes_per_pretoken * self.chunk_size as f64
                && self.tokenizer.lower_bound(text)? > self.chunk_size,
        )
    }

    fn split<'a>(
        &mut self,
        text: &'a str,
        start: usize,
        end: usize,
        count: Option<usize>,
        level: usize,
        out: &mut Vec<Chunk<'a>>,
    ) -> Result<()> {
        if let Some(count) = count.filter(|&c| c <= self.chunk_size) {
            out.push(Chunk::new(text, start, end, count));
            return Ok(());
        }
        let (Some(rule), Some(delims)) = (self.levels.get(level), self.delims.get(level)) else {
            return self.cut(text, start, end, out);
        };
        let pieces = pieces(&text[start..end], delims, rule.include, start);
        if pieces.len() < 2 {
            return self.split(text, start, end, count, level + 1, out);
        }
        let counts = self.counts(text, &pieces)?;
        let mut i = 0;
        while i < pieces.len() {
            if counts[i] > self.chunk_size {
                let count = Some(counts[i]).filter(|&c| c != usize::MAX);
                self.split(text, pieces[i].0, pieces[i].1, count, level + 1, out)?;
                i += 1;
                continue;
            }
            let mut j = i + 1;
            let mut sum = counts[i];
            while j < pieces.len() && sum.saturating_add(counts[j]) <= self.chunk_size {
                sum += counts[j];
                j += 1;
            }
            // A merged group can count differently from the sum of its parts.
            loop {
                let (s, e) = (pieces[i].0, pieces[j - 1].1);
                let real = if j - i == 1 {
                    counts[i]
                } else {
                    self.tokenizer.count(&text[s..e])?
                };
                if real <= self.chunk_size {
                    out.push(Chunk::new(text, s, e, real));
                    break;
                }
                j -= 1;
            }
            i = j;
        }
        Ok(())
    }

    /// Token counts of `pieces`, with `usize::MAX` for pieces the lower bound
    /// already puts over `chunk_size`.
    fn counts(&mut self, text: &str, pieces: &[(usize, usize)]) -> Result<Vec<usize>> {
        let mut counts = vec![usize::MAX; pieces.len()];
        let mut todo = Vec::with_capacity(pieces.len());
        for (i, &(s, e)) in pieces.iter().enumerate() {
            if !self.over(&text[s..e])? {
                todo.push(i);
            }
        }
        let slices: Vec<&str> = todo
            .iter()
            .map(|&i| &text[pieces[i].0..pieces[i].1])
            .collect();
        for (i, n) in todo.into_iter().zip(self.tokenizer.count_batch(&slices)?) {
            counts[i] = n;
        }
        Ok(counts)
    }

    /// Greedy cut at character boundaries: the longest prefix that fits, found
    /// by binary search over boundaries.
    fn cut<'a>(
        &mut self,
        text: &'a str,
        start: usize,
        end: usize,
        out: &mut Vec<Chunk<'a>>,
    ) -> Result<()> {
        let bounds: Vec<usize> = text[start..end]
            .char_indices()
            .map(|(i, _)| start + i)
            .skip(1)
            .chain([end])
            .collect();
        let mut from = start;
        let mut first = 0;
        while from < end {
            let (mut lo, mut hi) = (first, bounds.len() - 1);
            let mut best = None;
            while lo <= hi {
                let mid = (lo + hi) / 2;
                let n = self.tokenizer.count(&text[from..bounds[mid]])?;
                if n <= self.chunk_size {
                    best = Some((mid, n));
                    lo = mid + 1;
                } else if mid == 0 {
                    break;
                } else {
                    hi = mid - 1;
                }
            }
            // A single character over budget still becomes its own chunk.
            let (idx, n) = match best {
                Some(found) => found,
                None => (first, self.tokenizer.count(&text[from..bounds[first]])?),
            };
            out.push(Chunk::new(text, from, bounds[idx], n));
            from = bounds[idx];
            first = idx + 1;
        }
        Ok(())
    }
}

/// Byte ranges, offset by `base`, that cover `text` with no gaps: one per
/// delimiter match, with the delimiter on the side `include` says.
fn pieces(
    text: &str,
    delims: &PatternSplitter,
    include: Include,
    base: usize,
) -> Vec<(usize, usize)> {
    let include = match include {
        Include::Prev => IncludeDelim::Prev,
        Include::Next => IncludeDelim::Next,
    };
    delims
        .split(text.as_bytes(), include, 0)
        .into_iter()
        .filter(|&(s, e)| s < e)
        .map(|(s, e)| (base + s, base + e))
        .collect()
}
