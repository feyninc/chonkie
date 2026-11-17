//! The five correctness rules every chunker must hold, and the texts and
//! tokenizers to check them on. Each rule is its own function so a chunker's
//! tests can name the rule that broke; each returns the first violation as a
//! message that says which chunk broke it and how.
#![allow(dead_code)]

use chonkie::{Chunk, Tokenizer};

pub type Verdict = Result<(), String>;

fn show(text: &str) -> String {
    const MAX: usize = 60;
    match text.char_indices().nth(MAX) {
        Some((at, _)) => format!("{:?}… ({} bytes)", &text[..at], text.len()),
        None => format!("{text:?}"),
    }
}

/// Rule 1: with overlap 0, the chunk texts concatenate to the source.
pub fn rebuilds_source(source: &str, chunks: &[Chunk]) -> Verdict {
    let mut at = 0;
    for (i, chunk) in chunks.iter().enumerate() {
        let rest = &source[at..];
        if chunk.text.is_empty() {
            return Err(format!("chunk {i} is empty"));
        }
        if !rest.starts_with(chunk.text) {
            return Err(format!(
                "chunk {i} is {} but the source continues with {}",
                show(chunk.text),
                show(rest)
            ));
        }
        at += chunk.text.len();
    }
    match at == source.len() {
        true => Ok(()),
        false => Err(format!(
            "the chunks stop at byte {at}, the source has {}",
            source.len()
        )),
    }
}

/// Rule 2: `source[start..end] == text` for every chunk, and each chunk starts
/// where the one before it ended, from 0 to the end of the source.
pub fn offsets_slice_source(source: &str, chunks: &[Chunk]) -> Verdict {
    let mut at = 0;
    for (i, chunk) in chunks.iter().enumerate() {
        if chunk.start != at {
            return Err(format!(
                "chunk {i} starts at {}, expected {at}",
                chunk.start
            ));
        }
        if chunk.end < chunk.start || chunk.end > source.len() {
            return Err(format!(
                "chunk {i} has bad range {}..{}",
                chunk.start, chunk.end
            ));
        }
        if source.get(chunk.start..chunk.end) != Some(chunk.text) {
            return Err(format!(
                "chunk {i}: source[{}..{}] is not its text {}",
                chunk.start,
                chunk.end,
                show(chunk.text)
            ));
        }
        at = chunk.end;
    }
    match at == source.len() {
        true => Ok(()),
        false => Err(format!(
            "the last chunk ends at {at}, the source at {}",
            source.len()
        )),
    }
}

/// Rule 3: `token_count` is what the tokenizer counts for the chunk's text,
/// and is at most `chunk_size` unless the chunk is a single character, which
/// cannot be split further.
pub fn token_counts_exact(
    chunks: &[Chunk],
    tokenizer: &mut Tokenizer,
    chunk_size: usize,
) -> Verdict {
    for (i, chunk) in chunks.iter().enumerate() {
        let count = tokenizer.count(chunk.text).map_err(|e| e.to_string())?;
        if chunk.token_count != count {
            return Err(format!(
                "chunk {i} {} says {} tokens, the tokenizer counts {count}",
                show(chunk.text),
                chunk.token_count
            ));
        }
        if count > chunk_size && chunk.text.chars().count() > 1 {
            return Err(format!(
                "chunk {i} {} has {count} tokens, over chunk_size {chunk_size}",
                show(chunk.text)
            ));
        }
    }
    Ok(())
}

/// Rule 4: chunking the same input again gives the same chunks.
pub fn same_chunks(first: &[Chunk], again: &[Chunk]) -> Verdict {
    if first.len() != again.len() {
        return Err(format!(
            "{} chunks the first time, {} the second",
            first.len(),
            again.len()
        ));
    }
    match first.iter().zip(again).position(|(a, b)| a != b) {
        None => Ok(()),
        Some(i) => Err(format!(
            "chunk {i} changed: {:?} then {:?}",
            first[i], again[i]
        )),
    }
}

/// Rule 5: no codepoint is split and nothing is dropped. Every chunk boundary
/// falls on a char boundary, and each character of the source, spaces and
/// delimiters included, appears in the chunks as often as in the source.
/// Checked on its own, without leaning on rules 1 and 2.
pub fn keeps_every_char(source: &str, chunks: &[Chunk]) -> Verdict {
    for (i, chunk) in chunks.iter().enumerate() {
        for at in [chunk.start, chunk.end] {
            if !source.is_char_boundary(at) {
                return Err(format!(
                    "chunk {i} has an edge at byte {at}, inside a codepoint"
                ));
            }
        }
    }
    let mut seen = std::collections::BTreeMap::<char, isize>::new();
    for c in source.chars() {
        *seen.entry(c).or_default() += 1;
    }
    for c in chunks.iter().flat_map(|c| c.text.chars()) {
        *seen.entry(c).or_default() -= 1;
    }
    match seen.into_iter().find(|&(_, n)| n != 0) {
        None => Ok(()),
        Some((c, n)) if n > 0 => Err(format!(
            "{c:?} is in the source {n} more time(s) than in the chunks"
        )),
        Some((c, n)) => Err(format!(
            "{c:?} is in the chunks {} more time(s) than in the source",
            -n
        )),
    }
}

/// Edge cases plus seeded pseudo-random text.
pub fn corpus() -> Vec<String> {
    let mut texts: Vec<String> = [
        "",
        " ",
        "\n",
        "   \n\n\t  ",
        "a",
        "Hello, world.",
        "Hi.  Two spaces.   Three spaces.\n\n\nThree newlines.",
        "no-delimiters-at-all-just-one-very-long-token-like-a-url-https://example.com/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "Ünïcödé çàñ bé tricky: naïve café, Zürich, ﬁ ligature.",
        "日本語の文章です。句読点で区切ります。英語とは違います。",
        "Emoji 🦛🦛🦛 and ZWJ 👩‍👩‍👧‍👦 families, flags 🇺🇸🇮🇳 and skin tones 👍🏽.",
        "e\u{301}\u{301}\u{301} combining marks a\u{308}\u{308}",
        "\r\nwindows\r\nline\r\nendings\r\n",
        "fn main() {\n    println!(\"hi\");\n}\n",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    texts.push("The quick brown fox jumps over the lazy dog. ".repeat(200));
    texts.push("word ".repeat(1000));
    texts.push("x".repeat(5000));

    const PIECES: &[&str] = &[
        "the",
        "CHONK",
        "hippo",
        " ",
        " ",
        " ",
        "  ",
        "\n",
        "\n\n",
        "\t",
        ".",
        ". ",
        "! ",
        "? ",
        ",",
        ";",
        ":",
        "(",
        ")",
        "\"",
        "'",
        "-",
        "—",
        "...",
        "é",
        "日本",
        "🦛",
        "👩‍👧",
        "e\u{301}",
        "\r\n",
        "1234",
    ];
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    for len in [10, 100, 1000, 5000] {
        for _ in 0..4 {
            let mut text = String::new();
            for _ in 0..len {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                text.push_str(PIECES[(seed % PIECES.len() as u64) as usize]);
            }
            texts.push(text);
        }
    }
    texts
}

/// Tokenizers to test with: characters, plus every `*.json` in the
/// `CHONKIE_TEST_TOKENIZERS` directory when it is set.
pub fn tokenizers() -> Vec<(String, Box<dyn Fn() -> Tokenizer>)> {
    let mut all: Vec<(String, Box<dyn Fn() -> Tokenizer>)> =
        vec![("characters".into(), Box::new(Tokenizer::characters))];
    #[cfg(feature = "gpt2")]
    all.push(("gpt2 (built in)".into(), Box::new(Tokenizer::gpt2)));
    if let Ok(dir) = std::env::var("CHONKIE_TEST_TOKENIZERS") {
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        paths.sort();
        for path in paths
            .into_iter()
            .filter(|p| p.extension().is_some_and(|e| e == "json"))
        {
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            all.push((name, Box::new(move || Tokenizer::open(&path).unwrap())));
        }
    }
    all
}
