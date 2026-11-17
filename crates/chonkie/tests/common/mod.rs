//! The five correctness rules every chunker must hold, and the texts to check them on.
#![allow(dead_code)]

use chonkie::{Chunk, Tokenizer};

/// Checks rules 1, 2, 3 and 5 for chunks made with overlap 0.
/// Rule 5 follows from 1 and 2: the chunks are slices that rebuild the source.
pub fn check(source: &str, chunks: &[Chunk], tokenizer: &mut Tokenizer, chunk_size: usize) {
    let rebuilt: String = chunks.iter().map(|c| c.text).collect();
    assert_eq!(rebuilt, source, "rule 1: chunks must rebuild the source");

    let mut at = 0;
    for chunk in chunks {
        assert!(!chunk.text.is_empty(), "no empty chunks");
        assert_eq!(chunk.start, at, "rule 2: indices must be contiguous");
        assert_eq!(
            &source[chunk.start..chunk.end],
            chunk.text,
            "rule 2: text[start..end] == text"
        );
        at = chunk.end;

        let count = tokenizer.count(chunk.text).unwrap();
        assert_eq!(
            chunk.token_count, count,
            "rule 3: token_count must match the tokenizer"
        );
        let one_char = chunk.text.chars().count() == 1;
        assert!(
            chunk.token_count <= chunk_size || one_char,
            "rule 3: {count} tokens > {chunk_size}"
        );
    }
    assert_eq!(
        at,
        source.len(),
        "rule 2: chunks must end at the source end"
    );
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
