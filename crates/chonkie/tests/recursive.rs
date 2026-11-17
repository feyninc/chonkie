mod common;

use chonkie::{Chunk, Include, Level, RecursiveChunker, Tokenizer};
use common::Verdict;

const CHUNK_SIZES: [usize; 5] = [1, 2, 7, 64, 512];

/// Level sets to run every rule under: the defaults, the defaults with each
/// delimiter starting the next piece, and no levels at all (character cuts).
fn layouts() -> Vec<(&'static str, Vec<Level>)> {
    let next = Level::defaults()
        .into_iter()
        .map(|l| l.include(Include::Next))
        .collect();
    vec![
        ("default levels", Level::defaults()),
        ("Include::Next", next),
        ("no levels", vec![]),
    ]
}

/// What a rule check can use besides the text and its chunks.
struct Case<'a> {
    chunk_size: usize,
    chunker: &'a mut RecursiveChunker,
    /// A tokenizer the chunker never used, to count chunks independently.
    counter: &'a mut Tokenizer,
    /// Builds a new chunker, with a new tokenizer, configured the same way.
    fresh: &'a dyn Fn() -> RecursiveChunker,
}

/// Runs `check` on every tokenizer, chunk size, level set and corpus text,
/// and fails listing every case that broke the rule.
fn every_case(rule: &str, mut check: impl FnMut(&str, &[Chunk], Case) -> Verdict) {
    let corpus = common::corpus();
    let mut failures = Vec::new();
    for (name, make) in common::tokenizers() {
        let mut counter = make();
        for (layout, levels) in layouts() {
            for chunk_size in CHUNK_SIZES {
                let fresh = || {
                    RecursiveChunker::new(make())
                        .chunk_size(chunk_size)
                        .levels(levels.clone())
                };
                let mut chunker = fresh();
                for (i, text) in corpus.iter().enumerate() {
                    let chunks = chunker.chunk(text).unwrap();
                    let case = Case {
                        chunk_size,
                        chunker: &mut chunker,
                        counter: &mut counter,
                        fresh: &fresh,
                    };
                    if let Err(why) = check(text, &chunks, case) {
                        failures.push(format!(
                            "{name}, {layout}, chunk_size {chunk_size}, corpus text {i}: {why}"
                        ));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{rule} broke in {} case(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn rule_1_chunks_rebuild_the_source() {
    every_case("rule 1", |text, chunks, _| {
        common::rebuilds_source(text, chunks)
    });
}

#[test]
fn rule_2_offsets_slice_the_source() {
    every_case("rule 2", |text, chunks, _| {
        common::offsets_slice_source(text, chunks)
    });
}

#[test]
fn rule_3_token_counts_are_exact_and_fit() {
    every_case("rule 3", |_, chunks, case| {
        common::token_counts_exact(chunks, case.counter, case.chunk_size)
    });
}

#[test]
fn rule_4_same_input_gives_same_chunks() {
    // Again with the same chunker, whose tokenizer cache is now warm, and
    // with a fresh one, so no state carried between calls can change a chunk.
    every_case("rule 4", |text, chunks, case| {
        common::same_chunks(chunks, &case.chunker.chunk(text).unwrap())?;
        common::same_chunks(chunks, &(case.fresh)().chunk(text).unwrap())
    });
}

#[test]
fn rule_5_no_dropped_chars_or_split_codepoints() {
    every_case("rule 5", |text, chunks, _| {
        common::keeps_every_char(text, chunks)
    });
}

#[test]
fn keeps_paragraphs_whole_when_they_fit() {
    let text = "First paragraph.\n\nSecond paragraph.\n\nThird.";
    let chunks = RecursiveChunker::new(Tokenizer::characters())
        .chunk_size(20)
        .chunk(text)
        .unwrap();
    let texts: Vec<&str> = chunks.iter().map(|c| c.text).collect();
    assert_eq!(
        texts,
        ["First paragraph.\n\n", "Second paragraph.\n\n", "Third."]
    );
}

#[test]
fn keeps_whitespace_at_split_points() {
    // github.com/feyninc/chonkie/issues/681
    let text = "alpha beta  gamma   delta\tepsilon\n zeta";
    let chunks = RecursiveChunker::new(Tokenizer::characters())
        .chunk_size(6)
        .chunk(text)
        .unwrap();
    assert_eq!(chunks.iter().map(|c| c.text).collect::<String>(), text);
}

#[test]
fn empty_text_has_no_chunks() {
    assert!(
        RecursiveChunker::new(Tokenizer::characters())
            .chunk("")
            .unwrap()
            .is_empty()
    );
}

#[test]
fn rejects_bad_config() {
    assert!(
        RecursiveChunker::new(Tokenizer::characters())
            .chunk_size(0)
            .chunk("hi")
            .is_err()
    );
    let empty = vec![Level::new([""])];
    assert!(
        RecursiveChunker::new(Tokenizer::characters())
            .levels(empty)
            .chunk("hi")
            .is_err()
    );
}

#[test]
#[cfg(not(feature = "gpt2"))]
#[ignore = "downloads gpt2 from the Hugging Face Hub"]
fn default_tokenizer_is_gpt2() {
    let chunks = RecursiveChunker::new(Tokenizer::default())
        .chunk_size(4)
        .chunk("Hello world, CHONK!")
        .unwrap();
    assert_eq!(chunks.iter().map(|c| c.token_count).sum::<usize>(), 7);
}

#[test]
#[cfg(feature = "gpt2")]
fn built_in_gpt2_is_the_default() {
    let mut tok = Tokenizer::default();
    assert_eq!(tok.count("Hello world, CHONK!").unwrap(), 7);
    assert_eq!(Tokenizer::gpt2().count("").unwrap(), 0);
}

#[test]
fn calibration_keeps_the_chunks() {
    let text = common::corpus().concat();
    for (_, make) in common::tokenizers() {
        let plain = RecursiveChunker::new(make())
            .chunk_size(64)
            .chunk(&text)
            .unwrap();
        for sample in ["", "x", "The quick brown fox jumps over the lazy dog."] {
            let mut calibrated = RecursiveChunker::new(make())
                .chunk_size(64)
                .calibrate(sample)
                .unwrap();
            assert_eq!(plain, calibrated.chunk(&text).unwrap());
        }
    }
}

fn texts(chunks: &[Chunk]) -> Vec<String> {
    chunks.iter().map(|c| c.text.to_string()).collect()
}

#[test]
fn delimiters_go_on_the_side_include_says() {
    let text = "One. Two. Three.";
    let sentences = || vec![Level::new([". "])];
    let mut prev = RecursiveChunker::new(Tokenizer::characters())
        .chunk_size(6)
        .levels(sentences());
    let mut next = RecursiveChunker::new(Tokenizer::characters())
        .chunk_size(6)
        .levels(
            sentences()
                .into_iter()
                .map(|l| l.include(Include::Next))
                .collect(),
        );
    assert_eq!(
        texts(&prev.chunk(text).unwrap()),
        ["One. ", "Two. ", "Three."]
    );
    assert_eq!(
        texts(&next.chunk(text).unwrap()),
        ["One", ". Two", ". Thre", "e."]
    );
}

#[test]
fn text_that_fits_is_one_chunk() {
    for (name, make) in common::tokenizers() {
        let mut chunker = RecursiveChunker::new(make()).chunk_size(512);
        for text in [
            "a",
            " ",
            "Hello, world.",
            "Zürich 🦛 日本語.\n\nTwo paragraphs.",
        ] {
            let chunks = chunker.chunk(text).unwrap();
            assert_eq!(texts(&chunks), [text], "{name}");
            assert_eq!((chunks[0].start, chunks[0].end), (0, text.len()), "{name}");
        }
    }
}

#[test]
fn whitespace_only_text_is_kept() {
    for text in [" ", "\n\n\n", "  \t\r\n  \n\n   "] {
        for chunk_size in [1, 2, 64] {
            let chunks = RecursiveChunker::new(Tokenizer::characters())
                .chunk_size(chunk_size)
                .chunk(text)
                .unwrap();
            assert_eq!(texts(&chunks).concat(), text, "chunk_size {chunk_size}");
        }
    }
}

#[test]
fn multibyte_chars_are_never_cut() {
    let text = "日本🦛e\u{301}";
    let chunks = RecursiveChunker::new(Tokenizer::characters())
        .chunk_size(1)
        .chunk(text)
        .unwrap();
    assert_eq!(texts(&chunks), ["日", "本", "🦛", "e", "\u{301}"]);
}
