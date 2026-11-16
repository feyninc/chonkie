mod common;

use chonkie::{Include, Level, RecursiveChunker, Tokenizer};

fn run(make: &dyn Fn() -> Tokenizer, chunk_size: usize, levels: Vec<Level>) {
    let mut chunker = RecursiveChunker::new(make())
        .chunk_size(chunk_size)
        .levels(levels);
    let mut counter = make();
    for text in common::corpus() {
        let chunks = chunker.chunk(&text).unwrap();
        common::check(&text, &chunks, &mut counter, chunk_size);
        assert_eq!(
            chunks,
            chunker.chunk(&text).unwrap(),
            "rule 4: same input, same chunks"
        );
    }
}

#[test]
fn holds_the_rules() {
    for (name, make) in common::tokenizers() {
        for chunk_size in [1, 2, 7, 64, 512] {
            eprintln!("{name} chunk_size={chunk_size}");
            run(&*make, chunk_size, Level::defaults());
        }
    }
}

#[test]
fn holds_the_rules_with_next_and_no_levels() {
    let next: Vec<Level> = Level::defaults()
        .into_iter()
        .map(|l| l.include(Include::Next))
        .collect();
    for (_, make) in common::tokenizers() {
        run(&*make, 16, next.clone());
        run(&*make, 16, vec![]);
    }
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
#[ignore = "downloads gpt2 from the Hugging Face Hub"]
fn default_tokenizer_is_gpt2() {
    let chunks = RecursiveChunker::new(Tokenizer::default())
        .chunk_size(4)
        .chunk("Hello world, CHONK!")
        .unwrap();
    assert_eq!(chunks.iter().map(|c| c.token_count).sum::<usize>(), 7);
}
