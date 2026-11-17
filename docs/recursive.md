# 🦛 How the recursive chunker works

The recursive chunker cuts text at the most natural place that still fits: paragraphs first, then sentences, then pauses, then words, and characters only when nothing else is left. It does this without ever guessing a token count, and fast enough that the tokenizer is the bottleneck. This page covers the algorithm, the tricks that make it fast, and how the tests prove each guarantee.

## ✂️ The algorithm

A chunker has a `chunk_size` in tokens and a list of **levels**. Each level is a set of delimiters plus an `Include` side:

| Level | Delimiters | Splits into |
| ----- | ---------- | ----------- |
| 0 | `\n\n` `\r\n` `\n` `\r` | paragraphs and lines |
| 1 | `. ` `! ` `? ` | sentences |
| 2 | `{ } " [ ] < > ( ) : ; , — \| ~ - ... ` ' ` | clauses |
| 3 | ` ` | words |

`Include::Prev` keeps a delimiter at the end of the piece before it (`"Hi. " | "Yo."`), and `Include::Next` starts the next piece with it (`"Hi" | ". Yo"`). Either way the delimiter is kept, never dropped.

To chunk a span of text at level `l`:

1. **Fits?** If the span's token count is at most `chunk_size`, it is one chunk.
2. **Split** it at level `l`'s delimiters into pieces that cover the span with no gaps.
3. **Count** each piece's tokens.
4. **Merge** neighbouring pieces greedily while their counts add up to at most `chunk_size`. Then count the merged text exactly, because joining text can change how it tokenizes, and drop the last piece until the real count fits.
5. **Recurse** into any piece that is over `chunk_size` on its own, at level `l + 1`.
6. **Cut** text that has run out of levels (a long URL, a run of `x`) at character boundaries: binary search for the longest prefix that fits, then repeat on the rest.

Every chunk's `token_count` comes from counting that chunk's own text. Counts of pieces only decide where to try merging; they never end up in a chunk unless the chunk is exactly that piece.

Why recount a merge? Because tokens don't add up across a join. With gpt2, `"Hello. "` is 3 tokens (`Hello` `.` ` `) and `"World."` is 2, but `"Hello. World."` is 4 (`Hello` `.` ` World` `.`): the space joins the next word. Adding the parts is usually close, so it's a good first guess, but it's only a guess.

## ⚡ What makes it fast

On one core, recursive chunking spends most of its time in the tokenizer. The tricks below are about calling it less, and calling it on less text.

### 🔢 Pretokens are a free lower bound

Tokenizers first cut text into **pretokens** (roughly words, numbers, punctuation and whitespace runs), then encode each pretoken on its own. No token spans two pretokens, so:

\[
\text{pretokens}(t) \le \text{tokens}(t)
\]

If a piece already has more than `chunk_size` pretokens, it must be split, and there is no need to encode it. [tokie](https://github.com/feyninc/tokie) exposes this as `count_pretokens`. Its pretokenizer, pretokie, classifies 64 bytes at a time with AVX2 (letters, digits, spaces, newlines, apostrophes, non-ASCII), then walks the masks to find boundaries. That runs at about 800 MB/s on gpt2, 7–13× faster than encoding. tokie's tests check the inequality on 10MB of OpenWebText across 14 real tokenizers.

Even a cheap check costs something, so it only runs on pieces long enough to plausibly fail it: more than `bytes_per_pretoken × chunk_size` bytes. `bytes_per_pretoken` defaults to 4, about what English measures. `calibrate(sample)` measures it on your text instead. The threshold only decides which pieces get the cheap check, so calibration changes speed and never the chunks.

### 🔍 Delimiters are compiled once

Each level's delimiters are compiled into a [chunk](https://github.com/feyninc/chunk) `PatternSplitter` when the chunker is built: a double-array Aho-Corasick automaton that finds every delimiter in one pass, leftmost-first, so `\r\n` wins over `\r` and `\n\n` over `\n`. The recursion splits thousands of pieces per megabyte, and building the automaton for each one was about 8% of the runtime.

chunk's single-pattern paths go further with `memchr`: they search for delimiter bytes with SIMD (`memchr`, `memchr2`, `memchr3` for up to three bytes, a 256-entry lookup table beyond that) and only check whole patterns where a candidate byte is found. A prototype of that for `PatternSplitter` was another 13–16% faster end to end, which makes it the next thing to add to chunk.

### 🧵 One thread per chunker

A chunker counts thousands of short pieces. Handing those to a worker pool costs far more than counting them, so the chunker keeps tokie on the calling thread: with the pool, `chunk_size=128` dropped from about 100 MB/s to 15 MB/s on an 8-core machine. To use every core, run one chunker per thread on different texts.

### 📊 Numbers

5MB of OpenWebText, gpt2, one core. **Warm** is the best of 3 passes over the same text, so tokie's word cache is full; **cold** is the first pass, where tokie encodes every word for the first time.

| `chunk_size` | v2 warm | v2 cold | v1.7 + tiktoken |
| ------------ | ------: | ------: | --------------: |
| 128 | 105 MB/s | 44 MB/s | 8.2 MB/s |
| 512 | 113 MB/s | 46 MB/s | 12.1 MB/s |
| 2048 | 106 MB/s | 44 MB/s | 13.3 MB/s |

For scale, tokie encodes the same 5MB in one call at about 61 MB/s cold and 430 MB/s warm. Cold chunking runs at about 75% of the tokenizer's own speed. Warm, the remaining cost is mostly encoding text twice: once per piece and once per merged chunk.

## ✅ How the guarantees are tested

Each guarantee is its own test in `crates/chonkie/tests/recursive.rs`, and its own function in `crates/chonkie/tests/common/mod.rs` that the other chunkers will reuse. Each test runs over:

- **Tokenizers**: characters, the built-in gpt2 (with `--features gpt2`), and every `tokenizer.json` in `CHONKIE_TEST_TOKENIZERS` (CI uses gpt2, cl100k and BERT).
- **Chunk sizes**: 1, 2, 7, 64 and 512.
- **Levels**: the defaults, the defaults with `Include::Next`, and no levels at all, which forces character cuts.
- **Texts**: empty, whitespace only, CRLF, multi-byte Unicode (accents, combining marks, Japanese, emoji, ZWJ families, flags, skin tones), a long unsplittable URL, repeated text, and 16 seeded random texts built from words, delimiters, whitespace runs and multi-byte pieces.

A failure lists every case that broke, with the tokenizer, levels, chunk size, text and the offending chunk.

| Guarantee | Test | What it checks |
| --------- | ---- | -------------- |
| 1. 🧩 Reconstruction | `rule_1_chunks_rebuild_the_source` | Each chunk is non-empty and is exactly what the source continues with, and the chunks reach the end. |
| 2. 📍 Offsets | `rule_2_offsets_slice_the_source` | `source[start..end] == text`, each chunk starts where the last ended, from 0 to the end. |
| 3. 🔢 Token counts | `rule_3_token_counts_are_exact_and_fit` | A separate tokenizer recounts every chunk and gets `token_count`, which is at most `chunk_size` unless the chunk is one character. |
| 4. 🔁 Reproducibility | `rule_4_same_input_gives_same_chunks` | Chunking again with the same chunker (warm cache) and with a brand new one gives the same chunks. |
| 5. 🧼 No mangling | `rule_5_no_dropped_chars_or_split_codepoints` | Every chunk edge is a char boundary, and every character, spaces and delimiters included, appears as often in the chunks as in the source. |

The rules are checked separately on purpose: rule 5 doesn't lean on rules 1 and 2, so a bug that breaks one shows up as that rule. Smaller tests pin down behaviour the rules allow but don't fix: which side delimiters go on, text shorter than `chunk_size` staying whole, whitespace-only input, and multi-byte characters at `chunk_size=1`. `calibration_keeps_the_chunks` checks that `calibrate` never changes a chunk.
