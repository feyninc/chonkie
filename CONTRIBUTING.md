# Contributing

> "I like them big, I like them CONTRIBUTING" ~ Moto Moto, probably

**PRs are frozen.** We're building Chonkie v2. v1.7 is the last 1.x release; fixes go into 2.0. New pull requests on `main` will be closed until v2 is stable.

Read the [RFC](https://docs.chonkie.ai/oss/rfc-chonkie-v2) first.

## What belongs here

Chonkie is a chunking library. The core is Rust. Python and JS/TS are bindings — they don't implement chunkers.

We keep four chunkers: token, sentence, recursive, code. Tokie is the default tokenizer.

Don't add integrations, handshakes, chefs, embeddings, pipelines, or another chunker. Plumbing is easy to write in the app now. What's hard is getting the primitives right.

## Correctness

These are not optional:

1. With overlap 0, concatenating chunk texts equals the source.
2. `text[start:end]` equals `chunk.text`. Indices are contiguous.
3. `token_count` matches the tokenizer and never exceeds `chunk_size`.
4. Same input, same chunks.
5. Don't drop spaces or delimiters. Don't split unicode codepoints. Code chunks slice the original bytes.

If a change breaks any of these, it doesn't land.

## Issues

If something is broken in v2, [open an issue](https://github.com/chonkie-inc/chonkie/issues). All issues are helpful to have — bugs, missing edges, APIs that lie, docs that confuse. We want them.

Discord is [here](https://discord.gg/vH3SkRqmUz) if you'd rather talk first. Email: [support@chonkie.ai](mailto:support@chonkie.ai).
