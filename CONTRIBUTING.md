# Contributing

> "I like them big, I like them CONTRIBUTING" ~ Moto Moto, probably

**PRs are frozen.** We're building Chonkie v2. v1.7 is the last 1.x release; fixes go into 2.0. Until v2 is stable, the core team pushes directly to the `v2` branch, and new pull requests are closed automatically.

Read the [RFC](https://github.com/feyninc/chonkie/issues/656) first, and the [AI policy](AI_POLICY.md) before you open anything.

## What belongs here

Chonkie is a chunking library. The core is Rust. Python and JS/TS are bindings: they don't implement chunkers.

We keep four chunkers: token, sentence, recursive, code. [tokie](https://github.com/feyninc/tokie) is the default tokenizer. Delimiter search is chonkie's own, built on `memchr`.

Don't add integrations, handshakes, chefs, embeddings, pipelines, or another chunker. Plumbing is easy to write in the app now. What's hard is getting the primitives right.

## Correctness

These are not optional:

1. With overlap 0, concatenating chunk texts equals the source.
2. `text[start:end]` equals `chunk.text`. Indices are contiguous.
3. `token_count` matches the tokenizer and never exceeds `chunk_size`.
4. Same input, same chunks.
5. Don't drop spaces or delimiters. Don't split unicode codepoints. Code chunks slice the original bytes.

If a change breaks any of these, it doesn't land. Every chunker is tested against all five.

## Layout

v2 is one Cargo workspace. Crates land as they're written:

| Path | Role |
| --- | --- |
| `crates/chonkie` | Core: `Chunk`, the chunkers, tokenizer glue. |
| `crates/chonkie-python` | PyO3 bindings, built with maturin. Published to PyPI as `chonkie`. |
| `crates/chonkie-js` | JS/TS bindings. Published to npm; replaces chonkiejs. |
| `crates/chonkie-cli` | `chonkie` binary: CLI and HTTP server. |

## Dev setup

You need:

- Rust stable, via [rustup](https://rustup.rs), with `rustfmt` and `clippy`.
- For the Python bindings: Python 3.10+, [uv](https://docs.astral.sh/uv/), and [maturin](https://www.maturin.rs).
- For the JS bindings: Node.js LTS.

```bash
git clone https://github.com/feyninc/chonkie
cd chonkie
git checkout v2
```

Rust:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Python bindings:

```bash
cd crates/chonkie-python
uv venv && source .venv/bin/activate
maturin develop --release
pytest
```

Run fmt, clippy, and the tests before you push. CI runs the same.

## Commit messages

Every commit subject uses one of two prefixes:

```
feat(scope): short imperative summary
bug(scope): short imperative summary
```

`feat` is new behavior, a rewrite, docs, or a refactor. `bug` fixes incorrect behavior. No other prefixes: not `fix`, `docs`, `chore`, or `refactor`.

`scope` is one of `core`, `python`, `js`, `cli`, `server`, `docker`, `ci`, `docs`. The summary is lowercase and imperative, with no trailing period. Add a body when the subject doesn't say why.

```
feat(core): add the recursive chunker on top of chunk

bug(core): keep whitespace in recursive overlap
```

Pull request titles follow the same format.

## Pushing to `v2`

While v2 is being built, the core team pushes straight to `v2`. Keep each commit to one change, run fmt, clippy, and the tests first, and never force-push. If AI wrote a significant part of a commit, put the prompts in the commit body under `AI prompts:`.

## Pull requests

Once v2 is stable and the freeze lifts, changes come in as pull requests against `main`.

- One change per pull request. Keep it small enough to read in one sitting.
- A bug fix comes with a test that fails without it.
- Open it yourself and explain what changed and why in your own words. AI must not open pull requests or write comments.
- If AI wrote a significant part of the change, include the prompts you used under an **AI prompts** heading. See the [AI policy](AI_POLICY.md).

## Issues

If something is broken in v2, [open an issue](https://github.com/feyninc/chonkie/issues). Include the input, what you expected, and what Chonkie did. All issues are helpful to have: bugs, missing edges, APIs that lie, docs that confuse. We want them.

Discord is [here](https://discord.gg/vH3SkRqmUz) if you'd rather talk first. Email: [support@chonkie.ai](mailto:support@chonkie.ai).
