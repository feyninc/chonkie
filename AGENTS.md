# chonkie

Chonkie v2: a small chunking library with a Rust core and Python and JS/TS bindings. The plan is in the [v2 RFC](https://github.com/feyninc/chonkie/issues/656). v1.7 (Python) lives on `main`. v2 work happens on the `v2` branch.

License is MIT.

Read [CONTRIBUTING.md](CONTRIBUTING.md) and [AI_POLICY.md](AI_POLICY.md). Both apply to agents.

## Scope

Four chunkers: token, sentence, recursive, code. Nothing else. Do not add integrations, handshakes, chefs, embeddings, pipelines, cloud clients, or another chunker. Do not port v1 modules beyond those four.

The core stays small: 1-2k lines of Rust, under 1MB of its own code in the binary (dependencies such as tokie and std are not counted). Ask before adding a dependency. `tokie` is the tokenizer; the default is gpt2 (`tokiers/gpt2`), downloaded from the Hugging Face Hub on first use and cached. Nothing tokenizer-sized is embedded by default; the opt-in `gpt2` feature builds the gpt2 `.tokie` file (3.9MB of data) into the binary for offline and browser use. `chunk` is the foundation for the chunkers and stays an independent crate; don't vendor it.

## Crates

| Path | Role |
| --- | --- |
| `crates/chonkie` | Core: `Chunk`, the chunkers, tokenizer glue. |
| `crates/chonkie-python` | PyO3 bindings (maturin). The Python module is `chonkie`. |
| `crates/chonkie-js` | JS/TS bindings. Replaces chonkiejs. |
| `crates/chonkie-cli` | `chonkie` binary: CLI and HTTP server. |

Crates land as they are written; check the workspace `Cargo.toml` for what exists. Bindings wrap the core. They never implement chunking logic.

## Correctness

Every chunker must hold all five, and has tests for all five:

1. With overlap 0, concatenating chunk texts equals the source.
2. `text[start:end]` equals `chunk.text`. Indices are contiguous.
3. `token_count` matches the tokenizer and never exceeds `chunk_size`.
4. Same input, same chunks.
5. No dropped spaces or delimiters, no split codepoints. Code chunks slice the original bytes.

Test empty input, whitespace-only input, multi-byte unicode, and text shorter than `chunk_size`. Offsets are byte offsets in Rust; bindings convert them for their language and test that conversion.

## Commands

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test -p chonkie
```

Python bindings, from `crates/chonkie-python`: `maturin develop --release`, then `pytest`.

## Commit messages

Every commit subject uses one of these two prefixes:

```
feat(scope): short imperative summary
bug(scope): short imperative summary
```

`feat` is new behavior, a rewrite, docs, or a refactor. `bug` fixes incorrect behavior. Use no other prefix. That includes `docs`, `chore`, `fix`, and `refactor`.

`scope` is `core`, `python`, `js`, `cli`, `server`, `docker`, `ci`, or `docs`. The summary is lowercase and imperative, with no trailing period. Add a body when the subject does not say why. PR titles use the same format.

```
feat(core): add the recursive chunker on top of chunk

bug(core): keep whitespace in recursive overlap
```

Commit only when the user asks. Never open pull requests or issues, and never post comments on GitHub: a human does that. While v2 is being built, commits go straight to `v2` (no PRs; the freeze workflow closes them). Never force-push. Put the session's prompts in the commit body under `AI prompts:` when AI wrote a significant part, as the AI policy requires.
