<div align='center'>

![Chonkie Logo](./assets/chonkie-logo-thicc.png)

# Chonkie

[![PyPI version](https://img.shields.io/pypi/v/chonkie.svg)](https://pypi.org/project/chonkie/)
[![License](https://img.shields.io/github/license/feyninc/chonkie.svg)](https://github.com/feyninc/chonkie/blob/v2/LICENSE)
[![Discord](https://dcbadge.limes.pink/api/server/https://discord.gg/vH3SkRqmUz?style=flat)](https://discord.gg/vH3SkRqmUz)
[![GitHub stars](https://img.shields.io/github/stars/feyninc/chonkie.svg)](https://github.com/feyninc/chonkie/stargazers)

_Small, fast, correct text chunking._

</div>

> [!IMPORTANT]
> **Chonkie v2 is a rewrite in progress, and nothing on this branch is released yet.** The stable release is v1.7. Install it with `pip install "chonkie<2"`. Its code is on [`main`](https://github.com/feyninc/chonkie/tree/main) and its docs are at [docs.chonkie.ai](https://docs.chonkie.ai). The plan is in the [v2 RFC](https://github.com/feyninc/chonkie/issues/656).

## What v2 is

Chonkie v2 is a chunking library with a small Rust core and bindings for the places you use it:

- **Four chunkers:** token, sentence, recursive, and code.
- **[tokie](https://github.com/feyninc/tokie)** as the default tokenizer. It loads Hugging Face `tokenizer.json` files.
- **Built on [chunk](https://github.com/feyninc/chunk).**
- **Python and JS/TS bindings** over the same core.
- **A CLI, an HTTP server, and a small Docker image.**
- **Under 1MB** total binary.

## Guarantees

Every chunker holds these, and the test suite checks them:

1. **Reconstruction.** With no overlap, joining the chunks gives back the original text.
2. **Offsets.** `text[start:end]` is exactly the chunk's text.
3. **Token counts.** `token_count` matches the tokenizer and never exceeds `chunk_size`.
4. **Reproducibility.** The same input gives the same chunks.
5. **No mangling.** No dropped spaces or delimiters, and no split Unicode characters.

## What's gone

v1 grew from a chunking library into a framework. v2 keeps the chunking and drops the rest:

- **Other chunkers.** The semantic, neural, slumber, late, and the other v1 chunkers are gone. In our benchmarks, they did no better than recursive chunking.
- **Pipelines, chefs, refineries, embeddings, and handshakes.**
- **The 32+ vector DB and embedding integrations.**
- **The cloud client.**

If you depend on any of these, stay on v1.7. A 1.x to 2.0 migration guide will ship with the release.

## Install and usage

Not published yet. Python, JS/TS, CLI, and Docker instructions will land here as each one ships.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) and the [AI policy](AI_POLICY.md). Pull requests are frozen while v2 is built. Bug reports and questions are welcome in [issues](https://github.com/feyninc/chonkie/issues) and on [Discord](https://discord.gg/vH3SkRqmUz).

## Acknowledgements

Thanks to everyone who used Chonkie, filed issues, and sent fixes. v2 is shaped by your feedback.

And of course, special thanks to [Moto Moto](https://www.youtube.com/watch?v=I0zZC4wtqDQ&t=5s) for endorsing Chonkie with his famous quote:

> "I like them big, I like them chonkie." ~ Moto Moto

## Citation

If you use Chonkie in your research, please cite it as follows:

```bibtex
@software{chonkie2025,
  author = {Minhas, Bhavnick AND Nigam, Shreyash},
  title = {Chonkie: The lightweight ingestion library for fast, efficient and robust RAG pipelines},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/feyninc/chonkie}},
}
```
