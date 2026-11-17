<div align='center'>

![Chonkie Logo](./assets/chonkie-logo-thicc.png)

# 🦛 Chonkie ✨

[![PyPI version](https://img.shields.io/pypi/v/chonkie.svg)](https://pypi.org/project/chonkie/)
[![License](https://img.shields.io/github/license/feyninc/chonkie.svg)](https://github.com/feyninc/chonkie/blob/v2/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-chonkie.ai-blue.svg)](https://docs.chonkie.ai)
[![Downloads](https://static.pepy.tech/badge/chonkie)](https://pepy.tech/project/chonkie)
[![Discord](https://dcbadge.limes.pink/api/server/https://discord.gg/vH3SkRqmUz?style=flat)](https://discord.gg/vH3SkRqmUz)
[![GitHub stars](https://img.shields.io/github/stars/feyninc/chonkie.svg)](https://github.com/feyninc/chonkie/stargazers)

_The tiny, fast and correct chunking library that just works_

[Status](#-status) •
[Chunkers](#️-chunkers) •
[Guarantees](#-guarantees) •
[Installation](#-installation)

</div>

Tired of making your gazillionth chunker? Sick of the overhead of large libraries? Want to chunk your texts quickly and efficiently? Chonkie the mighty hippo is here to help! And this time, she's been on a diet. 🥗

**🦀 Rust core**: One small, readable core. Every binding CHONKs the same way </br>
**🪶 Light-weight**: Under 1MB of code. No bloat, just CHONK </br>
**⚡ Fast**: CHONK at the speed of light! zooooom </br>
**✅ Correct**: Chunks that add back up to your text, every single time </br>
**✨ Easy to use**: Install, Import, CHONK </br>
**🌍 Everywhere**: Python, JS/TS, a CLI, an HTTP server and a teeny Docker image </br>
**🦛 Cute CHONK mascot**: psst it's a pygmy hippo btw </br>
**❤️ [Moto Moto](#-acknowledgements)'s favorite chunking library** </br>

**Chonkie** is a chunking library that "**just works**" ✨

Chonkie started as a tiny chunking library and, well, got a little too CHONKY. 🍔 v2 goes back to what Chonkie does best: chunking.

- **🦀 A Rust core** of about 1–2k lines, built on [chunk](https://github.com/feyninc/chunk).
- **🔤 [tokie](https://github.com/feyninc/tokie)** as the default tokenizer. It loads any Hugging Face `tokenizer.json`.
- **🐍 Python and 🟨 JS/TS bindings** over the same core, so every language gets the same chunks.
- **💻 A CLI, 🌐 an HTTP server and 🐳 a small Docker image** for when you don't want to write code at all.

What's gone? 👋 The semantic, neural, slumber, late and other chunkers, which did no better than recursive chunking in our benchmarks. Pipelines, chefs, refineries, embeddings, handshakes, the 32+ integrations and the cloud client went with them. If you depend on any of those, v1.7 isn't going anywhere, and a 1.x to 2.0 migration guide will ship with the release.

## 🚧 Status

> [!IMPORTANT]
> **Chonkie v2 is a rewrite in progress, and nothing on this branch is released yet.** The stable release is v1.7. Install it with `pip install "chonkie<2"`. Its code is on [`main`](https://github.com/feyninc/chonkie/tree/main) and its docs are at [docs.chonkie.ai](https://docs.chonkie.ai). The full plan is in the [v2 RFC](https://github.com/feyninc/chonkie/issues/656).

## ✂️ Chunkers

Four chunkers. That's it. That's the list. 🦛

| Name        | Description                                                        |
| ----------- | ------------------------------------------------------------------ |
| `token`     | Splits text into fixed-size token chunks.                          |
| `sentence`  | Splits text into chunks on sentence boundaries.                    |
| `recursive` | Splits text hierarchically with customizable rules.                |
| `code`      | Splits code into structurally meaningful chunks.                   |

Curious how they CHONK so fast? 🏎️ [How the recursive chunker works](docs/recursive.md) walks through the algorithm, the speed tricks and the tests behind every guarantee.

## ✅ Guarantees

Every chunker holds these, and the test suite checks them. No ifs, no buts, no lost CHONKs:

1. **🧩 Reconstruction**: with no overlap, joining the chunks gives back the original text.
2. **📍 Offsets**: `text[start:end]` is exactly the chunk's text.
3. **🔢 Token counts**: `token_count` matches the tokenizer and never exceeds `chunk_size`.
4. **🔁 Reproducibility**: the same input gives the same chunks.
5. **🧼 No mangling**: no dropped spaces or delimiters, and no split Unicode characters.

## 📦 Installation

Not published yet! 🥚 Python, JS/TS, CLI and Docker instructions will hatch here as each one ships. Until then, `pip install "chonkie<2"` gets you v1.7.

## 🤝 Contributing

Want to help grow Chonkie? Check out [CONTRIBUTING.md](CONTRIBUTING.md) and our [AI policy](AI_POLICY.md). Pull requests are frozen while we build v2, but bug reports and questions are always welcome in [issues](https://github.com/feyninc/chonkie/issues) and on [Discord](https://discord.gg/vH3SkRqmUz).

Remember: No contribution is too small for this tiny hippo! 🦛

## 🙏 Acknowledgements

Chonkie would like to CHONK its way through a special thanks to all the users and contributors who have helped make this library what it is today! Your feedback, issue reports, and improvements have helped make Chonkie the CHONKIEST it can be.

And of course, special thanks to [Moto Moto](https://www.youtube.com/watch?v=I0zZC4wtqDQ&t=5s) for endorsing Chonkie with his famous quote:

> "I like them big, I like them chonkie." ~ Moto Moto

## 📝 Citation

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
