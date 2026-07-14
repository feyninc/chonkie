<div align='center'>

![Chonkie Logo](https://github.com/chonkie-inc/chonkie/blob/main/assets/chonkie_logo_br_transparent_bg.png?raw=true)

# 🦛 Chonkie ✨

[![PyPI version](https://img.shields.io/pypi/v/chonkie.svg)](https://pypi.org/project/chonkie/)
[![License](https://img.shields.io/github/license/chonkie-inc/chonkie.svg)](https://github.com/chonkie-inc/chonkie/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-chonkie.ai-blue.svg)](https://docs.chonkie.ai)
[![Package size](https://img.shields.io/badge/size-505KB-blue)](https://github.com/chonkie-inc/chonkie/blob/main/README.md#installation)
[![codecov](https://codecov.io/gh/chonkie-inc/chonkie/graph/badge.svg?token=V4EWIJWREZ)](https://codecov.io/gh/chonkie-inc/chonkie)
[![Downloads](https://static.pepy.tech/badge/chonkie)](https://pepy.tech/project/chonkie)
[![Discord](https://dcbadge.limes.pink/api/server/https://discord.gg/vH3SkRqmUz?style=flat)](https://discord.gg/vH3SkRqmUz)
[![GitHub stars](https://img.shields.io/github/stars/chonkie-inc/chonkie.svg)](https://github.com/chonkie-inc/chonkie/stargazers)

_The lightweight ingestion library for fast, efficient and robust RAG pipelines_

[Installation](#📦-installation) •
[Usage](#🚀-usage) •
[Chunkers](#✂️-chunkers) •
[Integrations](#🔌-integrations) •
[Benchmarks](#📊-benchmarks)

</div>

Tired of making your gazillionth chunker? Sick of the overhead of large libraries? Want to chunk your texts quickly and efficiently? Chonkie the mighty hippo is here to help!

**🚀 Feature-rich**: All the CHONKs you'd ever need </br>
**🔄 End-to-end**: Fetch, CHONK, refine, embed and ship straight to your vector DB! </br>
**✨ Easy to use**: Install, Import, CHONK </br>
**⚡ Fast**: CHONK at the speed of light! zooooom </br>
**🪶 Light-weight**: No bloat, just CHONK </br>
**🔌 32+ [integrations](#integrations)**: Works with your favorite tools and vector DBs out of the box! </br>
**💬 ️Multilingual**: Out-of-the-box support for 56 languages </br>
**☁️ Cloud-Friendly**: CHONK locally or in the [Cloud](https://labs.chonkie.ai) </br>
**🦛 Cute CHONK mascot**: psst it's a pygmy hippo btw </br>
**❤️ [Moto Moto](#acknowledgements)'s favorite python library** </br>

**Chonkie** is a chunking library that "**just works**" ✨

## 📦 Installation

### Basic Installation

Using pip:

```bash
pip install chonkie
```

Or using [uv](https://docs.astral.sh/uv/) (faster):

```bash
uv pip install chonkie
```

### Full Installation

Chonkie follows the rule of minimum installs.
Have a favorite chunker? Read our [docs](https://docs.chonkie.ai) to install only what you need.
Don't want to think about it? Simply install `all` (Not recommended for production environments).

Using pip:

```bash
pip install "chonkie[all]"
```

Or using [uv](https://docs.astral.sh/uv/):

```bash
uv pip install "chonkie[all]"
```

## 🚀 Usage

### Basic Usage

Here's a basic example to get you started:

```python
# First import the chunker you want from Chonkie
from chonkie import RecursiveChunker

# Initialize the chunker
chunker = RecursiveChunker()

# Chunk some text
chunks = chunker("Chonkie is the goodest boi! My favorite chunking hippo hehe.")

# Access chunks
for chunk in chunks:
    print(f"Chunk: {chunk.text}")
    print(f"Tokens: {chunk.token_count}")
```

### Pipeline Usage

You can also use the `chonkie.Pipeline` to chain components together and handle complex workflows. Read more about pipelines in the [docs](https://docs.chonkie.ai/oss/pipelines)!

```python
from chonkie import Pipeline

# Create a pipeline with multiple chunking and refinement steps
pipe = (
    Pipeline()
    .chunk_with("recursive", tokenizer="gpt2", chunk_size=2048, recipe="markdown")
    .chunk_with("semantic", chunk_size=512)
    .refine_with("overlap", context_size=128)
    .refine_with("embeddings", embedding_model="sentence-transformers/all-MiniLM-L6-v2")
)

# CHONK some Texts!
doc = pipe.run(texts="Chonkie is the goodest boi! My favorite chunking hippo hehe.")

# Access the processed chunks in the `doc` object
for chunk in doc.chunks:
    print(chunk.text)

# Run asynchronously for high-throughput applications
import asyncio

async def main():
    doc = await pipe.arun(texts="Chonkie runs fast!")
    print(len(doc.chunks))

asyncio.run(main())
```

Check out more usage examples in the [docs](https://docs.chonkie.ai)!

## 🌐 API Server

Run Chonkie as a self-hosted REST API for easy integration into any application:

```bash
# Install with API dependencies (includes catsu for multi-provider embeddings)
pip install "chonkie[api,semantic,code,catsu]"

# Start the server using the CLI
chonkie serve

# Or with custom options
chonkie serve --port 3000 --reload --log-level debug

# Or directly with uvicorn
uvicorn chonkie.api.main:app --host 0.0.0.0 --port 8000
```

Or use Docker:

```bash
docker compose up
```

The API provides endpoints for all chunkers, refineries, and **pipelines** — reusable workflow configurations stored in a local SQLite database.

```bash
# Create a reusable pipeline
curl -X POST http://localhost:8000/v1/pipelines \
  -H "Content-Type: application/json" \
  -d '{
    "name": "rag-chunker",
    "steps": [
      {"type": "chunk", "chunker": "semantic", "config": {"chunk_size": 512}},
      {"type": "refine", "refinery": "embeddings", "config": {"embedding_model": "text-embedding-3-small"}}
    ]
  }'

# List your pipelines
curl http://localhost:8000/v1/pipelines
```

Interactive documentation is available at `/docs` when the server is running.

## ✂️ Chunkers

Chonkie provides several chunkers to help you split your text efficiently for RAG applications. Here's a quick overview of the available chunkers:

| Name               | Alias       | Description                                                                                                                |
| ------------------ | ----------- | -------------------------------------------------------------------------------------------------------------------------- |
| `TokenChunker`     | `token`     | Splits text into fixed-size token chunks.                                                                                  |
| `FastChunker`      | `fast`      | SIMD-accelerated byte-based chunking at 100+ GB/s. Included in the default install.                                        |
| `SentenceChunker`  | `sentence`  | Splits text into chunks based on sentences.                                                                                |
| `RecursiveChunker` | `recursive` | Splits text hierarchically using customizable rules to create semantically meaningful chunks.                              |
| `SemanticChunker`  | `semantic`  | Splits text into chunks based on semantic similarity. Inspired by the work of [Greg Kamradt](https://github.com/gkamradt). |
| `LateChunker`      | `late`      | Embeds text and then splits it to have better chunk embeddings.                                                            |
| `CodeChunker`      | `code`      | Splits code into structurally meaningful chunks.                                                                           |
| `NeuralChunker`    | `neural`    | Splits text using a neural model.                                                                                          |
| `SlumberChunker`   | `slumber`   | Splits text using an LLM to find semantically meaningful chunks. Also known as _"AgenticChunker"_.                         |
| `TableChunker`     | `table`     | Chunks markdown tables by rows or character count.                                                                          |
| `TeraflopAIChunker`| `teraflopai`| Splits text using the TeraflopAI Segmentation API for domain-specific segmentation.                                        |

More on these methods and the approaches taken inside the [docs](https://docs.chonkie.ai)

## 🔌 Integrations

Chonkie boasts 45+ integrations across tokenizers, embedding providers, LLMs, refineries, porters, vector databases, and utilities, ensuring it fits seamlessly into your existing workflow.

<details>
<summary><strong>👨‍🍳 Chefs & 📁 Fetchers! Text preprocessing and data loading!</strong></summary>

Chefs handle text preprocessing, while Fetchers load data from various sources.

| Component | Class          | Description                                        | Optional Install  |
| --------- | -------------- | -------------------------------------------------- | ----------------- |
| `chef`    | `TextChef`     | Text preprocessing and cleaning.                   | `default`         |
| `chef`    | `MarkdownChef` | Parse markdown into structured MarkdownDocuments.   | `default`         |
| `chef`    | `TableChef`    | Process CSV/Excel files into MarkdownDocuments.     | `chonkie[table]`  |
| `chef`    | `MistralOCR`   | Extract text from images/PDFs via Mistral OCR API. | `chonkie[mistral]` |
| `fetcher` | `FileFetcher`  | Load text from files and directories.              | `default`         |

</details>
<details>
<summary><strong>🏭 Refine your CHONKs with Context and Embeddings! Chonkie supports 2+ refineries!</strong></summary>

Refineries help you post-process and enhance your chunks after initial chunking.

| Refinery Name | Class                | Description                                   | Optional Install    |
| ------------- | -------------------- | --------------------------------------------- | ------------------- |
| `overlap`     | `OverlapRefinery`    | Merge overlapping chunks based on similarity. | `default`           |
| `embeddings`  | `EmbeddingsRefinery` | Add embeddings to chunks using any provider.  | `chonkie[semantic]` |

</details>

<details>
<summary><strong>🐴 Exporting CHONKs! Chonkie supports 2+ Porters!</strong></summary>

Porters help you save your chunks easily.

| Porter Name | Class            | Description                            | Optional Install    |
| ----------- | ---------------- | -------------------------------------- | ------------------- |
| `json`      | `JSONPorter`     | Export chunks to a JSON file.          | `default`           |
| `datasets`  | `DatasetsPorter` | Export chunks to HuggingFace datasets. | `chonkie[datasets]` |

</details>

<details>
<summary><strong>🤝 Shake hands with your DB! Chonkie connects with 10+ vector stores!</strong></summary>

Handshakes provide a unified interface to ingest chunks directly into your favorite vector databases.

| Handshake Name | Class                  | Description                                  | Optional Install    |
| -------------- | ---------------------- | -------------------------------------------- | ------------------- |
| `chroma`       | `ChromaHandshake`      | Ingest chunks into ChromaDB.                 | `chonkie[chroma]`   |
| `elastic`      | `ElasticHandshake`     | Ingest chunks into Elasticsearch.            | `chonkie[elastic]`  |
| `mongodb`      | `MongoDBHandshake`     | Ingest chunks into MongoDB.                  | `chonkie[mongodb]`  |
| `pgvector`     | `PgvectorHandshake`    | Ingest chunks into PostgreSQL with pgvector. | `chonkie[pgvector]` |
| `pinecone`     | `PineconeHandshake`    | Ingest chunks into Pinecone.                 | `chonkie[pinecone]` |
| `qdrant`       | `QdrantHandshake`      | Ingest chunks into Qdrant.                   | `chonkie[qdrant]`   |
| `turbopuffer`  | `TurbopufferHandshake` | Ingest chunks into Turbopuffer.              | `chonkie[tpuf]`     |
| `weaviate`     | `WeaviateHandshake`    | Ingest chunks into Weaviate.                 | `chonkie[weaviate]` |
| `lancedb`      | `LanceDBHandshake`     | Ingest chunks into LanceDB.                  | `chonkie[lancedb]`  |
| `milvus`       | `MilvusHandshake`      | Ingest chunks into Milvus.                   | `chonkie[milvus]`   |

</details>
<details>
<summary><strong>🪓 Slice 'n' Dice! Chonkie supports 5+ ways to tokenize! </strong></summary>

Choose from supported tokenizers or provide your own custom token counting function. Flexibility first!

| Name           | Description                                                    | Optional Install      |
| -------------- | -------------------------------------------------------------- | --------------------- |
| `character`    | Basic character-level tokenizer. **Default tokenizer.**        | `default`             |
| `word`         | Basic word-level tokenizer.                                    | `default`             |
| `byte`         | Byte-level tokenizer operating on UTF-8 encoded bytes.         | `default`             |
| `tokenizers`   | Load any tokenizer from the Hugging Face `tokenizers` library. | `chonkie[tokenizers]` |
| `tiktoken`     | Use OpenAI's `tiktoken` library (e.g., for `gpt-4`).           | `chonkie[tiktoken]`   |
| `transformers` | Load tokenizers via `AutoTokenizer` from HF `transformers`.    | `chonkie[neural]`     |

`default` indicates that the feature is available with the default `pip install chonkie`.

To use a custom token counter, you can pass in any function that takes a string and returns an integer! Something like this:

```python
def custom_token_counter(text: str) -> int:
    return len(text)

chunker = RecursiveChunker(tokenizer=custom_token_counter)
```

You can use this to extend Chonkie to support any tokenization scheme you want!

</details>

<details>
<summary><strong>🧠 Embed like a boss! Chonkie links up with 16+ embedding pals!</strong></summary>

Seamlessly works with various embedding model providers. Bring your favorite embeddings to the CHONK party! Use `AutoEmbeddings` to load models easily.

| Provider / Alias        | Class                           | Description                            | Optional Install        |
| ----------------------- | ------------------------------- | -------------------------------------- | ----------------------- |
| `model2vec`             | `Model2VecEmbeddings`           | Use `Model2Vec` models.                | `chonkie[model2vec]`    |
| `sentence-transformers` | `SentenceTransformerEmbeddings` | Use any `sentence-transformers` model. | `chonkie[st]`           |
| `openai`                | `OpenAIEmbeddings`              | Use OpenAI's embedding API.            | `chonkie[openai]`       |
| `azure-openai`          | `AzureOpenAIEmbeddings`         | Use Azure OpenAI embedding service.    | `chonkie[azure-openai]` |
| `cohere`                | `CohereEmbeddings`              | Use Cohere's embedding API.            | `chonkie[cohere]`       |
| `gemini`                | `GeminiEmbeddings`              | Use Google's Gemini embedding API.     | `chonkie[gemini]`       |
| `jina`                  | `JinaEmbeddings`                | Use Jina AI's embedding API.           | `chonkie[jina]`         |
| `voyageai`              | `VoyageAIEmbeddings`            | Use Voyage AI's embedding API.         | `chonkie[voyageai]`     |
| `litellm`               | `LiteLLMEmbeddings`             | Use LiteLLM for 100+ embedding models. | `chonkie[litellm]`      |
| `catsu`                 | `CatsuEmbeddings`               | Unified adapter for 11+ providers.     | `chonkie[catsu]`        |
| `mistral`               | `MistralEmbeddings`             | Use Mistral's embedding API.           | `chonkie[catsu]`        |
| `together`              | `TogetherEmbeddings`            | Use Together AI's embedding API.       | `chonkie[catsu]`        |
| `mixedbread`            | `MixedbreadEmbeddings`          | Use Mixedbread's embedding API.        | `chonkie[catsu]`        |
| `nomic`                 | `NomicEmbeddings`               | Use Nomic's embedding API.             | `chonkie[catsu]`        |
| `deepinfra`             | `DeepInfraEmbeddings`           | Use DeepInfra's embedding API.         | `chonkie[catsu]`        |
| `cloudflare`            | `CloudflareEmbeddings`          | Use Cloudflare Workers AI embeddings.  | `chonkie[catsu]`        |

</details>

<details>
<summary><strong>🧞‍♂️ Power Up with Genies! Chonkie supports 5+ LLM providers!</strong></summary>

Genies provide interfaces to interact with Large Language Models (LLMs) for advanced chunking strategies or other tasks within the pipeline.

| Genie Name     | Class              | Description                                | Optional Install        |
| -------------- | ------------------ | ------------------------------------------ | ----------------------- |
| `gemini`       | `GeminiGenie`      | Interact with Google Gemini APIs.          | `chonkie[gemini]`       |
| `openai`       | `OpenAIGenie`      | Interact with OpenAI APIs.                 | `chonkie[openai]`       |
| `azure-openai` | `AzureOpenAIGenie` | Interact with Azure OpenAI APIs.           | `chonkie[azure-openai]` |
| `groq`         | `GroqGenie`        | Fast inference on Groq hardware.           | `chonkie[groq]`         |
| `cerebras`     | `CerebrasGenie`    | Fastest inference on Cerebras hardware.    | `chonkie[cerebras]`     |

You can also use the `OpenAIGenie` to interact with any LLM provider that supports the OpenAI API format, by simply changing the `model`, `base_url`, and `api_key` parameters. For example, here's how to use the `OpenAIGenie` to interact with the `Llama-4-Maverick` model via OpenRouter:

```python
from chonkie import OpenAIGenie

genie = OpenAIGenie(model="meta-llama/llama-4-maverick",
                    base_url="https://openrouter.ai/api/v1",
                    api_key="your_api_key")
```

Or, here's how to use the `OpenAIGenie` to interact with the `gpt-4o-mini` model via Requesty:

```python
from chonkie import OpenAIGenie

genie = OpenAIGenie(model="openai/gpt-4o-mini",
                    base_url="https://router.requesty.ai/v1",
                    api_key="your_api_key")
```

</details>

<details>
<summary><strong>🛠️ Utilities & Helpers! Chonkie includes handy tools!</strong></summary>

Additional utilities to enhance your chunking workflow.

| Utility Name | Class        | Description                                    | Optional Install |
| ------------ | ------------ | ---------------------------------------------- | ---------------- |
| `hub`        | `Hubbie`     | Simple wrapper for HuggingFace Hub operations. | `chonkie[hub]`   |
| `viz`        | `Visualizer` | Rich console visualizations for chunks.        | `chonkie[viz]`   |

</details>

With Chonkie's wide range of integrations, you can easily plug it into your existing infrastructure and start CHONKING!

## 🤖 AI Agent Skills & Plugins

Chonkie provides an official skill and plugin for AI coding agents, giving them deep knowledge of Chonkie's API, chunking strategies, and pipeline patterns — so they can help you build RAG pipelines faster.

**Supported agents:** Claude Code, Cursor, Gemini CLI, and more.

```bash
# Via skills.sh (works with Claude Code, Cursor, Copilot, and 20+ agents)
npx skills add chonkie-inc/skills

# Claude Code only
/plugin marketplace add chonkie-inc/skills
```

Once installed, your agent gains knowledge of all chunkers, the Pipeline API, tokenizer selection, embeddings refineries, vector DB handshakes, the REST API server, recipes, and async/batch processing patterns.

Learn more at [github.com/chonkie-inc/skills](https://github.com/chonkie-inc/skills).

## 📊 Benchmarks

> "I may be smol hippo, but I pack a big punch!" 🦛

Chonkie is not just cute, it's also fast and efficient! Here's how it stacks up against the competition:

**Size**📦

- **Wheel Size:** 505KB (vs 1-12MB for alternatives)
- **Installed Size:** 49MB (vs 80-171MB for alternatives)
- **With Semantic:** Still 10x lighter than the closest competition!

**Speed**⚡

- **Token Chunking:** 33x faster than the slowest alternative
- **Sentence Chunking:** Almost 2x faster than competitors
- **Semantic Chunking:** Up to 2.5x faster than others

Check out our detailed [benchmarks](BENCHMARKS.md) to see how Chonkie races past the competition! 🏃‍♂️💨

## 🤝 Contributing

Want to help grow Chonkie? Check out [CONTRIBUTING.md](CONTRIBUTING.md) to get started! Whether you're fixing bugs, adding features, or improving docs, every contribution helps make Chonkie a better CHONK for everyone.

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
  howpublished = {\url{https://github.com/chonkie-inc/chonkie}},
}
```
