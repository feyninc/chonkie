"""Tests for the TokenChunker class."""

from __future__ import annotations

from typing import cast

import pytest
import tiktoken
from tiktoken import Encoding
from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from chonkie import Chunk, TokenChunker


@pytest.fixture
def tiktokenizer() -> Encoding:
    """Fixture that returns a GPT-2 tokenizer from the tiktoken library."""
    return tiktoken.get_encoding("gpt2")


@pytest.fixture
def transformers_tokenizer() -> PreTrainedTokenizerFast:
    """Fixture that returns a GPT-2 tokenizer from the transformers library."""
    try:
        return cast(PreTrainedTokenizerFast, AutoTokenizer.from_pretrained("gpt2"))
    except (OSError, ValueError) as e:
        pytest.skip(f"Could not load HuggingFace tokenizer: {e}")


@pytest.fixture
def tokenizer() -> Tokenizer:
    """Fixture that returns a GPT-2 tokenizer from the tokenizers library."""
    try:
        return Tokenizer.from_pretrained("gpt2")
    except (OSError, ValueError) as e:
        pytest.skip(f"Could not load tokenizers tokenizer: {e}")


@pytest.fixture
def sample_text() -> str:
    """Fixture that returns a sample text for testing the TokenChunker."""
    text = """According to all known laws of aviation, there is no way a bee should be able to fly. Its wings are too small to get its fat little body off the ground. The bee, of course, flies anyway because bees don't care what humans think is impossible. Yellow, black. Yellow, black. Yellow, black. Yellow, black. Ooh, black and yellow! Let's shake it up a little. Barry! Breakfast is ready! Coming! Hang on a second. Hello? - Barry? - Adam? - Can you believe this is happening? - I can't. I'll pick you up. Looking sharp. Use the stairs. Your father paid good money for those. Sorry. I'm excited. Here's the graduate. We're very proud of you, son. A perfect report card, all B's. Very proud. Ma! I got a thing going here."""
    return text


@pytest.fixture
def sample_batch(sample_text: str) -> list[str]:
    """Fixture that returns a sample batch of 10 texts (500-1000 tokens each) for testing."""
    batch = []
    base_text = sample_text + " "  # Add space for separation when repeating

    # Create 10 texts with varying lengths within the range
    for i in range(10):
        repeats = 4 + (i % 3)  # Cycle through 4, 5, 6 repeats
        batch.append(base_text * repeats)

    return batch


@pytest.fixture
def sample_complex_markdown_text() -> str:
    """Fixture that returns a sample markdown text with complex formatting."""
    text = """# Heading 1
    This is a paragraph with some **bold text** and _italic text_. 
    ## Heading 2
    - Bullet point 1
    - Bullet point 2 with `inline code`
    ```python
    # Code block
    def hello_world():
        print("Hello, world!")
    ```
    Another paragraph with [a link](https://example.com) and an image:
    ![Alt text](https://example.com/image.jpg)
    > A blockquote with multiple lines
    > that spans more than one line.
    Finally, a paragraph at the end.
    """
    return text


def test_token_chunker_initialization_tok(tokenizer: Tokenizer) -> None:
    """Test that the TokenChunker can be initialized with a tokenizer."""
    chunker = TokenChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=128)

    assert chunker is not None
    assert chunker.tokenizer.tokenizer == tokenizer
    assert chunker.chunk_size == 512
    assert chunker.chunk_overlap == 128


def test_token_chunker_initialization_hftok(
    transformers_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """Test that the TokenChunker can be initialized with a tokenizer."""
    chunker = TokenChunker(tokenizer=transformers_tokenizer, chunk_size=512, chunk_overlap=128)

    assert chunker is not None
    assert chunker.tokenizer.tokenizer == transformers_tokenizer
    assert chunker.chunk_size == 512
    assert chunker.chunk_overlap == 128


def test_token_chunker_initialization_tik(tiktokenizer: Encoding) -> None:
    """Test that the TokenChunker can be initialized with a tokenizer."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)

    assert chunker is not None
    assert chunker.tokenizer.tokenizer == tiktokenizer
    assert chunker.chunk_size == 512
    assert chunker.chunk_overlap == 128


def test_token_chunker_chunking(tiktokenizer: Encoding, sample_text: str) -> None:
    """Test that the TokenChunker can chunk a sample text into tokens."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk(sample_text)

    assert len(chunks) > 0
    assert type(chunks[0]) is Chunk
    assert all([chunk.token_count <= 512 for chunk in chunks])
    assert all([chunk.token_count > 0 for chunk in chunks])
    assert all([chunk.text is not None for chunk in chunks])
    assert all([chunk.start_index is not None for chunk in chunks])
    assert all([chunk.end_index is not None for chunk in chunks])


def test_token_chunker_chunking_hf(
    transformers_tokenizer: PreTrainedTokenizerFast,
    sample_text: str,
) -> None:
    """Test that the TokenChunker can chunk a sample text into tokens."""
    chunker = TokenChunker(tokenizer=transformers_tokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk(sample_text)

    assert len(chunks) > 0
    assert type(chunks[0]) is Chunk
    assert all([chunk.token_count <= 512 for chunk in chunks])
    assert all([chunk.token_count > 0 for chunk in chunks])
    assert all([chunk.text is not None for chunk in chunks])
    assert all([chunk.start_index is not None for chunk in chunks])
    assert all([chunk.end_index is not None for chunk in chunks])


def test_token_chunker_chunking_tik(tiktokenizer: Encoding, sample_text: str) -> None:
    """Test that the TokenChunker can chunk a sample text into tokens."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk(sample_text)

    assert len(chunks) > 0
    assert type(chunks[0]) is Chunk
    assert all([chunk.token_count <= 512 for chunk in chunks])
    assert all([chunk.token_count > 0 for chunk in chunks])
    assert all([chunk.text is not None for chunk in chunks])
    assert all([chunk.start_index is not None for chunk in chunks])
    assert all([chunk.end_index is not None for chunk in chunks])


def test_token_chunker_empty_text(tiktokenizer: Encoding) -> None:
    """Test that the TokenChunker can handle empty text input."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk("")

    assert len(chunks) == 0


def test_token_chunker_single_token_text(tokenizer: Tokenizer) -> None:
    """Test that the TokenChunker can handle text with a single token."""
    chunker = TokenChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk("Hello")

    assert len(chunks) == 1
    assert chunks[0].token_count == 1
    assert chunks[0].text == "Hello"


def test_token_chunker_single_token_text_hf(
    transformers_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """Test that the TokenChunker can handle text with a single token."""
    chunker = TokenChunker(tokenizer=transformers_tokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk("Hello")

    assert len(chunks) == 1
    assert chunks[0].token_count == 1
    assert chunks[0].text == "Hello"


def test_token_chunker_single_token_text_tik(tiktokenizer: Encoding) -> None:
    """Test that the TokenChunker can handle text with a single token."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk("Hello")

    assert len(chunks) == 1
    assert chunks[0].token_count == 1
    assert chunks[0].text == "Hello"


def test_token_chunker_single_chunk_text(tokenizer: Tokenizer) -> None:
    """Test that the TokenChunker can handle text that fits within a single chunk."""
    chunker = TokenChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk("Hello, how are you?")

    assert len(chunks) == 1
    assert chunks[0].token_count == 6
    assert chunks[0].text == "Hello, how are you?"


def test_token_chunker_batch_chunking(tiktokenizer: Encoding, sample_batch: list[str]) -> None:
    """Test that the TokenChunker can chunk a batch of texts into tokens."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk_batch(sample_batch)

    assert len(chunks) > 0
    assert all([len(chunk) > 0 for chunk in chunks])
    assert all([type(chunk[0]) is Chunk for chunk in chunks])
    assert all([all([chunk.token_count <= 512 for chunk in chunks]) for chunks in chunks])
    assert all([all([chunk.token_count > 0 for chunk in chunks]) for chunks in chunks])
    assert all([all([chunk.text is not None for chunk in chunks]) for chunks in chunks])
    assert all([all([chunk.start_index is not None for chunk in chunks]) for chunks in chunks])
    assert all([all([chunk.end_index is not None for chunk in chunks]) for chunks in chunks])


def test_token_chunker_repr(tiktokenizer: Encoding) -> None:
    """Test that the TokenChunker has a string representation."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)

    assert repr(chunker) == (
        f"TokenChunker(tokenizer={chunker.tokenizer}, "
        f"chunk_size={chunker.chunk_size}, "
        f"chunk_overlap={chunker.chunk_overlap})"
    )


def test_token_chunker_call(tiktokenizer: Encoding, sample_text: str) -> None:
    """Test that the TokenChunker can be called directly."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker(sample_text)

    assert len(chunks) > 0
    assert type(chunks[0]) is Chunk
    assert all([chunk.token_count <= 512 for chunk in chunks])
    assert all([chunk.token_count > 0 for chunk in chunks])
    assert all([chunk.text is not None for chunk in chunks])
    assert all([chunk.start_index is not None for chunk in chunks])
    assert all([chunk.end_index is not None for chunk in chunks])


def verify_chunk_indices(chunks: list[Chunk], original_text: str):
    """Verify that chunk indices correctly map to the original text."""
    for i, chunk in enumerate(chunks):
        # Extract text using the indices
        extracted_text = original_text[chunk.start_index : chunk.end_index]
        # Exact match without stripping to catch whitespace issues
        assert chunk.text == extracted_text, (
            f"Chunk {i} text mismatch:\n"
            f"Chunk text: '{chunk.text}'\n"
            f"Extracted text: '{extracted_text}'\n"
            f"Indices: [{chunk.start_index}:{chunk.end_index}]"
        )


def test_token_chunker_indices(sample_text: str) -> None:
    """Test that TokenChunker's indices correctly map to original text."""
    chunker = TokenChunker(tokenizer="character", chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk(sample_text)
    verify_chunk_indices(chunks, sample_text)


def test_token_chunker_indices_complex_md(sample_complex_markdown_text: str) -> None:
    """Test that TokenChunker's indices correctly map to original text."""
    chunker = TokenChunker(tokenizer="character", chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk(sample_complex_markdown_text)
    verify_chunk_indices(chunks, sample_complex_markdown_text)


def test_token_chunker_token_counts(tiktokenizer: Encoding, sample_text: str) -> None:
    """Test that the TokenChunker correctly calculates token counts."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk(sample_text)
    assert all([chunk.token_count > 0 for chunk in chunks]), (
        "All chunks must have a positive token count"
    )
    assert all([chunk.token_count <= 512 for chunk in chunks]), (
        "All chunks must have a token count less than or equal to 512"
    )

    token_counts = [len(tiktokenizer.encode(chunk.text)) for chunk in chunks]
    assert all([
        chunk.token_count == token_count for chunk, token_count in zip(chunks, token_counts)
    ]), "All chunks must have a token count equal to the length of the encoded text"


def test_token_chunker_indices_batch(tiktokenizer: Encoding, sample_text: str) -> None:
    """Test that TokenChunker's indices correctly map to original text."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk_batch([sample_text] * 10)[-1]
    verify_chunk_indices(chunks, sample_text)


def test_token_chunker_return_type(tiktokenizer: Encoding, sample_text: str) -> None:
    """Test that TokenChunker returns Chunk objects by default."""
    chunker = TokenChunker(
        tokenizer=tiktokenizer,
        chunk_size=512,
        chunk_overlap=128,
    )
    chunks = chunker.chunk(sample_text)
    assert all([type(chunk) is Chunk for chunk in chunks])
    assert all([len(tiktokenizer.encode(chunk.text)) <= 512 for chunk in chunks])


def test_token_chunker_rejects_int_overlap_equal_to_chunk_size() -> None:
    """An int chunk_overlap equal to chunk_size is invalid (existing behavior)."""
    with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
        TokenChunker(tokenizer="character", chunk_size=10, chunk_overlap=10)


def test_token_chunker_rejects_float_overlap_resolving_to_chunk_size() -> None:
    """A float chunk_overlap of 1.0 resolves to chunk_size and must be rejected.

    Previously only the int path was validated, so this constructed a chunker with
    a zero stride and chunk() raised 'range() arg 3 must not be zero'.
    """
    with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
        TokenChunker(tokenizer="character", chunk_size=10, chunk_overlap=1.0)


def test_token_chunker_rejects_float_overlap_exceeding_chunk_size() -> None:
    """A float chunk_overlap > 1.0 resolves above chunk_size and must be rejected.

    Previously this produced a negative stride and chunk() silently returned zero
    chunks, dropping the entire input text.
    """
    with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
        TokenChunker(tokenizer="character", chunk_size=10, chunk_overlap=1.5)


def test_token_chunker_accepts_valid_fractional_overlap_and_chunks() -> None:
    """A valid fractional float overlap resolves correctly and chunks non-empty text."""
    chunker = TokenChunker(tokenizer="character", chunk_size=10, chunk_overlap=0.5)
    assert chunker.chunk_overlap == 5
    text = "x" * 40
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    assert "".join(dict.fromkeys([c.text[0] for c in chunks])) == "x"
    # Every original character is covered by the first chunk's start through the last chunk's end.
    assert chunks[0].start_index == 0
    assert chunks[-1].end_index == len(text)
