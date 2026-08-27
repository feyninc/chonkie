"""Tests for the TokenChunker class."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import pytest
import tiktoken
from tiktoken import Encoding
from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from chonkie import Chunk, TokenChunker
from chonkie.tokenizer import Tokenizer as ChonkieTokenizer
from chonkie.tokenizer import TokenizerEncoding, _byte_offsets_to_char_offsets

if TYPE_CHECKING:
    from tokie import Tokenizer as TokieTokenizer


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
def tokie_tokenizer() -> "TokieTokenizer":
    """Fixture that returns a byte-level GPT-2 tokenizer from tokie."""
    tokie = pytest.importorskip("tokie", reason="tokie not installed")
    try:
        return tokie.Tokenizer.from_pretrained("gpt2")
    except (OSError, RuntimeError, ValueError) as e:
        pytest.skip(f"Could not load tokie tokenizer: {e}")


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


class CountingTokenizer(ChonkieTokenizer):
    """Small tokenizer double for observing batch and offset encoding calls."""

    def __init__(self, offset_texts: set[str]) -> None:
        """Initialize the tokenizer with texts that support offsets."""
        super().__init__()
        self.offset_texts = offset_texts
        self.offset_calls: list[str] = []
        self.batch_calls: list[list[str]] = []

    def __repr__(self) -> str:
        """Return the tokenizer representation."""
        return "CountingTokenizer()"

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into individual characters."""
        return list(text)

    def encode(self, text: str) -> list[int]:
        """Encode each character as its Unicode code point."""
        return [ord(character) for character in text]

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        """Record and encode a batch of texts."""
        self.batch_calls.append(list(texts))
        return [self.encode(text) for text in texts]

    def encode_with_offsets(self, text: str) -> TokenizerEncoding | None:
        """Return character offsets for configured texts only."""
        self.offset_calls.append(text)
        if text not in self.offset_texts:
            return None
        return TokenizerEncoding(
            ids=self.encode(text),
            offsets=[(index, index + 1) for index in range(len(text))],
        )

    def decode(self, tokens: Sequence[int]) -> str:
        """Decode Unicode code points back into text."""
        return "".join(chr(token) for token in tokens)


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


def test_token_chunker_batch_avoids_duplicate_offset_encoding() -> None:
    """Batch chunking should not re-encode texts that provide offsets."""
    offset_text = "offset"
    fallback_text = "fallback"
    tokenizer = CountingTokenizer({offset_text})
    chunker = TokenChunker(tokenizer=tokenizer, chunk_size=2, chunk_overlap=0)

    chunks = chunker.chunk_batch([offset_text, fallback_text], show_progress_bar=False)

    assert tokenizer.offset_calls == [offset_text, fallback_text]
    assert tokenizer.batch_calls == [[fallback_text]]
    assert [chunk.text for chunk in chunks[0]] == ["of", "fs", "et"]
    assert [chunk.text for chunk in chunks[1]] == ["fa", "ll", "ba", "ck"]


def test_byte_offsets_to_char_offsets_expands_partial_multibyte_tokens() -> None:
    """Byte offsets split inside UTF-8 characters should expand to valid character spans."""
    text = "a🩺b"

    offsets = _byte_offsets_to_char_offsets(
        text,
        [
            (0, 2),
            (2, 4),
            (4, 6),
        ],
    )

    assert offsets == [(0, 2), (1, 2), (1, 3)]


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


def test_token_chunker_multibyte_offsets(
    tokie_tokenizer: "TokieTokenizer",
) -> None:
    """Test that byte-level token boundaries preserve multi-byte characters."""
    text = "a🩺 hello world"
    chunker = TokenChunker(tokenizer=tokie_tokenizer, chunk_size=2, chunk_overlap=0)

    chunks = chunker.chunk(text)
    batch_chunks = chunker.chunk_batch([text], show_progress_bar=False)[0]

    expected = [("a🩺", 0, 2), (" hello world", 2, 14)]

    for result in (chunks, batch_chunks):
        assert [(chunk.text, chunk.start_index, chunk.end_index) for chunk in result] == expected
        assert result
        assert all(chunk.text for chunk in result)
        assert all(chunk.end_index > chunk.start_index for chunk in result)
        assert all(chunk.text == text[chunk.start_index : chunk.end_index] for chunk in result)
        assert "".join(chunk.text for chunk in result) == text


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


def test_token_chunker_float_overlap_out_of_range() -> None:
    """A float chunk_overlap that resolves to >= chunk_size must be rejected.

    A float is treated as a fraction of chunk_size, so chunk_overlap=1.0 with
    chunk_size=100 resolves to an overlap of 100. That used to construct fine
    and then either crash chunk() with "range() arg 3 must not be zero" (step
    of zero) or silently drop the text (negative step). It should raise the
    same ValueError the int path already raises.
    """
    with pytest.raises(ValueError):
        TokenChunker(tokenizer="character", chunk_size=100, chunk_overlap=1.0)
    with pytest.raises(ValueError):
        TokenChunker(tokenizer="character", chunk_size=100, chunk_overlap=1.5)


def test_token_chunker_negative_overlap() -> None:
    """A negative chunk_overlap must be rejected instead of skipping tokens."""
    with pytest.raises(ValueError):
        TokenChunker(tokenizer="character", chunk_size=100, chunk_overlap=-0.5)
    # A small negative fraction resolves to int() == 0, so it must be rejected
    # on the input sign rather than the resolved token count.
    with pytest.raises(ValueError):
        TokenChunker(tokenizer="character", chunk_size=100, chunk_overlap=-0.001)


def test_token_chunker_float_overlap_valid() -> None:
    """A valid fractional overlap still resolves to a token count and chunks."""
    chunker = TokenChunker(tokenizer="character", chunk_size=100, chunk_overlap=0.2)
    assert chunker.chunk_overlap == 20
    chunks = chunker.chunk("hello world " * 50)
    assert len(chunks) > 0
