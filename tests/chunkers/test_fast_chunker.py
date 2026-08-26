"""Tests for the FastChunker class."""

from __future__ import annotations

import pytest

from chonkie import Chunk


@pytest.fixture
def sample_text() -> str:
    """Fixture that returns a sample text for testing the FastChunker."""
    text = """According to all known laws of aviation, there is no way a bee should be able to fly. Its wings are too small to get its fat little body off the ground. The bee, of course, flies anyway because bees don't care what humans think is impossible. Yellow, black. Yellow, black. Yellow, black. Yellow, black. Ooh, black and yellow! Let's shake it up a little. Barry! Breakfast is ready! Coming! Hang on a second. Hello? - Barry? - Adam? - Can you believe this is happening? - I can't. I'll pick you up. Looking sharp. Use the stairs. Your father paid good money for those. Sorry. I'm excited. Here's the graduate. We're very proud of you, son. A perfect report card, all B's. Very proud. Ma! I got a thing going here."""
    return text


@pytest.fixture
def sample_batch(sample_text: str) -> list[str]:
    """Fixture that returns a sample batch of texts for testing."""
    batch = []
    base_text = sample_text + " "

    for i in range(10):
        repeats = 4 + (i % 3)
        batch.append(base_text * repeats)

    return batch


@pytest.fixture
def paragraph_text() -> str:
    """Fixture that returns text with paragraph breaks."""
    return """First paragraph with multiple sentences.
This is still the first paragraph.

Second paragraph starts here.
More content in the second paragraph.

Third paragraph with its own content."""


def get_fast_chunker(**kwargs):  # type: ignore[no-untyped-def]
    """Import and create a FastChunker instance."""
    from chonkie import FastChunker

    return FastChunker(**kwargs)


def test_fast_chunker_initialization() -> None:
    """Test that the FastChunker can be initialized with default parameters."""
    chunker = get_fast_chunker()

    assert chunker is not None
    assert chunker.chunk_size == 4096
    assert chunker.delimiters == "\n.?"
    assert chunker.pattern is None
    assert chunker.prefix is False
    assert chunker.consecutive is False
    assert chunker.forward_fallback is False


def test_fast_chunker_initialization_custom() -> None:
    """Test that the FastChunker can be initialized with custom parameters."""
    chunker = get_fast_chunker(
        chunk_size=1024,
        delimiters=".!?",
        prefix=True,
        consecutive=True,
        forward_fallback=True,
    )

    assert chunker.chunk_size == 1024
    assert chunker.delimiters == ".!?"
    assert chunker.prefix is True
    assert chunker.consecutive is True
    assert chunker.forward_fallback is True


def test_fast_chunker_initialization_with_pattern() -> None:
    """Test that the FastChunker can be initialized with a pattern."""
    chunker = get_fast_chunker(chunk_size=2048, pattern="▁")

    assert chunker.chunk_size == 2048
    assert chunker.pattern == "▁"


def test_fast_chunker_chunking(sample_text: str) -> None:
    """Test that the FastChunker can chunk a sample text."""
    chunker = get_fast_chunker(chunk_size=256)
    chunks = chunker.chunk(sample_text)

    assert len(chunks) > 0
    assert type(chunks[0]) is Chunk
    assert all(len(chunk.text.encode("utf-8")) <= 256 for chunk in chunks)
    assert all(chunk.text is not None for chunk in chunks)
    assert all(chunk.start_index is not None for chunk in chunks)
    assert all(chunk.end_index is not None for chunk in chunks)


def test_fast_chunker_token_count_always_zero(sample_text: str) -> None:
    """Test that FastChunker always returns token_count=0."""
    chunker = get_fast_chunker(chunk_size=256)
    chunks = chunker.chunk(sample_text)

    assert all(chunk.token_count == 0 for chunk in chunks)


def test_fast_chunker_empty_text() -> None:
    """Test that the FastChunker can handle empty text input."""
    chunker = get_fast_chunker(chunk_size=512)
    chunks = chunker.chunk("")

    assert len(chunks) == 0


def test_fast_chunker_single_word() -> None:
    """Test that the FastChunker can handle text with a single word."""
    chunker = get_fast_chunker(chunk_size=512)
    chunks = chunker.chunk("Hello")

    assert len(chunks) == 1
    assert chunks[0].text == "Hello"
    assert chunks[0].token_count == 0


def test_fast_chunker_text_smaller_than_chunk_size() -> None:
    """Test that the FastChunker handles text smaller than chunk_size."""
    chunker = get_fast_chunker(chunk_size=4096)
    chunks = chunker.chunk("Hello, how are you?")

    assert len(chunks) == 1
    assert chunks[0].text == "Hello, how are you?"


def test_fast_chunker_sentence_delimiters(sample_text: str) -> None:
    """Test chunking with sentence-ending delimiters."""
    chunker = get_fast_chunker(chunk_size=256, delimiters=".!?")
    chunks = chunker.chunk(sample_text)

    assert len(chunks) > 0
    # Most chunks should end with a delimiter (except possibly the last one)
    for chunk in chunks[:-1]:
        text = chunk.text.rstrip()
        assert text[-1] in ".!?" or len(chunk.text.encode("utf-8")) == 256


def test_fast_chunker_newline_delimiters(paragraph_text: str) -> None:
    """Test chunking with newline delimiters."""
    chunker = get_fast_chunker(chunk_size=256, delimiters="\n")
    chunks = chunker.chunk(paragraph_text)

    assert len(chunks) > 0


def test_fast_chunker_prefix_option() -> None:
    """Test that prefix option puts delimiter at start of next chunk."""
    text = "Hello. World. Test."
    chunker_no_prefix = get_fast_chunker(chunk_size=10, delimiters=".")
    chunker_prefix = get_fast_chunker(chunk_size=10, delimiters=".", prefix=True)

    chunks_no_prefix = chunker_no_prefix.chunk(text)
    chunks_prefix = chunker_prefix.chunk(text)

    # With prefix=True, delimiters should appear at start of chunks (except first)
    # With prefix=False, delimiters should appear at end of chunks (except last)
    assert len(chunks_no_prefix) > 0
    assert len(chunks_prefix) > 0


def test_fast_chunker_pattern_option() -> None:
    """Test chunking with a multi-byte pattern."""
    text = "Hello▁World▁this▁is▁a▁test"
    chunker = get_fast_chunker(chunk_size=15, pattern="▁")
    chunks = chunker.chunk(text)

    assert len(chunks) > 0
    # Verify full text is preserved
    reconstructed = "".join(chunk.text for chunk in chunks)
    assert reconstructed == text


def test_fast_chunker_batch_chunking(sample_batch: list[str]) -> None:
    """Test that the FastChunker can chunk a batch of texts."""
    chunker = get_fast_chunker(chunk_size=512)
    chunks = chunker.chunk_batch(sample_batch)

    assert len(chunks) == len(sample_batch)
    assert all(len(doc_chunks) > 0 for doc_chunks in chunks)
    assert all(type(doc_chunks[0]) is Chunk for doc_chunks in chunks)


def test_fast_chunker_repr() -> None:
    """Test that the FastChunker has a correct string representation."""
    chunker = get_fast_chunker(
        chunk_size=1024,
        delimiters=".!?",
        pattern=None,
        prefix=True,
        consecutive=False,
        forward_fallback=True,
    )

    expected = (
        "FastChunker(chunk_size=1024, delimiters='.!?', "
        "pattern=None, prefix=True, "
        "consecutive=False, forward_fallback=True)"
    )
    assert repr(chunker) == expected


def test_fast_chunker_repr_with_pattern() -> None:
    """Test repr with pattern set."""
    chunker = get_fast_chunker(chunk_size=2048, pattern="▁")

    repr_str = repr(chunker)
    assert "pattern='▁'" in repr_str
    assert "chunk_size=2048" in repr_str


def test_fast_chunker_call(sample_text: str) -> None:
    """Test that the FastChunker can be called directly."""
    chunker = get_fast_chunker(chunk_size=256)
    chunks = chunker(sample_text)

    assert len(chunks) > 0
    assert type(chunks[0]) is Chunk


def verify_chunk_indices(chunks: list[Chunk], original_text: str) -> None:
    """Verify that chunk indices correctly map to the original text."""
    for i, chunk in enumerate(chunks):
        extracted_text = original_text[chunk.start_index : chunk.end_index]
        assert chunk.text == extracted_text, (
            f"Chunk {i} text mismatch:\n"
            f"Chunk text: '{chunk.text}'\n"
            f"Extracted text: '{extracted_text}'\n"
            f"Indices: [{chunk.start_index}:{chunk.end_index}]"
        )


def test_fast_chunker_indices(sample_text: str) -> None:
    """Test that FastChunker's indices correctly map to original text."""
    chunker = get_fast_chunker(chunk_size=256)
    chunks = chunker.chunk(sample_text)
    verify_chunk_indices(chunks, sample_text)


def test_fast_chunker_indices_with_pattern() -> None:
    """Test indices with pattern-based chunking."""
    text = "Hello▁World▁this▁is▁a▁test▁sentence"
    chunker = get_fast_chunker(chunk_size=20, pattern="▁")
    chunks = chunker.chunk(text)
    verify_chunk_indices(chunks, text)


def test_fast_chunker_full_text_reconstruction(sample_text: str) -> None:
    """Test that all chunks together reconstruct the original text."""
    chunker = get_fast_chunker(chunk_size=256)
    chunks = chunker.chunk(sample_text)

    reconstructed = "".join(chunk.text for chunk in chunks)
    assert reconstructed == sample_text


def test_fast_chunker_consecutive_delimiters() -> None:
    """Test chunking with consecutive delimiter handling."""
    text = "word1   word2    word3"
    chunker = get_fast_chunker(chunk_size=10, pattern=" ", consecutive=True)
    chunks = chunker.chunk(text)

    assert len(chunks) > 0
    reconstructed = "".join(chunk.text for chunk in chunks)
    assert reconstructed == text


def test_fast_chunker_forward_fallback() -> None:
    """Test forward fallback when no delimiter in backward window."""
    text = "verylongword short more"
    chunker = get_fast_chunker(chunk_size=10, pattern=" ", forward_fallback=True)
    chunks = chunker.chunk(text)

    assert len(chunks) > 0
    reconstructed = "".join(chunk.text for chunk in chunks)
    assert reconstructed == text


def test_fast_chunker_unicode_text() -> None:
    """Test chunking with unicode text."""
    text = "Hello 世界! This is a test with émojis 🦛 and spëcial characters."
    chunker = get_fast_chunker(chunk_size=30)
    chunks = chunker.chunk(text)

    assert len(chunks) > 0
    reconstructed = "".join(chunk.text for chunk in chunks)
    assert reconstructed == text


def test_fast_chunker_multibyte_split_boundary() -> None:
    """Test that a size cut inside a multi-byte UTF-8 char does not raise.

    chonkie-core returns byte offsets, and a hard size cut can fall inside a
    multi-byte character. Slicing the raw bytes there used to raise
    UnicodeDecodeError; offsets are now snapped to character boundaries.
    """
    # chunk_size=2 forces cuts inside every 2-byte "é"; no delimiter is present.
    text = "café café café café "
    chunker = get_fast_chunker(chunk_size=2)
    chunks = chunker.chunk(text)

    assert len(chunks) > 0
    reconstructed = "".join(chunk.text for chunk in chunks)
    assert reconstructed == text
    assert all(chunk.text == text[chunk.start_index : chunk.end_index] for chunk in chunks)

    # A run of 3-byte CJK characters longer than a realistic chunk_size.
    cjk = "这是一段没有分隔符的中文文本" * 20
    cjk_chunks = get_fast_chunker(chunk_size=64).chunk(cjk)
    assert "".join(chunk.text for chunk in cjk_chunks) == cjk


def test_fast_chunker_no_tokenizer_attribute() -> None:
    """Test that FastChunker has _tokenizer set to None."""
    chunker = get_fast_chunker()
    assert chunker._tokenizer is None
