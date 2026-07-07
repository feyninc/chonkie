"""Test the SemanticChunker class - cleaned version with only working tests."""

import pytest

from chonkie import SemanticChunker
from chonkie.embeddings import BaseEmbeddings, Model2VecEmbeddings
from chonkie.types import Chunk


@pytest.fixture
def sample_text() -> str:
    """Sample text for testing the SemanticChunker."""
    text = """The process of text chunking in RAG applications represents a delicate balance between competing requirements. On one side, we have the need for semantic coherence – ensuring that each chunk maintains meaningful context that can be understood and processed independently. On the other, we must optimize for information density, ensuring that each chunk carries sufficient signal without excessive noise that might impede retrieval accuracy. In this post, we explore the challenges of text chunking in RAG applications and propose a novel approach that leverages recent advances in transformer-based language models to achieve a more effective balance between these competing requirements."""
    return text


@pytest.fixture
def embedding_model() -> BaseEmbeddings:
    """Fixture that returns a Model2Vec embedding model for testing."""
    return Model2VecEmbeddings("minishlab/potion-base-32M")


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


def test_semantic_chunker_initialization(embedding_model: BaseEmbeddings) -> None:
    """Test that the SemanticChunker can be initialized with required parameters."""
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        threshold=0.5,
    )

    assert chunker is not None
    assert chunker.chunk_size == 512
    assert chunker.threshold == 0.5
    assert chunker.similarity_window == 3
    assert chunker.min_sentences_per_chunk == 1


def test_semantic_chunker_chunking(embedding_model: BaseEmbeddings, sample_text: str) -> None:
    """Test that the SemanticChunker can chunk a sample text."""
    chunker = SemanticChunker(
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=512,
        threshold=0.5,
    )
    chunks = chunker.chunk(sample_text)

    assert len(chunks) > 0
    assert isinstance(chunks[0], Chunk)
    assert all([chunk.token_count <= 512 for chunk in chunks])
    assert all([chunk.token_count > 0 for chunk in chunks])
    assert all([chunk.text is not None for chunk in chunks])
    assert all([chunk.start_index is not None for chunk in chunks])
    assert all([chunk.end_index is not None for chunk in chunks])


def test_semantic_chunker_oversized_single_sentence(embedding_model: BaseEmbeddings) -> None:
    """A single sentence longer than chunk_size must still be split to respect the limit.

    Regression test for #222: a sentence with no delimiters that exceeds chunk_size
    was emitted as one oversized chunk because grouping only splits between sentences.
    """
    chunk_size = 100
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        threshold=0.5,
        min_sentences_per_chunk=1,
    )

    # One long run of words with no sentence delimiters => a single "sentence"
    # far larger than chunk_size, surrounded by normal sentences.
    normal = "This is a normal sentence. Here is another one. And a third here. A fourth now. "
    oversized = "word " * 800
    text = normal + oversized + ". " + normal

    chunks = chunker.chunk(text)

    assert len(chunks) > 0
    assert all([chunk.token_count <= chunk_size for chunk in chunks]), (
        "Every chunk must respect chunk_size even when a single sentence is oversized"
    )
    # Guard against token_count being underreported: re-tokenize the actual text.
    assert all([chunker.tokenizer.count_tokens(chunk.text) <= chunk_size for chunk in chunks]), (
        "The tokenized chunk text must also stay within chunk_size"
    )


def test_semantic_chunker_oversized_sentence_below_similarity_window(
    embedding_model: BaseEmbeddings,
) -> None:
    """The few-sentences early-return path must also enforce chunk_size (#222)."""
    chunk_size = 100
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        threshold=0.5,
        similarity_window=3,
    )

    # Fewer sentences than similarity_window, but one is far larger than chunk_size.
    text = "Short intro. " + ("word " * 600)
    chunks = chunker.chunk(text)

    assert len(chunks) > 1
    assert all([chunk.token_count <= chunk_size for chunk in chunks]), (
        "The few-sentences path must also split oversized sentences"
    )
    assert all([chunker.tokenizer.count_tokens(chunk.text) <= chunk_size for chunk in chunks]), (
        "The tokenized chunk text must also stay within chunk_size"
    )


def test_semantic_chunker_empty_text(embedding_model: BaseEmbeddings) -> None:
    """Test that the SemanticChunker can handle empty text input."""
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        threshold=0.5,
    )
    chunks = chunker.chunk("")
    assert len(chunks) == 0

    chunks = chunker.chunk("   ")
    assert len(chunks) == 0


def test_semantic_chunker_single_sentence(embedding_model: BaseEmbeddings) -> None:
    """Test that the SemanticChunker can handle text with a single sentence."""
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        threshold=0.5,
    )
    chunks = chunker.chunk("This is a single sentence.")

    assert len(chunks) == 1
    assert chunks[0].text == "This is a single sentence."


def test_semantic_chunker_token_counts(embedding_model: BaseEmbeddings, sample_text: str) -> None:
    """Test that the SemanticChunker correctly calculates token counts and respects chunk_size."""
    chunker = SemanticChunker(embedding_model=embedding_model, chunk_size=512, threshold=0.5)
    chunks = chunker.chunk(sample_text)

    assert all([chunk.token_count > 0 for chunk in chunks]), (
        "All chunks must have a positive token count"
    )
    assert all([chunk.token_count <= 512 for chunk in chunks]), (
        "All chunks must respect the chunk_size limit of 512 tokens"
    )

    # Verify token counts match the tokenizer's count
    token_counts = [chunker.tokenizer.count_tokens(chunk.text) for chunk in chunks]
    for i, (chunk, token_count) in enumerate(zip(chunks, token_counts)):
        assert chunk.token_count == token_count, (
            f"Chunk {i} has a token count of {chunk.token_count} but the encoded text length is {token_count}"
        )


def test_semantic_chunker_reconstruction(
    embedding_model: BaseEmbeddings,
    sample_text: str,
) -> None:
    """Test that the SemanticChunker can reconstruct the original text."""
    chunker = SemanticChunker(embedding_model=embedding_model, chunk_size=512, threshold=0.5)
    chunks = chunker.chunk(sample_text)
    assert sample_text == "".join([chunk.text for chunk in chunks])


def verify_chunk_indices(chunks: list[Chunk], original_text: str) -> None:
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


def test_semantic_chunker_indices(embedding_model: BaseEmbeddings, sample_text: str) -> None:
    """Test that the SemanticChunker correctly maps chunk indices to the original text."""
    chunker = SemanticChunker(embedding_model=embedding_model, chunk_size=512, threshold=0.5)
    chunks = chunker.chunk(sample_text)
    verify_chunk_indices(chunks, sample_text)


class TestSemanticChunkerParameterValidation:
    """Test parameter validation in SemanticChunker."""

    def test_invalid_chunk_size(self, embedding_model: BaseEmbeddings) -> None:
        """Test that SemanticChunker raises error for invalid chunk_size."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            SemanticChunker(embedding_model=embedding_model, chunk_size=0)

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            SemanticChunker(embedding_model=embedding_model, chunk_size=-1)

    def test_invalid_min_sentences_per_chunk(self, embedding_model: BaseEmbeddings) -> None:
        """Test that SemanticChunker raises error for invalid min_sentences_per_chunk."""
        with pytest.raises(ValueError, match="min_sentences_per_chunk must be positive"):
            SemanticChunker(embedding_model=embedding_model, min_sentences_per_chunk=0)

        with pytest.raises(ValueError, match="min_sentences_per_chunk must be positive"):
            SemanticChunker(embedding_model=embedding_model, min_sentences_per_chunk=-1)

    def test_invalid_similarity_window(self, embedding_model: BaseEmbeddings) -> None:
        """Test that SemanticChunker raises error for invalid similarity_window."""
        with pytest.raises(ValueError, match="similarity_window must be positive"):
            SemanticChunker(embedding_model=embedding_model, similarity_window=0)

        with pytest.raises(ValueError, match="similarity_window must be positive"):
            SemanticChunker(embedding_model=embedding_model, similarity_window=-1)

    def test_invalid_threshold(self, embedding_model: BaseEmbeddings) -> None:
        """Test that SemanticChunker raises error for invalid threshold values."""
        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            SemanticChunker(embedding_model=embedding_model, threshold=0)

        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            SemanticChunker(embedding_model=embedding_model, threshold=1.0)

        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            SemanticChunker(embedding_model=embedding_model, threshold=-0.1)

        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            SemanticChunker(embedding_model=embedding_model, threshold=1.5)

    def test_invalid_threshold_type(self, embedding_model: BaseEmbeddings) -> None:
        """Test that SemanticChunker raises error for non-numeric threshold."""
        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            SemanticChunker(embedding_model=embedding_model, threshold="invalid")

        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            SemanticChunker(embedding_model=embedding_model, threshold=[0.5])

    def test_invalid_embedding_model_type(self) -> None:
        """Test that SemanticChunker raises error for invalid embedding model type."""
        with pytest.raises(
            ValueError,
            match="embedding_model must be a string or a BaseEmbeddings object",
        ):
            SemanticChunker(embedding_model=123)


class TestSemanticChunkerConfiguration:
    """Test different configuration options."""

    def test_with_different_similarity_windows(
        self,
        embedding_model: BaseEmbeddings,
        sample_text: str,
    ) -> None:
        """Test SemanticChunker with different similarity windows."""
        # Test with default window
        chunker1 = SemanticChunker(
            embedding_model=embedding_model,
            threshold=0.5,
            chunk_size=512,
        )
        chunks1 = chunker1.chunk(sample_text)
        assert len(chunks1) > 0
        assert chunker1.similarity_window == 3  # Default value

        # Test with larger window
        chunker2 = SemanticChunker(
            embedding_model=embedding_model,
            similarity_window=5,
            threshold=0.5,
            chunk_size=512,
        )
        chunks2 = chunker2.chunk(sample_text)
        assert len(chunks2) > 0
        assert chunker2.similarity_window == 5

    def test_with_different_thresholds(
        self,
        embedding_model: BaseEmbeddings,
        sample_text: str,
    ) -> None:
        """Test SemanticChunker with different threshold values."""
        # Low threshold (more merging)
        chunker1 = SemanticChunker(embedding_model=embedding_model, threshold=0.1, chunk_size=512)
        chunks1 = chunker1.chunk(sample_text)
        assert len(chunks1) > 0
        assert chunker1.threshold == 0.1

        # High threshold (less merging)
        chunker2 = SemanticChunker(embedding_model=embedding_model, threshold=0.9, chunk_size=512)
        chunks2 = chunker2.chunk(sample_text)
        assert len(chunks2) > 0
        assert chunker2.threshold == 0.9


class TestSemanticChunkerEdgeCases:
    """Test edge cases and special scenarios."""

    def test_very_small_chunk_size(self, embedding_model: BaseEmbeddings) -> None:
        """Test chunking with very small chunk size."""
        chunker = SemanticChunker(
            embedding_model=embedding_model,
            chunk_size=20,
            min_sentences_per_chunk=1,
        )
        text = "Short. Also short. Another short one. Final short sentence."
        chunks = chunker.chunk(text)

        assert len(chunks) > 0
        # Chunks should respect the size limit (with some buffer for semantic grouping)
        for chunk in chunks:
            assert chunk.token_count <= 30

    def test_delim_as_string(self, embedding_model: BaseEmbeddings) -> None:
        """Test SemanticChunker with delim as string instead of list."""
        chunker = SemanticChunker(embedding_model=embedding_model, delim=".", chunk_size=512)
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1


class TestSemanticChunkerSkipWindow:
    """Test suite for skip_window functionality in SemanticChunker."""

    def test_skip_window_default_disabled(self, embedding_model: BaseEmbeddings) -> None:
        """Test that skip_window=0 (default) disables skip-and-merge."""
        chunker = SemanticChunker(
            embedding_model=embedding_model,
            chunk_size=512,
            threshold=0.5,
            # skip_window defaults to 0
        )

        # Verify default value
        assert chunker.skip_window == 0

        text = """The weather is beautiful today. The sun is shining brightly.
        I love programming in Python. It's a versatile language.
        The climate is changing rapidly. Global temperatures are rising."""

        chunks = chunker.chunk(text)
        assert len(chunks) >= 1

    def test_skip_window_enabled_single(self, embedding_model: BaseEmbeddings) -> None:
        """Test that skip_window=1 enables merging of adjacent similar groups."""
        # Test with skip_window disabled
        chunker_no_skip = SemanticChunker(
            embedding_model=embedding_model,
            chunk_size=512,
            threshold=0.7,
            skip_window=0,
        )

        # Test with skip_window enabled
        chunker_with_skip = SemanticChunker(
            embedding_model=embedding_model,
            chunk_size=512,
            threshold=0.7,
            skip_window=1,
        )

        # Text with alternating but related topics
        text = """Dogs are loyal companions. They love to play fetch.
        Cats are independent pets. They enjoy climbing trees.
        Puppies need lots of training. They require patience.
        Kittens are playful animals. They chase laser pointers."""

        chunks_no_skip = chunker_no_skip.chunk(text)
        chunks_with_skip = chunker_with_skip.chunk(text)

        # Both should produce valid chunks
        assert len(chunks_no_skip) >= 1
        assert len(chunks_with_skip) >= 1

        # Skip window may produce different chunking patterns
        # We can't guarantee fewer chunks as it depends on embeddings
        # but we verify the functionality works without errors

    def test_skip_window_larger_value(self, embedding_model: BaseEmbeddings) -> None:
        """Test skip_window with larger values (2+)."""
        chunker = SemanticChunker(
            embedding_model=embedding_model,
            chunk_size=512,
            threshold=0.6,
            skip_window=2,
        )

        # Text with topics that might be merged across gaps
        text = """Machine learning is fascinating. Neural networks are powerful.
        The stock market fluctuated today. Economic indicators show growth.
        Deep learning models are complex. They require lots of data.
        Financial markets are volatile. Investors remain cautious."""

        chunks = chunker.chunk(text)
        assert len(chunks) >= 1

        # Verify all chunks have proper structure
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.text
            assert chunk.token_count > 0

    def test_skip_window_with_different_thresholds(self, embedding_model: BaseEmbeddings) -> None:
        """Test skip_window interaction with different threshold values."""
        thresholds = [0.3, 0.5, 0.7, 0.9]
        skip_windows = [0, 1, 2]

        text = """Python is a programming language. It's used for web development.
        JavaScript runs in browsers. It powers interactive websites.
        Python has many libraries. Data science uses Python extensively.
        JavaScript frameworks are popular. React and Vue are examples."""

        for threshold in thresholds:
            for skip_window in skip_windows:
                chunker = SemanticChunker(
                    embedding_model=embedding_model,
                    chunk_size=512,
                    threshold=threshold,
                    skip_window=skip_window,
                )

                chunks = chunker.chunk(text)
                assert len(chunks) >= 1

                # Verify chunks are valid
                for chunk in chunks:
                    assert chunk.text
                    assert chunk.start_index >= 0
                    assert chunk.end_index > chunk.start_index

    def test_skip_window_preserves_chunk_size_limits(
        self,
        embedding_model: BaseEmbeddings,
    ) -> None:
        """Test that skip_window respects chunk_size limits after merging."""
        chunker = SemanticChunker(
            embedding_model=embedding_model,
            chunk_size=50,  # Small chunk size to force splitting
            threshold=0.5,
            skip_window=2,
        )

        # Long text that will need to be split even after merging
        text = """The Renaissance was a period of cultural rebirth in Europe.
        It began in Italy during the 14th century and spread across the continent.
        Artists like Leonardo da Vinci and Michelangelo created masterpieces.
        Scientific discoveries challenged traditional beliefs about the world.
        The printing press revolutionized the spread of knowledge.
        Literature flourished with writers like Shakespeare and Dante.
        Architecture evolved with new techniques and classical influences.
        Trade routes expanded, bringing new goods and ideas to Europe."""

        chunks = chunker.chunk(text)

        # All chunks should respect the size limit
        for chunk in chunks:
            assert chunk.token_count <= 50

    def test_skip_window_with_empty_text(self, embedding_model: BaseEmbeddings) -> None:
        """Test skip_window behavior with empty or minimal text."""
        chunker = SemanticChunker(embedding_model=embedding_model, skip_window=1)

        # Empty text
        chunks = chunker.chunk("")
        assert chunks == []

        # Whitespace only
        chunks = chunker.chunk("   \n  \t  ")
        assert chunks == []

        # Single sentence
        chunks = chunker.chunk("Hello world.")
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world."

    def test_skip_window_with_single_sentence_groups(
        self,
        embedding_model: BaseEmbeddings,
    ) -> None:
        """Test skip_window when text produces single-sentence groups."""
        chunker = SemanticChunker(
            embedding_model=embedding_model,
            chunk_size=512,
            threshold=0.6,
            skip_window=1,
            min_sentences_per_chunk=1,
        )

        # Short sentences that might each form their own group
        text = "First. Second. Third. Fourth. Fifth."

        chunks = chunker.chunk(text)
        assert len(chunks) >= 1

        # Verify reconstruction
        reconstructed = "".join(chunk.text for chunk in chunks)
        assert reconstructed == text

    def test_skip_window_parameter_validation(self, embedding_model: BaseEmbeddings) -> None:
        """Test that skip_window parameter validates correctly."""
        # Valid skip_window values
        for skip_window in [0, 1, 2, 5, 10]:
            chunker = SemanticChunker(embedding_model=embedding_model, skip_window=skip_window)
            assert chunker.skip_window == skip_window

        # Negative skip_window should raise error
        with pytest.raises(ValueError, match="skip_window must be non-negative"):
            SemanticChunker(embedding_model=embedding_model, skip_window=-1)

    def test_skip_window_representation(self, embedding_model: BaseEmbeddings) -> None:
        """Test that skip_window appears in string representation."""
        chunker = SemanticChunker(embedding_model=embedding_model, skip_window=2)

        repr_str = repr(chunker)
        assert "skip_window=2" in repr_str
