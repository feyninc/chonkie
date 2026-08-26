"""Comprehensive tests for the SlumberChunker."""

import json
from typing import Any, Optional
from unittest.mock import Mock, patch

import pytest

from chonkie.chunker import SlumberChunker
from chonkie.genie import BaseGenie
from chonkie.types import Chunk, RecursiveLevel, RecursiveRules


class MockGenie(BaseGenie):
    """Mock genie for testing SlumberChunker."""

    def __init__(self, split_responses: Optional[list[int]] = None) -> None:
        """Initialize mock genie with predefined responses.

        Args:
            split_responses: List of split indices to return. Defaults to [1].

        """
        self.split_responses = split_responses or [1]
        self.call_count = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        """Generate a text response."""
        return f"Mock response for: {prompt[:50]}..."

    def generate_json(self, prompt: str, schema: Any) -> Any:
        """Generate a JSON response with split_index."""
        self.prompts.append(prompt)
        response_index = self.call_count % len(self.split_responses)
        split_index = self.split_responses[response_index]
        self.call_count += 1
        return {"split_index": split_index}


@pytest.fixture
def mock_genie() -> MockGenie:
    """Fixture for mock genie."""
    return MockGenie()


@pytest.fixture
def sample_text() -> str:
    """Fixture providing sample text for testing."""
    return """The quick brown fox jumps over the lazy dog. This is the first paragraph.
    
This is the second paragraph with more content. It contains multiple sentences. Some are longer than others.

The third paragraph discusses different topics. It talks about technology and innovation. The digital revolution has changed everything.

Finally, the fourth paragraph concludes our sample text. It wraps up the discussion nicely."""


@pytest.fixture
def long_sample_text() -> str:
    """Fixture providing longer sample text for testing."""
    return """Chapter 1: Introduction

The world of natural language processing has evolved dramatically over the past decade. Machine learning models have become increasingly sophisticated, enabling new applications and use cases that were previously impossible.

Chapter 2: Text Chunking

Text chunking is a fundamental preprocessing step in many NLP applications. It involves dividing large documents into smaller, manageable pieces while preserving semantic coherence. This process is crucial for applications like RAG systems, document analysis, and information retrieval.

The SlumberChunker represents an innovative approach to text chunking. Unlike traditional methods that rely solely on static rules or simple token counts, it leverages AI to make intelligent decisions about where to split text. This AI-driven approach can better understand context and maintain semantic boundaries.

Chapter 3: Implementation Details

The chunker works by first creating candidate splits using recursive rules. These candidates are then processed by an AI model that determines the optimal split points based on semantic coherence. The process continues iteratively until the entire document is processed.

Chapter 4: Performance Considerations

When dealing with large documents, performance becomes a critical factor. The SlumberChunker includes optimizations like batch processing and configurable chunk sizes to balance accuracy with processing speed.

Chapter 5: Conclusion

The SlumberChunker demonstrates how AI can enhance traditional text processing tasks. By combining rule-based preprocessing with intelligent decision-making, it achieves better results than purely algorithmic approaches."""


class TestSlumberChunkerInitialization:
    """Test SlumberChunker initialization."""

    def test_default_initialization(self, mock_genie: MockGenie) -> None:
        """Test default initialization parameters."""
        chunker = SlumberChunker(genie=mock_genie)

        assert chunker.genie == mock_genie
        assert chunker.chunk_size == 2048
        assert chunker.candidate_size == 128
        assert chunker.min_characters_per_chunk == 24
        assert chunker.verbose is True
        assert isinstance(chunker.rules, RecursiveRules)
        assert chunker.template is not None
        assert chunker.sep == "✄"
        assert chunker._CHARS_PER_TOKEN == 6.5
        assert chunker._use_multiprocessing is False

    def test_custom_initialization(self, mock_genie: MockGenie) -> None:
        """Test initialization with custom parameters."""
        custom_rules = RecursiveRules(
            levels=[
                RecursiveLevel(delimiters=["\n\n", "\n", "."]),
                RecursiveLevel(whitespace=True),  # Use whitespace=True instead of delimiter=" "
            ],
        )

        chunker = SlumberChunker(
            genie=mock_genie,
            tokenizer="gpt2",
            chunk_size=2048,
            rules=custom_rules,
            candidate_size=256,
            min_characters_per_chunk=50,
            verbose=False,
        )

        assert chunker.genie == mock_genie
        assert chunker.chunk_size == 2048
        assert chunker.candidate_size == 256
        assert chunker.min_characters_per_chunk == 50
        assert chunker.verbose is False
        assert chunker.rules == custom_rules

    def test_default_genie_initialization(self) -> None:
        """Test that default GeminiGenie is created when none provided."""
        # Import the specific local slumber module
        import os
        import sys

        sys.path.insert(0, os.path.join(os.getcwd(), "src"))

        # Clear any cached imports
        if "chonkie.chunker.slumber" in sys.modules:
            del sys.modules["chonkie.chunker.slumber"]

        # Import the local slumber module directly
        from chonkie.chunker import slumber as slumber_module

        with patch.object(slumber_module, "GeminiGenie") as mock_gemini:
            mock_gemini.return_value = Mock()
            SlumberChunker()  # Create chunker to trigger genie initialization
            assert mock_gemini.called


class TestSlumberChunkerInternalMethods:
    """Test SlumberChunker internal methods."""

    def test_split_text_whitespace(self, mock_genie: MockGenie) -> None:
        """Test text splitting with whitespace delimiter."""
        chunker = SlumberChunker(genie=mock_genie)

        # Test with whitespace and include_delim="prev"
        level = RecursiveLevel(whitespace=True, include_delim="prev")
        text = "hello world test"
        splits = chunker._split_text(text, level)
        # After merging short splits, the result may be different
        assert isinstance(splits, list)
        assert len(splits) > 0

        # Test with whitespace and include_delim="next"
        level = RecursiveLevel(whitespace=True, include_delim="next")
        splits = chunker._split_text(text, level)
        assert isinstance(splits, list)
        assert len(splits) > 0

        # Test with whitespace and no include_delim
        level = RecursiveLevel(whitespace=True, include_delim=None)
        splits = chunker._split_text(text, level)
        assert isinstance(splits, list)
        assert len(splits) > 0

    def test_split_text_delimiters(self, mock_genie: MockGenie) -> None:
        """Test text splitting with custom delimiters."""
        chunker = SlumberChunker(genie=mock_genie)

        # Test with delimiters and include_delim="prev"
        level = RecursiveLevel(delimiters=["\n\n", "."], include_delim="prev")
        text = "First paragraph.\n\nSecond paragraph."
        splits = chunker._split_text(text, level)
        # Should split into multiple parts
        assert isinstance(splits, list)
        assert len(splits) >= 1

        # Test with delimiters and include_delim="next"
        level = RecursiveLevel(delimiters=["\n\n", "."], include_delim="next")
        splits = chunker._split_text(text, level)
        assert isinstance(splits, list)
        assert len(splits) >= 1

        # Test with delimiters and no include_delim
        level = RecursiveLevel(delimiters=["\n\n", "."], include_delim=None)
        splits = chunker._split_text(text, level)
        assert isinstance(splits, list)
        assert len(splits) >= 1

    def test_split_text_token_based(self, mock_genie: MockGenie) -> None:
        """Test text splitting with token-based approach."""
        chunker = SlumberChunker(genie=mock_genie, chunk_size=10)

        # Test with no delimiters (should use token-based splitting)
        level = RecursiveLevel()
        text = "This is a long text that should be split into multiple token-based chunks."
        splits = chunker._split_text(text, level)

        # Should have multiple splits for long text
        assert len(splits) > 1

        # Verify splits can be reconstructed
        reconstructed = "".join(splits)
        assert reconstructed == text

    def test_split_text_merge_short_splits(self, mock_genie: MockGenie) -> None:
        """Test merging of short splits."""
        chunker = SlumberChunker(genie=mock_genie, min_characters_per_chunk=10)

        level = RecursiveLevel(whitespace=True)  # Use whitespace instead of delimiter=" "
        text = "a b c d e f g h i j k l m n o p"  # Short words
        splits = chunker._split_text(text, level)

        # Short splits should be merged, but last split might be an exception
        assert isinstance(splits, list)
        assert len(splits) >= 1
        # Most splits should meet minimum length or be the last split
        non_last_splits = splits[:-1]
        for split in non_last_splits:
            assert len(split) >= chunker.min_characters_per_chunk

    def test_recursive_split_base_case(self, mock_genie: MockGenie) -> None:
        """Test recursive split base case."""
        chunker = SlumberChunker(genie=mock_genie)

        # Test with level beyond rules length
        text = "Short text"
        chunks = chunker._recursive_split(text, level=100, offset=0)

        assert len(chunks) == 1
        assert isinstance(chunks[0], Chunk)
        assert chunks[0].text == text
        assert chunks[0].start_index == 0
        assert chunks[0].end_index == len(text)
        assert chunks[0].token_count > 0

    def test_recursive_split_recursive_case(self, mock_genie: MockGenie) -> None:
        """Test recursive split with large candidate."""
        chunker = SlumberChunker(genie=mock_genie, candidate_size=5)

        # Create text that will exceed candidate_size
        text = "This is a very long text that should trigger recursive splitting behavior."
        chunks = chunker._recursive_split(text, level=0, offset=10)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.start_index >= 10  # Offset should be applied

    def test_prepare_splits(self, mock_genie: MockGenie) -> None:
        """Test preparation of splits for genie input."""
        chunker = SlumberChunker(genie=mock_genie)

        chunks = [
            Chunk(text="First chunk", start_index=0, end_index=11, token_count=2),
            Chunk(text="Second chunk\nwith newline", start_index=11, end_index=35, token_count=4),
            Chunk(text="Third chunk", start_index=35, end_index=46, token_count=2),
        ]

        prepared = chunker._prepare_splits(chunks)

        assert len(prepared) == 3
        assert prepared[0] == "ID 0: First chunk"
        assert prepared[1] == "ID 1: Second chunk with newline"  # Newlines replaced with space
        assert prepared[2] == "ID 2: Third chunk"

    def test_get_cumulative_token_counts(self, mock_genie: MockGenie) -> None:
        """Test cumulative token count calculation."""
        chunker = SlumberChunker(genie=mock_genie)

        chunks = [
            Chunk(text="First", start_index=0, end_index=5, token_count=1),
            Chunk(text="Second", start_index=5, end_index=11, token_count=2),
            Chunk(text="Third", start_index=11, end_index=16, token_count=3),
        ]

        cumulative = chunker._get_cumulative_token_counts(chunks)

        assert cumulative == [0, 1, 3, 6]  # [0, 0+1, 1+2, 3+3]


class TestSlumberChunkerChunking:
    """Test SlumberChunker chunking functionality."""

    def test_chunk_simple_text(self, sample_text: str) -> None:
        """Test chunking simple text."""
        mock_genie = MockGenie([2, 4])  # Split at indices 2 and 4
        chunker = SlumberChunker(genie=mock_genie, verbose=False)

        chunks = chunker.chunk(sample_text)

        assert len(chunks) >= 1
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

        # Verify chunks cover the entire text
        total_length = sum(chunk.end_index - chunk.start_index for chunk in chunks)
        assert total_length <= len(sample_text)

        # Verify chunk properties
        for chunk in chunks:
            assert chunk.start_index >= 0
            assert chunk.end_index <= len(sample_text)
            assert chunk.start_index < chunk.end_index
            assert chunk.token_count > 0
            assert len(chunk.text) == chunk.end_index - chunk.start_index

    def test_chunk_with_progress_bar(self, sample_text: str) -> None:
        """Test chunking with verbose progress bar."""
        mock_genie = MockGenie([1])
        chunker = SlumberChunker(genie=mock_genie, verbose=True)

        # Import the specific local slumber module
        import os
        import sys

        sys.path.insert(0, os.path.join(os.getcwd(), "src"))

        # Import the local slumber module directly
        from chonkie.chunker import slumber as slumber_module

        # Patch tqdm from the local slumber module
        with patch.object(slumber_module, "tqdm") as mock_tqdm:
            mock_progress = Mock()
            mock_progress.n = 0  # Set the n attribute that's used in the update calculation
            mock_tqdm.return_value = mock_progress

            chunker.chunk(sample_text)  # Process text to trigger progress bar

            # Verify tqdm was called for progress bar
            assert mock_tqdm.called
            assert mock_progress.update.called

    def test_chunk_genie_response_correction(self, sample_text: str) -> None:
        """Test correction of invalid genie responses."""
        # Mock genie that returns invalid split indices
        mock_genie = MockGenie([0, -1])  # Invalid indices
        chunker = SlumberChunker(genie=mock_genie, verbose=False)

        chunks = chunker.chunk(sample_text)

        # Should still produce valid chunks despite invalid responses
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.start_index >= 0

    def test_chunk_large_text(self, long_sample_text: str) -> None:
        """Test chunking large text with multiple iterations."""
        mock_genie = MockGenie([3, 2, 4, 1])  # Various split points
        chunker = SlumberChunker(
            genie=mock_genie,
            chunk_size=512,  # Smaller chunks to trigger multiple iterations
            verbose=False,
        )

        chunks = chunker.chunk(long_sample_text)

        assert len(chunks) >= 2  # Should split large text

        # Verify chunks are sequential and non-overlapping
        for i in range(len(chunks) - 1):
            assert chunks[i].end_index <= chunks[i + 1].start_index

    def test_chunk_empty_text(self, mock_genie: MockGenie) -> None:
        """Test chunking empty text."""
        chunker = SlumberChunker(genie=mock_genie, verbose=False)

        chunks = chunker.chunk("")

        # Should handle empty text gracefully
        assert isinstance(chunks, list)

    def test_chunk_single_word(self, mock_genie: MockGenie) -> None:
        """Test chunking single word."""
        chunker = SlumberChunker(genie=mock_genie, verbose=False)

        chunks = chunker.chunk("word")

        assert len(chunks) == 1
        assert chunks[0].text == "word"
        assert chunks[0].start_index == 0
        assert chunks[0].end_index == 4

    def test_chunk_bisect_edge_case(self, sample_text: str) -> None:
        """Test edge case where group_end_index equals current_pos."""
        # This test aims to trigger line 241: group_end_index += 1
        mock_genie = MockGenie([1])

        # Use a very small chunk_size to increase chances of hitting the edge case
        chunker = SlumberChunker(
            genie=mock_genie,
            chunk_size=1,  # Very small chunk size
            candidate_size=1,  # Very small candidate size
            verbose=False,
        )

        chunks = chunker.chunk(sample_text)

        # Should still produce valid chunks despite edge case
        assert len(chunks) >= 1
        assert all(isinstance(chunk, Chunk) for chunk in chunks)


class TestSlumberChunkerEdgeCases:
    """Test SlumberChunker edge cases and error conditions."""

    def test_genie_json_parsing_error(self, sample_text: str) -> None:
        """Test handling of genie JSON parsing errors with retry and fallback."""

        class ErrorGenie(BaseGenie):
            def generate(self, prompt: str) -> str:
                return "invalid response"

            def generate_json(self, prompt: str, schema: Any) -> Any:
                raise json.JSONDecodeError("Invalid JSON", "", 0)

        chunker = SlumberChunker(genie=ErrorGenie(), verbose=False)

        # Should not raise - retry logic falls back gracefully
        chunks = chunker.chunk(sample_text)
        assert len(chunks) >= 1
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

    def test_very_small_candidate_size(self, sample_text: str) -> None:
        """Test with very small candidate size."""
        mock_genie = MockGenie([1])
        chunker = SlumberChunker(
            genie=mock_genie,
            candidate_size=1,  # Very small
            verbose=False,
        )

        chunks = chunker.chunk(sample_text)

        # Should still produce valid chunks
        assert len(chunks) >= 1
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

    def test_chunk_size_larger_than_text(self, mock_genie: MockGenie) -> None:
        """Test with chunk size larger than text."""
        chunker = SlumberChunker(
            genie=mock_genie,
            chunk_size=10000,  # Very large
            verbose=False,
        )

        text = "Short text"
        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_out_of_bounds_genie_response(self, sample_text: str) -> None:
        """Test handling of out-of-bounds genie response."""
        # Mock genie that returns an out-of-bounds index (e.g., 1000)
        mock_genie = MockGenie([1000])
        chunker = SlumberChunker(genie=mock_genie, verbose=False)

        # Should not raise IndexError and fallback gracefully
        chunks = chunker.chunk(sample_text)

        assert len(chunks) >= 1
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

    def test_different_tokenizers(self, sample_text: str) -> None:
        """Test with different tokenizers."""
        mock_genie = MockGenie([1])

        # Test with different tokenizer
        chunker = SlumberChunker(genie=mock_genie, tokenizer="gpt2", verbose=False)

        chunks = chunker.chunk(sample_text)
        assert len(chunks) >= 1

    def test_custom_rules_integration(self, sample_text: str) -> None:
        """Test integration with custom recursive rules."""
        mock_genie = MockGenie([2])
        custom_rules = RecursiveRules(
            levels=[
                RecursiveLevel(delimiters=["\n\n"]),
                RecursiveLevel(delimiters=["\n"]),
                RecursiveLevel(delimiters=["."]),
                RecursiveLevel(whitespace=True),
            ],
        )

        chunker = SlumberChunker(genie=mock_genie, rules=custom_rules, verbose=False)

        chunks = chunker.chunk(sample_text)

        assert len(chunks) >= 1
        assert all(isinstance(chunk, Chunk) for chunk in chunks)


class TestSlumberChunkerPromptGeneration:
    """Test SlumberChunker prompt generation and genie interaction."""

    def test_prompt_format(self, sample_text: str) -> None:
        """Test that prompts are formatted correctly."""
        mock_genie = MockGenie([1])
        chunker = SlumberChunker(genie=mock_genie, verbose=False)

        chunker.chunk(sample_text)

        # Check that genie was called with properly formatted prompts
        assert len(mock_genie.prompts) > 0

        for prompt in mock_genie.prompts:
            assert "<passages>" in prompt
            assert "</passages>" in prompt
            assert "ID " in prompt
            assert "split_index" in prompt

    def test_prompt_template_customization(self, sample_text: str) -> None:
        """Test customization of prompt template."""
        mock_genie = MockGenie([1])
        chunker = SlumberChunker(genie=mock_genie, verbose=False)

        custom_template = """Custom template: {passages}
        Return JSON with split_index."""
        chunker.template = custom_template

        chunker.chunk(sample_text)

        # Check that custom template was used
        for prompt in mock_genie.prompts:
            assert "Custom template:" in prompt


class TestSlumberChunkerRepresentation:
    """Test SlumberChunker string representation."""

    def test_repr(self, mock_genie: MockGenie) -> None:
        """Test __repr__ method."""
        chunker = SlumberChunker(
            genie=mock_genie,
            chunk_size=2048,
            candidate_size=256,
            min_characters_per_chunk=50,
        )

        repr_str = repr(chunker)

        assert "SlumberChunker" in repr_str
        assert "genie=" in repr_str
        assert "chunk_size=2048" in repr_str
        assert "candidate_size=256" in repr_str
        assert "min_characters_per_chunk=50" in repr_str

    def test_repr_default_values(self, mock_genie: MockGenie) -> None:
        """Test __repr__ with default values."""
        chunker = SlumberChunker(genie=mock_genie)

        repr_str = repr(chunker)

        assert "chunk_size=2048" in repr_str
        assert "candidate_size=128" in repr_str
        assert "min_characters_per_chunk=24" in repr_str


class TestSlumberChunkerIntegration:
    """Integration tests for SlumberChunker."""

    def test_end_to_end_chunking(self, long_sample_text: str) -> None:
        """Test end-to-end chunking process."""
        # Create a more sophisticated mock genie
        mock_genie = MockGenie([3, 2, 4, 1, 5])

        chunker = SlumberChunker(
            genie=mock_genie,
            chunk_size=1024,
            candidate_size=128,
            verbose=False,
        )

        chunks = chunker.chunk(long_sample_text)

        # Comprehensive validation
        assert len(chunks) > 0

        # Check chunk properties
        total_chars = 0
        for i, chunk in enumerate(chunks):
            assert isinstance(chunk, Chunk)
            assert len(chunk.text) > 0
            assert chunk.token_count > 0
            assert chunk.start_index >= 0
            assert chunk.end_index > chunk.start_index

            # Check sequential nature
            if i > 0:
                assert chunk.start_index >= chunks[i - 1].end_index

            total_chars += len(chunk.text)

        # Text should be reasonably preserved (allowing for some processing differences)
        assert total_chars >= len(long_sample_text) * 0.95

    def test_chunking_with_different_rules(self, sample_text: str) -> None:
        """Test chunking with different recursive rules."""
        mock_genie = MockGenie([2])

        # Test with paragraph-focused rules
        paragraph_rules = RecursiveRules(
            levels=[
                RecursiveLevel(delimiters=["\n\n"]),
                RecursiveLevel(delimiters=["."]),
            ],
        )

        chunker = SlumberChunker(genie=mock_genie, rules=paragraph_rules, verbose=False)

        chunks = chunker.chunk(sample_text)

        assert len(chunks) >= 1
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

        # Test with sentence-focused rules
        sentence_rules = RecursiveRules(
            levels=[
                RecursiveLevel(delimiters=[".", "!", "?"]),
                RecursiveLevel(whitespace=True),
            ],
        )

        chunker.rules = sentence_rules
        chunks_sentences = chunker.chunk(sample_text)

        assert len(chunks_sentences) >= 1
        assert all(isinstance(chunk, Chunk) for chunk in chunks_sentences)


class MockTextGenie(BaseGenie):
    """Mock genie that only supports generate() (text generation), not JSON.

    Used for testing text extraction mode.
    """

    def __init__(self, split_responses: Optional[list[int]] = None) -> None:
        """Initialize mock genie with predefined responses.

        Args:
            split_responses: List of split indices to return as plain text.

        """
        self.split_responses = split_responses or [1]
        self.call_count = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        """Generate a text response with just the index number."""
        self.prompts.append(prompt)
        response_index = self.call_count % len(self.split_responses)
        split_index = self.split_responses[response_index]
        self.call_count += 1
        return str(split_index)

    # Note: generate_json is NOT overridden, so it raises NotImplementedError


class TestSlumberChunkerTextMode:
    """Test SlumberChunker text extraction mode."""

    def test_text_mode_initialization(self) -> None:
        """Test initialization with text mode."""
        genie = MockTextGenie()
        chunker = SlumberChunker(genie=genie, extract_mode="text", verbose=False)

        assert chunker.extract_mode == "text"
        assert chunker.Split is None  # Pydantic is not needed for text mode

    def test_json_mode_initialization(self) -> None:
        """Test initialization with JSON mode."""
        genie = MockGenie()
        chunker = SlumberChunker(genie=genie, extract_mode="json", verbose=False)

        assert chunker.extract_mode == "json"
        assert chunker.Split is not None  # Pydantic schema is needed for JSON

    def test_auto_mode_detection_text(self) -> None:
        """Test auto mode detects text-only genie."""
        genie = MockTextGenie()
        chunker = SlumberChunker(genie=genie, extract_mode="auto", verbose=False)

        # Should auto-detect text mode since MockTextGenie doesn't override generate_json
        assert chunker.extract_mode == "text"

    def test_auto_mode_detection_json(self) -> None:
        """Test auto mode detects JSON-capable genie."""
        genie = MockGenie()
        chunker = SlumberChunker(genie=genie, extract_mode="auto", verbose=False)

        # Should auto-detect JSON mode since MockGenie overrides generate_json
        assert chunker.extract_mode == "json"

    def test_text_mode_chunking(self, sample_text: str) -> None:
        """Test chunking with text extraction mode."""
        genie = MockTextGenie([2, 3])
        chunker = SlumberChunker(genie=genie, extract_mode="text", verbose=False)

        chunks = chunker.chunk(sample_text)

        assert len(chunks) >= 1
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

    def test_extract_index_from_text_direct(self) -> None:
        """Test _extract_index_from_text with direct integer."""
        genie = MockGenie()
        chunker = SlumberChunker(genie=genie, extract_mode="text", verbose=False)

        assert chunker._extract_index_from_text("5") == 5
        assert chunker._extract_index_from_text("  3  ") == 3
        assert chunker._extract_index_from_text("0") == 0

    def test_extract_index_from_text_with_noise(self) -> None:
        """Test _extract_index_from_text extracts first integer."""
        genie = MockGenie()
        chunker = SlumberChunker(genie=genie, extract_mode="text", verbose=False)

        assert chunker._extract_index_from_text("The answer is 7") == 7
        assert chunker._extract_index_from_text("Split at: 3") == 3

    def test_extract_index_from_text_invalid(self) -> None:
        """Test _extract_index_from_text with invalid input."""
        genie = MockGenie()
        chunker = SlumberChunker(genie=genie, extract_mode="text", verbose=False)

        with pytest.raises(ValueError):
            chunker._extract_index_from_text("no numbers here")

        with pytest.raises(ValueError):
            chunker._extract_index_from_text("")

    def test_text_mode_retry_on_failure(self, sample_text: str) -> None:
        """Test text extraction retries on parsing failure."""

        class FlakeyTextGenie(BaseGenie):
            def __init__(self) -> None:
                self.call_count = 0

            def generate(self, prompt: str) -> str:
                self.call_count += 1
                if self.call_count < 3:
                    return "Invalid response without numbers"
                return "1"

        genie = FlakeyTextGenie()
        chunker = SlumberChunker(genie=genie, extract_mode="text", max_retries=5, verbose=False)

        chunks = chunker.chunk("Short text to chunk here.")
        assert len(chunks) >= 1

    def test_text_mode_max_retries_fallback(self, sample_text: str) -> None:
        """Test fallback when max retries exceeded."""

        class AlwaysFailTextGenie(BaseGenie):
            def generate(self, prompt: str) -> str:
                return "This will never parse as a number"

        genie = AlwaysFailTextGenie()
        chunker = SlumberChunker(genie=genie, extract_mode="text", max_retries=2, verbose=False)

        # Should still produce chunks using fallback
        chunks = chunker.chunk(sample_text)
        assert len(chunks) >= 1

    def test_repr_includes_extract_mode(self) -> None:
        """Test __repr__ includes extract_mode and max_retries."""
        genie = MockGenie()
        chunker = SlumberChunker(genie=genie, extract_mode="text", max_retries=5, verbose=False)

        repr_str = repr(chunker)
        assert "extract_mode=text" in repr_str
        assert "max_retries=5" in repr_str

    def test_invalid_extract_mode(self) -> None:
        """Test that invalid extract_mode raises ValueError."""
        genie = MockGenie()

        with pytest.raises(ValueError):
            SlumberChunker(genie=genie, extract_mode="invalid", verbose=False)  # type: ignore
