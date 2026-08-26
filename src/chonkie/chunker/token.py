"""Module containing TokenChunker class.

This module provides a TokenChunker class for splitting text into chunks of a specified token size.

"""

from bisect import bisect_right
from typing import Generator, Sequence, Union

from tqdm import trange

from chonkie.chunker.base import BaseChunker
from chonkie.logger import get_logger
from chonkie.pipeline import chunker
from chonkie.tokenizer import TokenizerProtocol
from chonkie.types import Chunk

logger = get_logger(__name__)


@chunker("token")
class TokenChunker(BaseChunker):
    """Chunker that splits text into chunks of a specified token size.

    Args:
        tokenizer: The tokenizer instance to use for encoding/decoding
        chunk_size: Maximum number of tokens per chunk
        chunk_overlap: Number of tokens to overlap between chunks

    """

    def __init__(
        self,
        tokenizer: Union[str, TokenizerProtocol] = "character",
        chunk_size: int = 2048,
        chunk_overlap: Union[int, float] = 0,
    ) -> None:
        """Initialize the TokenChunker with configuration parameters.

        Args:
            tokenizer: The tokenizer instance to use for encoding/decoding
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks

        Raises:
            ValueError: If chunk_size <= 0 or chunk_overlap >= chunk_size

        """
        super().__init__(tokenizer)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if isinstance(chunk_overlap, int) and chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        # Assign the values if they make sense
        self.chunk_size = chunk_size
        self.chunk_overlap = (
            chunk_overlap if isinstance(chunk_overlap, int) else int(chunk_overlap * chunk_size)
        )

        self._use_multiprocessing = False

    def _create_chunks(
        self,
        chunk_texts: Sequence[str],
        token_groups: list[list[int]],
        token_counts: list[int],
    ) -> list[Chunk]:
        """Create chunks from a list of texts."""
        # Find the overlap lengths for index calculation
        if self.chunk_overlap > 0:
            # we get the overlap texts, that gives you the start_index for the next chunk
            # if the token group is smaller than the overlap, we just use the whole token group
            overlap_texts = self.tokenizer.decode_batch([
                token_group[-self.chunk_overlap :]
                if (len(token_group) > self.chunk_overlap)
                else token_group
                for token_group in token_groups
            ])
            overlap_lengths = [len(overlap_text) for overlap_text in overlap_texts]
        else:
            overlap_lengths = [0] * len(token_groups)

        # Create the chunks
        chunks = []
        current_index = 0
        for chunk_text, overlap_length, token_count in zip(
            chunk_texts,
            overlap_lengths,
            token_counts,
        ):
            start_index = current_index
            end_index = start_index + len(chunk_text)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_index=start_index,
                    end_index=end_index,
                    token_count=token_count,
                ),
            )
            current_index = end_index - overlap_length

        return chunks

    def _create_chunks_with_offsets(
        self,
        text: str,
        tokens: Sequence[int],
        offsets: Sequence[tuple[int, int]],
    ) -> list[Chunk]:
        """Create character-safe chunks from tokenizer offsets."""
        safe_boundaries = [0]
        for index in range(1, len(tokens)):
            previous_start, previous_end = offsets[index - 1]
            current_start, _ = offsets[index]
            if previous_end > previous_start and previous_end == current_start:
                safe_boundaries.append(index)
        safe_boundaries.append(len(tokens))

        chunks = []
        start = 0
        while start < len(tokens):
            requested_end = min(start + self.chunk_size, len(tokens))
            end = safe_boundaries[bisect_right(safe_boundaries, requested_end) - 1]
            if end <= start:
                end = safe_boundaries[bisect_right(safe_boundaries, start)]

            start_index = offsets[start][0]
            end_index = offsets[end - 1][1]
            chunks.append(
                Chunk(
                    text=text[start_index:end_index],
                    start_index=start_index,
                    end_index=end_index,
                    token_count=end - start,
                )
            )

            if end == len(tokens):
                break

            requested_start = max(0, end - self.chunk_overlap)
            next_start = safe_boundaries[bisect_right(safe_boundaries, requested_start) - 1]
            if next_start <= start:
                next_start = safe_boundaries[bisect_right(safe_boundaries, start)]
            start = next_start

        return chunks

    def _chunk_with_offsets(self, text: str) -> tuple[list[Chunk], int] | None:
        """Chunk text with source offsets when the tokenizer provides them."""
        encode_with_offsets = getattr(self.tokenizer, "encode_with_offsets", None)
        if callable(encode_with_offsets):
            try:
                offset_tokens, offsets = encode_with_offsets(text)
            except (NotImplementedError, ValueError):
                return None
            else:
                if offset_tokens and len(offsets) == len(offset_tokens):
                    chunks = self._create_chunks_with_offsets(text, offset_tokens, offsets)
                    return chunks, len(offset_tokens)

        return None

    def _chunk_tokens(self, tokens: Sequence[int]) -> list[Chunk]:
        """Chunk token IDs using the decode-based fallback."""
        token_groups = list(self._token_group_generator(tokens))
        token_counts = [len(token_group) for token_group in token_groups]
        chunk_texts = self.tokenizer.decode_batch(token_groups)
        return self._create_chunks(chunk_texts, token_groups, token_counts)

    def _token_group_generator(self, tokens: Sequence[int]) -> Generator[list[int], None, None]:
        """Generate chunks from a list of tokens."""
        for start in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            end = min(start + self.chunk_size, len(tokens))
            yield list(tokens[start:end])
            if end == len(tokens):
                break

    def chunk(self, text: str) -> list[Chunk]:
        """Split text into overlapping chunks of specified token size.

        Args:
            text: Input text to be chunked

        Returns:
            List of Chunk objects containing the chunked text and metadata

        """
        if not text.strip():
            return []

        logger.debug(f"Chunking text of length {len(text)} with chunk_size={self.chunk_size}")

        offset_chunks = self._chunk_with_offsets(text)
        if offset_chunks is not None:
            chunks, token_count = offset_chunks
        else:
            text_tokens = self.tokenizer.encode(text)
            chunks = self._chunk_tokens(text_tokens)
            token_count = len(text_tokens)

        logger.info(f"Created {len(chunks)} chunks from {token_count} tokens")
        return chunks

    def _process_batch(self, texts: list[str]) -> list[list[Chunk]]:
        """Process a batch of texts."""
        result: list[list[Chunk] | None] = [None] * len(texts)
        fallback_indices = []
        fallback_texts = []

        for index, text in enumerate(texts):
            offset_chunks = self._chunk_with_offsets(text)
            if offset_chunks is not None:
                result[index] = offset_chunks[0]
            else:
                fallback_indices.append(index)
                fallback_texts.append(text)

        if fallback_texts:
            tokens_list = self.tokenizer.encode_batch(fallback_texts)
            for index, tokens in zip(fallback_indices, tokens_list):
                result[index] = self._chunk_tokens(tokens) if tokens else []

        return [chunks if chunks is not None else [] for chunks in result]

    def chunk_batch(  # ty: ignore[invalid-method-override]
        self,
        texts: list[str],
        batch_size: int = 1,
        show_progress_bar: bool = True,
    ) -> list[list[Chunk]]:
        """Split a batch of texts into their respective chunks.

        Args:
            texts: List of input texts to be chunked
            batch_size: Number of texts to process in a single batch
            show_progress_bar: Whether to show a progress bar

        Returns:
            List of lists of Chunk objects containing the chunked text and metadata

        """
        chunks: list = []
        for i in trange(
            0,
            len(texts),
            batch_size,
            desc="🦛",
            disable=not show_progress_bar,
            unit="batch",
            bar_format="{desc} ch{bar:20}nk {percentage:3.0f}% • {n_fmt}/{total_fmt} batches chunked [{elapsed}<{remaining}, {rate_fmt}] 🌱",
            ascii=" o",
        ):
            batch_texts = texts[i : min(i + batch_size, len(texts))]
            chunks.extend(self._process_batch(batch_texts))
        return chunks

    def __call__(  # ty: ignore[invalid-method-override]
        self,
        text: Union[str, list[str]],
        batch_size: int = 1,
        show_progress_bar: bool = True,
    ) -> Union[list[Chunk], list[list[Chunk]]]:
        """Make the TokenChunker callable directly.

        Args:
            text: Input text or list of texts to be chunked
            batch_size: Number of texts to process in a single batch
            show_progress_bar: Whether to show a progress bar (for batch chunking)

        Returns:
            List of Chunk objects or list of lists of Chunk

        """
        if isinstance(text, str):
            return self.chunk(text)
        elif isinstance(text, list) and isinstance(text[0], str):
            return self.chunk_batch(text, batch_size, show_progress_bar)
        else:
            raise ValueError("Invalid input type. Expected a string or a list of strings.")

    def __repr__(self) -> str:
        """Return a string representation of the TokenChunker."""
        return (
            f"TokenChunker(tokenizer={self.tokenizer}, "
            f"chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap})"
        )
