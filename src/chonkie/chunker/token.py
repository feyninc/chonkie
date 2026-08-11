"""Module containing TokenChunker class.

This module provides a TokenChunker class for splitting text into chunks of a specified token size.

"""

from typing import Generator, Sequence, Union

from tqdm import trange

from chonkie.chunker.base import BaseChunker
from chonkie.logger import get_logger
from chonkie.pipeline import chunker
from chonkie.tokenizer import TokenizerEncoding, TokenizerProtocol
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
        text_spans: Sequence[tuple[int, int]] | None = None,
    ) -> list[Chunk]:
        """Create chunks from a list of texts."""
        if text_spans is not None:
            return [
                Chunk(
                    text=chunk_text,
                    start_index=start_index,
                    end_index=end_index,
                    token_count=token_count,
                )
                for chunk_text, (start_index, end_index), token_count in zip(
                    chunk_texts,
                    text_spans,
                    token_counts,
                )
            ]

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

    def _token_group_ranges(
        self,
        token_count: int,
        offsets: Sequence[tuple[int, int]] | None = None,
    ) -> list[tuple[int, int]]:
        """Return token ranges, expanding boundaries inside a character."""
        if offsets is None or len(offsets) != token_count:
            step = self.chunk_size - self.chunk_overlap
            ranges = []
            for start in range(0, token_count, step):
                end = min(start + self.chunk_size, token_count)
                ranges.append((start, end))
                if end == token_count:
                    break
            return ranges

        ranges = []
        start = 0
        while start < token_count:
            start_character = offsets[start][0]
            while start > 0 and offsets[start - 1][1] > start_character:
                start -= 1

            end = min(start + self.chunk_size, token_count)
            end_character = offsets[end - 1][1]
            while end < token_count and offsets[end][0] < end_character:
                end += 1
                end_character = max(end_character, offsets[end - 1][1])

            ranges.append((start, end))
            if end == token_count:
                break

            next_start = (
                end if self.chunk_overlap == 0 else max(start + 1, end - self.chunk_overlap)
            )
            if next_start < token_count:
                next_start_character = offsets[next_start][0]
                while next_start > 0 and offsets[next_start - 1][1] > next_start_character:
                    next_start -= 1
            if next_start <= start:
                next_start = end
            start = next_start

        return ranges

    def _token_group_generator(self, tokens: Sequence[int]) -> Generator[list[int], None, None]:
        """Generate chunks from a list of tokens."""
        for start, end in self._token_group_ranges(len(tokens)):
            yield list(tokens[start:end])

    def _chunk_with_offsets(self, text: str) -> list[Chunk] | None:
        """Chunk text using backend-provided offsets when available."""
        encoded: TokenizerEncoding | None = self.tokenizer.encode_with_offsets(text)
        if encoded is None:
            return None

        tokens = list(encoded.ids)
        if not tokens:
            return []

        offsets = list(encoded.offsets)
        if len(offsets) != len(tokens):
            return None

        ranges = self._token_group_ranges(len(tokens), offsets)
        token_groups = [tokens[start:end] for start, end in ranges]
        text_spans = [(offsets[start][0], offsets[end - 1][1]) for start, end in ranges]
        chunk_texts = [text[start:end] for start, end in text_spans]
        token_counts = [end - start for start, end in ranges]
        return self._create_chunks(
            chunk_texts,
            token_groups,
            token_counts,
            text_spans=text_spans,
        )

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
            logger.info(f"Created {len(offset_chunks)} chunks using tokenizer offsets")
            return offset_chunks

        # Encode full text
        text_tokens = self.tokenizer.encode(text)

        # Calculate token groups and counts
        token_groups = list(self._token_group_generator(text_tokens))
        token_counts = [len(toks) for toks in token_groups]

        # decode the token groups into the chunk texts
        chunk_texts = self.tokenizer.decode_batch(token_groups)

        # Create the chunks from the token groups and token counts
        chunks = self._create_chunks(chunk_texts, token_groups, token_counts)

        logger.info(f"Created {len(chunks)} chunks from {len(text_tokens)} tokens")
        return chunks

    def _process_batch(self, texts: list[str]) -> list[list[Chunk]]:
        """Process a batch of texts."""
        # encode the texts into tokens in a batch
        tokens_list = self.tokenizer.encode_batch(texts)
        result: list = []

        for text, tokens in zip(texts, tokens_list):
            if not tokens:
                result.append([])
                continue

            offset_chunks = self._chunk_with_offsets(text)
            if offset_chunks is not None:
                result.append(offset_chunks)
                continue

            # get the token groups
            token_groups = list(self._token_group_generator(tokens))

            # get the token counts
            token_counts = [len(token_group) for token_group in token_groups]

            # decode the token groups into the chunk texts
            chunk_texts = self.tokenizer.decode_batch(token_groups)

            # create the chunks from the token groups and token counts
            chunks = self._create_chunks(chunk_texts, token_groups, token_counts)
            result.append(chunks)

        return result

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
