"""SemanticChunker with advanced peak detection and window embedding calculation.

This chunker uses peak detection to find split points instead of a simple threshold,
and calculates window embeddings directly rather than approximating them from sentence embeddings.
It uses Savitzky-Golay filtering for smoother boundary detection.
"""

import os
from typing import Any, Literal, Optional, Union

import chonkie_core
import numpy as np

from chonkie.embeddings import AutoEmbeddings, BaseEmbeddings
from chonkie.logger import get_logger
from chonkie.pipeline import chunker
from chonkie.types import Chunk, Sentence
from chonkie.utils import Hubbie

from .base import BaseChunker, split_text_by_delimiters

logger = get_logger(__name__)


@chunker("semantic")
class SemanticChunker(BaseChunker):
    """SemanticChunker uses peak detection to find split points and direct window embedding calculation.

    This chunker improves on traditional semantic chunking by using Savitzky-Golay filtering
    for smoother boundary detection and calculating window embeddings directly for more accurate
    semantic similarity computation.
    """

    def __init__(
        self,
        embedding_model: Union[str, BaseEmbeddings] = "minishlab/potion-base-32M",
        threshold: float = 0.8,
        chunk_size: int = 2048,
        similarity_window: int = 3,
        min_sentences_per_chunk: int = 1,
        min_characters_per_sentence: int = 24,
        delim: Union[str, list[str]] = [". ", "! ", "? ", "\n"],
        include_delim: Optional[Literal["prev", "next"]] = "prev",
        skip_window: int = 0,
        filter_window: int = 5,
        filter_polyorder: int = 3,
        filter_tolerance: float = 0.2,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the SemanticChunker.

        Args:
            embedding_model: Name of the sentence-transformers model to load
            threshold: Threshold for semantic similarity (0-1)
            chunk_size: Maximum tokens allowed per chunk
            similarity_window: Number of sentences to consider for similarity threshold calculation
            min_sentences_per_chunk: Minimum number of sentences per chunk
            min_characters_per_sentence: Minimum number of characters per sentence
            delim: Delimiter to use for sentence splitting
            include_delim: Whether to include the delimiter in the sentence
            skip_window: Number of groups to skip when merging (0=disabled, >0=enabled)
            filter_window: Window length for the Savitzky-Golay filter
            filter_polyorder: Polynomial order for the Savitzky-Golay filter
            filter_tolerance: Tolerance for the Savitzky-Golay filter
            **kwargs: Additional keyword arguments

        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if similarity_window <= 0:
            raise ValueError("similarity_window must be positive")
        if min_sentences_per_chunk <= 0:
            raise ValueError("min_sentences_per_chunk must be positive")
        if skip_window < 0:
            raise ValueError("skip_window must be non-negative")
        if not isinstance(threshold, (int, float)) or threshold <= 0 or threshold >= 1:
            raise ValueError("threshold must be between 0 and 1")
        if type(delim) not in [str, list]:
            raise ValueError("delim must be a string or list of strings")
        if filter_window <= 0:
            raise ValueError("filter_window must be positive")
        if filter_polyorder < 0 or filter_polyorder >= filter_window:
            raise ValueError("filter_polyorder must be non-negative and less than filter_window")
        if filter_tolerance <= 0 or filter_tolerance >= 1:
            raise ValueError("filter_tolerance must be between 0 and 1")

        if isinstance(embedding_model, str):
            self.embedding_model = AutoEmbeddings.get_embeddings(embedding_model)
        elif isinstance(embedding_model, BaseEmbeddings):
            self.embedding_model = embedding_model
        else:
            raise ValueError("embedding_model must be a string or a BaseEmbeddings object")

        # Initialize the tokenizer and chunker
        tokenizer = self.embedding_model.get_tokenizer()
        super().__init__(tokenizer)

        # Initialize the chunker parameters
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.similarity_window = similarity_window
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.skip_window = skip_window
        self.delim = delim
        self.include_delim = include_delim
        self.sep = "✄"
        self.min_characters_per_sentence = min_characters_per_sentence
        self.filter_window = filter_window
        self.filter_polyorder = filter_polyorder
        self.filter_tolerance = filter_tolerance

        # Set the multiprocessing flag to False
        self._use_multiprocessing = False

    @classmethod
    def from_recipe(
        cls,
        name: str = "default",
        lang: Optional[str] = "en",
        path: str | os.PathLike | None = None,
        embedding_model: Union[str, BaseEmbeddings] = "minishlab/potion-base-32M",
        threshold: float = 0.8,
        chunk_size: int = 2048,
        similarity_window: int = 3,
        min_sentences_per_chunk: int = 1,
        min_characters_per_sentence: int = 24,
        delim: Union[str, list[str]] = [". ", "! ", "? ", "\n"],
        include_delim: Optional[Literal["prev", "next"]] = "prev",
        skip_window: int = 0,
        filter_window: int = 5,
        filter_polyorder: int = 3,
        filter_tolerance: float = 0.2,
        **kwargs: dict[str, Any],
    ) -> "SemanticChunker":
        """Create a SemanticChunker from a recipe.

        Args:
            name: The name of the recipe to use.
            lang: The language that the recipe should support.
            path: The path to the recipe to use.
            embedding_model: The embedding model to use.
            threshold: The threshold to use for semantic similarity.
            chunk_size: The maximum tokens allowed per chunk.
            similarity_window: The number of sentences to consider for similarity threshold calculation.
            min_sentences_per_chunk: The minimum number of sentences per chunk.
            min_characters_per_sentence: The minimum number of characters per sentence.
            delim: The delimiter to use for sentence splitting.
            include_delim: Whether to include the delimiter in the sentence.
            skip_window: Window size for merging non-consecutive groups (0 to disable)
            filter_window: Window length for the Savitzky-Golay filter
            filter_polyorder: Polynomial order for the Savitzky-Golay filter
            filter_tolerance: Tolerance for the Savitzky-Golay filter
            **kwargs: Additional keyword arguments

        """
        hub = Hubbie()
        recipe = hub.get_recipe(name, lang, path)
        return cls(
            embedding_model=embedding_model,
            threshold=threshold,
            chunk_size=chunk_size,
            similarity_window=similarity_window,
            min_sentences_per_chunk=min_sentences_per_chunk,
            min_characters_per_sentence=min_characters_per_sentence,
            delim=recipe["recipe"]["delimiters"],
            include_delim=recipe["recipe"]["include_delim"],
            skip_window=skip_window,
            filter_window=filter_window,
            filter_polyorder=filter_polyorder,
            filter_tolerance=filter_tolerance,
            **kwargs,
        )

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using shared delimiter splitting.

        Args:
            text: Input text to be split into sentences

        Returns:
            List of sentences

        """
        return split_text_by_delimiters(
            text,
            delimiters=self.delim,
            include_delim=self.include_delim or "prev",
            min_chars=self.min_characters_per_sentence,
        )

    def _prepare_sentences(self, text: str) -> list[Sentence]:
        """Prepare the sentences for chunking."""
        # Handle empty or whitespace-only text
        if not text or text.isspace():
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        token_counts = self.tokenizer.count_tokens_batch(sentences)
        return [
            Sentence(text=s, start_index=i, end_index=i + len(s), token_count=tc)
            for (i, (s, tc)) in enumerate(zip(sentences, token_counts))
        ]

    def _get_sentence_embeddings(self, sentences: list[Sentence]) -> list[np.ndarray]:
        """Get the embeddings for the sentences."""
        return self.embedding_model.embed_batch([
            s.text for s in sentences[self.similarity_window :]
        ])

    def _get_window_embeddings(self, sentences: list[Sentence]) -> list[np.ndarray]:
        """Get the embeddings for the window."""
        paragraphs = []
        for i in range(len(sentences) - self.similarity_window):
            paragraphs.append("".join([s.text for s in sentences[i : i + self.similarity_window]]))
        return self.embedding_model.embed_batch(paragraphs)

    def _get_similarity(self, sentences: list[Sentence]) -> list[float]:
        """Get the similarity between the window and the sentence embeddings."""
        window_embeddings = self._get_window_embeddings(sentences)
        sentence_embeddings = self._get_sentence_embeddings(sentences)
        similarities = [
            float(self.embedding_model.similarity(w, s))
            for w, s in zip(window_embeddings, sentence_embeddings)
        ]
        return similarities

    def _get_split_indices(self, similarities: Union[list[float], np.ndarray]) -> list[int]:
        """Get split indices using optimized Savitzky-Golay filter with interpolation."""
        # Convert to numpy array if needed
        if not isinstance(similarities, np.ndarray):
            similarities = np.asarray(similarities, dtype=np.float64)

        # Handle case where data is too small for the filter window
        if len(similarities) == 0:
            return []
        if len(similarities) < self.filter_window:
            # If data is too small for filter, return boundaries only
            return []

        # Use optimized Rust implementation with interpolation
        minima_indices, minima_values = chonkie_core.find_local_minima_interpolated(
            similarities,
            window_size=self.filter_window,
            poly_order=self.filter_polyorder,
            tolerance=self.filter_tolerance,
        )

        # Handle empty case
        if len(minima_indices) == 0:
            return []

        # Filter by percentile and minimum distance
        filtered_indices, _ = chonkie_core.filter_split_indices(
            minima_indices,
            minima_values,
            self.threshold,
            self.min_sentences_per_chunk,
        )
        split_indices_list = filtered_indices.tolist()  # Convert numpy array to list

        # Add boundaries with window offset
        return (
            [0]
            + [int(i + self.similarity_window) for i in split_indices_list]
            + [len(similarities) + self.similarity_window]
        )

    def _compute_group_embeddings_batch(self, groups: list[list[Sentence]]) -> list[np.ndarray]:
        """Compute embeddings for all groups in batch.

        Args:
            groups: List of sentence groups

        Returns:
            List of embedding vectors, one for each group

        """
        if not groups:
            return []

        # Combine sentences in each group into a single text
        group_texts = []
        for group in groups:
            combined_text = "".join([s.text for s in group])
            group_texts.append(combined_text)

        # Get embeddings for all groups in batch
        embeddings = self.embedding_model.embed_batch(group_texts)
        return embeddings

    def _get_windowed_similarity(
        self,
        sentences: list[Sentence],
    ) -> Union[list[float], np.ndarray]:
        """Alternative similarity computation using windowed cross-similarity.

        This can be more robust than pairwise window-sentence comparison.
        """
        # Get embeddings for all sentences
        embeddings = self.embedding_model.embed_batch([s.text for s in sentences])
        # Stack embeddings into a 2D numpy array for the Rust extension
        embeddings_array = np.stack(embeddings, axis=0)
        result = chonkie_core.windowed_cross_similarity(
            embeddings_array, self.similarity_window * 2 + 1
        )
        return result

    def _skip_and_merge(self, groups: list[list[Sentence]]) -> list[list[Sentence]]:
        """Merge similar groups considering skip window.

        Args:
            groups: List of sentence groups to potentially merge

        Returns:
            List of merged groups

        """
        if len(groups) <= 1 or self.skip_window == 0:
            return groups

        # Get embeddings for all groups in batch for efficiency
        group_texts = ["".join([s.text for s in group]) for group in groups]
        embeddings = self.embedding_model.embed_batch(group_texts)

        merged_groups = []
        i = 0

        while i < len(groups):
            if i == len(groups) - 1:
                # Last group, can't merge with anything
                merged_groups.append(groups[i])
                break

            # Calculate skip index ensuring it's valid
            skip_index = min(i + self.skip_window + 1, len(groups) - 1)

            # Find the best merge candidate within the skip window
            best_similarity = -1.0
            best_idx = -1

            # Check similarity with all groups within skip window
            for j in range(i + 1, min(skip_index + 1, len(groups))):
                similarity = float(self.embedding_model.similarity(embeddings[i], embeddings[j]))
                if similarity >= self.threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_idx = j

            if best_idx != -1:
                # Merge groups from i to best_idx (inclusive)
                merged = []
                for k in range(i, best_idx + 1):
                    merged.extend(groups[k])
                merged_groups.append(merged)
                i = best_idx + 1  # Skip past merged groups
            else:
                # No merge possible, add current group
                merged_groups.append(groups[i])
                i += 1

        return merged_groups

    def _group_sentences(
        self,
        sentences: list[Sentence],
        split_indices: list[int],
    ) -> list[list[Sentence]]:
        """Group the sentences based on the split indices.

        Simply groups sentences between split points without any size checking.
        Size enforcement happens later in _split_groups.
        """
        groups = []

        # Handle empty split_indices
        if not split_indices:
            # Return all sentences as one group if no splits
            if sentences:
                groups.append(sentences)
            return groups

        # Create groups from sentences between split indices
        for i in range(len(split_indices) - 1):
            group = sentences[split_indices[i] : split_indices[i + 1]]
            if group:  # Only add non-empty groups
                groups.append(group)

        # Add the last group if there are remaining sentences
        if len(split_indices) > 0 and split_indices[-1] < len(sentences):
            remaining = sentences[split_indices[-1] :]
            if remaining:
                groups.append(remaining)

        return groups

    def _split_oversized_sentence(self, sentence: Sentence) -> list[Sentence]:
        """Split a single sentence that exceeds chunk_size into token-sized pieces.

        A sentence longer than chunk_size can never fit into any group, so grouping
        alone cannot enforce the limit. We split its text at the token level, the same
        way TokenChunker does, so no resulting piece exceeds chunk_size.

        Args:
            sentence: A sentence whose token_count exceeds chunk_size

        Returns:
            List of sentences, each with token_count <= chunk_size

        """
        tokens = self.tokenizer.encode(sentence.text)
        token_groups = [
            tokens[i : i + self.chunk_size] for i in range(0, len(tokens), self.chunk_size)
        ]
        texts = self.tokenizer.decode_batch(token_groups)
        return [
            Sentence(
                text=text,
                start_index=sentence.start_index,
                end_index=sentence.start_index + len(text),
                token_count=len(group),
            )
            for text, group in zip(texts, token_groups)
        ]

    def _split_groups(self, groups: list[list[Sentence]]) -> list[list[Sentence]]:
        """Split groups that exceed chunk_size into smaller groups.

        Args:
            groups: List of sentence groups

        Returns:
            List of groups where each respects the chunk_size limit

        """
        final_groups = []

        for group in groups:
            token_count = sum([s.token_count for s in group])

            if token_count <= self.chunk_size:
                final_groups.append(group)
            else:
                # Split the group into smaller chunks that respect chunk_size
                current_group = []
                current_token_count = 0

                for sentence in group:
                    # A single sentence larger than chunk_size can't fit in any
                    # group, so split it into token-sized pieces first.
                    pieces = (
                        self._split_oversized_sentence(sentence)
                        if sentence.token_count > self.chunk_size
                        else [sentence]
                    )

                    for piece in pieces:
                        if current_token_count + piece.token_count <= self.chunk_size:
                            current_group.append(piece)
                            current_token_count += piece.token_count
                        else:
                            if current_group:
                                final_groups.append(current_group)
                            current_group = [piece]
                            current_token_count = piece.token_count

                if current_group:
                    final_groups.append(current_group)

        return final_groups

    def _create_chunks(self, sentence_groups: list[list[Sentence]]) -> list[Chunk]:
        """Create a chunk from the sentence groups."""
        chunks = []
        current_index = 0
        for group in sentence_groups:
            text = "".join([s.text for s in group])
            token_count = sum([s.token_count for s in group])
            chunks.append(
                Chunk(
                    text=text,
                    start_index=current_index,
                    end_index=current_index + len(text),
                    token_count=token_count,
                ),
            )
            current_index += len(text)
        return chunks

    def chunk(self, text: str) -> list[Chunk]:
        """Chunk the text into semantic chunks."""
        # Handle empty text
        if not text or text.isspace():
            logger.debug("Empty or whitespace-only text provided")
            return []

        logger.debug(f"Starting semantic chunking for text of length {len(text)}")

        # Prepare the sentences
        sentences = self._prepare_sentences(text)
        logger.debug(f"Prepared {len(sentences)} sentences for semantic analysis")

        # Handle edge cases - too few sentences
        if len(sentences) <= self.similarity_window:
            # If we have any sentences, group them together, still enforcing
            # chunk_size so a few large sentences don't produce an oversized chunk.
            if sentences:
                return self._create_chunks(self._split_groups([sentences]))
            else:
                return []

        # Get the similarities
        similarities = self._get_similarity(sentences)

        # Get the split indices
        split_indices = self._get_split_indices(similarities)

        # Group the sentences into chunks based on split indices
        sentence_groups = self._group_sentences(sentences, split_indices)

        # Apply skip-and-merge if skip_window > 0
        if self.skip_window > 0:
            sentence_groups = self._skip_and_merge(sentence_groups)

        # Split groups that exceed chunk_size
        final_groups = self._split_groups(sentence_groups)

        # Create the chunks
        chunks = self._create_chunks(final_groups)

        logger.info(f"Created {len(chunks)} semantic chunks from {len(sentences)} sentences")
        # Return the chunks
        return chunks

    def __repr__(self) -> str:
        """Return a string representation of the SemanticChunker."""
        return (
            f"SemanticChunker(model={self.embedding_model}, "
            f"chunk_size={self.chunk_size}, "
            f"threshold={self.threshold}, "
            f"similarity_window={self.similarity_window}, "
            f"min_sentences_per_chunk={self.min_sentences_per_chunk}, "
            f"skip_window={self.skip_window}, "
            f"filter_window={self.filter_window}, "
            f"filter_polyorder={self.filter_polyorder}, "
            f"filter_tolerance={self.filter_tolerance})"
        )
