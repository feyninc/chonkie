"""Base Class for All Chunkers."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Optional, Sequence, Union

import chonkie_core
from tqdm import tqdm

from chonkie.logger import get_logger
from chonkie.tokenizer import AutoTokenizer, TokenizerProtocol
from chonkie.types import Chunk, Document

logger = get_logger(__name__)


def split_text_by_delimiters(
    text: str,
    delimiters: Union[str, list[str]],
    include_delim: Optional[str] = "prev",
    min_chars: int = 1,
) -> list[str]:
    """Split text at delimiter boundaries using chonkie-core.

    This is the shared implementation used by SentenceChunker,
    SemanticChunker, RecursiveChunker, and SlumberChunker.

    Args:
        text: Input text to be split.
        delimiters: Delimiter(s) to split on. Can be a single string or list of strings.
        include_delim: Whether to include delimiters in the current chunk ("prev"),
            the next chunk ("next"), or to drop them entirely (None).
            Defaults to "prev".
        min_chars: Minimum number of characters per split.

    Returns:
        List of non-empty text splits.

    """
    if isinstance(delimiters, str):
        delimiters = [delimiters]

    text_bytes = text.encode("utf-8")
    has_complex_delimiters = any(len(d) > 1 or not d.isascii() for d in delimiters)

    if has_complex_delimiters:
        patterns: list[bytes] = [d.encode("utf-8") for d in delimiters]
        offsets = chonkie_core.split_pattern_offsets(
            text_bytes,
            patterns=patterns,
            include_delim=include_delim,
            min_chars=min_chars,
        )
    else:
        delim_bytes = "".join(delimiters).encode("utf-8")
        offsets = chonkie_core.split_offsets(
            text_bytes,
            delimiters=delim_bytes,
            include_delim=include_delim,
            min_chars=min_chars,
        )

    splits = [text_bytes[start:end].decode("utf-8") for start, end in offsets]
    return [s for s in splits if s]


class BaseChunker(ABC):
    """Base class for all chunkers."""

    def __init__(self, tokenizer: Union[str, TokenizerProtocol] = "gpt2"):
        """Initialize the chunker with any necessary parameters.

        Args:
            tokenizer: The tokenizer to use. Can be:
                - A string identifier (e.g., "gpt2", "character", "word")
                - An object implementing TokenizerProtocol (encode, decode, tokenize methods)

        """
        self._tokenizer = AutoTokenizer(tokenizer)
        self._use_multiprocessing = False
        logger.debug(
            f"Initialized {self.__class__.__name__}",
            tokenizer=str(tokenizer)[:50],
        )

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Get the tokenizer instance."""
        return self._tokenizer

    def __repr__(self) -> str:
        """Return a string representation of the chunker."""
        return f"{self.__class__.__name__}()"

    def __call__(
        self,
        text: Union[str, Sequence[str]],
        show_progress: bool = True,
    ) -> Union[list[Chunk], list[list[Chunk]]]:
        """Call the chunker with the given arguments.

        Args:
            text (Union[str, Sequence[str]]): The text to chunk.
            show_progress (bool): Whether to show progress.

        Returns:
            If the input is a string, return a list of Chunks.
            If the input is a list of strings, return a list of lists of Chunks.

        """
        if isinstance(text, str):
            return self.chunk(text)
        elif isinstance(text, Sequence):
            return self.chunk_batch(text, show_progress)
        else:
            raise ValueError("Input must be a string or a list of strings.")

    def _get_optimal_worker_count(self) -> int:
        """Get the optimal number of workers for parallel processing."""
        try:
            from multiprocessing import cpu_count

            cpu_cores = cpu_count()
            worker_count = min(8, max(1, cpu_cores * 3 // 4))
            logger.debug(
                f"Using {worker_count} workers for parallel processing",
                cpu_cores=cpu_cores,
            )
            return worker_count
        except Exception:
            logger.warning(
                "Failed to calculate optimal worker count, using 1 worker",
                exc_info=True,
            )
            return 1

    def _sequential_batch_processing(
        self,
        texts: Sequence[str],
        show_progress: bool = True,
    ) -> list[list[Chunk]]:
        """Process a batch of texts sequentially."""
        logger.info(f"Starting sequential batch processing of {len(texts)} texts")
        results = [
            self.chunk(t)
            for t in tqdm(
                texts,
                desc="🦛",
                disable=not show_progress,
                unit="doc",
                bar_format="{desc} ch{bar:20}nk {percentage:3.0f}% • {n_fmt}/{total_fmt} docs chunked [{elapsed}<{remaining}, {rate_fmt}] 🌱",
                ascii=" o",
            )
        ]
        total_chunks = sum(len(r) for r in results)
        logger.info(
            f"Completed sequential processing: {total_chunks} total chunks from {len(texts)} texts",
        )
        return results

    def _parallel_batch_processing(
        self,
        texts: Sequence[str],
        show_progress: bool = True,
    ) -> list[list[Chunk]]:
        """Process a batch of texts using multiprocessing."""
        from multiprocessing import Pool

        num_workers = self._get_optimal_worker_count()
        total = len(texts)
        chunk_size = max(1, min(total // (num_workers * 16), 10))

        logger.info(
            f"Starting parallel batch processing of {total} texts",
            workers=num_workers,
            chunk_size=chunk_size,
        )

        with Pool(processes=num_workers) as pool:
            results = []
            with tqdm(
                total=total,
                desc="🦛",
                disable=not show_progress,
                unit="doc",
                bar_format="{desc} ch{bar:20}nk {percentage:3.0f}% • {n_fmt}/{total_fmt} docs chunked [{elapsed}<{remaining}, {rate_fmt}] 🌱",
                ascii=" o",
            ) as progress_bar:
                for result in pool.imap(self.chunk, texts, chunksize=chunk_size):
                    results.append(result)
                    progress_bar.update()

            total_chunks = sum(len(r) for r in results)
            logger.info(
                f"Completed parallel processing: {total_chunks} total chunks from {total} texts",
            )
            return results

    @abstractmethod
    def chunk(self, text: str) -> list[Chunk]:
        """Chunk the given text.

        Args:
            text (str): The text to chunk.

        Returns:
            list[Chunk]: A list of Chunks.

        """
        pass

    def chunk_batch(self, texts: Sequence[str], show_progress: bool = True) -> list[list[Chunk]]:
        """Chunk a batch of texts.

        Args:
            texts (Sequence[str]): The texts to chunk.
            show_progress (bool): Whether to show progress.

        Returns:
            list[list[Chunk]]: A list of lists of Chunks.

        """
        # simple handles of empty and single text cases
        if len(texts) == 0:
            return []
        if len(texts) == 1:
            return [self.chunk(texts[0])]

        # Now for the remaining, check the self._multiprocessing bool flag
        if self._use_multiprocessing:
            return self._parallel_batch_processing(texts, show_progress)
        else:
            return self._sequential_batch_processing(texts, show_progress)

    async def achunk(self, text: str) -> list[Chunk]:
        """Chunk the given text asynchronously.

        Args:
            text (str): The text to chunk.

        Returns:
            list[Chunk]: A list of Chunks.

        """
        return await asyncio.to_thread(self.chunk, text)

    async def achunk_batch(
        self, texts: Sequence[str], show_progress: bool = True
    ) -> list[list[Chunk]]:
        """Chunk a batch of texts asynchronously.

        Args:
            texts (Sequence[str]): The texts to chunk.
            show_progress (bool): Whether to show progress.

        Returns:
            list[list[Chunk]]: A list of lists of Chunks.

        """
        return await asyncio.to_thread(self.chunk_batch, texts, show_progress)

    @staticmethod
    def _merge_new_chunks(
        original_chunks: list[Chunk], new_chunk_batches: list[list[Chunk]]
    ) -> list[Chunk]:
        """Merge new chunks batches into a single list, shifting indices.

        Args:
            original_chunks: The original chunks from the document.
            new_chunk_batches: The new batches of chunks corresponding to each original chunk.

        Returns:
            list[Chunk]: The merged and shifted chunks.

        """
        return [
            replace(
                c,
                start_index=c.start_index + old_chunk.start_index,
                end_index=c.end_index + old_chunk.start_index,
            )
            for old_chunk, new_chunks in zip(original_chunks, new_chunk_batches)
            for c in new_chunks
        ]

    @staticmethod
    def _propagate_document_metadata(document: Document) -> None:
        """Merge ``document.metadata`` into each chunk (existing chunk keys take precedence)."""
        if not document.metadata or not document.chunks:
            return
        doc_meta = document.metadata
        for chunk in document.chunks:
            chunk.metadata = {**doc_meta, **chunk.metadata}

    def chunk_document(self, document: Document) -> Document:
        """Chunk a document.

        After chunking, non-empty ``document.metadata`` is shallow-merged into each
        chunk's :attr:`~chonkie.types.Chunk.metadata` (chunk keys override on conflict).

        Args:
            document: The document to chunk.

        Returns:
            The document with chunks populated.

        """
        # If the document has chunks already, then we need to re-chunk the content
        if document.chunks:
            chunk_results = [self.chunk(c.text) for c in document.chunks]
            document.chunks = self._merge_new_chunks(document.chunks, chunk_results)
        else:
            document.chunks = self.chunk(document.content)
        self._propagate_document_metadata(document)
        return document

    async def achunk_document(self, document: Document) -> Document:
        """Chunk a document asynchronously.

        Args:
            document: The document to chunk.

        Returns:
            The document with chunks populated.

        """
        # If the document has chunks already, then we need to re-chunk the content
        if document.chunks:
            tasks = [self.achunk(c.text) for c in document.chunks]
            chunk_results = await asyncio.gather(*tasks)
            document.chunks = self._merge_new_chunks(document.chunks, chunk_results)
        else:
            document.chunks = await self.achunk(document.content)
        self._propagate_document_metadata(document)
        return document
