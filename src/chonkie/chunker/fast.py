"""Fast chunker powered by chonkie-core."""

from bisect import bisect_left
from typing import Any, Dict, List, Optional, Sequence

import chonkie_core

from chonkie.chunker.base import BaseChunker
from chonkie.pipeline import chunker
from chonkie.types import Chunk


@chunker("fast")
class FastChunker(BaseChunker):
    r"""Fast byte-based chunker using SIMD-accelerated boundary detection.

    Unlike other chonkie chunkers that use token counts, FastChunker uses
    byte size limits for maximum performance (~100+ GB/s throughput).

    This is a thin wrapper around chonkie-core's chunking functionality.

    Args:
        chunk_size: Target chunk size in bytes (default: 4096)
        delimiters: Delimiter characters for splitting (default: "\n.?")
        pattern: Multi-byte pattern to split on (overrides delimiters)
        prefix: Put delimiter at start of next chunk (default: False)
        consecutive: Split at START of consecutive runs (default: False)
        forward_fallback: Search forward if no delimiter in backward window

    Example:
        >>> chunker = FastChunker(chunk_size=1024)
        >>> chunks = chunker("Your long document here...")
        >>> for chunk in chunks:
        ...     print(chunk.text[:50])

    """

    def __init__(
        self,
        chunk_size: int = 4096,
        delimiters: str = "\n.?",
        pattern: Optional[str] = None,
        prefix: bool = False,
        consecutive: bool = False,
        forward_fallback: bool = False,
    ):
        """Initialize the FastChunker."""
        # Don't call super().__init__() - we don't need a tokenizer
        # But set required attributes for BaseChunker compatibility
        self._tokenizer = None
        self._use_multiprocessing = False

        self.chunk_size = chunk_size
        self.delimiters = delimiters
        self.pattern = pattern
        self.prefix = prefix
        self.consecutive = consecutive
        self.forward_fallback = forward_fallback

    def __repr__(self) -> str:
        """Return a string representation of the chunker."""
        return (
            f"FastChunker(chunk_size={self.chunk_size}, delimiters={self.delimiters!r}, "
            f"pattern={self.pattern!r}, prefix={self.prefix}, "
            f"consecutive={self.consecutive}, forward_fallback={self.forward_fallback})"
        )

    def chunk(self, text: str) -> List[Chunk]:
        """Chunk text at delimiter boundaries.

        Args:
            text: Input text to chunk

        Returns:
            List of Chunk objects

        """
        if not text:
            return []

        # Build kwargs for chonkie-core
        kwargs: Dict[str, Any] = {
            "size": self.chunk_size,
            "prefix": self.prefix,
            "consecutive": self.consecutive,
            "forward_fallback": self.forward_fallback,
        }

        if self.pattern:
            kwargs["pattern"] = self.pattern
        else:
            kwargs["delimiters"] = self.delimiters

        # Encode to bytes for chonkie-core (which works with byte offsets)
        text_bytes = text.encode("utf-8")

        # Get chunk offsets from chonkie-core (these are byte offsets)
        offsets = chonkie_core.chunk_offsets(text_bytes, **kwargs)

        # chonkie-core returns byte offsets, and a hard size cut can land inside a
        # multi-byte UTF-8 character; slicing the raw bytes there raises
        # UnicodeDecodeError. Pure-ASCII text has byte offsets == char offsets, so
        # the common case needs no remapping.
        if len(text_bytes) == len(text):
            return [
                Chunk(text=text[start:end], start_index=start, end_index=end, token_count=0)
                for start, end in offsets
                if end > start
            ]

        # Otherwise snap each byte offset up to a character boundary so a chunk
        # never splits a multi-byte character, then slice the decoded text.
        char_starts = [i for i, b in enumerate(text_bytes) if (b & 0xC0) != 0x80]
        char_starts.append(len(text_bytes))

        chunks = []
        for start, end in offsets:
            char_start = bisect_left(char_starts, start)
            char_end = bisect_left(char_starts, end)
            if char_end <= char_start:
                continue
            chunks.append(
                Chunk(
                    text=text[char_start:char_end],
                    start_index=char_start,
                    end_index=char_end,
                    token_count=0,
                )
            )
        return chunks

    def chunk_batch(self, texts: Sequence[str], show_progress: bool = True) -> List[List[Chunk]]:
        """Chunk a batch of texts.

        Args:
            texts: The texts to chunk.
            show_progress: Whether to show progress (ignored, always fast).

        Returns:
            A list of lists of Chunks.

        """
        return [self.chunk(text) for text in texts]
