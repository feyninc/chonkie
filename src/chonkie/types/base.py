"""Custom base types for Chonkie."""

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Union
from uuid import uuid4

import numpy as np


# Function to generate the IDs for the Chonkie  types
def generate_id(prefix: str) -> str:
    """Generate a UUID for a given prefix."""
    return f"{prefix}_{uuid4().hex}"


@dataclass
class Chunk:
    """Chunks with metadata.

    Attributes:
        text (str): The text of the chunk.
        start_index (int): The starting index of the chunk in the original text.
        end_index (int): The ending index of the chunk in the original text.
        token_count (int): The number of tokens in the chunk.
        context (Optional[str]): Optional context metadata for the chunk.
        embedding (Union[list[float], np.ndarray, None]): Optional embedding vector for the chunk,
            either as a list of floats or a numpy array.
        metadata (dict[str, Any]): Arbitrary per-chunk metadata; ``BaseChunker.chunk_document``
            merges in the parent :class:`~chonkie.types.Document` metadata (chunk keys win
            on duplicate keys).

    """

    id: str = field(default_factory=lambda: generate_id("chnk"))
    text: str = field(default="")
    start_index: int = field(default=0)
    end_index: int = field(default=0)
    token_count: int = field(default=0)
    context: Optional[str] = field(default=None)
    embedding: Union[list[float], np.ndarray, None] = field(default=None)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return the length of the text."""
        return len(self.text)

    def __str__(self) -> str:
        """Return a string representation of the Chunk."""
        return self.text

    def _preview_embedding(self) -> str:
        """Create a preview string for the embedding.

        Shows first 3 and last 2 values for long embeddings,
        or full embedding for short ones.

        Returns:
            A string preview of the embedding

        """
        if self.embedding is None:
            return ""

        try:
            # Check if it's array-like with length
            if hasattr(self.embedding, "__len__") and hasattr(self.embedding, "__getitem__"):
                emb_len = len(self.embedding)
                if emb_len > 5:
                    # Show first 3 and last 2 values
                    preview = f"[{self.embedding[0]:.4f}, {self.embedding[1]:.4f}, {self.embedding[2]:.4f}, ..., {self.embedding[-2]:.4f}, {self.embedding[-1]:.4f}]"
                else:
                    # Show all values if 5 or fewer
                    preview = "[" + ", ".join(f"{v:.4f}" for v in self.embedding) + "]"

                # Add shape info if available
                if hasattr(self.embedding, "shape"):
                    preview += f" shape={self.embedding.shape}"

                return preview
            else:
                return str(self.embedding)
        except Exception:
            return "<embedding>"

    def __repr__(self) -> str:
        """Return a detailed string representation of the Chunk."""
        repr = (
            f"Chunk(text='{self.text}', token_count={self.token_count}, "
            f"start_index={self.start_index}, end_index={self.end_index}"
        )
        if self.context:
            repr += f", context={self.context}"
        if self.embedding is not None:
            repr += f", embedding={self._preview_embedding()}"
        return repr + ")"

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over the chunk's text."""
        return iter(self.text)

    def __getitem__(self, index: int) -> str:
        """Return a slice of the chunk's text."""
        return self.text[index]

    def to_dict(self) -> dict:
        """Return the Chunk as a dictionary."""
        result = self.__dict__.copy()
        result["context"] = self.context
        # Convert embedding to list if it has tolist method (numpy array)
        if self.embedding is not None:
            if isinstance(self.embedding, np.ndarray):
                result["embedding"] = self.embedding.tolist()  # ty: ignore[no-matching-overload]
            else:
                result["embedding"] = self.embedding
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        """Create a Chunk object from a dictionary."""
        return cls(
            id=data.get("id", generate_id("chnk")),
            text=data["text"],
            start_index=data["start_index"],
            end_index=data["end_index"],
            token_count=data["token_count"],
            context=data.get("context", None),
            embedding=data.get("embedding", None),
            metadata=dict(data.get("metadata") or {}),
        )

    def copy(self) -> "Chunk":
        """Return a deep copy of the chunk."""
        return Chunk.from_dict(self.to_dict())
