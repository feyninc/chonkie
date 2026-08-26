"""Regression test: a float chunk_overlap >= 1.0 must be rejected, not silently drop content.

TokenChunker.__init__ guarded `chunk_overlap >= chunk_size` only for int inputs. A float
like 1.5 is turned into int(1.5 * chunk_size), which can exceed chunk_size, making the step
(chunk_size - chunk_overlap) negative -> range() is empty -> chunk() returns [] and the whole
document is silently dropped from a RAG index. The docstring promises ValueError when
chunk_overlap >= chunk_size; this enforces it for floats too.

  with the fix -> PASS (raises ValueError at construction; valid fractional overlap still works)
  without it   -> FAIL (no raise; chunk() silently returns [])
"""
import pytest

from chonkie import TokenChunker


def test_float_overlap_ge_one_is_rejected():
    with pytest.raises(ValueError):
        TokenChunker(tokenizer="character", chunk_size=10, chunk_overlap=1.5)


def test_valid_fractional_overlap_still_chunks():
    chunker = TokenChunker(tokenizer="character", chunk_size=10, chunk_overlap=0.1)  # -> 1 token
    chunks = chunker.chunk("a" * 100)
    assert len(chunks) > 0
