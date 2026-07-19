"""Table chunker for processing markdown tables."""

import re
from typing import Union

from chonkie.chunker.base import BaseChunker
from chonkie.logger import get_logger
from chonkie.pipeline import chunker
from chonkie.tokenizer import RowTokenizer, TokenizerProtocol
from chonkie.types import Chunk, Document, MarkdownDocument

logger = get_logger(__name__)


@chunker("table")
class TableChunker(BaseChunker):
    """Chunker that chunks tables based on character count on each row or number of rows."""

    def __init__(
        self,
        tokenizer: Union[str, TokenizerProtocol] = "row",
        chunk_size: int = 3,
    ) -> None:
        """Initialize the TableChunker with configuration parameters.

        Args:
            tokenizer: The tokenizer to use for chunking. Use "row" for row-based
                chunking (default), or any other tokenizer string/instance for
                token-based chunking.
            chunk_size: The maximum size of each chunk. When using "row" tokenizer,
                this is the maximum number of data rows per chunk. For other
                tokenizers, this is the maximum number of tokens.

        """
        super().__init__(tokenizer)

        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0.")
        if chunk_size == 3 and tokenizer != "row":
            logger.warning(
                "Using default chunk size of 3 with a token-based tokenizer may not be optimal. "
                "Consider using a larger chunk_size for token-based chunking.",
            )

        self.chunk_size = chunk_size
        self.newline_pattern = re.compile(r"\n(?=\|)")
        self.sep = "✄"
        self._is_row_tokenizer = isinstance(self.tokenizer.tokenizer, RowTokenizer)

    def _split_markdown_table(self, table: str) -> tuple[str, list[str]]:
        table = table.strip()
        # insert separator right after the newline that precedes a pipe
        raw = self.newline_pattern.sub(rf"\n{self.sep}", table)
        chunks = [c for c in raw.split(self.sep) if c]  # keep empty strings away
        header = "".join(chunks[:2])  # header line + separator line
        rows = chunks[2:]  # data rows still contain their trailing \n
        return header, rows

    def _find_html_rows(self, body_content: str) -> list[str]:
        """Extract <tr>...</tr> rows using plain string search (O(n), no ReDoS)."""
        rows = []
        lower = body_content.lower()
        pos = 0
        while True:
            tr_start = lower.find("<tr", pos)
            if tr_start == -1:
                break
            tr_tag_end = lower.find(">", tr_start)
            if tr_tag_end == -1:
                break
            close_start = lower.find("</tr>", tr_tag_end + 1)
            if close_start == -1:
                break
            rows.append(body_content[tr_start : close_start + 5])  # 5 = len("</tr>")
            pos = close_start + 5
        return rows

    def _split_html_table(self, table: str) -> tuple[str, list[str]]:
        table = table.strip()
        lower = table.lower()
        # Find the start of <tbody...> using plain string search (O(n), no ReDoS)
        tbody_start = lower.find("<tbody")
        if tbody_start != -1:
            tbody_tag_end = lower.find(">", tbody_start)
            if tbody_tag_end != -1:
                header = table[: tbody_tag_end + 1]
                tbody_close = lower.find("</tbody>", tbody_tag_end + 1)
                body_end = tbody_close if tbody_close != -1 else len(table)
                body_content = table[tbody_tag_end + 1 : body_end]
                rows = self._find_html_rows(body_content)
                return header, rows
        # If no tbody, assume everything after the first matching row is data
        rows = self._find_html_rows(table)
        if not rows:
            return table, []
        header = table[: table.find(rows[0])]
        return header, rows

    def chunk(self, text: str) -> list[Chunk]:
        """Chunk the table into smaller tables based on the chunk size.

        Args:
            text: The input markdown or HTML table as a string.

        Returns:
            list[Chunk]: A list of table chunks.

        """
        logger.debug(f"Starting table chunking for table of length {len(text)}")
        chunks: list[Chunk] = []
        # Basic validation
        if not text.strip():
            logger.warning("No table content found. Skipping chunking.")
            return []

        # Detect table type (plain string check avoids ReDoS)
        is_html = "<table" in text.lower()

        if is_html:
            header, data_rows = self._split_html_table(text)
            footer = "</tbody></table>" if "</tbody>" in text.lower() else "</table>"
            if len(data_rows) < 1:
                logger.warning("HTML table must have at least one data row. Skipping chunking.")
                return []
        else:
            rows = text.strip().split("\n")
            if len(rows) < 3:  # Need header, separator, and at least one data row
                logger.warning(
                    "Table must have at least a header, separator, and one data row. Skipping chunking.",
                )
                return []
            header, data_rows = self._split_markdown_table(text)
            footer = ""

        # row based table chunking
        if self._is_row_tokenizer:
            if len(data_rows) <= self.chunk_size:
                return [
                    Chunk(
                        text=text,
                        token_count=len(data_rows),
                        start_index=0,
                        end_index=len(text),
                    ),
                ]
            else:
                # Track character position. The first chunk's span includes the
                # header (which physically sits at the start of the text), so it
                # begins at index 0; subsequent chunks span only their data rows.
                # This mirrors the token-based path below and keeps chunk spans
                # contiguous over the whole table starting from index 0.
                header_len = len(header)
                current_char_index = 0

                for i in range(0, len(data_rows), self.chunk_size):
                    chunk_rows = data_rows[i : i + self.chunk_size]
                    chunk_text = header + "".join(chunk_rows) + footer
                    span_len = len("".join(chunk_rows)) + (header_len if i == 0 else 0)

                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            token_count=len(chunk_rows),
                            start_index=current_char_index,
                            end_index=current_char_index + span_len,
                        ),
                    )
                    current_char_index += span_len

            return chunks

        # tokenizer based table chunking
        else:
            # Check if the table size is smaller than the chunk size
            table_token_count = self.tokenizer.count_tokens(text.strip())
            if table_token_count <= self.chunk_size:
                return [
                    Chunk(
                        text=text,
                        token_count=table_token_count,
                        start_index=0,
                        end_index=len(text),
                    ),
                ]

            header_token_count = self.tokenizer.count_tokens(header)
            footer_token_count = self.tokenizer.count_tokens(footer) if footer else 0
            current_token_count = header_token_count + footer_token_count
            current_index = 0
            current_chunk = [header]

            # split data rows into chunks
            for row in data_rows:
                row_size = self.tokenizer.count_tokens(row)
                # if adding this row exceeds chunk size
                if current_token_count + row_size >= self.chunk_size and len(current_chunk) > 1:
                    # only create a new chunk if the current chunk has more than just the header
                    # if the current chunk only has the header, we need to add the row anyway
                    if chunks == []:
                        chunk_text = "".join(current_chunk) + footer
                        chunk = Chunk(
                            text=chunk_text,
                            start_index=current_index,
                            end_index=current_index + len("".join(current_chunk)),
                            token_count=current_token_count,
                        )
                        chunks.append(chunk)
                        current_index = chunk.end_index
                    else:
                        chunk_text = "".join(current_chunk) + footer
                        chunk_len = len("".join(current_chunk)) - len(header)
                        chunk = Chunk(
                            text=chunk_text,
                            start_index=current_index,
                            end_index=current_index + chunk_len,
                            token_count=current_token_count,
                        )
                        chunks.append(chunk)
                        current_index = chunk.end_index
                    current_chunk = [header, row]
                    current_token_count = header_token_count + footer_token_count + row_size
                # if the current chunk is not full, we need to add the row to the current chunk
                else:
                    current_chunk.append(row)
                    current_token_count += row_size

            # if the current chunk is not full, we need to add the row to the current chunk
            if len(current_chunk) > 1:
                chunk_text = "".join(current_chunk) + footer
                chunk_len = (
                    len("".join(current_chunk)) - len(header) if chunks != [] else len(chunk_text)
                )
                chunk = Chunk(
                    text=chunk_text,
                    start_index=current_index,
                    end_index=current_index + chunk_len,
                    token_count=current_token_count,
                )
                chunks.append(chunk)

            logger.info(f"Created {len(chunks)} table chunks from table")
            return chunks

    def chunk_document(self, document: Document) -> Document:
        """Chunk a document."""
        logger.debug(
            f"Chunking document with {len(document.content) if hasattr(document, 'content') else 0} characters",
        )
        if isinstance(document, MarkdownDocument):
            if document.tables:
                logger.debug(f"Processing MarkdownDocument with {len(document.tables)} tables")
                for table in document.tables:
                    chunks = self.chunk(table.content)
                    for chunk in chunks:
                        chunk.start_index = table.start_index + chunk.start_index
                        chunk.end_index = table.start_index + chunk.end_index
                    document.chunks.extend(chunks)
                document.chunks.sort(key=lambda x: x.start_index)
            BaseChunker._propagate_document_metadata(document)
            return document
        return super().chunk_document(document)

    def __repr__(self) -> str:
        """Return a string representation of the TableChunker."""
        return f"TableChunker(tokenizer={self.tokenizer}, chunk_size={self.chunk_size})"
