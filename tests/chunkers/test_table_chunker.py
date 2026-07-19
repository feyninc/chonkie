"""Tests for the TableChunker class."""

from __future__ import annotations

import pytest

from chonkie import Chunk, RecursiveChunker, TableChunker
from chonkie.types import Document, MarkdownDocument, MarkdownTable


@pytest.fixture
def sample_table() -> str:
    """Fixture that returns a sample markdown table for testing."""
    table = """| Name | Age | City | Country | Occupation |
|------|-----|------|---------|------------|
| John | 25 | New York | USA | Engineer |
| Alice | 30 | London | UK | Designer |
| Bob | 35 | Paris | France | Manager |
| Carol | 28 | Tokyo | Japan | Developer |
| David | 40 | Berlin | Germany | Architect |
| Eva | 32 | Sydney | Australia | Analyst |
| Frank | 45 | Toronto | Canada | Consultant |
| Grace | 29 | Rome | Italy | Writer |
| Henry | 38 | Madrid | Spain | Teacher |
| Iris | 33 | Amsterdam | Netherlands | Researcher |"""
    return table


@pytest.fixture
def large_table() -> str:
    """Fixture that returns a large markdown table that should be chunked."""
    header = """| ID | First Name | Last Name | Email | Phone | Address | City | State | ZIP | Country | Department | Position | Salary | Start Date |
|-----|------------|-----------|-------|-------|---------|------|-------|-----|---------|------------|----------|--------|------------|"""

    rows = []
    for i in range(20):
        rows.append(
            f"| {i + 1:03d} | Person{i + 1} | Lastname{i + 1} | person{i + 1}@email.com | 555-{i + 1:04d} | {i + 1} Main St | City{i + 1} | ST | {10000 + i} | Country{i + 1} | Dept{i + 1} | Position{i + 1} | ${50000 + i * 1000} | 2023-01-{(i % 28) + 1:02d} |",
        )

    return header + "\n" + "\n".join(rows)


def test_table_chunker_initialization() -> None:
    """Test that the TableChunker can be initialized with default parameters."""
    chunker = TableChunker(tokenizer="character", chunk_size=2048)

    assert chunker is not None
    assert chunker.chunk_size == 2048
    assert hasattr(chunker, "tokenizer")


def test_table_chunker_initialization_with_params() -> None:
    """Test that the TableChunker can be initialized with custom parameters."""
    chunker = TableChunker(tokenizer="character", chunk_size=500)

    assert chunker is not None
    assert chunker.chunk_size == 500


def test_table_chunker_invalid_chunk_size() -> None:
    """Test that the TableChunker raises an error for invalid chunk size."""
    with pytest.raises(ValueError, match="Chunk size must be greater than 0"):
        TableChunker(chunk_size=0)

    with pytest.raises(ValueError, match="Chunk size must be greater than 0"):
        TableChunker(chunk_size=-1)


def test_table_chunker_small_table(sample_table: str) -> None:
    """Test that a small table returns a single chunk."""
    chunker = TableChunker(tokenizer="character", chunk_size=2048)
    chunks = chunker.chunk(sample_table)

    assert len(chunks) == 1
    assert chunks[0].text == sample_table
    assert chunks[0].start_index == 0
    assert chunks[0].end_index == len(sample_table)
    assert chunks[0].token_count == len(sample_table)


def test_table_chunker_large_table(large_table: str) -> None:
    """Test that a large table gets chunked into multiple pieces."""
    chunker = TableChunker(tokenizer="character", chunk_size=500)
    chunks = chunker.chunk(large_table)

    assert len(chunks) > 1
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk.token_count <= 500 for chunk in chunks)

    # Verify all chunks have the header
    header_lines = large_table.split("\n")[:2]
    expected_header = "\n".join(header_lines)

    for chunk in chunks:
        chunk_lines = chunk.text.split("\n")
        actual_header = "\n".join(chunk_lines[:2])
        assert actual_header == expected_header, (
            f"Chunk missing proper header: {chunk.text[:100]}..."
        )


def test_table_chunker_index_calculation(large_table: str) -> None:
    """Test that index calculations are correct when headers are added to chunks."""
    chunker = TableChunker(tokenizer="character", chunk_size=500)
    chunks = chunker.chunk(large_table)

    # Verify indices are sequential and non-overlapping (except for headers)
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        chunks[i + 1]

        # For all chunks after the first, the start should be where previous ended
        if i > 0:
            prev_chunk = chunks[i - 1]
            assert current_chunk.start_index == prev_chunk.end_index

        # Verify we can extract meaningful text from the original using indices
        if i == 0:
            # First chunk should start at 0
            assert current_chunk.start_index == 0

        # Check that end index is reasonable
        assert current_chunk.end_index > current_chunk.start_index


def test_table_chunker_preserves_content() -> None:
    """Test that all original data rows are preserved across chunks."""
    table = """| Name | Value |
|------|-------|
| A | 1 |
| B | 2 |
| C | 3 |
| D | 4 |
| E | 5 |"""

    chunker = TableChunker(tokenizer="character", chunk_size=50)  # Force chunking
    chunks = chunker.chunk(table)

    # Extract all data rows from chunks (skip header rows)
    all_data_rows = []
    for chunk in chunks:
        lines = chunk.text.split("\n")
        data_lines = lines[2:]  # Skip header and separator
        # Filter out empty strings that come from trailing newlines
        data_lines = [line for line in data_lines if line.strip()]
        all_data_rows.extend(data_lines)

    # Get original data rows
    original_lines = table.split("\n")
    original_data = original_lines[2:]

    # Should have same data rows (accounting for duplicates from multiple chunks)
    unique_data_rows = list(dict.fromkeys(all_data_rows))  # Remove duplicates, preserve order
    assert set(unique_data_rows) == set(original_data)


def test_table_chunker_invalid_table(caplog) -> None:
    """Test that the TableChunker handles invalid tables appropriately."""
    chunker = TableChunker(tokenizer="character", chunk_size=500)

    # Table with no rows (just header)
    chunks = chunker.chunk("| Name | Value |\n|------|-------|")
    assert len(chunks) == 0
    assert "Table must have at least a header, separator, and one data row" in caplog.text
    caplog.clear()

    # Single line (no table structure)
    chunks = chunker.chunk("Just a single line")
    assert len(chunks) == 0
    assert "Table must have at least a header, separator, and one data row" in caplog.text


def test_table_chunker_empty_input(caplog) -> None:
    """Test that the TableChunker handles empty input."""
    chunker = TableChunker(tokenizer="character", chunk_size=500)

    assert not chunker.chunk("")
    assert "No table content found" in caplog.text


def test_table_chunker_exact_chunk_size() -> None:
    """Test table chunking when rows exactly fit the chunk size."""
    # Create a table where each row is exactly a known size
    table = """| A | B |
|---|---|
| 1 | 2 |
| 3 | 4 |"""

    # Set chunk size to fit header + one row
    header_size = len("| A | B |\n|---|---|")
    row_size = len("\n| 1 | 2 |")
    chunk_size = header_size + row_size

    chunker = TableChunker(tokenizer="character", chunk_size=chunk_size)
    chunks = chunker.chunk(table)

    # Should create multiple chunks since we have 2 data rows
    assert len(chunks) >= 2

    # Each chunk should have the header
    for chunk in chunks:
        assert "| A | B |" in chunk.text
        assert "|---|---|" in chunk.text


def verify_chunk_indices(chunks: list[Chunk], original_text: str) -> None:
    """Verify that chunk indices correctly represent positions in original text."""
    # For table chunker, we need to account for the fact that indices represent
    # logical positions in the original table, not literal string positions
    # since headers are repeated in each chunk

    # Basic sanity checks
    assert all(chunk.start_index >= 0 for chunk in chunks)
    assert all(chunk.end_index > chunk.start_index for chunk in chunks)

    # Indices should be increasing (non-overlapping content)
    for i in range(len(chunks) - 1):
        assert chunks[i].end_index <= chunks[i + 1].start_index


def test_table_chunker_indices_consistency(large_table: str) -> None:
    """Test that TableChunker's indices are consistent and reasonable."""
    chunker = TableChunker(tokenizer="character", chunk_size=400)
    chunks = chunker.chunk(large_table)

    verify_chunk_indices(chunks, large_table)


def test_table_chunker_call_method(sample_table: str) -> None:
    """Test that the TableChunker can be called directly."""
    chunker = TableChunker(tokenizer="character", chunk_size=2048)
    chunks = chunker(sample_table)

    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].text == sample_table


def test_table_chunker_repr() -> None:
    """Test that the TableChunker has a string representation."""
    chunker = TableChunker(tokenizer="character", chunk_size=500)

    repr_str = repr(chunker)
    assert "TableChunker" in repr_str
    assert "500" in repr_str


# ==================== Edge Case Tests ====================


def test_table_chunker_single_row() -> None:
    """Test table with exactly one data row."""
    table = """| Name | Value |
|------|-------|
| A | 1 |"""

    chunker = TableChunker(tokenizer="character", chunk_size=100)
    chunks = chunker.chunk(table)

    assert len(chunks) == 1
    assert chunks[0].text == table


def test_table_chunker_very_wide_table() -> None:
    """Test table with many columns that might cause formatting issues."""
    table = """| C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 |
|----|----|----|----|----|----|----|----|----|-----|
| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |"""

    chunker = TableChunker(tokenizer="character", chunk_size=100)
    chunks = chunker.chunk(table)

    # Should chunk due to size constraint
    assert len(chunks) >= 1
    # All chunks should have header
    for chunk in chunks:
        assert "| C1 | C2 |" in chunk.text
        assert "|----|----|" in chunk.text


def test_table_chunker_very_long_row() -> None:
    """Test table with a single row that exceeds chunk size."""
    table = """| Name | Description |
|------|-------------|
| Item | This is an extremely long description that goes on and on and contains lots of information that will definitely exceed the chunk size limit we set for this test |"""

    chunker = TableChunker(tokenizer="character", chunk_size=50)
    chunks = chunker.chunk(table)

    # Should create chunk even though row exceeds size
    assert len(chunks) >= 1
    # First chunk should contain the long row
    assert "extremely long description" in chunks[0].text


def test_table_chunker_irregular_spacing() -> None:
    """Test table with irregular spacing and alignment."""
    table = """| Name|Age|City |
|---|---|---|
|John  |25|NYC|
| Alice|30| London   |
|  Bob  | 35 |  Paris  |"""

    chunker = TableChunker(tokenizer="character", chunk_size=500)
    chunks = chunker.chunk(table)

    assert len(chunks) >= 1
    # Check that all rows are preserved
    assert "John" in "".join(c.text for c in chunks)
    assert "Alice" in "".join(c.text for c in chunks)
    assert "Bob" in "".join(c.text for c in chunks)


def test_table_chunker_special_characters() -> None:
    """Test table containing special characters and symbols."""
    table = """| Symbol | Meaning |
|--------|---------|
| @ | At sign |
| # | Hash |
| $ | Dollar |
| % | Percent |
| & | Ampersand |
| * | Asterisk |
| | | Pipe (escaped) |"""

    chunker = TableChunker(tokenizer="character", chunk_size=80)
    chunks = chunker.chunk(table)

    # Verify all symbols are present
    combined_text = "".join(c.text for c in chunks)
    assert "@" in combined_text
    assert "#" in combined_text
    assert "$" in combined_text
    assert "%" in combined_text


def test_table_chunker_unicode_content() -> None:
    """Test table with unicode and emoji content."""
    table = """| Name | Country | Flag |
|------|---------|------|
| Tokyo | Japan | 🇯🇵 |
| Paris | France | 🇫🇷 |
| Berlin | Germany | 🇩🇪 |
| Москва | Россия | 🇷🇺 |
| 北京 | 中国 | 🇨🇳 |"""

    chunker = TableChunker(tokenizer="character", chunk_size=100)
    chunks = chunker.chunk(table)

    # Verify unicode is preserved
    combined_text = "".join(c.text for c in chunks)
    assert "Tokyo" in combined_text
    assert "Москва" in combined_text
    assert "北京" in combined_text
    assert "🇯🇵" in combined_text


def test_table_chunker_empty_cells() -> None:
    """Test table with empty cells."""
    table = """| Name | Value | Description |
|------|-------|-------------|
| A | 1 | |
| B | | Some text |
| C | | |
| D | 4 | Complete |"""

    chunker = TableChunker(tokenizer="character", chunk_size=60)
    chunks = chunker.chunk(table)

    assert len(chunks) >= 1
    # Verify structure is maintained
    for chunk in chunks:
        lines = chunk.text.strip().split("\n")
        # Each line should have the same number of pipe characters
        if len(lines) > 2:  # If there are data rows
            pipe_counts = [line.count("|") for line in lines]
            assert len(set(pipe_counts)) == 1  # All should be the same


def test_table_chunker_exact_boundary_conditions() -> None:
    """Test chunking at exact boundary conditions."""
    # Create a table where rows fit exactly at chunk boundaries
    table = """| A | B |
|---|---|
| 1 | 2 |
| 3 | 4 |
| 5 | 6 |"""

    header_and_sep = "| A | B |\n|---|---|"
    row_size = len("\n| 1 | 2 |")

    # Set chunk size to fit header + exactly 2 rows
    chunk_size = len(header_and_sep) + (row_size * 2)

    chunker = TableChunker(tokenizer="character", chunk_size=chunk_size)
    chunks = chunker.chunk(table)

    # Should split into chunks respecting boundary
    assert len(chunks) >= 1
    for chunk in chunks:
        assert "| A | B |" in chunk.text


def test_table_chunker_whitespace_only_cells() -> None:
    """Test table with cells containing only whitespace."""
    table = """| Name | Value |
|------|-------|
| A |   |
|   | B |
|  |  |"""

    chunker = TableChunker(tokenizer="character", chunk_size=100)
    chunks = chunker.chunk(table)

    assert len(chunks) >= 1
    # Structure should be maintained
    for chunk in chunks:
        assert "|" in chunk.text


def test_table_chunker_trailing_newlines() -> None:
    """Test table with trailing newlines."""
    table = """| Name | Value |
|------|-------|
| A | 1 |
| B | 2 |

"""

    chunker = TableChunker(tokenizer="character", chunk_size=100)
    chunks = chunker.chunk(table)

    assert len(chunks) >= 1
    # Should handle gracefully
    assert "| A | 1 |" in "".join(c.text for c in chunks)


def test_table_chunker_numeric_edge_cases() -> None:
    """Test table with various numeric formats."""
    table = """| Number | Value |
|--------|-------|
| 0 | Zero |
| -1 | Negative |
| 3.14159 | Pi |
| 1e10 | Scientific |
| 0xFF | Hex |"""

    chunker = TableChunker(tokenizer="character", chunk_size=70)
    chunks = chunker.chunk(table)

    combined_text = "".join(c.text for c in chunks)
    assert "0" in combined_text
    assert "-1" in combined_text
    assert "3.14159" in combined_text


def test_table_chunker_markdown_document_empty_tables() -> None:
    """Test MarkdownDocument with empty tables list."""
    doc = MarkdownDocument(content="# Title\n\nSome text", tables=[])
    chunker = TableChunker(tokenizer="character", chunk_size=100)

    result = chunker.chunk_document(doc)

    # Should return document with original chunks (empty in this case)
    assert isinstance(result, MarkdownDocument)


def test_table_chunker_markdown_document_multiple_tables() -> None:
    """Test MarkdownDocument with multiple tables."""
    table1 = """| A | B |
|---|---|
| 1 | 2 |"""

    table2 = """| X | Y |
|---|---|
| 3 | 4 |"""

    content = f"# Title\n\n{table1}\n\nSome text\n\n{table2}"

    doc = MarkdownDocument(
        content=content,
        tables=[
            MarkdownTable(content=table1, start_index=9, end_index=9 + len(table1)),
            MarkdownTable(
                content=table2,
                start_index=9 + len(table1) + 12,
                end_index=9 + len(table1) + 12 + len(table2),
            ),
        ],
    )

    chunker = TableChunker(tokenizer="character", chunk_size=100)
    result = chunker.chunk_document(doc)

    # Should process both tables
    assert len(result.chunks) >= 2
    # Chunks should be sorted by start_index
    for i in range(len(result.chunks) - 1):
        assert result.chunks[i].start_index <= result.chunks[i + 1].start_index


def test_table_chunker_plain_document() -> None:
    """Test that TableChunker handles plain Document objects."""
    table = """| Name | Value |
|------|-------|
| A | 1 |
| B | 2 |"""

    doc = Document(content=table)
    chunker = TableChunker(tokenizer="character", chunk_size=100)

    result = chunker.chunk_document(doc)

    assert isinstance(result, Document)
    assert len(result.chunks) >= 1


# ==================== Integration Tests ====================


def test_table_chunker_after_recursive_chunker() -> None:
    """Test using TableChunker after RecursiveChunker on a document with tables."""
    # Create a document with text and tables mixed
    table1 = """| Product | Price |
|---------|-------|
| Apple | $1.50 |
| Banana | $0.75 |
| Orange | $1.25 |"""

    table2 = """| City | Population |
|------|------------|
| NYC | 8000000 |
| LA | 4000000 |
| Chicago | 2700000 |"""

    content = f"""# Product Catalog

This is our product catalog with fresh fruits.

{table1}

## City Statistics

Here are some major cities and their populations.

{table2}

Thank you for shopping with us!"""

    # First, create chunks using RecursiveChunker
    recursive_chunker = RecursiveChunker(tokenizer="character", chunk_size=200)
    recursive_chunks = recursive_chunker.chunk(content)

    # Verify recursive chunker created multiple chunks
    assert len(recursive_chunks) > 1
    initial_chunk_count = len(recursive_chunks)

    # Now find which chunks contain tables
    table1_start = content.index(table1)
    table2_start = content.index(table2)

    # Create MarkdownDocument with table locations
    doc = MarkdownDocument(
        content=content,
        tables=[
            MarkdownTable(
                content=table1,
                start_index=table1_start,
                end_index=table1_start + len(table1),
            ),
            MarkdownTable(
                content=table2,
                start_index=table2_start,
                end_index=table2_start + len(table2),
            ),
        ],
        chunks=recursive_chunks,
    )

    # Apply TableChunker
    table_chunker = TableChunker(tokenizer="character", chunk_size=100)
    result = table_chunker.chunk_document(doc)

    # Verify both chunkers' results are present
    assert len(result.chunks) >= initial_chunk_count  # Should have at least as many chunks

    # Verify chunks are sorted by start_index
    for i in range(len(result.chunks) - 1):
        assert result.chunks[i].start_index <= result.chunks[i + 1].start_index

    # Verify table content is preserved
    table_chunks = [c for c in result.chunks if "| Product |" in c.text or "| City |" in c.text]
    assert len(table_chunks) > 0

    # Verify content from both tables is present
    all_text = "".join(c.text for c in result.chunks)
    assert "Apple" in all_text
    assert "NYC" in all_text


def test_table_chunker_quality_after_recursive() -> None:
    """Test the quality of results when TableChunker is used after RecursiveChunker."""
    # Create a realistic document with mixed content
    large_table = """| ID | Name | Email | Department | Salary |
|-----|---------|-----------------|------------|--------|
| 001 | Alice | alice@corp.com | Engineering | 120000 |
| 002 | Bob | bob@corp.com | Sales | 95000 |
| 003 | Carol | carol@corp.com | Marketing | 85000 |
| 004 | David | david@corp.com | Engineering | 115000 |
| 005 | Eve | eve@corp.com | HR | 75000 |
| 006 | Frank | frank@corp.com | Sales | 92000 |
| 007 | Grace | grace@corp.com | Engineering | 125000 |
| 008 | Henry | henry@corp.com | Marketing | 88000 |"""

    content = f"""# Employee Directory

Welcome to our employee directory. Below you'll find information about all current employees.

{large_table}

## Notes

All employee information is confidential. Please maintain appropriate data privacy standards.

For questions about this directory, contact HR at hr@corp.com."""

    # First pass: RecursiveChunker
    recursive_chunker = RecursiveChunker(tokenizer="character", chunk_size=300)
    doc = Document(content=content)
    recursive_result = recursive_chunker.chunk_document(doc)

    # Second pass: Extract tables and apply TableChunker
    table_start = content.index(large_table)
    markdown_doc = MarkdownDocument(
        content=content,
        tables=[
            MarkdownTable(
                content=large_table,
                start_index=table_start,
                end_index=table_start + len(large_table),
            ),
        ],
        chunks=recursive_result.chunks.copy(),
    )

    table_chunker = TableChunker(tokenizer="character", chunk_size=200)
    final_result = table_chunker.chunk_document(markdown_doc)

    # Quality checks
    # 1. No content should be lost
    all_chunk_text = "".join(c.text for c in final_result.chunks)
    assert "Alice" in all_chunk_text
    assert "Henry" in all_chunk_text
    assert "Welcome to our employee directory" in all_chunk_text

    # 2. Table chunks should have headers
    table_chunks = [c for c in final_result.chunks if "| ID | Name |" in c.text]
    assert len(table_chunks) > 0

    for table_chunk in table_chunks:
        # Each table chunk should have header and separator
        assert "| ID | Name | Email | Department | Salary |" in table_chunk.text
        assert "|-----|" in table_chunk.text

    # 3. Chunks should respect size constraints (with some tolerance for headers)
    for chunk in final_result.chunks:
        # Allow some overflow for table headers
        assert chunk.token_count <= 300 or "| ID | Name |" in chunk.text

    # 4. All chunks should be sorted by start index
    sorted_chunks = sorted(final_result.chunks, key=lambda x: x.start_index)
    assert final_result.chunks == sorted_chunks

    # 5. Verify that we have chunks from both recursive and table chunkers
    # Table chunks will have the table header
    # Recursive chunks may or may not have table content
    assert len(table_chunks) >= 1  # At least one table chunk from TableChunker


def test_table_chunker_very_small_chunk_size() -> None:
    """Test table chunker with chunk size smaller than header."""
    table = """| Name | Value |
|------|-------|
| A | 1 |
| B | 2 |"""

    # Set chunk size smaller than header
    chunker = TableChunker(tokenizer="character", chunk_size=20)
    chunks = chunker.chunk(table)

    # Should still create chunks (header must be in each)
    assert len(chunks) > 0
    # Every chunk should have the header even if it exceeds chunk_size
    for chunk in chunks:
        assert "| Name | Value |" in chunk.text


# Test TableChunker with row tokenizer
def test_table_chunker_row_tokenizer(sample_table: str) -> None:
    """Test TableChunker with tokenizer='row' chunks by rows, preserving header."""
    chunker = TableChunker(tokenizer="row", chunk_size=3)
    chunks = chunker.chunk(sample_table)

    # Should split into chunks with max 3 data rows each, header always present
    assert len(chunks) > 1
    for chunk in chunks:
        lines = chunk.text.strip().split("\n")
        # Header and separator always present
        assert lines[0].startswith("| Name")
        assert lines[1].startswith("|------")
        # Data rows count per chunk should be <= chunk_size
        data_rows = lines[2:]
        assert len(data_rows) <= 3
        # All original data rows should be present across chunks
    all_chunked_rows = [line for chunk in chunks for line in chunk.text.strip().split("\n")[2:]]
    original_rows = sample_table.strip().split("\n")[2:]
    assert set(all_chunked_rows) == set(original_rows)


@pytest.fixture
def html_table() -> str:
    """Fixture that returns an HTML table string."""
    return """<table>
  <thead>
    <tr><th>ID</th><th>Name</th><th>Role</th></tr>
  </thead>
  <tbody>
    <tr><td>1</td><td>Alice</td><td>Admin</td></tr>
    <tr><td>2</td><td>Bob</td><td>User</td></tr>
    <tr><td>3</td><td>Charlie</td><td>Guest</td></tr>
    <tr><td>4</td><td>David</td><td>User</td></tr>
    <tr><td>5</td><td>Eve</td><td>Admin</td></tr>
  </tbody>
</table>"""


def test_table_chunker_html_table(html_table: str) -> None:
    """Test chunking an HTML table."""
    chunker = TableChunker(tokenizer="character", chunk_size=100)
    chunks = chunker.chunk(html_table)

    assert len(chunks) > 1
    for chunk in chunks:
        assert "<table>" in chunk.text
        assert "</table>" in chunk.text
        assert "<thead>" in chunk.text
        assert "ID" in chunk.text

    # All data rows should be present across chunks
    all_content = "".join(chunks[i].text for i in range(len(chunks)))
    assert "Alice" in all_content
    assert "Eve" in all_content


def test_table_chunker_html_table_no_tbody() -> None:
    """Test chunking an HTML table without tbody tags (exercises _split_html_table else branch)."""
    table = """<table>
  <tr><th>ID</th><th>Name</th><th>Role</th></tr>
  <tr><td>1</td><td>Alice</td><td>Admin</td></tr>
  <tr><td>2</td><td>Bob</td><td>User</td></tr>
  <tr><td>3</td><td>Charlie</td><td>Guest</td></tr>
  <tr><td>4</td><td>David</td><td>User</td></tr>
  <tr><td>5</td><td>Eve</td><td>Admin</td></tr>
</table>"""
    chunker = TableChunker(tokenizer="character", chunk_size=100)
    chunks = chunker.chunk(table)

    assert len(chunks) > 1
    for chunk in chunks:
        assert "<table>" in chunk.text
        assert "</table>" in chunk.text
    all_content = "".join(c.text for c in chunks)
    assert "Alice" in all_content
    assert "Eve" in all_content


def test_table_chunker_html_table_row_based(html_table: str) -> None:
    """Test row-based chunking for HTML tables."""
    chunker = TableChunker(tokenizer="row", chunk_size=2)
    chunks = chunker.chunk(html_table)

    # 5 data rows with chunk_size=2 → 3 chunks (2, 2, 1)
    assert len(chunks) == 3
    for chunk in chunks:
        assert "<table>" in chunk.text
        assert "</table>" in chunk.text
        assert "<thead>" in chunk.text
        assert chunk.token_count <= 2
    all_content = "".join(c.text for c in chunks)
    assert "Alice" in all_content
    assert "Eve" in all_content


def test_table_chunker_html_table_fits_single_chunk_character(html_table: str) -> None:
    """Test that an HTML table smaller than chunk_size is returned as a single chunk (character tokenizer)."""
    chunker = TableChunker(tokenizer="character", chunk_size=10_000)
    chunks = chunker.chunk(html_table)

    assert len(chunks) == 1
    assert chunks[0].text == html_table


def test_table_chunker_html_table_fits_single_chunk_row_based(html_table: str) -> None:
    """Test that an HTML table with fewer rows than chunk_size is returned as a single chunk (row tokenizer)."""
    chunker = TableChunker(tokenizer="row", chunk_size=10)
    chunks = chunker.chunk(html_table)

    assert len(chunks) == 1
    assert chunks[0].token_count == 5  # html_table has 5 data rows in tbody
    assert chunks[0].text == html_table


def test_table_chunker_html_table_empty_tbody() -> None:
    """Test that an HTML table with an empty tbody returns an empty chunk list."""
    table = """<table>
  <thead>
    <tr><th>ID</th><th>Name</th></tr>
  </thead>
  <tbody>
  </tbody>
</table>"""
    chunker = TableChunker(tokenizer="row", chunk_size=3)
    chunks = chunker.chunk(table)

    assert chunks == []


def test_table_chunker_html_table_malformed_no_closing_tag() -> None:
    """Test that an HTML table missing the closing </table> tag is handled without crashing."""
    table = """<table>
  <tbody>
    <tr><td>1</td><td>Alice</td></tr>
    <tr><td>2</td><td>Bob</td></tr>
    <tr><td>3</td><td>Charlie</td></tr>
  </tbody>"""
    chunker = TableChunker(tokenizer="row", chunk_size=2)
    # Should not crash; <table> tag is enough for HTML detection
    chunks = chunker.chunk(table)

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    all_content = "".join(c.text for c in chunks)
    assert "Alice" in all_content
    assert "Charlie" in all_content


def test_table_chunker_markdown_document_no_tables_preserves_chunks() -> None:
    """Test that TableChunker is a no-op on MarkdownDocument with no tables."""
    doc = MarkdownDocument(
        content="# Title\n\nSome prose text here.\n",
        chunks=[
            Chunk(text="Some prose text here.", start_index=10, end_index=30, token_count=4),
        ],
        tables=[],
    )

    chunker = TableChunker(tokenizer="row", chunk_size=3)
    result = chunker.chunk_document(doc)

    assert len(result.chunks) == 1
    assert result.chunks[0].text == "Some prose text here."


def test_table_chunker_row_tokenizer_indices():
    table = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n| 5 | 6 |\n| 7 | 8 |"
    chunker = TableChunker(tokenizer="row", chunk_size=2)
    chunks = chunker.chunk(table)
    assert len(chunks) > 1
    # First chunk must start at 0 (matches the character path and every other chunker).
    assert chunks[0].start_index == 0
    # Spans are contiguous and reconstruct the full table (no lost/duplicated regions).
    for i in range(len(chunks) - 1):
        assert chunks[i].end_index == chunks[i + 1].start_index
    assert chunks[-1].end_index == len(table)
    assert "".join(table[c.start_index : c.end_index] for c in chunks) == table
