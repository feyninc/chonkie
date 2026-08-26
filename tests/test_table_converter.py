"""Tests for table converter utilities."""

from chonkie.utils.table_converter import (
    html_table_to_json,
    markdown_table_to_json,
)


class TestMarkdownTableToJson:
    """Tests for markdown_table_to_json function."""

    def test_basic(self) -> None:
        """Test basic conversion with numeric type inference."""
        table = """| Name | Age | Score |
|------|-----|-------|
| Alice | 30 | 95.5 |
| Bob | 25 | 100 |"""

        result = markdown_table_to_json(table)

        assert len(result) == 2
        assert result[0]["Name"] == "Alice"
        assert result[0]["Age"] == 30
        assert result[0]["Score"] == 95.5
        assert result[1]["Name"] == "Bob"

    def test_negative_numbers(self) -> None:
        """Test negative numbers are parsed correctly."""
        table = """| Value | Change |
|------|--------|
| 100 | -10 |
| -50 | 25 |"""

        result = markdown_table_to_json(table)

        assert result[0]["Value"] == 100
        assert result[0]["Change"] == -10
        assert result[1]["Value"] == -50

    def test_empty_and_missing_cells(self) -> None:
        """Test handling of empty cells and column alignment."""
        table = """| Name | Value | Notes |
|------|-------|-------|
| Alice | 100 | |
| Bob | | Extra |"""

        result = markdown_table_to_json(table)

        assert result[0]["Name"] == "Alice"
        assert result[0]["Value"] == 100
        assert result[1]["Name"] == "Bob"
        assert result[1]["Value"] is None or result[1]["Value"] == ""
        assert result[1]["Notes"] == "Extra"

    def test_special_content(self) -> None:
        """Test special characters and unicode."""
        table = """| City | Symbol |
|------|--------|
| 北京 | @ |
| Paris | $ |"""

        result = markdown_table_to_json(table)

        assert result[0]["City"] == "北京"
        assert result[1]["Symbol"] == "$"

    def test_empty_table(self) -> None:
        """Test empty and header-only tables."""
        assert markdown_table_to_json("") == []
        assert markdown_table_to_json("| Name |\n|------|") == []
        # Empty header cells must not leak pandas EmptyDataError.
        assert markdown_table_to_json("|\n|-|") == []
        assert markdown_table_to_json("||\n|-|") == []
        # strip("|") leaves inner pipes for 3+ blank columns; still empty JSON.
        assert markdown_table_to_json("| | | |\n|---|---|---|") == []
        assert markdown_table_to_json("|  |  |  |  |\n|---|---|---|---|") == []


class TestHTMLTableToJson:
    """Tests for html_table_to_json function."""

    def test_basic(self) -> None:
        """Test basic HTML conversion with numeric inference."""
        html = """<table>
  <thead>
    <tr><th>Name</th><th>Age</th></tr>
  </thead>
  <tbody>
    <tr><td>Alice</td><td>30</td></tr>
    <tr><td>Bob</td><td>25</td></tr>
  </tbody>
</table>"""

        result = html_table_to_json(html)

        assert result is not None
        assert len(result) == 2
        assert result[0]["Name"] == "Alice"
        assert result[0]["Age"] == 30
        assert result[1]["Name"] == "Bob"

    def test_without_thead(self) -> None:
        """Test HTML table without thead uses first row as headers."""
        html = "<table><tr><th>Name</th><th>Age</th></tr><tr><td>Bob</td><td>25</td></tr></table>"

        result = html_table_to_json(html)

        assert result is not None
        assert result[0]["Name"] == "Bob"

    def test_no_valid_table(self) -> None:
        """Test that invalid/missing tables return None."""
        assert html_table_to_json("<table><tr><td>Alice</td></tr></table>") is None
        assert html_table_to_json("not a table") is None
