"""Table converter utilities for transforming tables to different formats."""

import math
from io import StringIO


def _read_markdown_table(table_content: str):
    """Read markdown table into DataFrame."""
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "Pandas is required to use the table converter. Please install it with `pip install chonkie[table]`.",
        ) from e

    lines = [line.strip("|").strip() for line in table_content.split("\n") if line.strip()]
    # Empty/blank headers (incl. multi-col "| | | |") yield EmptyDataError from pandas.
    if len(lines) < 2 or not lines[0].replace("|", "").strip():
        return pd.DataFrame()

    csv_content = "\n".join([lines[0]] + lines[2:])
    df = pd.read_csv(StringIO(csv_content), sep="|", skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    return df


def _read_html_table(table_content: str):
    """Read HTML table into DataFrame."""
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "Pandas is required to use the table converter. Please install it with `pip install chonkie[table]`.",
        ) from e

    try:
        tables = pd.read_html(StringIO(table_content))
        if not tables:
            return None
        df = tables[0]
        if df.columns[0] == 0 and len(df.columns) == 1:
            return None
        return df
    except Exception:
        return None


def _clean_for_json(value):
    """Convert pandas values to JSON-serializable types."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def markdown_table_to_json(table_content: str) -> list[dict]:
    """Convert a markdown table to a JSON-serializable list of dictionaries.

    Each row becomes a dictionary with column names as keys.
    Numeric values are automatically converted to int/float.

    Args:
        table_content: The markdown table content as a string.

    Returns:
        A list of dictionaries, one per data row.

    Example:
        >>> table = '''
        ... | Name | Score |
        ... |------|-------|
        ... | Alice | 100 |
        ... | Bob | 95 |
        ... '''
        >>> markdown_table_to_json(table)
        [{'Name': 'Alice', 'Score': 100}, {'Name': 'Bob', 'Score': 95}]

    """
    df = _read_markdown_table(table_content)
    if df.empty:
        return []
    records = df.to_dict(orient="records")
    return [
        {
            k.strip(): _clean_for_json(v.strip() if isinstance(v, str) else v)
            for k, v in record.items()
        }
        for record in records
    ]


def html_table_to_json(table_content: str) -> list[dict] | None:
    """Convert an HTML table to a JSON-serializable list of dictionaries.

    Each row becomes a dictionary with column names as keys.
    Numeric values are automatically converted to int/float.

    Args:
        table_content: The HTML table content as a string.

    Returns:
        A list of dictionaries, one per data row, or None if no valid table found.

    Example:
        >>> html = '<table><thead><tr><th>Name</th><th>Age</th></tr></thead><tbody><tr><td>Alice</td><td>30</td></tr></tbody></table>'
        >>> html_table_to_json(html)
        [{'Name': 'Alice', 'Age': 30}]

    """
    df = _read_html_table(table_content)
    if df is None or df.empty:
        return None
    records = df.to_dict(orient="records")
    return [
        {
            k.strip(): _clean_for_json(v.strip() if isinstance(v, str) else v)
            for k, v in record.items()
        }
        for record in records
    ]
