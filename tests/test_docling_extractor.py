"""Unit tests for Docling table -> DataFrame conversion (no model download)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from pdf_extractor.docling_extractor import (
    DoclingExtractionError,
    DoclingTableExtractor,
    _parse_marker_markdown_tables,
    _save_dataframes_to_excel,
    extract_tables_from_pdf,
)


def _cell(
    text: str,
    *,
    row: int,
    col: int,
    row_span: int = 1,
    col_span: int = 1,
    column_header: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        row_span=row_span,
        col_span=col_span,
        start_row_offset_idx=row,
        start_col_offset_idx=col,
        column_header=column_header,
    )


def _fake_extractor() -> DoclingTableExtractor:
    """Build extractor without calling DocumentConverter / touching disk."""
    with patch.object(DoclingTableExtractor, "__init__", lambda self, pdf_path: None):
        ext = DoclingTableExtractor.__new__(DoclingTableExtractor)
    ext._pdf_path = None
    ext._converter = None
    ext._raw_result = None
    return ext


def test_table_to_dataframe_single_header() -> None:
    ext = _fake_extractor()
    # 1 header row + 1 data row, 3 cols
    h0 = _cell("A", row=0, col=0, column_header=True)
    h1 = _cell("B", row=0, col=1, column_header=True)
    h2 = _cell("C", row=0, col=2, column_header=True)
    d0 = _cell("1", row=1, col=0)
    d1 = _cell("2", row=1, col=1)
    d2 = _cell("3", row=1, col=2)
    table = SimpleNamespace(
        data=SimpleNamespace(
            num_cols=3,
            grid=[[h0, h1, h2], [d0, d1, d2]],
        )
    )
    df = ext._table_to_dataframe(table)
    assert list(df.columns) == ["A", "B", "C"]
    assert df.shape == (1, 3)
    assert df.iloc[0].tolist() == ["1", "2", "3"]


def test_table_to_dataframe_multiindex_with_colspan() -> None:
    ext = _fake_extractor()
    # Row0: "Group" colspan=2 | "Solo"
    # Row1: "A" | "B" | "C"
    # Data: 1 | 2 | 3
    g = _cell("Group", row=0, col=0, col_span=2, column_header=True)
    s = _cell("Solo", row=0, col=2, column_header=True)
    a = _cell("A", row=1, col=0, column_header=True)
    b = _cell("B", row=1, col=1, column_header=True)
    c = _cell("C", row=1, col=2, column_header=True)
    # Grid already expands spans like Docling: same object in spanned slots
    grid = [
        [g, g, s],
        [a, b, c],
        [_cell("1", row=2, col=0), _cell("2", row=2, col=1), _cell("3", row=2, col=2)],
    ]
    table = SimpleNamespace(data=SimpleNamespace(num_cols=3, grid=grid))
    df = ext._table_to_dataframe(table)
    assert isinstance(df.columns, pd.MultiIndex)
    assert list(df.columns) == [("Group", "A"), ("Group", "B"), ("Solo", "C")]
    assert df.iloc[0].tolist() == ["1", "2", "3"]


def test_save_to_excel_writes_table_1(tmp_path) -> None:
    out = tmp_path / "output_tables.xlsx"
    dfs = [pd.DataFrame({"x": [1], "y": [2]})]
    path = _save_dataframes_to_excel(dfs, out)
    assert path.exists()
    loaded = pd.read_excel(path, sheet_name="Table_1")
    assert list(loaded.columns) == ["x", "y"]
    assert len(loaded) == 1


def test_parse_marker_markdown_tables() -> None:
    md = """
| A | B |
|---|---|
| 1 | 2 |
"""
    tables = _parse_marker_markdown_tables(md)
    assert len(tables) == 1
    assert list(tables[0].columns) == ["A", "B"]


def test_extract_tables_from_pdf_marker_fallback(tmp_path) -> None:
    pdf = tmp_path / "empty.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    out = tmp_path / "out.xlsx"

    with patch.object(
        DoclingTableExtractor,
        "__init__",
        side_effect=DoclingExtractionError("boom"),
    ):
        with patch(
            "pdf_extractor.docling_extractor._marker_fallback",
            return_value=[pd.DataFrame({"a": [1]})],
        ):
            dfs = extract_tables_from_pdf(pdf, output_path=out, apply_ocr_cleanup=False)

    assert len(dfs) == 1
    assert out.exists()
    sheets = pd.ExcelFile(out).sheet_names
    assert "Table_1" in sheets


def test_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        DoclingTableExtractor("definitely_missing_file_xyz.pdf")
