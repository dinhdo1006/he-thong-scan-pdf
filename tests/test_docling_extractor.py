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
    _select_ocr_options,
    extract_tables_from_pdf,
    flatten_docling_dataframe,
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


def _fake_extractor(
    *, header_row_count: int | None = None, expected_cols: int = 9
) -> DoclingTableExtractor:
    """Build extractor without calling DocumentConverter / touching disk."""
    with patch.object(DoclingTableExtractor, "__init__", lambda self, pdf_path: None):
        ext = DoclingTableExtractor.__new__(DoclingTableExtractor)
    ext._pdf_path = None
    ext._converter = None
    ext._raw_result = None
    ext._header_row_count_override = header_row_count
    ext._expected_cols = expected_cols
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


# --- Fixed header-row override (Phase 2: TableFormer header misclassification) ---


def test_header_row_count_override_ignores_tableformer_misclassification() -> None:
    """
    TableFormer sometimes tags label/data rows as `column_header=True` on
    low-quality scans (e.g. 4 rows instead of 2), which smears everything
    into one giant header cell. `header_row_count` must override that.
    """
    ext = _fake_extractor(header_row_count=2)
    # 3 rows wrongly marked as headers by TableFormer + 1 real data row.
    h0 = _cell("Group", row=0, col=0, column_header=True)
    h1 = _cell("A", row=1, col=0, column_header=True)
    mislabeled = _cell("1.1", row=2, col=0, column_header=True)  # should be data
    d0 = _cell("Real data", row=3, col=0)
    table = SimpleNamespace(data=SimpleNamespace(num_cols=1, grid=[[h0], [h1], [mislabeled], [d0]]))

    df = ext._table_to_dataframe(table)

    # Only rows 0-1 became header; rows 2-3 stayed in the body.
    assert isinstance(df.columns, pd.MultiIndex)
    assert list(df.columns) == [("Group", "A")]
    assert df.iloc[0].tolist() == ["1.1"]
    assert df.iloc[1].tolist() == ["Real data"]


def test_header_row_count_none_falls_back_to_auto_detect() -> None:
    ext = _fake_extractor(header_row_count=None)
    h0 = _cell("A", row=0, col=0, column_header=True)
    d0 = _cell("1", row=1, col=0)
    table = SimpleNamespace(data=SimpleNamespace(num_cols=1, grid=[[h0], [d0]]))

    df = ext._table_to_dataframe(table)

    assert list(df.columns) == ["A"]
    assert df.iloc[0].tolist() == ["1"]


# --- Per-page extraction (Phase 2: don't drop a whole page's table) ---


def test_table_page_no_reads_prov_and_converts_to_zero_index() -> None:
    table = SimpleNamespace(prov=[SimpleNamespace(page_no=2)])
    assert DoclingTableExtractor._table_page_no(table) == 1


def test_table_page_no_none_when_no_prov() -> None:
    table = SimpleNamespace(prov=[])
    assert DoclingTableExtractor._table_page_no(table) is None


def test_extract_with_pages_keeps_valid_and_drops_misaligned_by_page() -> None:
    """
    One valid 3-col table on page 1 and one misaligned 2-col table (expects
    3) on page 2 -- extract_with_pages() must keep the former, tagged with
    its page, and drop the latter without touching the valid one.
    """
    ext = _fake_extractor(header_row_count=1, expected_cols=3)

    def _row(text, row, col, **kw):
        return _cell(text, row=row, col=col, **kw)

    good_grid = [
        [_row("A", 0, 0, column_header=True), _row("B", 0, 1, column_header=True), _row("C", 0, 2, column_header=True)],
        [_row("1", 1, 0), _row("2", 1, 1), _row("3", 1, 2)],
    ]
    bad_grid = [
        [_row("X", 0, 0, column_header=True), _row("Y", 0, 1, column_header=True)],
        [_row("9", 1, 0), _row("8", 1, 1)],
    ]
    good_table = SimpleNamespace(
        data=SimpleNamespace(num_cols=3, grid=good_grid),
        prov=[SimpleNamespace(page_no=1)],
    )
    bad_table = SimpleNamespace(
        data=SimpleNamespace(num_cols=2, grid=bad_grid),
        prov=[SimpleNamespace(page_no=2)],
    )
    ext._raw_result = SimpleNamespace(
        document=SimpleNamespace(tables=[good_table, bad_table])
    )
    ext._converter = SimpleNamespace(convert=lambda _p: ext._raw_result)
    ext._pdf_path = "fake.pdf"

    results = ext.extract_with_pages()

    assert len(results) == 1
    page_no, df = results[0]
    assert page_no == 0  # page 1 -> 0-indexed
    assert list(df.columns) == ["A", "B", "C"]


# --- flatten_docling_dataframe ---------------------------------------------


def test_flatten_docling_dataframe_multiindex_header() -> None:
    columns = pd.MultiIndex.from_tuples(
        [("Group", "A"), ("Group", "B"), ("Solo", "Solo")]
    )
    df = pd.DataFrame([["1", "2", "3"]], columns=columns)

    flat = flatten_docling_dataframe(df, expected_cols=3)

    assert list(flat.columns) == ["Group_A", "Group_B", "Solo"]
    assert flat.iloc[0].tolist() == ["1", "2", "3"]


def test_flatten_docling_dataframe_despans_repeated_body_values() -> None:
    # A colspan cell echoes "Tổng cộng" into 3 physical columns; only the
    # left-most one should keep the text after flattening.
    df = pd.DataFrame(
        [["1", "Mô tả", "Tổng cộng", "Tổng cộng", "Tổng cộng"]],
        columns=["STT", "Mô_tả", "col_2", "col_3", "col_4"],
    )

    flat = flatten_docling_dataframe(df, expected_cols=5)

    row = flat.iloc[0].tolist()
    assert row[2] == "Tổng cộng"
    assert pd.isna(row[3])
    assert pd.isna(row[4])
    assert len(flat.columns) == 5  # physical grid width is preserved


def test_flatten_docling_dataframe_raises_on_column_mismatch() -> None:
    df = pd.DataFrame([["1", "2", "3"]], columns=["A", "B", "C"])

    with pytest.raises(ValueError):
        flatten_docling_dataframe(df, expected_cols=9)


def test_flatten_docling_dataframe_none_raises() -> None:
    with pytest.raises(ValueError):
        flatten_docling_dataframe(None, expected_cols=9)  # type: ignore[arg-type]


# --- OCR engine selection ---------------------------------------------------


def test_select_ocr_options_falls_back_to_none_without_engines() -> None:
    with patch.dict("sys.modules", {"easyocr": None, "tesserocr": None, "pytesseract": None}):
        result = _select_ocr_options(["vi"])
    assert result is None


def test_select_ocr_options_uses_easyocr_when_available() -> None:
    import sys
    import types

    fake_easyocr = types.ModuleType("easyocr")
    with patch.dict(sys.modules, {"easyocr": fake_easyocr}):
        result = _select_ocr_options(["vi"])
    assert result is not None
    assert list(result.lang) == ["vi"]
