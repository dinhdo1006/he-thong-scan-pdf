"""
Tests for `UnifiedPDFPipeline._extract_tables`'s per-page fallback cascade
(Phase 2: don't silently drop a page's table just because another page's
Docling extraction succeeded).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from pdf_extractor.unified_pipeline import (
    BACKEND_DOCLING,
    BACKEND_HYBRID,
    BACKEND_PADDLE,
    UnifiedPDFPipeline,
)


def _pipeline() -> UnifiedPDFPipeline:
    """Bypass `__init__` -- `_extract_tables` doesn't need Marker/table_settings."""
    return UnifiedPDFPipeline.__new__(UnifiedPDFPipeline)


def test_extract_tables_docling_covers_all_pages(tmp_path) -> None:
    df0 = pd.DataFrame({"a": ["1"]})
    df1 = pd.DataFrame({"a": ["2"]})
    mock_extractor = MagicMock()
    mock_extractor.extract_with_pages.return_value = [(0, df0), (1, df1)]

    with patch("pdf_extractor.unified_pipeline._DOCLING_AVAILABLE", True), patch(
        "pdf_extractor.unified_pipeline.DoclingTableExtractor", return_value=mock_extractor
    ):
        dfs, backend = _pipeline()._extract_tables(Path("f.pdf"), tmp_path, [0, 1])

    assert backend == BACKEND_DOCLING
    assert dfs == [df0, df1]


def test_extract_tables_fills_missing_page_with_paddle(tmp_path) -> None:
    """Page 1's Docling table failed validation -- Paddle should fill only that gap."""
    df0 = pd.DataFrame({"a": ["1"]})
    paddle_df1 = pd.DataFrame({"a": ["paddle-2"]})
    mock_extractor = MagicMock()
    mock_extractor.extract_with_pages.return_value = [(0, df0)]

    with patch("pdf_extractor.unified_pipeline._DOCLING_AVAILABLE", True), patch(
        "pdf_extractor.unified_pipeline.DoclingTableExtractor", return_value=mock_extractor
    ), patch(
        "pdf_extractor.unified_pipeline._run_paddle_fallback_with_pages",
        return_value=[(1, paddle_df1)],
    ):
        dfs, backend = _pipeline()._extract_tables(Path("f.pdf"), tmp_path, [0, 1])

    assert backend == BACKEND_HYBRID
    assert dfs == [df0, paddle_df1]


def test_extract_tables_docling_zero_tables_paddle_covers_all(tmp_path) -> None:
    paddle_df0 = pd.DataFrame({"a": ["p0"]})
    mock_extractor = MagicMock()
    mock_extractor.extract_with_pages.return_value = []

    with patch("pdf_extractor.unified_pipeline._DOCLING_AVAILABLE", True), patch(
        "pdf_extractor.unified_pipeline.DoclingTableExtractor", return_value=mock_extractor
    ), patch(
        "pdf_extractor.unified_pipeline._run_paddle_fallback_with_pages",
        return_value=[(0, paddle_df0)],
    ):
        dfs, backend = _pipeline()._extract_tables(Path("f.pdf"), tmp_path, [0])

    assert backend == BACKEND_PADDLE
    assert dfs == [paddle_df0]


def test_extract_tables_paddle_fails_keeps_partial_docling(tmp_path) -> None:
    """
    Even if PaddleOCR itself crashes, a Docling page that DID succeed must
    still ship -- never regress to "0 tables" just because the fallback for
    a *different* page failed too.
    """
    df0 = pd.DataFrame({"a": ["1"]})
    mock_extractor = MagicMock()
    mock_extractor.extract_with_pages.return_value = [(0, df0)]

    with patch("pdf_extractor.unified_pipeline._DOCLING_AVAILABLE", True), patch(
        "pdf_extractor.unified_pipeline.DoclingTableExtractor", return_value=mock_extractor
    ), patch(
        "pdf_extractor.unified_pipeline._run_paddle_fallback_with_pages",
        side_effect=RuntimeError("paddle boom"),
    ):
        dfs, backend = _pipeline()._extract_tables(Path("f.pdf"), tmp_path, [0, 1])

    assert backend == BACKEND_DOCLING
    assert dfs == [df0]


def test_extract_tables_unplaced_docling_tables_appended_last(tmp_path) -> None:
    """Tables with unknown page_no (no provenance) are still kept, at the end."""
    df0 = pd.DataFrame({"a": ["1"]})
    unplaced = pd.DataFrame({"a": ["?"]})
    mock_extractor = MagicMock()
    mock_extractor.extract_with_pages.return_value = [(0, df0), (None, unplaced)]

    with patch("pdf_extractor.unified_pipeline._DOCLING_AVAILABLE", True), patch(
        "pdf_extractor.unified_pipeline.DoclingTableExtractor", return_value=mock_extractor
    ):
        dfs, backend = _pipeline()._extract_tables(Path("f.pdf"), tmp_path, [0])

    assert backend == BACKEND_DOCLING
    assert dfs == [df0, unplaced]
