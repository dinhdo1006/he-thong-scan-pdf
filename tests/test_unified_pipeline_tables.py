"""
Tests for `UnifiedPDFPipeline._extract_tables`'s per-page fallback cascade
and form-agnostic quality merge (Phase 2 + 3).
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


def _clean_df(label: str) -> pd.DataFrame:
    return pd.DataFrame({"STT": [label], "Mo_ta": ["noi dung"], "So_tien": ["1000"]})


def _dirty_df() -> pd.DataFrame:
    # Mimics Docling collapsed header soup that must lose a quality contest.
    return pd.DataFrame(
        [["1", "x", "10"]],
        columns=[
            "Số TT",
            "trong hoyết Tình |_Ủy thác THA |_Trả đơn THA đã giải qhyết bằng các biện pháp " * 2,
            "l",
        ],
    )


def test_extract_tables_docling_covers_all_pages(tmp_path) -> None:
    df0 = _clean_df("1")
    df1 = _clean_df("2")
    mock_extractor = MagicMock()
    mock_extractor.extract_with_pages.return_value = [(0, df0), (1, df1)]

    with patch("pdf_extractor.unified_pipeline._DOCLING_AVAILABLE", True), patch(
        "pdf_extractor.unified_pipeline.DoclingTableExtractor", return_value=mock_extractor
    ), patch(
        "pdf_extractor.unified_pipeline._run_paddle_fallback_with_pages"
    ) as paddle_mock:
        dfs, backend = _pipeline()._extract_tables(Path("f.pdf"), tmp_path, [0, 1])

    paddle_mock.assert_not_called()
    assert backend == BACKEND_DOCLING
    assert dfs == [df0, df1]


def test_extract_tables_fills_missing_page_with_paddle(tmp_path) -> None:
    """Page 1 missing from Docling -- Paddle fills the gap."""
    df0 = _clean_df("1")
    paddle_df1 = _clean_df("paddle-2")
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


def test_extract_tables_prefers_higher_quality_on_same_page(tmp_path) -> None:
    """Dirty Docling + cleaner Paddle on the same page -> Paddle wins."""
    dirty = _dirty_df()
    clean = _clean_df("paddle")
    mock_extractor = MagicMock()
    mock_extractor.extract_with_pages.return_value = [(0, dirty)]

    with patch("pdf_extractor.unified_pipeline._DOCLING_AVAILABLE", True), patch(
        "pdf_extractor.unified_pipeline.DoclingTableExtractor", return_value=mock_extractor
    ), patch(
        "pdf_extractor.unified_pipeline._run_paddle_fallback_with_pages",
        return_value=[(0, clean)],
    ):
        dfs, backend = _pipeline()._extract_tables(Path("f.pdf"), tmp_path, [0])

    assert backend == BACKEND_PADDLE
    assert dfs == [clean]


def test_extract_tables_prefers_wider_paddle_over_narrow_docling(tmp_path) -> None:
    """Page 2: Docling 3-col must lose to Paddle 9-col so stitch can work."""
    page0 = pd.DataFrame(
        [["1", "mo ta"] + [""] * 7],
        columns=[f"c{i}" for i in range(9)],
    )
    narrow_docling = pd.DataFrame(
        [["1.3.2", "Thu tai san", "x"]],
        columns=["STT", "Mo_ta", "x"],
    )
    wide_paddle = pd.DataFrame(
        [["1.3.2", "Thu tai san"] + [""] * 7],
        columns=[f"c{i}" for i in range(9)],
    )
    mock_extractor = MagicMock()
    mock_extractor.extract_with_pages.return_value = [(0, page0), (1, narrow_docling)]

    with patch("pdf_extractor.unified_pipeline._DOCLING_AVAILABLE", True), patch(
        "pdf_extractor.unified_pipeline.DoclingTableExtractor", return_value=mock_extractor
    ), patch(
        "pdf_extractor.unified_pipeline._run_paddle_fallback_with_pages",
        return_value=[(0, page0), (1, wide_paddle)],
    ):
        dfs, backend = _pipeline()._extract_tables(Path("f.pdf"), tmp_path, [0, 1])

    assert backend == BACKEND_HYBRID
    assert len(dfs[0].columns) == 9
    assert len(dfs[1].columns) == 9
    assert dfs[1].equals(wide_paddle)


def test_extract_tables_keeps_clean_docling_over_weaker_paddle(tmp_path) -> None:
    """When Paddle runs for another reason, cleaner Docling still wins that page."""
    clean_docling = _clean_df("docling")
    weak_paddle = pd.DataFrame(
        [["x"]],
        columns=["a|_b|_c " + ("z" * 80)],
    )
    mock_extractor = MagicMock()
    # Page 1 missing forces Paddle; page 0 has both.
    mock_extractor.extract_with_pages.return_value = [(0, clean_docling)]

    with patch("pdf_extractor.unified_pipeline._DOCLING_AVAILABLE", True), patch(
        "pdf_extractor.unified_pipeline.DoclingTableExtractor", return_value=mock_extractor
    ), patch(
        "pdf_extractor.unified_pipeline._run_paddle_fallback_with_pages",
        return_value=[(0, weak_paddle), (1, _clean_df("p1"))],
    ):
        dfs, backend = _pipeline()._extract_tables(Path("f.pdf"), tmp_path, [0, 1])

    assert backend == BACKEND_HYBRID
    assert dfs[0].equals(clean_docling)
    assert dfs[1].iloc[0, 0] == "p1"


def test_extract_tables_docling_zero_tables_paddle_covers_all(tmp_path) -> None:
    paddle_df0 = _clean_df("p0")
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
    df0 = _clean_df("1")
    mock_extractor = MagicMock()
    mock_extractor.extract_with_pages.return_value = [(0, df0)]

    with patch("pdf_extractor.unified_pipeline._DOCLING_AVAILABLE", True), patch(
        "pdf_extractor.unified_pipeline.DoclingTableExtractor", return_value=mock_extractor
    ), patch(
        "pdf_extractor.unified_pipeline._run_paddle_fallback_with_pages",
        side_effect=RuntimeError("paddle boom"),
    ):
        # Missing page 1 forces Paddle path; on failure keep Docling page 0.
        dfs, backend = _pipeline()._extract_tables(Path("f.pdf"), tmp_path, [0, 1])

    assert backend == BACKEND_DOCLING
    assert dfs == [df0]


def test_extract_tables_unplaced_docling_tables_appended_last(tmp_path) -> None:
    df0 = _clean_df("1")
    unplaced = _clean_df("?")
    mock_extractor = MagicMock()
    mock_extractor.extract_with_pages.return_value = [(0, df0), (None, unplaced)]

    with patch("pdf_extractor.unified_pipeline._DOCLING_AVAILABLE", True), patch(
        "pdf_extractor.unified_pipeline.DoclingTableExtractor", return_value=mock_extractor
    ), patch(
        "pdf_extractor.unified_pipeline._run_paddle_fallback_with_pages"
    ) as paddle_mock:
        dfs, backend = _pipeline()._extract_tables(Path("f.pdf"), tmp_path, [0])

    paddle_mock.assert_not_called()
    assert backend == BACKEND_DOCLING
    assert dfs == [df0, unplaced]
