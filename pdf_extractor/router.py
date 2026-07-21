"""
Smart Router — thin, content-driven entrypoint over `unified_pipeline.py`.

--------------------------------------------------------------------------
MIGRATION NOTE (this version):
Earlier versions of this router classified documents by a hardcoded literal
keyword ("Mẫu số: B06") found on page 1, and routed matches to a single
dedicated table pipeline. That was itself a form of hardcoding: it only
worked for documents that happened to contain that exact string, and any
other table-bearing PDF silently fell through to a pipeline that never even
looked for tables.

That keyword logic has been REMOVED ENTIRELY. Classification is now 100%
content-driven:
    - `classify_pdf()` asks `pdfplumber` whether ANY page of the document
      physically contains a table (via `grid_table_extractor.
      detect_table_pages`, which uses `page.find_tables()`).
    - `route_and_extract()` no longer picks between two hardcoded pipelines
      itself -- it delegates the whole run to `UnifiedPDFPipeline`
      (`unified_pipeline.py`), which ALWAYS extracts text via Marker and,
      ONLY if tables were structurally found, extracts them via a
      VLM -> PaddleOCR -> pdfplumber-grid fallback chain.

No document template, keyword, or per-form configuration is used anywhere
in this decision path -- the same code path handles Form B06, an invoice, a
bank statement, or plain running text.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .grid_table_extractor import DEFAULT_TABLE_SETTINGS, detect_table_pages
from .unified_pipeline import UnifiedPDFPipeline, UnifiedResult

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "."


class PipelineChoice:
    """Content-driven classification result (no template/keyword involved)."""

    HAS_TABLES = "HAS_TABLES"
    TEXT_ONLY = "TEXT_ONLY"


def classify_pdf(pdf_path: str | Path, table_settings: dict = DEFAULT_TABLE_SETTINGS) -> str:
    """
    Purely structural classification: does ANY page contain a physical table?

    Uses pdfplumber's ruled-line table detection -- no keyword, template
    name, or coordinate assumption is involved. This is informational only;
    `route_and_extract()` performs the same detection internally (via
    `UnifiedPDFPipeline`) regardless of whether this function was called.

    Returns:
        PipelineChoice.HAS_TABLES if at least one page has a detected table;
        otherwise PipelineChoice.TEXT_ONLY.
    """
    try:
        pages_with_tables = detect_table_pages(pdf_path, table_settings=table_settings)
    except Exception as exc:
        logger.warning("Router: failed to scan %s for tables (%s) -- assuming TEXT_ONLY.", pdf_path, exc)
        return PipelineChoice.TEXT_ONLY

    if pages_with_tables:
        logger.info("Router: table(s) detected on page(s) %s -> HAS_TABLES.", [p + 1 for p in pages_with_tables])
        return PipelineChoice.HAS_TABLES

    logger.info("Router: no tables detected anywhere -> TEXT_ONLY.")
    return PipelineChoice.TEXT_ONLY


# A single shared pipeline instance so the (large) VLM inside it is loaded
# once per process and reused across every `route_and_extract()` call.
_default_pipeline: Optional[UnifiedPDFPipeline] = None


def _get_default_pipeline() -> UnifiedPDFPipeline:
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = UnifiedPDFPipeline()
    return _default_pipeline


def route_and_extract(
    pdf_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> UnifiedResult:
    """
    End-to-end entrypoint: every PDF goes through the same content-driven
    pipeline (Marker for text; VLM/Paddle/grid for tables only if present).

    Kept for backward compatibility with code that imports `route_and_extract`
    from `pdf_extractor` -- delegates entirely to `UnifiedPDFPipeline`.

    Args:
        pdf_path: Path to the input PDF.
        output_dir: Directory for output files (created if missing).

    Returns:
        `UnifiedResult` with output paths, table count, which pages had
        tables, and which backend ultimately produced them.
    """
    return _get_default_pipeline().run(pdf_path, output_dir=output_dir)
