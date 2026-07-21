"""
Unified Generic Pipeline: content-driven orchestration for ANY input PDF.

Every PDF goes through the same fixed sequence:
    Step A (always):     Marker           -> document Markdown.
    Step B (always):     pdfplumber scan  -> which pages contain a table.
    Step C (if B found): VLM -> PaddleOCR -> pdfplumber grid (first usable wins).

Output: ONE plain-text `.txt` file only (prose + tables in reading order).
If the GPU/ML table backends fail, Marker's table blocks are kept so content
is never silently dropped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from .exporters import DEFAULT_OUTPUT_TXT, compose_document_txt, resolve_output_path, save_unified_txt
from .grid_table_extractor import (
    DEFAULT_TABLE_SETTINGS,
    detect_table_pages,
    extract_pdf_tables_to_excel,
)
from .marker_extractor import MarkerExtractor
from .markdown_parser import MarkdownBlockParser
from .paddle_extractor import PaddleExtractionError
from .paddle_extractor import extract as paddle_extract
from .vlm_extractor import VLMExtractionError, VLMTableExtractor

logger = logging.getLogger(__name__)

BACKEND_NONE = "none"
BACKEND_VLM = "vlm"
BACKEND_PADDLE = "paddle"
BACKEND_GRID = "grid"
BACKEND_MARKER = "marker"


@dataclass
class UnifiedResult:
    """Result of one `UnifiedPDFPipeline.run()` call."""

    output_path: Path
    table_count: int
    pages_with_tables: List[int] = field(default_factory=list)
    table_backend_used: str = BACKEND_NONE
    used_marker_table_fallback: bool = False


def _run_paddle_fallback(pdf_path: Path, out_dir: Path) -> List[pd.DataFrame]:
    tmp_path = out_dir / ".tmp_paddle_fallback.xlsx"
    try:
        return paddle_extract(pdf_path, output_path=tmp_path, apply_ocr_cleanup=True)
    finally:
        tmp_path.unlink(missing_ok=True)


def _run_grid_fallback(pdf_path: Path, out_dir: Path, table_settings: dict) -> List[pd.DataFrame]:
    tmp_path = out_dir / ".tmp_grid_fallback.xlsx"
    try:
        return extract_pdf_tables_to_excel(
            pdf_path,
            output_path=tmp_path,
            table_settings=table_settings,
            apply_ocr_cleanup=True,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


class UnifiedPDFPipeline:
    """Content-driven orchestrator producing a single `.txt` file."""

    def __init__(
        self,
        marker_extractor: Optional[MarkerExtractor] = None,
        vlm_extractor: Optional[VLMTableExtractor] = None,
        table_settings: dict = DEFAULT_TABLE_SETTINGS,
    ) -> None:
        self.marker = marker_extractor or MarkerExtractor()
        self.vlm = vlm_extractor or VLMTableExtractor()
        self.table_settings = table_settings

    def _detect_table_pages(self, pdf_path: Path) -> List[int]:
        return detect_table_pages(pdf_path, table_settings=self.table_settings)

    def _extract_tables(
        self,
        pdf_path: Path,
        out_dir: Path,
        page_indices: List[int],
    ) -> Tuple[List[pd.DataFrame], str]:
        try:
            dataframes = self.vlm.extract_pages(pdf_path, page_indices=page_indices)
            if dataframes:
                logger.info(
                    "VLM extracted %d table(s) from page(s) %s.",
                    len(dataframes),
                    [p + 1 for p in page_indices],
                )
                return dataframes, BACKEND_VLM
            logger.warning("VLM ran but returned no usable tables -- falling back to PaddleOCR.")
        except VLMExtractionError as exc:
            logger.warning("VLM unavailable/failed (%s) -- falling back to PaddleOCR.", exc)

        try:
            dataframes = _run_paddle_fallback(pdf_path, out_dir)
            if dataframes:
                logger.info("PaddleOCR fallback extracted %d table(s).", len(dataframes))
                return dataframes, BACKEND_PADDLE
            logger.warning("PaddleOCR fallback found no tables -- falling back to pdfplumber grid extraction.")
        except PaddleExtractionError as exc:
            logger.warning(
                "PaddleOCR fallback unavailable/failed (%s) -- falling back to pdfplumber grid extraction.", exc
            )

        dataframes = _run_grid_fallback(pdf_path, out_dir, self.table_settings)
        logger.info("pdfplumber grid fallback extracted %d table(s).", len(dataframes))
        return dataframes, BACKEND_GRID

    def run(
        self,
        pdf_path: str | Path,
        output: str | Path = DEFAULT_OUTPUT_TXT,
        *,
        output_filename: str = DEFAULT_OUTPUT_TXT,
        skip_tables: bool = False,
    ) -> UnifiedResult:
        """
        Process one PDF and write a single `.txt` file.

        Args:
            pdf_path: Input PDF.
            output: Output `.txt` path, or a directory (writes `output.txt` inside).
            output_filename: File name when `output` is a directory.
            skip_tables: Skip VLM/Paddle/grid (Marker text + Marker tables only).
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        output_path = resolve_output_path(output, default_name=output_filename)
        work_dir = output_path.parent

        markdown = self.marker.extract_markdown(pdf_path)
        pages_with_tables = self._detect_table_pages(pdf_path)

        dataframes: List[pd.DataFrame] = []
        backend = BACKEND_NONE
        used_marker_fallback = False

        if not pages_with_tables:
            logger.info("No tables detected -- skipping GPU table extraction.")
            self.marker.unload()
        elif skip_tables:
            logger.info("skip_tables=True -- using Marker's tables only.")
            self.marker.unload()
            used_marker_fallback = True
            backend = BACKEND_MARKER
        else:
            self.marker.unload()
            dataframes, backend = self._extract_tables(pdf_path, work_dir, pages_with_tables)
            if not dataframes:
                logger.error(
                    "Table page(s) %s detected but every backend returned zero tables. "
                    "Falling back to Marker's table blocks in the final .txt.",
                    [p + 1 for p in pages_with_tables],
                )
                used_marker_fallback = True
                backend = BACKEND_MARKER

        txt_content = compose_document_txt(markdown, dataframes, apply_ocr_cleanup=True)
        save_unified_txt(txt_content, output_path)

        marker_table_count = sum(1 for _ in MarkdownBlockParser().parse_blocks(markdown) if _.kind == "table")
        table_count = len(dataframes) if dataframes else marker_table_count

        return UnifiedResult(
            output_path=output_path,
            table_count=table_count,
            pages_with_tables=pages_with_tables,
            table_backend_used=backend,
            used_marker_table_fallback=used_marker_fallback,
        )


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Content-driven PDF pipeline -> single plain-text .txt output."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to the input PDF file.")
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT_TXT,
        help="Output .txt file path, or a directory (writes output.txt inside).",
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Fast test: Marker only, skip VLM/Paddle/grid extraction.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    import logging as _logging

    args = _build_arg_parser().parse_args(argv)
    _logging.basicConfig(
        level=_logging.DEBUG if args.verbose else _logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        result = UnifiedPDFPipeline().run(args.input, output=args.output, skip_tables=args.skip_tables)
    except Exception as exc:
        _logging.error("Pipeline failed: %s", exc)
        return 1

    print("Done.")
    print(f"  Output:            {result.output_path}")
    if result.pages_with_tables:
        print(f"  Table page(s):     {[p + 1 for p in result.pages_with_tables]}")
        print(f"  Backend:           {result.table_backend_used}")
        if result.used_marker_table_fallback:
            print("  Note: ML backends failed or skipped -- tables from Marker OCR.")
    else:
        print("  No tables detected in PDF structure scan.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
