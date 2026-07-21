"""
Unified Generic Pipeline: content-driven orchestration for ANY input PDF.

No filename keyword, template name, or per-document configuration is used
anywhere in this module. Every PDF -- regardless of what it is -- goes
through the exact same fixed sequence of steps:

    Step A (always):     Marker           -> general document text/Markdown.
    Step B (always):     pdfplumber scan  -> which pages physically contain
                                              a table (`grid_table_extractor.
                                              detect_table_pages`).
    Step C (ONLY if B found >= 1 table page):
        C1: VLM (Qwen2-VL), restricted to the flagged pages only.
        C2: If C1 is unavailable/fails/finds nothing -> PaddleOCR
            PP-Structure (whole document) as a backup.
        C3: If C2 is unavailable/fails/finds nothing -> pure pdfplumber
            grid extraction (whole document) as a last resort -- this
            backend has no ML dependency at all, so it always works if a
            table has visible ruled lines.
    If Step B found NO tables anywhere, Step C (and the GPU) is skipped
    entirely -- table extraction is not attempted "just in case".

Outputs, written into `output_dir`:
    output_text.txt          -- prose (Marker's Markdown minus any pipe-table
                                 blocks it happened to also recognize, so
                                 table content isn't duplicated between the
                                 text file and the tables workbook)
    output_markdown.md       -- full intermediate Markdown (unless disabled)
    output_tables.xlsx        + `_preview.md` + `_table_N.csv` companions --
                                 written even when there are zero tables, so
                                 downstream consumers always find a valid file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from .exporters import TextExporter, export_document, export_tables_preview
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

DEFAULT_TEXT_FILENAME = "output_text.txt"
DEFAULT_MARKDOWN_FILENAME = "output_markdown.md"
DEFAULT_TABLES_FILENAME = "output_tables.xlsx"
DEFAULT_DOCX_FILENAME = "output.docx"

# Prepended to the saved intermediate Markdown so it's never mistaken for the
# verified table output -- Marker's own table guesses on scanned/complex forms
# are frequently wrong (merged cells, OCR typos); the real, checked tables
# live in `output_tables.xlsx` / `_preview.md` / `output.docx` instead.
_MARKDOWN_NOTE = (
    "<!-- LƯU Ý: đây là Markdown thô của Marker (chỉ dùng để đọc văn bản).\n"
    "     Mọi bảng hiển thị dưới đây là Marker tự đoán và có thể SAI cấu trúc/OCR.\n"
    "     Bảng đã được kiểm tra/trích xuất thật nằm ở output_tables.xlsx,\n"
    "     output_tables_preview.md, và output.docx — hãy đánh giá bảng ở đó. -->\n\n"
)

# Table-extraction backend identifiers, in fallback order.
BACKEND_NONE = "none"
BACKEND_VLM = "vlm"
BACKEND_PADDLE = "paddle"
BACKEND_GRID = "grid"


@dataclass
class UnifiedResult:
    """Paths + basic stats produced by one `UnifiedPDFPipeline.run()` call."""

    text_path: Path
    markdown_path: Optional[Path]
    tables_path: Path
    docx_path: Path
    table_count: int
    pages_with_tables: List[int] = field(default_factory=list)
    table_backend_used: str = BACKEND_NONE


def _run_paddle_fallback(pdf_path: Path, out_dir: Path) -> List[pd.DataFrame]:
    """Run the PaddleOCR PP-Structure backend, discarding its own throwaway workbook."""
    tmp_path = out_dir / ".tmp_paddle_fallback.xlsx"
    try:
        return paddle_extract(pdf_path, output_path=tmp_path, apply_ocr_cleanup=True)
    finally:
        tmp_path.unlink(missing_ok=True)


def _run_grid_fallback(pdf_path: Path, out_dir: Path, table_settings: dict) -> List[pd.DataFrame]:
    """Run the pure-pdfplumber grid backend, discarding its own throwaway workbook."""
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
    """
    Content-driven orchestrator.

    Decides, PER DOCUMENT (never per-template/keyword), whether table
    extraction is needed at all, and which backend actually produced usable
    results. Safe to reuse across many `.run()` calls: the underlying VLM
    model is loaded lazily on first use and cached on `self.vlm`.
    """

    def __init__(
        self,
        marker_extractor: Optional[MarkerExtractor] = None,
        markdown_parser: Optional[MarkdownBlockParser] = None,
        vlm_extractor: Optional[VLMTableExtractor] = None,
        table_settings: dict = DEFAULT_TABLE_SETTINGS,
    ) -> None:
        self.marker = marker_extractor or MarkerExtractor()
        self.markdown_parser = markdown_parser or MarkdownBlockParser()
        self.vlm = vlm_extractor or VLMTableExtractor()
        self.table_settings = table_settings

    # ------------------------------------------------------------------
    # Step A: general document text (always runs)
    # ------------------------------------------------------------------
    def _extract_text(self, pdf_path: Path, out_dir: Path, save_markdown: bool) -> Tuple[str, Optional[Path]]:
        markdown = self.marker.extract_markdown(pdf_path)

        markdown_path: Optional[Path] = None
        if save_markdown:
            markdown_path = out_dir / DEFAULT_MARKDOWN_FILENAME
            markdown_path.write_text(_MARKDOWN_NOTE + markdown, encoding="utf-8")
            logger.info("Saved intermediate Markdown -> %s", markdown_path)

        # Reuse the existing pipe-table/text splitter purely to STRIP any
        # tables Marker's own Markdown happens to contain, so the text file
        # never duplicates content that Step B/C already extracts properly
        # (with real column/row structure) into the tables workbook.
        parsed = self.markdown_parser.parse(markdown)
        return parsed.text, markdown_path

    # ------------------------------------------------------------------
    # Step B: structural table detection (cheap, CPU-only, always runs)
    # ------------------------------------------------------------------
    def _detect_table_pages(self, pdf_path: Path) -> List[int]:
        return detect_table_pages(pdf_path, table_settings=self.table_settings)

    # ------------------------------------------------------------------
    # Step C: table extraction with a 3-tier fallback chain
    # ------------------------------------------------------------------
    def _extract_tables(
        self,
        pdf_path: Path,
        out_dir: Path,
        page_indices: List[int],
    ) -> Tuple[List[pd.DataFrame], str]:
        # C1 -- VLM, restricted to ONLY the flagged pages.
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

        # C2 -- PaddleOCR PP-Structure (whole document; independent model stack).
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

        # C3 -- pure pdfplumber grid extraction: no ML dependency, always available.
        dataframes = _run_grid_fallback(pdf_path, out_dir, self.table_settings)
        logger.info("pdfplumber grid fallback extracted %d table(s).", len(dataframes))
        return dataframes, BACKEND_GRID

    # ------------------------------------------------------------------
    # End-to-end entrypoint
    # ------------------------------------------------------------------
    def run(
        self,
        pdf_path: str | Path,
        output_dir: str | Path = ".",
        *,
        save_markdown: bool = True,
        text_filename: str = DEFAULT_TEXT_FILENAME,
        tables_filename: str = DEFAULT_TABLES_FILENAME,
        docx_filename: str = DEFAULT_DOCX_FILENAME,
        skip_tables: bool = False,
    ) -> UnifiedResult:
        """
        Process one PDF end-to-end and write outputs into `output_dir`.

        Args:
            pdf_path: Input PDF path.
            output_dir: Directory for output files (created if missing). Must
                be a directory, NOT a `.xlsx`/file path -- if a file with that
                exact name already exists, a clear `NotADirectoryError` is
                raised instead of the confusing raw `[Errno 17] File exists`.
            save_markdown: Also write the intermediate Markdown file.
            text_filename: Plain text output file name.
            tables_filename: Excel workbook file name (CSV/Markdown preview
                companions are derived from this name automatically).
            skip_tables: If True, run Marker + the cheap pdfplumber table scan
                as usual, but never load/run the VLM/PaddleOCR/grid extraction
                backends -- writes an empty placeholder tables file instead.
                Useful for fast iteration (no GPU model loading) when you only
                want to sanity-check the text pipeline and wiring.

        Returns:
            `UnifiedResult` with output paths, table count, which pages had
            tables, and which backend ultimately produced them.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        out_dir = Path(output_dir).expanduser().resolve()
        if out_dir.is_file():
            raise NotADirectoryError(
                f"--output-dir must be a DIRECTORY, but '{out_dir}' is already an existing FILE. "
                f"Xoá/đổi tên file đó, hoặc chọn một thư mục khác (ví dụ: -o ./output)."
            )
        out_dir.mkdir(parents=True, exist_ok=True)

        # Step A -- always: general text via Marker.
        text, markdown_path = self._extract_text(pdf_path, out_dir, save_markdown)
        text_path = TextExporter().save(text, out_dir / text_filename)

        # Step B -- always: cheap structural scan for physical tables.
        pages_with_tables = self._detect_table_pages(pdf_path)
        tables_path = (out_dir / tables_filename).expanduser().resolve()
        docx_path = (out_dir / docx_filename).expanduser().resolve()

        dataframes: List[pd.DataFrame] = []
        backend = BACKEND_NONE

        if not pages_with_tables:
            logger.info(
                "No tables detected anywhere in %s -- skipping table extraction (GPU untouched).", pdf_path
            )
            # Marker is no longer needed for this document -- free VRAM.
            self.marker.unload()
        elif skip_tables:
            logger.info(
                "skip_tables=True -- %d table page(s) detected but extraction backends "
                "(VLM/PaddleOCR/grid) are NOT run.",
                len(pages_with_tables),
            )
            self.marker.unload()
        else:
            # Critical on 16 GiB cards: Marker + Qwen2-VL-7B cannot coexist in VRAM.
            # Drop Marker BEFORE Step C loads the VLM.
            self.marker.unload()

            dataframes, backend = self._extract_tables(pdf_path, out_dir, pages_with_tables)
            if not dataframes:
                logger.error(
                    "Table page(s) %s were structurally/visually detected, but EVERY "
                    "backend (VLM -> PaddleOCR -> pdfplumber grid) returned zero usable "
                    "tables for %s. This is NOT the same as 'document has no tables' -- "
                    "check the WARNING lines above (VLM raw response, OOM, PaddleOCR/grid "
                    "errors) for the actual cause before trusting an empty output_tables.xlsx.",
                    [p + 1 for p in pages_with_tables],
                    pdf_path,
                )

        export_tables_preview(dataframes, tables_path)
        export_document(text, dataframes, docx_path)

        return UnifiedResult(
            text_path=text_path,
            markdown_path=markdown_path,
            tables_path=tables_path,
            docx_path=docx_path,
            table_count=len(dataframes),
            pages_with_tables=pages_with_tables,
            table_backend_used=backend,
        )


# ---------------------------------------------------------------------------
# CLI (standalone usage / debugging)
# ---------------------------------------------------------------------------
def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Content-driven PDF pipeline: Marker for text (always), then "
            "VLM -> PaddleOCR -> pdfplumber-grid for tables (only if a "
            "pdfplumber scan physically finds one). No keyword or template "
            "configuration involved."
        )
    )
    parser.add_argument("--input", "-i", required=True, help="Path to the input PDF file.")
    parser.add_argument("--output-dir", "-o", default="./output", help="Directory for output files.")
    parser.add_argument("--no-markdown", action="store_true", help="Do not save the intermediate Markdown file.")
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Fast test mode: run Marker + table detection only, skip VLM/PaddleOCR/grid extraction (no GPU model load).",
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
        result = UnifiedPDFPipeline().run(
            args.input,
            output_dir=args.output_dir,
            save_markdown=not args.no_markdown,
            skip_tables=args.skip_tables,
        )
    except Exception as exc:
        _logging.error("Pipeline failed: %s", exc)
        return 1

    print("Done.")
    print(f"  Text:              {result.text_path}")
    if result.markdown_path:
        print(f"  Markdown:          {result.markdown_path}")
    print(f"  Word document:     {result.docx_path}")
    if result.pages_with_tables:
        print(f"  Tables detected on page(s): {[p + 1 for p in result.pages_with_tables]}")
        print(f"  Backend used:      {result.table_backend_used}")
        print(f"  Tables ({result.table_count}):      {result.tables_path}")
    else:
        print("  No tables detected -- table extraction skipped (GPU untouched).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
