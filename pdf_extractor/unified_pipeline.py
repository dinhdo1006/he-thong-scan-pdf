"""
Unified Generic Pipeline: content-driven orchestration for ANY input PDF.

Every PDF goes through the same fixed sequence:
    Step A (always):     Marker           -> document Markdown (prose + anchors).
    Step B (always):     pdfplumber/visual scan -> which pages contain a table.
    Step C (if B found): VLM (full page) -> VLM crop retry -> Paddle -> grid.

On Windows CPU, Marker (torch) and Paddle collide in one process -- Step A and
the Paddle branch of Step C run in isolated subprocesses.

Primary outputs: `output.txt` + `output_tables.xlsx` (form grids).
Optional: `output.docx` with real Word tables.
If GPU/ML table backends fail, Marker's table blocks are kept with a loud warning.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from .exporters import (
    DEFAULT_OUTPUT_DOCX,
    DEFAULT_OUTPUT_TXT,
    DEFAULT_OUTPUT_XLSX,
    compose_document_txt,
    export_document_from_markdown,
    export_tables_preview,
    resolve_output_path,
    save_unified_txt,
)
from .grid_table_extractor import (
    DEFAULT_TABLE_SETTINGS,
    detect_table_pages,
    extract_pdf_tables_to_excel,
)
from .isolated_backends import (
    _skip_marker,
    extract_markdown_isolated,
    extract_paddle_tables_isolated,
)
from .marker_extractor import MarkerExtractor
from .markdown_parser import MarkdownBlockParser
from .paddle_extractor import PaddleExtractionError
from .paddle_extractor import extract as paddle_extract
from .text_fallback import extract_plaintext_fallback, plaintext_as_markdown
from .vlm_extractor import VLMConfig, VLMExtractionError, VLMTableExtractor

logger = logging.getLogger(__name__)

BACKEND_NONE = "none"
BACKEND_VLM = "vlm"
BACKEND_PADDLE = "paddle"
BACKEND_GRID = "grid"
BACKEND_MARKER = "marker"
BACKEND_FAILED = "failed"


@dataclass
class UnifiedResult:
    """Result of one `UnifiedPDFPipeline.run()` call."""

    output_path: Path
    table_count: int
    pages_with_tables: List[int] = field(default_factory=list)
    table_backend_used: str = BACKEND_NONE
    used_marker_table_fallback: bool = False
    docx_path: Optional[Path] = None
    xlsx_path: Optional[Path] = None
    txt_path: Optional[Path] = None


def _needs_process_isolation() -> bool:
    """True when Marker/torch and Paddle must not share one interpreter."""
    return sys.platform.startswith("win")


def _run_paddle_fallback(pdf_path: Path, out_dir: Path) -> List[pd.DataFrame]:
    if _needs_process_isolation():
        return extract_paddle_tables_isolated(pdf_path, out_dir)

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
    """Content-driven orchestrator: txt + xlsx primary, optional docx."""

    def __init__(
        self,
        marker_extractor: Optional[MarkerExtractor] = None,
        vlm_extractor: Optional[VLMTableExtractor] = None,
        table_settings: dict = DEFAULT_TABLE_SETTINGS,
        *,
        skip_marker: Optional[bool] = None,
        force_marker: bool = False,
    ) -> None:
        self.marker = marker_extractor or MarkerExtractor()
        # Full-page by default -- scanned B06 forms lose the grid when cropped.
        self.vlm = vlm_extractor or VLMTableExtractor(VLMConfig(crop_to_table=False))
        self.table_settings = table_settings
        self.skip_marker = skip_marker
        self.force_marker = force_marker

    def _detect_table_pages(self, pdf_path: Path) -> List[int]:
        return detect_table_pages(pdf_path, table_settings=self.table_settings)

    def _extract_markdown(self, pdf_path: Path, work_dir: Path) -> str:
        if _skip_marker(skip_marker=self.skip_marker, force_marker=self.force_marker):
            logger.info(
                "Skipping Marker (default on CPU / SKIP_MARKER). "
                "Using PyMuPDF prose; tables still come from Paddle/VLM. "
                "Set FORCE_MARKER=1 or pass --force-marker to enable Marker."
            )
            return plaintext_as_markdown(extract_plaintext_fallback(pdf_path))

        if _needs_process_isolation():
            logger.info("Windows: running Marker in an isolated subprocess (torch/paddle DLL split).")
            return extract_markdown_isolated(
                pdf_path,
                work_dir,
                skip_marker=False,
                force_marker=True,
            )
        try:
            return self.marker.extract_markdown(pdf_path)
        except Exception as exc:
            logger.warning("Marker failed (%s) -- using PyMuPDF text fallback.", exc)
            return plaintext_as_markdown(extract_plaintext_fallback(pdf_path))


    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _extract_tables(
        self,
        pdf_path: Path,
        out_dir: Path,
        page_indices: List[int],
    ) -> Tuple[List[pd.DataFrame], str]:
        # --- VLM (GPU only; Qwen2-VL is impractical on CPU for this form) ---
        if self._cuda_available():
            try:
                logger.info(
                    "Table backend: trying VLM on page(s) %s (full page)...",
                    [p + 1 for p in page_indices],
                )
                dataframes = self.vlm.extract_pages(
                    pdf_path, page_indices=page_indices, crop_to_table=False
                )
                if dataframes:
                    logger.info("VLM (full page) extracted %d table(s).", len(dataframes))
                    return dataframes, BACKEND_VLM
                logger.warning("VLM full-page returned 0 tables -- retrying with table-region crop.")
                dataframes = self.vlm.extract_pages(
                    pdf_path, page_indices=page_indices, crop_to_table=True
                )
                if dataframes:
                    logger.info("VLM (cropped) extracted %d table(s).", len(dataframes))
                    return dataframes, BACKEND_VLM
                logger.warning("VLM returned no usable tables -- falling back to PaddleOCR.")
            except VLMExtractionError as exc:
                logger.warning("VLM unavailable/failed (%s) -- falling back to PaddleOCR.", exc)
        else:
            logger.warning(
                "No CUDA GPU detected -- skipping VLM and using PaddleOCR PP-StructureV3 first."
            )

        # --- PaddleOCR PP-Structure (v3 on paddleocr>=3; works on CPU) ---
        try:
            logger.info("Table backend: trying PaddleOCR PP-Structure...")
            dataframes = _run_paddle_fallback(pdf_path, out_dir)
            if dataframes:
                logger.info("PaddleOCR extracted %d table(s).", len(dataframes))
                return dataframes, BACKEND_PADDLE
            logger.warning("PaddleOCR found no tables -- falling back to pdfplumber grid.")
        except (PaddleExtractionError, RuntimeError, OSError) as exc:
            logger.warning(
                "PaddleOCR unavailable/failed (%s) -- falling back to pdfplumber grid. "
                "Install with: pip install paddlepaddle==3.2.2 'paddleocr>=3.0' 'paddlex[ocr]'",
                exc,
            )

        # --- pdfplumber lines (native/vector PDFs only; scans usually get 0) ---
        logger.info("Table backend: trying pdfplumber ruled-line grid...")
        dataframes = _run_grid_fallback(pdf_path, out_dir, self.table_settings)
        if dataframes:
            logger.info("pdfplumber grid extracted %d table(s).", len(dataframes))
            return dataframes, BACKEND_GRID

        logger.error(
            "All image/grid table backends returned 0 tables on page(s) %s. "
            "Will fall back to Marker OCR tables (often garbled on scans).",
            [p + 1 for p in page_indices],
        )
        return [], BACKEND_GRID

    def run(
        self,
        pdf_path: str | Path,
        output: str | Path = DEFAULT_OUTPUT_TXT,
        *,
        output_filename: str = DEFAULT_OUTPUT_TXT,
        skip_tables: bool = False,
        skip_marker: Optional[bool] = None,
        force_marker: bool = False,
        write_txt: bool = True,
        write_xlsx: bool = True,
        write_docx: bool = False,
    ) -> UnifiedResult:
        """
        Process one PDF.

        Default deliverables: `output.txt` + `output_tables.xlsx`.
        Set `write_docx=True` (or pass a `.docx` `-o` path) for Word output.
        """
        if skip_marker is not None:
            self.skip_marker = skip_marker
        if force_marker:
            self.force_marker = True
        pdf_path = Path(pdf_path)
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        requested = Path(output).expanduser()
        suffix = requested.suffix.lower()
        want_docx_primary = suffix == ".docx"

        if suffix == ".docx":
            docx_path = resolve_output_path(requested, default_name=DEFAULT_OUTPUT_DOCX)
            work_dir = docx_path.parent
            txt_path = work_dir / DEFAULT_OUTPUT_TXT
            write_docx = True
        elif suffix == ".txt":
            txt_path = resolve_output_path(requested, default_name=DEFAULT_OUTPUT_TXT)
            work_dir = txt_path.parent
            docx_path = work_dir / DEFAULT_OUTPUT_DOCX
        else:
            work_dir = requested
            work_dir.mkdir(parents=True, exist_ok=True)
            txt_path = work_dir / DEFAULT_OUTPUT_TXT
            docx_path = work_dir / DEFAULT_OUTPUT_DOCX

        xlsx_path = work_dir / DEFAULT_OUTPUT_XLSX

        markdown = self._extract_markdown(pdf_path, work_dir)
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
            table_error: Optional[str] = None
            try:
                dataframes, backend = self._extract_tables(pdf_path, work_dir, pages_with_tables)
            except Exception as exc:
                table_error = str(exc)
                logger.error("Table extraction failed: %s", exc)
                dataframes, backend = [], BACKEND_FAILED

            if not dataframes:
                # Marker is skipped by default -- do NOT pretend Marker saved us.
                used_marker_fallback = False
                backend = BACKEND_FAILED
                logger.error(
                    "Table pages %s were detected but every backend returned 0 tables. "
                    "Install/fix: pip install paddlepaddle==3.2.2 'paddleocr>=3.0' 'paddlex[ocr]'. "
                    "Detail: %s",
                    [p + 1 for p in pages_with_tables],
                    table_error or "empty result",
                )

        final_tables: List[pd.DataFrame] = []
        written_docx: Optional[Path] = None
        if write_docx or want_docx_primary:
            written_docx, final_tables = export_document_from_markdown(
                markdown, dataframes, docx_path, apply_ocr_cleanup=True
            )
        else:
            from .exporters import assemble_document_blocks

            blocks = assemble_document_blocks(markdown, dataframes, apply_ocr_cleanup=True)
            final_tables = [
                b.content for b in blocks if b.kind == "table" and isinstance(b.content, pd.DataFrame)
            ]

        # Safety net: if assemble dropped tables but backends returned grids, keep them.
        if dataframes and not final_tables:
            logger.warning(
                "Document assembler dropped %d extracted table(s); exporting them anyway.",
                len(dataframes),
            )
            final_tables = list(dataframes)

        written_xlsx: Optional[Path] = None
        if write_xlsx:
            empty_reason = None
            if not final_tables and pages_with_tables and not skip_tables:
                empty_reason = (
                    "Table pages were detected but extraction backends returned 0 tables. "
                    "Check PaddleOCR install / logs (not a Marker OCR table)."
                )
            written = export_tables_preview(
                final_tables,
                xlsx_path,
                write_csv=True,
                write_markdown=False,
                empty_reason=empty_reason,
            )
            written_xlsx = written["xlsx"]

        written_txt: Optional[Path] = None
        if write_txt or not want_docx_primary:
            txt_content = compose_document_txt(markdown, dataframes, apply_ocr_cleanup=True)
            written_txt = save_unified_txt(txt_content, txt_path)

        marker_table_count = sum(1 for _ in MarkdownBlockParser().parse_blocks(markdown) if _.kind == "table")
        table_count = len(final_tables) if final_tables else marker_table_count

        primary = written_docx if want_docx_primary and written_docx else (written_txt or written_docx or xlsx_path)
        return UnifiedResult(
            output_path=primary if primary is not None else txt_path,
            table_count=table_count,
            pages_with_tables=pages_with_tables,
            table_backend_used=backend,
            used_marker_table_fallback=used_marker_fallback,
            docx_path=written_docx,
            xlsx_path=written_xlsx,
            txt_path=written_txt,
        )


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Content-driven PDF pipeline -> output.txt + output_tables.xlsx."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to the input PDF file.")
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT_TXT,
        help="Output .txt path, or a directory (writes output.txt + output_tables.xlsx).",
    )
    parser.add_argument(
        "--docx",
        action="store_true",
        help="Also write output.docx with real Word tables.",
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Fast test: Marker only, skip VLM/Paddle/grid extraction.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    from .logging_config import configure_app_logging

    args = _build_arg_parser().parse_args(argv)
    configure_app_logging(verbose=args.verbose)

    try:
        result = UnifiedPDFPipeline().run(
            args.input, output=args.output, skip_tables=args.skip_tables, write_docx=args.docx
        )
    except Exception as exc:
        logging.error("Pipeline failed: %s", exc)
        return 1

    print("Done.")
    if result.txt_path:
        print(f"  Text:              {result.txt_path}")
    if result.xlsx_path:
        print(f"  Tables (xlsx):     {result.xlsx_path}")
    if result.docx_path:
        print(f"  DOCX:              {result.docx_path}")
    if result.pages_with_tables:
        print(f"  Table page(s):     {[p + 1 for p in result.pages_with_tables]}")
        print(f"  Backend:           {result.table_backend_used}")
        if result.used_marker_table_fallback:
            print("  WARNING: tables from Marker OCR only -- install/fix VLM or PaddleOCR.")
    else:
        print("  No tables detected in PDF structure scan.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
