"""
Zero-shot, deep-learning-based table extraction using PaddleOCR's PP-Structure.

This module REPLACES the old `coordinate_extractor.py`. Instead of relying on
hardcoded/anchor-derived x0-y0 coordinates to bucket words into rows and
columns, PP-Structure performs full LAYOUT ANALYSIS + TABLE STRUCTURE
RECOGNITION on a rendered page image and hands back the table already
reconstructed as HTML (`<table><tr><td colspan=... rowspan=...>`).

Why this is "zero-shot":
    - No anchor header strings to configure per document template.
    - No absolute or relative coordinate ranges to calibrate.
    - `colspan` / `rowspan` (merged cells) are handled by the model itself and
      then by `pd.read_html()` -- no manual coordinate math anywhere.

Pipeline:
    1. render_pdf_to_images() -> rasterize every PDF page to a BGR numpy array
       (via PyMuPDF, already a project dependency -- no poppler/pdf2image
       needed).
    2. get_engine()           -> lazily construct a single, reusable
       `PPStructure(show_log=False, layout=True)` instance.
    3. analyze_page()         -> run PP-Structure layout analysis on one page
       image, returning its raw list of layout blocks.
    4. filter_table_blocks()  -> keep only blocks where `res['type'] == 'table'`.
    5. table_block_to_dataframe() -> pull `res['res']['html']` out of a table
       block and parse it with `pd.read_html()`.
    6. clean_ocr_errors()     -> reuse the existing OCR fix-up dictionary
       (see `ocr_cleanup.py`) to clean recognized text.
    7. export_to_excel()      -> write every recovered table to
       `output_tables.xlsx` (one sheet per table).

100% local/offline once the PP-Structure model weights have been downloaded
the first time -- no external API calls happen during extraction itself.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .ocr_cleanup import clean_ocr_errors

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = "output_tables.xlsx"

# Rendering resolution for rasterizing PDF pages before feeding PP-Structure.
# Higher DPI improves recognition accuracy on small text, at the cost of speed.
DEFAULT_DPI = 200


class PaddleExtractionError(Exception):
    """Raised when the PP-Structure pipeline cannot be initialized or run."""


# ---------------------------------------------------------------------------
# Step 1: PDF -> page images (no poppler / pdf2image dependency needed)
# ---------------------------------------------------------------------------
def render_pdf_to_images(pdf_path: str | Path, dpi: int = DEFAULT_DPI) -> List[np.ndarray]:
    """
    Rasterize every page of a PDF into a BGR numpy array (OpenCV convention).

    PP-Structure expects images in the same format `cv2.imread()` would
    produce (H x W x 3, BGR channel order), so this converts PyMuPDF's native
    RGB pixmap samples accordingly.

    Args:
        pdf_path: Path to the input PDF.
        dpi: Rendering resolution. 200 DPI is a good accuracy/speed trade-off
            for typical scanned forms; raise it if small text is misread.

    Returns:
        One BGR numpy array per page, in page order.

    Raises:
        FileNotFoundError: If the PDF does not exist.
    """
    import fitz  # PyMuPDF -- already a project dependency

    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    zoom = dpi / 72.0  # PyMuPDF's base unit is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)

    images: List[np.ndarray] = []
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            bgr = rgb[:, :, ::-1].copy()  # RGB -> BGR
            images.append(bgr)
    finally:
        doc.close()

    logger.info("Rendered %d page(s) from %s at %d DPI.", len(images), pdf_path, dpi)
    return images


# ---------------------------------------------------------------------------
# Step 2: Lazy PP-Structure engine (heavy model weights loaded once)
# ---------------------------------------------------------------------------
_engine_cache: Dict[str, Any] = {}


def get_engine():
    """
    Lazily construct (and cache) a single `PPStructure` instance.

    Loading PP-Structure's layout + table-recognition models is expensive, so
    this is done once per process and reused across pages/documents.

    Raises:
        PaddleExtractionError: If `paddleocr` is not installed.
    """
    if "engine" in _engine_cache:
        return _engine_cache["engine"]

    try:
        from paddleocr import PPStructure
    except ImportError as exc:
        raise PaddleExtractionError(
            "paddleocr is not installed. Run: pip install paddlepaddle paddleocr"
        ) from exc

    logger.info("Loading PP-Structure layout engine (show_log=False, layout=True)...")
    engine = PPStructure(show_log=False, layout=True)
    _engine_cache["engine"] = engine
    logger.info("PP-Structure engine ready.")
    return engine


# ---------------------------------------------------------------------------
# Step 3: Layout analysis + table block filtering
# ---------------------------------------------------------------------------
def analyze_page(image: np.ndarray, engine=None) -> List[Dict[str, Any]]:
    """
    Run PP-Structure layout analysis on a single page image.

    Returns:
        The raw list of layout blocks PP-Structure detected (text, title,
        figure, table, ...). Never raises for "no blocks found" -- an empty
        image simply yields an empty list.
    """
    engine = engine or get_engine()
    try:
        return engine(image) or []
    except Exception as exc:  # pragma: no cover - defensive: model/runtime errors
        logger.warning("PP-Structure failed to analyze a page: %s", exc)
        return []


def filter_table_blocks(layout_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only the layout blocks PP-Structure classified as `table`."""
    tables = [block for block in layout_blocks if block.get("type") == "table"]
    logger.debug("Filtered %d table block(s) out of %d layout block(s).", len(tables), len(layout_blocks))
    return tables


# ---------------------------------------------------------------------------
# Step 4: Table block (HTML) -> DataFrame
# ---------------------------------------------------------------------------
def table_block_to_dataframe(table_block: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Convert one PP-Structure table block into a pandas DataFrame.

    PP-Structure's table-recognition sub-model already resolved merged cells
    (colspan/rowspan) into the returned HTML, so `pd.read_html()` recovers the
    correct grid shape with zero coordinate math on our side.

    Returns:
        The parsed DataFrame, or `None` if this block had no usable HTML
        (e.g. the table-recognition sub-model failed on this crop) -- callers
        should skip `None` results rather than raise, so one bad table on one
        page doesn't abort the whole document.
    """
    html = table_block.get("res", {}).get("html")
    if not html:
        logger.warning("Table block on page %s has no HTML output -- skipping.", table_block.get("page"))
        return None

    try:
        # StringIO avoids pandas' deprecation warning about passing raw HTML
        # strings directly, and works identically across pandas versions.
        parsed_tables = pd.read_html(io.StringIO(html))
    except ValueError as exc:
        # pd.read_html raises ValueError when it cannot find any <table> tag,
        # e.g. a degenerate/empty table crop.
        logger.warning("Could not parse table HTML into a DataFrame: %s", exc)
        return None

    if not parsed_tables:
        return None

    return parsed_tables[0]


# ---------------------------------------------------------------------------
# Step 5: Export
# ---------------------------------------------------------------------------
def export_to_excel(dataframes: List[pd.DataFrame], output_path: str | Path) -> Path:
    """
    Write every recovered table to its own sheet in one .xlsx workbook.

    If no tables were found anywhere in the document, a small placeholder
    workbook is written instead of raising, so callers always get a valid
    file to point users at.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not dataframes:
        pd.DataFrame({"info": ["No tables were detected by PP-Structure in this document."]}).to_excel(
            output_path, index=False, sheet_name="Info", engine="openpyxl"
        )
        logger.warning("No tables detected -- wrote placeholder workbook to %s", output_path)
        return output_path

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for i, df in enumerate(dataframes, start=1):
            sheet_name = f"Table_{i}"[:31]  # Excel hard-limits sheet names to 31 chars
            df.to_excel(writer, index=False, sheet_name=sheet_name)

    logger.info("Exported %d table(s) to %s", len(dataframes), output_path)
    return output_path


# ---------------------------------------------------------------------------
# Step 6: End-to-end entrypoint (this is what router.py calls)
# ---------------------------------------------------------------------------
def extract(
    pdf_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    dpi: int = DEFAULT_DPI,
    apply_ocr_cleanup: bool = True,
) -> List[pd.DataFrame]:
    """
    Full pipeline: PDF -> page images -> PP-Structure layout -> table HTML ->
    DataFrame -> OCR cleanup -> Excel export.

    Args:
        pdf_path: Path to the input PDF.
        output_path: Destination .xlsx path.
        dpi: Page rendering resolution passed to `render_pdf_to_images`.
        apply_ocr_cleanup: If True (default), run every DataFrame through
            `clean_ocr_errors()` before exporting.

    Returns:
        One cleaned DataFrame per table found across the whole document (may
        be an empty list if the document genuinely has no tables -- this is
        NOT an error condition, just an empty result).
    """
    pdf_path = Path(pdf_path)
    engine = get_engine()
    dataframes: List[pd.DataFrame] = []

    try:
        page_images = render_pdf_to_images(pdf_path, dpi=dpi)
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise PaddleExtractionError(f"Failed to render '{pdf_path}' to images: {exc}") from exc

    for page_index, image in enumerate(page_images):
        layout_blocks = analyze_page(image, engine=engine)
        table_blocks = filter_table_blocks(layout_blocks)

        if not table_blocks:
            logger.info("Page %d: no tables detected -- skipping.", page_index + 1)
            continue

        for table_index, table_block in enumerate(table_blocks):
            table_block["page"] = page_index + 1  # for logging only
            df = table_block_to_dataframe(table_block)
            if df is None or df.empty:
                logger.info(
                    "Page %d, table %d: empty/unparsable -- skipping.",
                    page_index + 1,
                    table_index + 1,
                )
                continue

            if apply_ocr_cleanup:
                df = clean_ocr_errors(df)

            dataframes.append(df)

    export_to_excel(dataframes, output_path)
    return dataframes


# ---------------------------------------------------------------------------
# CLI (standalone usage / debugging, independent of the router)
# ---------------------------------------------------------------------------
def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot table extraction via PaddleOCR PP-Structure (layout "
            "analysis + table structure recognition). No coordinates, no "
            "anchors, no per-document configuration required."
        )
    )
    parser.add_argument("--input", "-i", required=True, help="Path to the input PDF file.")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_PATH, help="Path to the output .xlsx file.")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Page rendering resolution.")
    parser.add_argument("--no-ocr-cleanup", action="store_true", help="Skip the OCR post-processing replacement pass.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        dataframes = extract(
            args.input,
            output_path=args.output,
            dpi=args.dpi,
            apply_ocr_cleanup=not args.no_ocr_cleanup,
        )
    except (FileNotFoundError, PaddleExtractionError) as exc:
        logging.error(str(exc))
        return 1

    print(f"Done. Exported {len(dataframes)} table(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
