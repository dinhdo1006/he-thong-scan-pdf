"""
Zero-shot, grid/border-based table extraction.

Unlike `coordinate_extractor.py` (anchor-text based) and the even older hardcoded
absolute-x0-range approach, this module makes NO assumptions about:
  - specific header text ("Số TT", "Tiêu chí", "Số thực thu", ...)
  - specific absolute or relative coordinates
  - a specific document template (Form B06, an invoice, a bank statement, ...)

It relies purely on the PHYSICAL RULED LINES (table borders) that `pdfplumber`
detects on the page via `page.extract_tables()`. Any PDF with visibly drawn table
borders can be parsed the same way, with ZERO per-document configuration. This is
what makes it "zero-shot": point it at a brand-new document type and it works,
or it doesn't (falls back to another pipeline) -- there is no template to write
or coordinate range to calibrate first.

Trade-off (documented, not hidden): tables WITHOUT visible ruling lines (pure
whitespace/alignment-based tables, common in some scanned forms) are NOT reliably
found by the "lines" strategy used here. For those, the anchor-based
`coordinate_extractor.py` pipeline (or Pipeline 1 / Marker) remains the fallback.

Pipeline:
    1. load_pdf()                 -> open the PDF with pdfplumber (context-managed).
    2. extract_raw_tables()       -> page.extract_tables() per page -> raw grids.
    3. table_to_dataframe()       -> one raw grid -> one clean pandas DataFrame.
    4. clean_ocr_errors()         -> reuse the existing OCR fix-up dict (ocr_cleanup.py).
    5. export_to_excel()          -> one sheet per detected table.

100% offline: no network calls, no AI/LLM, nothing leaves the machine. Safe for
confidential documents.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import pdfplumber

from .ocr_cleanup import clean_ocr_errors

logger = logging.getLogger(__name__)

# pdfplumber's table-finding strategy: detect cell boundaries purely from the
# VECTOR LINES/RECTS actually drawn on the page, never from text alignment or
# whitespace gaps. This is the crux of "zero hardcoding" -- the grid comes from
# whatever borders truly exist on THIS page, for ANY document, instead of from a
# config file we maintain per template.
DEFAULT_TABLE_SETTINGS: Dict[str, str] = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
}


@dataclass
class RawTable:
    """One table exactly as pdfplumber found it, before any cleanup."""

    page: int  # 0-based page index
    index_on_page: int  # 0-based table index within that page (a page can have >1 table)
    rows: List[List[Optional[str]]]  # raw grid; pdfplumber uses None for empty cells


def load_pdf(pdf_path: str | Path) -> pdfplumber.PDF:
    """
    Open a PDF file with pdfplumber.

    Returns a `pdfplumber.PDF` context manager -- callers should use it with
    `with load_pdf(path) as pdf:` (or call `.close()` manually) to ensure the
    underlying file handle is released.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    logger.info("Loading PDF (pdfplumber, offline, grid/border mode): %s", pdf_path)
    return pdfplumber.open(pdf_path)


def extract_raw_tables(
    pdf: pdfplumber.PDF,
    table_settings: Dict[str, str] = DEFAULT_TABLE_SETTINGS,
) -> List[RawTable]:
    """
    Detect every bordered table on every page, purely from ruled lines.

    `page.extract_tables(table_settings)` returns a list of tables per page;
    each table is a list of rows, and each row is a list of cell strings (or
    `None` for a grid cell with no detected text). No knowledge of column
    headers, expected row counts, or absolute/relative coordinates is required
    anywhere in this call -- pdfplumber infers the entire grid from the page's
    own vector graphics. This is exactly why the same call generalizes across
    very different document types (Form B06, invoices, bank statements, ...)
    without any per-document configuration.
    """
    raw_tables: List[RawTable] = []
    for page_index, page in enumerate(pdf.pages):
        page_tables = page.extract_tables(table_settings=table_settings)
        for table_index, rows in enumerate(page_tables):
            if not rows:
                continue  # pdfplumber can report an empty table for stray/degenerate line intersections
            raw_tables.append(RawTable(page=page_index, index_on_page=table_index, rows=rows))

    logger.info(
        "Detected %d bordered table(s) across %d page(s) (strategy=%s).",
        len(raw_tables),
        len(pdf.pages),
        table_settings,
    )
    return raw_tables


def _dedupe_headers(header_row: List[Optional[str]]) -> List[str]:
    """
    Turn a raw header row into unique, non-empty, pandas-safe column names.

    Handles, gracefully and generically (no document-specific logic):
      - `None` / empty-string cells      -> "Column_<n>"
      - duplicate header text            -> suffixed with "_2", "_3", ...
      - embedded newlines (multi-line headers inside one merged cell, common
        in bordered forms) -> collapsed to a single space
    """
    seen: Dict[str, int] = {}
    clean_names: List[str] = []
    for position, raw_name in enumerate(header_row):
        name = (raw_name or "").replace("\n", " ").strip()
        if not name:
            name = f"Column_{position + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        clean_names.append(name)
    return clean_names


def table_to_dataframe(table: RawTable, header_row: bool = True) -> pd.DataFrame:
    """
    Convert one RawTable's raw list-of-lists into a clean pandas DataFrame.

    Args:
        table: The RawTable to convert.
        header_row: If True (default), the table's first row is treated as
            column headers. If False, generic "Column_1", "Column_2", ...
            names are generated and every row (including the first) is data --
            use this for tables that have no header row at all.

    Empty cells (pdfplumber's `None` marker for "no text detected in this grid
    cell") are converted to empty strings here, so every downstream step (OCR
    cleanup, Excel export, further pandas processing) only ever has to deal
    with plain strings -- no `None`-vs-`""` special-casing anywhere else.
    """
    rows = table.rows
    if not rows:
        return pd.DataFrame()

    if header_row:
        columns = _dedupe_headers(rows[0])
        data_rows = rows[1:]
    else:
        width = max(len(r) for r in rows)
        columns = [f"Column_{i + 1}" for i in range(width)]
        data_rows = rows

    # pdfplumber can occasionally emit a row that's shorter than the header
    # (e.g. around merged/spanned cells). Pad rather than raise, so no row is
    # ever silently dropped just because its width didn't match exactly.
    normalized_rows: List[List[str]] = []
    for row in data_rows:
        padded = list(row) + [None] * (len(columns) - len(row))
        normalized_rows.append(
            [("" if cell is None else str(cell).replace("\n", " ").strip()) for cell in padded[: len(columns)]]
        )

    df = pd.DataFrame(normalized_rows, columns=columns)
    # Provenance columns -- always useful when a workbook has many small tables
    # spread across pages, and cost nothing in terms of hardcoding.
    df.insert(0, "table_index", table.index_on_page)
    df.insert(0, "page", table.page + 1)
    return df


def tables_to_dataframes(tables: List[RawTable], header_row: bool = True) -> List[pd.DataFrame]:
    """Convert every detected RawTable into its own clean DataFrame."""
    return [table_to_dataframe(t, header_row=header_row) for t in tables]


# --- Visual (pixel-based) fallback: for SCANNED pages ------------------------
# `page.find_tables()` above only ever sees VECTOR line/rect objects. A scanned
# page (a photographed/scanned form flattened into one big embedded raster
# image, which is extremely common for older government forms) has table
# borders that are just printed PIXELS -- there is no vector object for
# pdfplumber to find, so the structural scan above silently returns "no
# table" even when a table is clearly visible to the human eye. The functions
# below are a cheap, template-agnostic (no per-document config) heuristic
# that renders the page and looks for a dense grid of horizontal + vertical
# dark-pixel bands, so scanned documents aren't silently skipped.
DEFAULT_VISUAL_DPI = 150
DEFAULT_VISUAL_DARK_THRESHOLD = 180  # 0-255 grayscale; below this counts as "ink"
DEFAULT_VISUAL_LINE_FRACTION = 0.2  # a row/column needs >= 20% ink to count as a candidate ruling
DEFAULT_VISUAL_MIN_CLUSTERS = 5  # need this many distinct horizontal AND vertical bands to call it a grid


def _render_page_grayscale(pdf_path: str | Path, page_index: int, dpi: int = DEFAULT_VISUAL_DPI) -> np.ndarray:
    """Render one page to a 2D grayscale numpy array (0=black, 255=white) via PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    try:
        pix = doc[page_index].get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    finally:
        doc.close()


def _line_cluster_spans(ink_fraction: np.ndarray, threshold: float, gap: int = 3) -> List[Tuple[int, int]]:
    """
    Group indices where `ink_fraction` exceeds `threshold` into (start, end)
    spans, merging runs separated by up to `gap` pixels. A single thick ruled
    line renders as several adjacent rows/columns above threshold -- this
    collapses that into ONE span, so span count roughly tracks "how many
    ruling lines", not "how many pixels".
    """
    above = np.where(ink_fraction > threshold)[0]
    if len(above) == 0:
        return []
    spans: List[Tuple[int, int]] = []
    start = prev = above[0]
    for idx in above[1:]:
        if idx - prev > gap:
            spans.append((int(start), int(prev)))
            start = idx
        prev = idx
    spans.append((int(start), int(prev)))
    return spans


def _count_line_clusters(ink_fraction: np.ndarray, threshold: float, gap: int = 3) -> int:
    """Count distinct ruling-line bands (see `_line_cluster_spans`)."""
    return len(_line_cluster_spans(ink_fraction, threshold, gap=gap))


def page_looks_like_scanned_table(
    pdf_path: str | Path,
    page_index: int,
    dpi: int = DEFAULT_VISUAL_DPI,
    dark_threshold: int = DEFAULT_VISUAL_DARK_THRESHOLD,
    line_fraction: float = DEFAULT_VISUAL_LINE_FRACTION,
    min_clusters: int = DEFAULT_VISUAL_MIN_CLUSTERS,
) -> bool:
    """
    Pixel-based heuristic: does this page visually contain a dense ruled grid?

    Used ONLY as a fallback when `find_tables()` (vector-based) finds nothing
    anywhere in the document -- i.e. exactly the scanned/flattened-image case
    it structurally cannot see. Trades perfect precision for not silently
    skipping table extraction on scanned forms.
    """
    gray = _render_page_grayscale(pdf_path, page_index, dpi=dpi)
    dark = gray < dark_threshold
    height, width = dark.shape
    row_ink_fraction = dark.sum(axis=1) / width
    col_ink_fraction = dark.sum(axis=0) / height
    horizontal_bands = _count_line_clusters(row_ink_fraction, line_fraction)
    vertical_bands = _count_line_clusters(col_ink_fraction, line_fraction)
    return horizontal_bands >= min_clusters and vertical_bands >= min_clusters


def find_table_bbox(
    pdf_path: str | Path,
    page_index: int,
    dpi: int = DEFAULT_VISUAL_DPI,
    dark_threshold: int = DEFAULT_VISUAL_DARK_THRESHOLD,
    line_fraction: float = DEFAULT_VISUAL_LINE_FRACTION,
    min_clusters: int = DEFAULT_VISUAL_MIN_CLUSTERS,
    padding_ratio: float = 0.02,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Locate the bounding box of the densest ruled-grid region on a page, in
    PDF POINT coordinates (x0, y0, x1, y1) -- suitable for a `fitz.Rect` clip.

    Purely pixel/ink-density driven (same signal as
    `page_looks_like_scanned_table`): the box spans from the first to the
    last detected horizontal/vertical ruling-line band, plus a small margin.
    No template, header text, or column count is assumed -- this works for
    ANY grid of ruled lines, on ANY document. Returns `None` if the page does
    not look like it has a dense ruled grid at all (same threshold as
    `page_looks_like_scanned_table`), so callers can fall back to the full
    page image instead of cropping away real content.

    Cropping the page to just this region before handing it to a VLM removes
    unrelated noise (letterhead, signatures, stamps) so the model can focus
    its limited attention/tokens on the actual grid.
    """
    gray = _render_page_grayscale(pdf_path, page_index, dpi=dpi)
    dark = gray < dark_threshold
    height, width = dark.shape
    row_ink_fraction = dark.sum(axis=1) / width
    col_ink_fraction = dark.sum(axis=0) / height
    row_spans = _line_cluster_spans(row_ink_fraction, line_fraction)
    col_spans = _line_cluster_spans(col_ink_fraction, line_fraction)

    if len(row_spans) < min_clusters or len(col_spans) < min_clusters:
        return None

    y0_px, y1_px = row_spans[0][0], row_spans[-1][1]
    x0_px, x1_px = col_spans[0][0], col_spans[-1][1]

    pad_y = int((y1_px - y0_px) * padding_ratio) + 5
    pad_x = int((x1_px - x0_px) * padding_ratio) + 5
    y0_px = max(0, y0_px - pad_y)
    y1_px = min(height, y1_px + pad_y)
    x0_px = max(0, x0_px - pad_x)
    x1_px = min(width, x1_px + pad_x)

    scale = 72.0 / dpi  # pixels (at `dpi`) -> PDF points
    return (x0_px * scale, y0_px * scale, x1_px * scale, y1_px * scale)


def detect_table_pages(
    pdf_path: str | Path,
    table_settings: Dict[str, str] = DEFAULT_TABLE_SETTINGS,
    use_visual_fallback: bool = True,
) -> List[int]:
    """
    Quick structural scan: which 0-based page indices physically contain >= 1 table.

    Uses `page.find_tables()` -- structure/bbox detection only, no cell text
    extraction -- so this is cheap to run on every page of every document
    before deciding whether the (expensive) VLM/Paddle table pipeline is even
    needed. This is the content-driven signal the orchestrator
    (`unified_pipeline.py`) uses instead of any filename/keyword heuristic.

    Args:
        pdf_path: Path to the PDF to scan.
        table_settings: Forwarded to pdfplumber's table detection.
        use_visual_fallback: If the vector-based scan finds ZERO tables on
            EVERY page (the telltale sign of a scanned/flattened document),
            also run the cheap pixel-based `page_looks_like_scanned_table`
            heuristic before giving up -- so scanned forms with real, visible
            table borders aren't silently skipped just because pdfplumber has
            no vector line objects to find.

    Returns:
        Sorted list of 0-based page indices that contain at least one
        detected table. Empty list if the document has no tables at all
        (or if it could not be opened/scanned -- callers should treat that
        the same as "no tables found").
    """
    pages_with_tables: List[int] = []
    with load_pdf(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for page_index, page in enumerate(pdf.pages):
            try:
                found = page.find_tables(table_settings=table_settings)
            except Exception as exc:  # pragma: no cover - defensive: malformed page content
                logger.warning("pdfplumber failed to scan page %d for tables: %s", page_index + 1, exc)
                continue
            if found:
                pages_with_tables.append(page_index)

    if not pages_with_tables and use_visual_fallback and total_pages > 0:
        logger.info(
            "No vector-based tables found anywhere -- falling back to pixel-based "
            "scan detection (likely a scanned/flattened document)."
        )
        for page_index in range(total_pages):
            try:
                if page_looks_like_scanned_table(pdf_path, page_index):
                    pages_with_tables.append(page_index)
            except Exception as exc:  # pragma: no cover - defensive: rendering failure
                logger.warning("Visual table scan failed on page %d: %s", page_index + 1, exc)

    logger.info(
        "Table scan: %d/%d page(s) contain a physical table: %s",
        len(pages_with_tables),
        total_pages,
        [p + 1 for p in pages_with_tables],
    )
    return pages_with_tables


def export_to_excel(
    dataframes: List[pd.DataFrame],
    output_path: str | Path,
    sheet_name_prefix: str = "Table",
) -> Path:
    """
    Write every detected table to its own sheet in one .xlsx workbook.

    If no tables were found at all, a small placeholder workbook is written
    instead of raising -- callers always get a valid output file to point users
    at, with a clear explanation of why it's (almost) empty.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not dataframes:
        pd.DataFrame({"info": ["No bordered tables were detected in this document."]}).to_excel(
            output_path, index=False, sheet_name="Info", engine="openpyxl"
        )
        logger.warning("No tables detected -- wrote placeholder workbook to %s", output_path)
        return output_path

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for i, df in enumerate(dataframes, start=1):
            # Excel hard-limits sheet names to 31 characters.
            sheet_name = f"{sheet_name_prefix}_{i}"[:31]
            df.to_excel(writer, index=False, sheet_name=sheet_name)

    logger.info("Exported %d table(s) to %s", len(dataframes), output_path)
    return output_path


def extract_pdf_tables_to_excel(
    pdf_path: str | Path,
    output_path: str | Path = "output_grid_tables.xlsx",
    table_settings: Dict[str, str] = DEFAULT_TABLE_SETTINGS,
    header_row: bool = True,
    apply_ocr_cleanup: bool = True,
) -> List[pd.DataFrame]:
    """
    Full pipeline: detect every bordered table -> DataFrame -> OCR cleanup -> Excel.

    This is the single entrypoint most callers (including the fallback chain
    in `unified_pipeline.py`) should use. `apply_ocr_cleanup=False` skips the
    OCR post-processing pass entirely; when True (default), the wrong->right
    mapping is loaded dynamically from `ocr_corrections.json` (see
    `ocr_cleanup.py`) -- no dictionary is hardcoded here.
    """
    with load_pdf(pdf_path) as pdf:
        raw_tables = extract_raw_tables(pdf, table_settings=table_settings)

    dataframes = tables_to_dataframes(raw_tables, header_row=header_row)

    if apply_ocr_cleanup:
        dataframes = [clean_ocr_errors(df) for df in dataframes]

    export_to_excel(dataframes, output_path)
    return dataframes


# ---------------------------------------------------------------------------
# CLI (standalone usage / debugging, independent of the router)
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot, border/grid-based table extraction (pdfplumber + pandas). "
            "Works on any PDF with visibly ruled table lines -- no template, no "
            "anchor text, no hardcoded coordinates required."
        )
    )
    parser.add_argument("--input", "-i", required=True, help="Path to the input PDF file.")
    parser.add_argument("--output", "-o", default="output_grid_tables.xlsx", help="Path to the output .xlsx file.")
    parser.add_argument(
        "--no-header-row",
        action="store_true",
        help="Treat every detected row as data (skip first-row-as-header detection).",
    )
    parser.add_argument("--no-ocr-cleanup", action="store_true", help="Skip the OCR post-processing replacement pass.")
    parser.add_argument(
        "--debug-tables",
        action="store_true",
        help="Print detected table shapes (page/index/rows/cols) instead of exporting.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    with load_pdf(args.input) as pdf:
        raw_tables = extract_raw_tables(pdf)

    if args.debug_tables:
        for t in raw_tables:
            cols = len(t.rows[0]) if t.rows else 0
            print(f"page={t.page + 1} table_index={t.index_on_page} rows={len(t.rows)} cols={cols}")
        if not raw_tables:
            print("No bordered tables detected.")
        return 0

    dataframes = tables_to_dataframes(raw_tables, header_row=not args.no_header_row)
    if not args.no_ocr_cleanup:
        dataframes = [clean_ocr_errors(df) for df in dataframes]

    export_to_excel(dataframes, args.output)
    print(f"Done. Exported {len(dataframes)} table(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
