"""
Docling TableFormer table extractor (replaces ``vlm_extractor.py``).

Uses IBM Docling offline for table structure recognition (TableFormer in
ACCURATE mode) with native OCR enabled (EasyOCR, falling back to Tesseract)
using explicit Vietnamese language codes. Running OCR gives TableFormer real
text anchors instead of guessing column geometry blindly, and preserves
diacritics (dấu thanh) that would otherwise be lost.

Multi-level headers with colspan/rowspan are converted into a pandas
``MultiIndex`` and then flattened + validated against the expected physical
column count via ``flatten_docling_dataframe`` -- tables that don't match are
dropped so the pipeline's Tier 2/3 cascade (PaddleOCR / pdfplumber) can take
over. Results are written to ``output_tables.xlsx`` via openpyxl.
"""

from __future__ import annotations

import logging
import pathlib
import warnings  # required by module contract; used for future deprecations
from typing import Any

import pandas as pd
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrOptions,
    PdfPipelineOptions,
    TableFormerMode,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from .ocr_cleanup import clean_ocr_errors

logger = logging.getLogger(__name__)

# Optional hints only -- the extractor is form-agnostic. Callers that know a
# specific template (e.g. B06-THA = 9 cols / 2 header rows) may still pass
# these explicitly; they are NOT defaults that reject other tables.
EXPECTED_COL_COUNT = 9
EXPECTED_HEADER_ROWS = 2

# Vietnamese language codes for the two OCR engine families Docling supports.
# EasyOCR uses ISO 639-1 ("vi"); Tesseract uses the 3-letter ISO 639-2 code
# ("vie") plus its language pack must be installed separately.
DEFAULT_EASYOCR_LANG = ["vi"]
DEFAULT_TESSERACT_LANG = ["vie"]


class DoclingExtractionError(RuntimeError):
    """Raised when Docling fails to process the PDF."""

    pass


def _select_ocr_options(easyocr_lang: list[str] | None = None) -> OcrOptions | None:
    """
    Pick the best available OCR engine for scanned Vietnamese PDFs.

    Default path always prefers EasyOCR ``lang=["vi"]``. If EasyOCR cannot
    be imported or configured (missing package / model load error), fall
    back to Tesseract ``lang=["vie"]``. Returns ``None`` only when both
    families fail -- caller must then set ``do_ocr=False`` and warn loudly.

    Returns:
        A configured ``OcrOptions`` instance, or ``None`` if no OCR engine
        is available.
    """
    easyocr_lang = easyocr_lang or DEFAULT_EASYOCR_LANG

    # 1) EasyOCR first (bundled language models, no external binary).
    try:
        import easyocr  # noqa: F401

        options = EasyOcrOptions(lang=easyocr_lang, force_full_page_ocr=False)
        logger.info("OCR engine: EasyOCR (lang=%s).", easyocr_lang)
        return options
    except Exception as exc:
        logger.info(
            "EasyOCR unavailable (%s) -- trying Tesseract for Vietnamese OCR.",
            exc,
        )

    # 2) Tesseract fallback (tesserocr, then CLI wrapper).
    try:
        import tesserocr  # noqa: F401

        options = TesseractOcrOptions(lang=DEFAULT_TESSERACT_LANG)
        logger.info("OCR engine: Tesseract (tesserocr, lang=%s).", DEFAULT_TESSERACT_LANG)
        return options
    except Exception as exc:
        logger.debug("tesserocr unavailable (%s).", exc)

    try:
        import pytesseract  # noqa: F401

        options = TesseractCliOcrOptions(lang=DEFAULT_TESSERACT_LANG)
        logger.info("OCR engine: Tesseract (CLI, lang=%s).", DEFAULT_TESSERACT_LANG)
        return options
    except Exception as exc:
        logger.debug("pytesseract unavailable (%s).", exc)

    logger.error(
        "No OCR engine found (tried EasyOCR lang=%s, Tesseract lang=%s). "
        "Install with: pip install easyocr  (or Tesseract-OCR + the 'vie' language pack).",
        easyocr_lang,
        DEFAULT_TESSERACT_LANG,
    )
    return None


def _warn_docling_without_ocr(pdf_path: str | pathlib.Path | None) -> None:
    """Emit the required console warning when Docling runs without OCR anchors."""
    pdf_label = str(pdf_path) if pdf_path is not None else "(unknown PDF)"
    message = (
        "[WARN] Docling running without OCR anchor — column collapse "
        f"risk high | pdf={pdf_label}"
    )
    # logger.warning reaches console via configure_app_logging; print keeps
    # the exact token visible even if a caller muted the package logger.
    logger.warning(message)
    print(message, flush=True)


def _build_pipeline_options(
    *,
    ocr_lang: list[str] | None = None,
    disable_ocr: bool = False,
    pdf_path: str | pathlib.Path | None = None,
) -> PdfPipelineOptions:
    """
    Build PdfPipelineOptions with native OCR + Vietnamese text anchors enabled.

    ``do_table_structure=True`` keeps TableFormer in ACCURATE mode.
    By default ``do_ocr=True`` (EasyOCR ``vi``, else Tesseract ``vie``).
    ``disable_ocr=True`` forces ``do_ocr=False`` for debug (``--docling-no-ocr``).
    Only when both OCR engines fail (and OCR is not explicitly disabled) do
    we allow ``do_ocr=False``, with a loud column-collapse warning.
    """
    options = PdfPipelineOptions()
    options.do_table_structure = True
    options.table_structure_options.mode = TableFormerMode.ACCURATE

    if disable_ocr:
        options.do_ocr = False
        logger.info(
            "Docling OCR disabled by flag (--docling-no-ocr / disable_ocr=True) for %s.",
            pdf_path or "(unknown PDF)",
        )
        return options

    ocr_options = _select_ocr_options(ocr_lang)
    if ocr_options is not None:
        options.do_ocr = True
        options.ocr_options = ocr_options
    else:
        # Last resort only -- Tier 2/3 fallbacks still apply, but column
        # collapse risk is high on scanned forms without text anchors.
        options.do_ocr = False
        _warn_docling_without_ocr(pdf_path)
    return options


def _flatten_header_tuple(levels: tuple[Any, ...]) -> str:
    """Join one MultiIndex column's levels into a single string, in order."""
    parts: list[str] = []
    for level in levels:
        text = "" if level is None else str(level).strip()
        if not text or text.lower().startswith("unnamed:"):
            continue
        if parts and parts[-1] == text:
            # Same label repeated top-to-bottom (colspan/rowspan echo) —
            # keep it once instead of "Group_Group".
            continue
        parts.append(text)
    return "_".join(parts) if parts else "col"


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return str(value).strip() == ""


def _cells_equal(a: Any, b: Any) -> bool:
    if _is_blank(a) or _is_blank(b):
        return False
    return str(a).strip() == str(b).strip()


def _degap_spanned_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove colspan-echo duplicates from body rows without changing shape.

    Docling (and pandas' own MultiIndex expansion) repeats a spanning
    cell's text into every physical column it covers. Keep that text only
    in the left-most column of each run of identical, non-empty, adjacent
    values and blank out the rest with ``NaN`` — this is what keeps the
    *physical* column count intact instead of letting the span visually
    "collapse" the grid.
    """
    n_cols = len(df.columns)
    if n_cols < 2 or df.empty:
        return df

    out = df.copy()
    for row_pos in range(len(out)):
        run_start = 0
        for col_pos in range(1, n_cols + 1):
            still_running = (
                col_pos < n_cols
                and _cells_equal(
                    out.iat[row_pos, col_pos], out.iat[row_pos, run_start]
                )
            )
            if still_running:
                continue
            if col_pos - run_start > 1:
                for dup_pos in range(run_start + 1, col_pos):
                    out.iat[row_pos, dup_pos] = None
            run_start = col_pos
    return out


def flatten_docling_dataframe(
    df: pd.DataFrame,
    expected_cols: int | None = None,
    *,
    strict_cols: bool = False,
) -> pd.DataFrame:
    """
    Flatten a Docling table DataFrame into a single-level physical grid.

    Docling represents multi-row headers with colspan/rowspan as a pandas
    ``MultiIndex``. This function:

    1. **Header flatten** — join MultiIndex levels into one string per column
       without changing the physical column count.
    2. **Body de-span** — keep spanning text only in the left-most cell of
       each run; blank the echoed copies so alignment is preserved.

    Args:
        df: Docling export (or any DataFrame with optional MultiIndex columns).
        expected_cols: Optional width hint. When set and ``strict_cols`` is
            True, a mismatch raises ``ValueError``. When ``strict_cols`` is
            False (default), only a warning is logged — so tables of any
            width remain usable (generic extraction, not form-locked).
        strict_cols: If True and ``expected_cols`` is set, enforce exact width.

    Returns:
        A DataFrame with single-level string columns (any width).

    Raises:
        ValueError: If ``df`` is None, or if ``strict_cols`` and width mismatch.
    """
    if df is None:
        raise ValueError("flatten_docling_dataframe() received None, expected a DataFrame")

    flat = df.copy()

    if isinstance(flat.columns, pd.MultiIndex):
        flat.columns = [_flatten_header_tuple(t) for t in flat.columns.tolist()]
    else:
        flat.columns = [str(c) for c in flat.columns]

    flat = _degap_spanned_rows(flat)

    n_cols = len(flat.columns)
    if expected_cols is not None and n_cols != expected_cols:
        msg = (
            f"Flattened Docling table has {n_cols} column(s), expected "
            f"{expected_cols}."
        )
        if strict_cols:
            raise ValueError(
                msg + " Physical grid is likely misaligned by an unresolved "
                "colspan/rowspan."
            )
        logger.warning("%s Keeping the table (generic mode; not form-locked).", msg)

    return flat


class DoclingTableExtractor:
    """Extract tables from a PDF using Docling TableFormer (offline)."""

    def __init__(
        self,
        pdf_path: str | pathlib.Path,
        *,
        ocr_lang: list[str] | None = None,
        expected_cols: int | None = None,
        header_row_count: int | None = None,
        strict_cols: bool = False,
        disable_ocr: bool = False,
    ) -> None:
        """
        Initialize the extractor for a single PDF.

        Args:
            pdf_path: Path to the input PDF file.
            ocr_lang: EasyOCR language codes to use (defaults to
                ``["vi"]``). Ignored if EasyOCR is unavailable and Docling
                falls back to Tesseract, which always uses ``["vie"]``.
            expected_cols: Optional width hint for logging / strict checks.
                Default ``None`` accepts any column count (generic tables).
            header_row_count: Optional fixed header depth. Default ``None``
                uses TableFormer's ``column_header`` auto-detect so forms
                with 1 or N header rows still work. Pass an int only when
                a known template needs an override.
            strict_cols: When True with ``expected_cols``, drop tables whose
                flattened width mismatches (legacy form-locked behavior).
            disable_ocr: When True, force ``do_ocr=False`` (CLI
                ``--docling-no-ocr``) for debug comparisons without OCR
                anchors. Default False — always try EasyOCR then Tesseract.

        Raises:
            FileNotFoundError: If ``pdf_path`` does not exist.
        """
        self._pdf_path = pathlib.Path(pdf_path)
        if not self._pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {self._pdf_path}")

        self._expected_cols = expected_cols
        self._strict_cols = strict_cols
        self._header_row_count_override = header_row_count
        self._disable_ocr = bool(disable_ocr)
        pipeline_options = _build_pipeline_options(
            ocr_lang=ocr_lang,
            disable_ocr=self._disable_ocr,
            pdf_path=self._pdf_path,
        )
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        self._raw_result: Any = None

    @staticmethod
    def _table_page_no(table: Any) -> int | None:
        """
        Return the 0-indexed page number a Docling ``TableItem`` was found on.

        Uses ``table.prov[0].page_no`` (Docling provenance is 1-indexed).
        Returns ``None`` if unavailable so callers can degrade gracefully
        instead of mis-mapping a table to the wrong page.
        """
        prov = getattr(table, "prov", None) or []
        if not prov:
            return None
        page_no = getattr(prov[0], "page_no", None)
        if page_no is None:
            return None
        try:
            return int(page_no) - 1
        except (TypeError, ValueError):
            return None

    def extract_with_pages(self) -> list[tuple[int | None, pd.DataFrame]]:
        """
        Run Docling conversion and return each table tagged with its page.

        Returns:
            A list of ``(page_no, dataframe)`` tuples, one per table that
            converted successfully. ``page_no`` is 0-indexed. Tables that
            fail hard conversion are skipped; width mismatches only drop
            when ``strict_cols=True``.

        Raises:
            DoclingExtractionError: On hard conversion failures.
        """
        try:
            self._raw_result = self._converter.convert(str(self._pdf_path))
        except Exception as exc:
            raise DoclingExtractionError(
                f"Docling failed to convert {self._pdf_path}: {exc}"
            ) from exc

        document = self._raw_result.document
        tables = getattr(document, "tables", None) or []
        results: list[tuple[int | None, pd.DataFrame]] = []
        strict = bool(getattr(self, "_strict_cols", False))
        expected = getattr(self, "_expected_cols", None)
        for table in tables:
            page_no = self._table_page_no(table)
            try:
                raw_df = self._table_to_dataframe(table)
                flat_df = flatten_docling_dataframe(
                    raw_df, expected_cols=expected, strict_cols=strict
                )
                results.append((page_no, flat_df))
            except ValueError as exc:
                logger.warning(
                    "Skipping table on page %s (%s) -- cascade may fill from PaddleOCR.",
                    page_no + 1 if page_no is not None else "?",
                    exc,
                )
            except Exception as exc:
                logger.warning(
                    "Skipping table on page %s that failed DataFrame conversion: %s",
                    page_no + 1 if page_no is not None else "?",
                    exc,
                )
        return results

    def extract(self) -> list[pd.DataFrame]:
        """
        Run Docling conversion on ``self._pdf_path``.

        Returns:
            A list of pandas DataFrames, one per table found in the document.
            If no tables are found, returns ``[]``. See
            ``extract_with_pages()`` for per-page provenance.

        Raises:
            DoclingExtractionError: On hard conversion failures.
        """
        return [df for _, df in self.extract_with_pages()]

    def _resolve_header_row_count(self, grid: list[list[Any]]) -> int:
        """
        Decide how many leading grid rows are the column header.

        Uses ``self._header_row_count_override`` when set (clamped so at
        least one data row always remains), otherwise falls back to
        TableFormer's own ``column_header`` classification.
        """
        # getattr guards against extractors built via `__new__` in tests,
        # which bypass `__init__` and never set this attribute.
        override = getattr(self, "_header_row_count_override", None)
        if override is None:
            return self._count_header_rows(grid)
        n_rows = len(grid)
        return max(0, min(int(override), max(n_rows - 1, 0)))

    def _count_header_rows(self, grid: list[list[Any]]) -> int:
        """Count leading rows where all non-empty cells are column headers."""
        header_row_count = 0
        for row in grid:
            non_empty = [cell for cell in row if (getattr(cell, "text", None) or "").strip()]
            if not non_empty:
                # Empty row in the middle of headers — stop.
                break
            if all(bool(getattr(cell, "column_header", False)) for cell in non_empty):
                header_row_count += 1
            else:
                break
        return header_row_count

    @staticmethod
    def _fill_span_matrix(
        grid: list[list[Any]],
        row_start: int,
        row_end: int,
        n_cols: int,
    ) -> list[list[str]]:
        """
        Build a 2D string matrix for rows ``[row_start, row_end)``, expanding
        rowspan/colspan from each unique anchor cell.
        """
        n_rows = max(0, row_end - row_start)
        matrix: list[list[str]] = [["" for _ in range(n_cols)] for _ in range(n_rows)]
        seen: set[tuple[int, int]] = set()

        for abs_row in range(row_start, min(row_end, len(grid))):
            for cell in grid[abs_row]:
                start_r = int(getattr(cell, "start_row_offset_idx", abs_row))
                start_c = int(getattr(cell, "start_col_offset_idx", 0))
                key = (start_r, start_c)
                if key in seen:
                    continue
                seen.add(key)

                text = str(getattr(cell, "text", "") or "")
                row_span = max(1, int(getattr(cell, "row_span", 1) or 1))
                col_span = max(1, int(getattr(cell, "col_span", 1) or 1))

                for dr in range(row_span):
                    for dc in range(col_span):
                        rr = start_r + dr - row_start
                        cc = start_c + dc
                        if 0 <= rr < n_rows and 0 <= cc < n_cols:
                            matrix[rr][cc] = text
        return matrix

    def _table_to_dataframe(self, table: Any) -> pd.DataFrame:
        """
        Convert a single Docling TableItem to a pandas DataFrame.

        Correctly handles colspan and rowspan to produce a MultiIndex when
        multiple header rows are present.
        """
        data = getattr(table, "data", None)
        if data is None:
            return pd.DataFrame()

        grid = getattr(data, "grid", None) or []
        if not grid:
            return pd.DataFrame()

        n_cols = int(getattr(data, "num_cols", 0) or len(grid[0]))
        if n_cols <= 0:
            return pd.DataFrame()

        header_row_count = self._resolve_header_row_count(grid)
        n_rows = len(grid)

        if header_row_count > 0:
            header_matrix = self._fill_span_matrix(grid, 0, header_row_count, n_cols)
            if header_row_count == 1:
                columns: Any = header_matrix[0]
            else:
                tuples = list(zip(*header_matrix))
                columns = pd.MultiIndex.from_tuples(tuples)
        else:
            columns = [f"col_{i}" for i in range(n_cols)]

        data_rows = self._fill_span_matrix(grid, header_row_count, n_rows, n_cols)
        return pd.DataFrame(data_rows, columns=columns)

    def save_to_excel(
        self,
        dataframes: list[pd.DataFrame],
        output_path: str | pathlib.Path = "output_tables.xlsx",
    ) -> pathlib.Path:
        """
        Write all extracted DataFrames to a single Excel workbook.

        Each DataFrame goes to a separate sheet named Table_1, Table_2, etc.

        Args:
            dataframes: Tables to write.
            output_path: Destination workbook path.

        Returns:
            The resolved output path.
        """
        import openpyxl  # noqa: F401 — keep openpyxl off the module import surface

        return _save_dataframes_to_excel(dataframes, output_path)


def _parse_marker_markdown_tables(markdown: str) -> list[pd.DataFrame]:
    """Parse pipe-delimited markdown tables from Marker output."""
    if not markdown or not str(markdown).strip():
        return []

    lines = str(markdown).splitlines()
    tables: list[pd.DataFrame] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("|") and "|" in line[1:]:
            block: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            if len(block) < 2:
                continue
            # Skip markdown separator row (|---|---|)
            body_lines = [block[0]]
            for row in block[1:]:
                cells = [c.strip() for c in row.strip("|").split("|")]
                if cells and all(set(c) <= {"-", ":", " "} for c in cells):
                    continue
                body_lines.append(row)
            try:
                from io import StringIO

                text = "\n".join(body_lines)
                df = pd.read_csv(
                    StringIO(text),
                    sep="|",
                    engine="python",
                    skipinitialspace=True,
                )
                # Drop empty edge columns produced by leading/trailing pipes
                df = df.dropna(axis=1, how="all")
                df.columns = [str(c).strip() for c in df.columns]
                for col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].map(
                            lambda v: v.strip() if isinstance(v, str) else v
                        )
                if not df.empty:
                    tables.append(df)
            except Exception as exc:
                logger.warning("Failed to parse Marker markdown table: %s", exc)
            continue
        i += 1
    return tables


def _apply_ocr_cleanup(dataframes: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Apply ``clean_ocr_errors`` to each non-empty DataFrame."""
    return [
        clean_ocr_errors(df) if not df.empty else df
        for df in dataframes
    ]


def _log_dataframe_summary(dataframes: list[pd.DataFrame]) -> None:
    """Log shape/column info and warn on unexpected column counts."""
    for i, df in enumerate(dataframes):
        assert df is not None
        sheet_name = f"Table_{i + 1}"
        col_names: Any = list(df.columns)
        logger.info(
            "Table %d -> sheet %s: %d rows, %d columns, columns=%s",
            i + 1,
            sheet_name,
            len(df),
            len(df.columns),
            col_names,
        )
        if len(df.columns) != EXPECTED_COL_COUNT and not df.empty:
            logger.warning(
                "Table %d has %d columns, expected %d. "
                "Check for colspan detection errors.",
                i + 1,
                len(df.columns),
                EXPECTED_COL_COUNT,
            )


def extract_tables_from_pdf(
    pdf_path: str | pathlib.Path,
    output_path: str | pathlib.Path = "output_tables.xlsx",
    apply_ocr_cleanup: bool = True,
) -> list[pd.DataFrame]:
    """
    High-level orchestration: Docling extract -> OCR cleanup -> Excel save.

    Falls back to Marker markdown table parsing when Docling fails or finds
    no tables. Returns the final list of cleaned DataFrames.

    Args:
        pdf_path: Path to the input PDF.
        output_path: Destination Excel workbook path.
        apply_ocr_cleanup: When True, run ``clean_ocr_errors`` on each
            non-empty DataFrame before saving.

    Returns:
        List of extracted (and optionally cleaned) DataFrames.
    """
    dataframes: list[pd.DataFrame] = []
    extractor: DoclingTableExtractor | None = None

    try:
        # --- TIER 1: Docling happy path ---
        try:
            extractor = DoclingTableExtractor(pdf_path)
            dataframes = extractor.extract()
            if not dataframes:
                logger.warning("Docling returned 0 tables. Activating fallback.")
                raise DoclingExtractionError("No tables detected")
        except DoclingExtractionError as exc:
            # --- TIER 2: Marker fallback ---
            logger.error(
                "Docling extraction failed: %s. Attempting Marker fallback.",
                exc,
            )
            dataframes = _marker_fallback(pdf_path)
        except Exception as exc:
            logger.error(
                "Docling extraction failed: %s. Attempting Marker fallback.",
                exc,
            )
            dataframes = _marker_fallback(pdf_path)

        if apply_ocr_cleanup:
            dataframes = _apply_ocr_cleanup(dataframes)

        _log_dataframe_summary(dataframes)

        if extractor is not None:
            extractor.save_to_excel(dataframes, output_path=output_path)
        else:
            DoclingTableExtractor.save_to_excel(
                object.__new__(DoclingTableExtractor),
                dataframes,
                output_path=output_path,
            )
        return dataframes
    finally:
        # --- TIER 3: always ---
        logger.debug("extract_tables_from_pdf completed for: %s", pdf_path)


def _marker_fallback(pdf_path: str | pathlib.Path) -> list[pd.DataFrame]:
    """Attempt Marker markdown table extraction; return empty DF if unavailable."""
    try:
        from marker.convert import convert_single_pdf  # type: ignore
    except ImportError:
        logger.error("Marker not installed. Returning empty DataFrame.")
        return [pd.DataFrame()]

    try:
        markdown = convert_single_pdf(str(pdf_path))
        if isinstance(markdown, tuple):
            markdown = markdown[0]
        tables = _parse_marker_markdown_tables(str(markdown))
        if tables:
            logger.warning(
                "Used Marker fallback — colspan metadata will be lost"
            )
            return tables
        logger.warning(
            "Marker fallback produced 0 tables for %s. Returning empty DataFrame.",
            pdf_path,
        )
        return [pd.DataFrame()]
    except Exception as exc:
        logger.error("Marker fallback failed for %s: %s", pdf_path, exc)
        return [pd.DataFrame()]


def _save_dataframes_to_excel(
    dataframes: list[pd.DataFrame],
    output_path: str | pathlib.Path,
) -> pathlib.Path:
    """Write DataFrames to Excel without requiring a live Docling converter."""
    output_path = pathlib.Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        if not dataframes:
            pd.DataFrame().to_excel(writer, sheet_name="Table_1", index=False)
        else:
            for i, df in enumerate(dataframes):
                df.to_excel(writer, sheet_name=f"Table_{i + 1}", index=False)
    logger.info("Wrote Docling tables to %s", output_path)
    return output_path
