"""
Docling TableFormer table extractor (replaces ``vlm_extractor.py``).

Uses IBM Docling offline for table structure recognition (TableFormer in
ACCURATE mode). Multi-level headers with colspan/rowspan are converted into a
pandas ``MultiIndex``. Results are written to ``output_tables.xlsx`` via
openpyxl.
"""

from __future__ import annotations

import logging
import pathlib
import warnings  # required by module contract; used for future deprecations
from typing import Any

import pandas as pd
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption

from .ocr_cleanup import clean_ocr_errors

logger = logging.getLogger(__name__)

EXPECTED_COL_COUNT = 9  # Form B06 has 9 columns


class DoclingExtractionError(RuntimeError):
    """Raised when Docling fails to process the PDF."""

    pass


def _build_pipeline_options() -> PdfPipelineOptions:
    """Build PdfPipelineOptions for offline TableFormer structure recognition."""
    options = PdfPipelineOptions()
    options.do_table_structure = True
    options.table_structure_options.mode = TableFormerMode.ACCURATE
    # OCR is handled separately by the existing pipeline; Docling is used
    # here ONLY for table structure recognition, not for text extraction.
    options.do_ocr = False
    return options


class DoclingTableExtractor:
    """Extract tables from a PDF using Docling TableFormer (offline)."""

    def __init__(self, pdf_path: str | pathlib.Path) -> None:
        """
        Initialize the extractor for a single PDF.

        Args:
            pdf_path: Path to the input PDF file.

        Raises:
            FileNotFoundError: If ``pdf_path`` does not exist.
        """
        self._pdf_path = pathlib.Path(pdf_path)
        if not self._pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {self._pdf_path}")

        pipeline_options = _build_pipeline_options()
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        self._raw_result: Any = None

    def extract(self) -> list[pd.DataFrame]:
        """
        Run Docling conversion on ``self._pdf_path``.

        Returns:
            A list of pandas DataFrames, one per table found in the document.
            Each DataFrame uses a pandas MultiIndex on columns when the table
            has multi-level (spanning) headers. If no tables are found,
            returns ``[]``.

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
        dataframes: list[pd.DataFrame] = []
        for table in tables:
            try:
                dataframes.append(self._table_to_dataframe(table))
            except Exception as exc:
                logger.warning(
                    "Skipping table that failed DataFrame conversion: %s",
                    exc,
                )
        return dataframes

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

        header_row_count = self._count_header_rows(grid)
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
