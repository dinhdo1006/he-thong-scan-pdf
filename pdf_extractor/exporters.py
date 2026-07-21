"""Export parsed tables and text to plain-text (and optional debug formats)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from .markdown_parser import MarkdownBlockParser
from .ocr_cleanup import clean_ocr_errors, clean_text_ocr_errors

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_TXT = "output.txt"


def resolve_output_path(output: str | Path, default_name: str = DEFAULT_OUTPUT_TXT) -> Path:
    """
    Resolve CLI `-o` to a single `.txt` file path.

    - `result.txt`           -> that file
    - `./output` (directory) -> `./output/output.txt`
    """
    path = Path(output).expanduser()
    if path.suffix.lower() == ".txt":
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return (path / default_name).resolve()


def _markdown_text_to_plain(text: str) -> str:
    """Light cleanup: headings and inline HTML from Marker -> plain text."""
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("##"):
            stripped = re.sub(r"^#+\s*", "", stripped)
        stripped = stripped.replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ")
        lines.append(stripped)
    return "\n".join(lines).strip()


def _split_multi_value_cells(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-process a Marker-fallback DataFrame where one cell contains two
    formatted-number values separated by spaces (e.g. "10.620.000   10.370.000").

    For each column, if the column contains ANY cell that looks like
    "NUMBER  NUMBER" (2+ whitespace-separated number-like tokens), the
    column is replaced by two sub-columns named "col" and "col_2".
    Generic: no column-name or template assumption.
    """
    NUMBER_LIKE = re.compile(r"^[\d.,]+$")

    def _try_split(cell: str) -> list[str] | None:
        """Return [part1, part2, ...] if cell is 2+ number tokens, else None."""
        parts = cell.strip().split()
        if len(parts) >= 2 and all(NUMBER_LIKE.match(p) for p in parts):
            return parts
        return None

    rows = len(df)
    new_cols: list[str] = []
    column_data: list[list[str]] = []

    # Iterate by position to avoid duplicate-column issues with df[name]
    for col_pos, col in enumerate(df.columns):
        raw: list[str] = [str(v) for v in df.iloc[:, col_pos]]
        splits: list[list[str] | None] = [_try_split(v) for v in raw]
        max_parts = max((len(s) for s in splits if s is not None), default=1)

        if max_parts >= 2:
            for part_idx in range(max_parts):
                if part_idx == 0:
                    sub = col
                else:
                    # Ensure the generated name is unique across ALL columns
                    candidate = f"{col}_{part_idx + 1}"
                    suffix = part_idx + 1
                    while candidate in new_cols:
                        suffix += 1
                        candidate = f"{col}_{suffix}"
                    sub = candidate
                new_cols.append(sub)
                col_vals: list[str] = []
                for i in range(rows):
                    s = splits[i]
                    if s is not None and part_idx < len(s):
                        col_vals.append(s[part_idx])
                    elif s is None and part_idx == 0:
                        col_vals.append(raw[i])
                    else:
                        col_vals.append("")
                column_data.append(col_vals)
        else:
            new_cols.append(col)
            column_data.append(raw)

    # Build using a list-of-lists to avoid duplicate-column confusion
    result = pd.DataFrame(dict(enumerate(column_data)))
    result.columns = pd.Index(new_cols)
    return result


def dataframe_to_plaintext_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a tab-separated plain-text grid.

    Uses positional (iloc) access to safely handle DataFrames with
    duplicate column names.
    """
    if df.empty or len(df.columns) == 0:
        return ""

    def _clean(s: str) -> str:
        return s.replace("\n", " ").replace("<br>", " ").replace("<br/>", " ").replace("\t", " ")

    columns = [_clean(str(c)) for c in df.columns]
    lines = ["\t".join(columns)]
    for row_idx in range(len(df)):
        cells = [_clean(str(df.iloc[row_idx, col_idx])) for col_idx in range(len(df.columns))]
        lines.append("\t".join(cells))
    return "\n".join(lines)


def compose_document_txt(
    markdown: str,
    extracted_tables: list[pd.DataFrame],
    *,
    apply_ocr_cleanup: bool = True,
) -> str:
    """
    Build ONE plain-text document from Marker Markdown + optional extracted tables.

    Walks Markdown blocks in reading order. For each table block:
      - use the next extracted DataFrame when the VLM/grid backend succeeded;
      - otherwise keep Marker's table for that block (so content is never lost).

  Prose blocks are lightly cleaned (headings, <br>) and run through
  `ocr_corrections.json`.
    """
    blocks = MarkdownBlockParser().parse_blocks(markdown)
    extracted_queue = list(extracted_tables)
    parts: list[str] = []

    for block in blocks:
        if block.kind == "text":
            plain = _markdown_text_to_plain(str(block.content))
            if apply_ocr_cleanup:
                plain = clean_text_ocr_errors(plain)
            if plain.strip():
                parts.append(plain)
            continue

        df = extracted_queue.pop(0) if extracted_queue else block.content
        assert isinstance(df, pd.DataFrame)
        df = _split_multi_value_cells(df)
        if apply_ocr_cleanup:
            df = clean_ocr_errors(df)
        table_txt = dataframe_to_plaintext_table(df)
        if table_txt.strip():
            parts.append(table_txt)

    # Rare: backend returned MORE tables than Marker blocks (e.g. stitched pages).
    while extracted_queue:
        df = extracted_queue.pop(0)
        df = _split_multi_value_cells(df)
        if apply_ocr_cleanup:
            df = clean_ocr_errors(df)
        table_txt = dataframe_to_plaintext_table(df)
        if table_txt.strip():
            parts.append(table_txt)

    body = "\n\n".join(parts)
    return body.rstrip() + ("\n" if body.strip() else "")


def save_unified_txt(content: str, output_path: str | Path) -> Path:
    """Write the final unified document to a UTF-8 `.txt` file."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")
    except Exception as exc:
        raise IOError(f"Failed to write text file '{path}': {exc}") from exc
    logger.info("Saved unified text (%d chars) -> %s", len(content), path)
    return path


def export_tables_preview(
    tables: list[pd.DataFrame],
    output_xlsx: str | Path,
    *,
    write_csv: bool = True,
    write_markdown: bool = True,
) -> dict[str, Path]:
    """
    Write tables for both Excel and VS Code-friendly previews.

    VS Code does not render .xlsx well; companions:
      - `output_tables_preview.md`  — Markdown tables (best preview in VS Code)
      - `output_table_1.csv`, ...   — plain CSV (open as text / Excel)

    Returns paths of files that were written (keys: xlsx, markdown, csv_dir).
    """
    xlsx_path = Path(output_xlsx).expanduser().resolve()
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    # --- Excel (open with LibreOffice / Excel, not VS Code) ---
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            if not tables:
                pd.DataFrame({"info": ["No tables found in document."]}).to_excel(
                    writer, sheet_name="No_Tables", index=False
                )
            else:
                for idx, df in enumerate(tables, start=1):
                    df.to_excel(writer, sheet_name=f"Table_{idx}"[:31], index=False)
    except Exception as exc:
        raise IOError(f"Failed to write Excel file '{xlsx_path}': {exc}") from exc
    written["xlsx"] = xlsx_path
    logger.info("Saved %d table(s) -> %s", len(tables), xlsx_path)

    stem = xlsx_path.stem  # e.g. output_tables
    out_dir = xlsx_path.parent

    # --- CSV (one file per table; UTF-8 with BOM for Excel on Windows) ---
    if write_csv:
        if not tables:
            csv_path = out_dir / f"{stem}_empty.csv"
            pd.DataFrame({"info": ["No tables found in document."]}).to_csv(
                csv_path, index=False, encoding="utf-8-sig"
            )
            written["csv"] = csv_path
        else:
            csv_paths: list[Path] = []
            for idx, df in enumerate(tables, start=1):
                csv_path = out_dir / f"{stem}_table_{idx}.csv"
                df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                csv_paths.append(csv_path)
            written["csv"] = csv_paths[0]
            logger.info("Saved %d CSV file(s) under %s", len(csv_paths), out_dir)

    # --- Markdown preview (easiest to read inside VS Code) ---
    if write_markdown:
        md_path = out_dir / f"{stem}_preview.md"
        if not tables:
            md_body = "# Tables preview\n\n_No tables found in document._\n"
        else:
            parts = ["# Tables preview\n"]
            for idx, df in enumerate(tables, start=1):
                parts.append(f"\n## Table {idx}\n\n")
                parts.append(_dataframe_to_markdown(df))
                parts.append("\n")
            md_body = "".join(parts)
        md_path.write_text(md_body, encoding="utf-8")
        written["markdown"] = md_path
        logger.info("Saved Markdown preview -> %s", md_path)

    return written


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GFM pipe table (no `tabulate` dependency)."""
    columns = [str(c) for c in df.columns]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    rows: list[str] = []
    for _, row in df.iterrows():
        cells = [str(row[c]).replace("\n", " ").replace("|", "\\|") for c in df.columns]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows]) if columns else "_empty table_"


class TableExporter:
    """Write one or more DataFrames into Excel + VS Code-friendly companions."""

    def save(
        self,
        tables: list[pd.DataFrame],
        output_path: str | Path,
    ) -> Path:
        """
        Save each DataFrame to Excel, plus CSV and Markdown previews.

        Returns:
            Path of the written .xlsx file (companions sit next to it).
        """
        written = export_tables_preview(tables, output_path)
        return written["xlsx"]


def export_document(
    text: str,
    tables: list[pd.DataFrame],
    output_path: str | Path,
) -> Path:
    """
    Assemble one readable Word (.docx) file from ALREADY-STRUCTURED data:
    plain prose paragraphs + real Word tables (one `python-docx` table per
    DataFrame, one cell per DataFrame cell).

    Deliberately NOT built by converting Marker's raw Markdown -- Marker's
    own table guesses are unreliable on scanned/complex forms (merged cells,
    OCR typos), so converting that Markdown to Word would just bake the same
    mistakes into a different file format. This function only ever consumes:
      - `text`: prose with tables already stripped out (see
        `MarkdownBlockParser` / `unified_pipeline._extract_text`)
      - `tables`: the DataFrames actually produced by the table-extraction
        backend (VLM / PaddleOCR / pdfplumber grid), which is what carries a
        real, verified column/row structure.

    No layout reconstruction (merged header cells, stamp/signature
    positioning, etc.) is attempted -- this produces a plain, readable
    document: paragraphs, then each table as a normal Word table, in order.
    """
    from docx import Document

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = Document()

    for paragraph in text.split("\n\n"):
        stripped = paragraph.strip()
        if not stripped:
            continue
        for line in stripped.splitlines():
            if line.strip():
                doc.add_paragraph(line.strip())
        doc.add_paragraph("")

    if not tables:
        doc.add_paragraph("(Không phát hiện bảng nào trong tài liệu này.)")
    else:
        for idx, df in enumerate(tables, start=1):
            doc.add_heading(f"Table {idx}", level=2)
            n_rows, n_cols = len(df), len(df.columns)
            if n_cols == 0:
                continue
            table = doc.add_table(rows=n_rows + 1, cols=n_cols)
            table.style = "Table Grid"

            header_cells = table.rows[0].cells
            for col_idx, col_name in enumerate(df.columns):
                header_cells[col_idx].text = str(col_name)
                for run in header_cells[col_idx].paragraphs[0].runs:
                    run.bold = True

            for row_idx, row in enumerate(df.itertuples(index=False), start=1):
                row_cells = table.rows[row_idx].cells
                for col_idx, value in enumerate(row):
                    row_cells[col_idx].text = "" if value is None else str(value)

            doc.add_paragraph("")

    try:
        doc.save(str(output_path))
    except Exception as exc:
        raise IOError(f"Failed to write Word document '{output_path}': {exc}") from exc

    logger.info("Saved Word document (%d table(s)) -> %s", len(tables), output_path)
    return output_path


class DocumentExporter:
    """Assemble prose + extracted tables into one readable `.docx` file."""

    def save(self, text: str, tables: list[pd.DataFrame], output_path: str | Path) -> Path:
        return export_document(text, tables, output_path)


class TextExporter:
    """Write non-table Markdown text to a UTF-8 plain text file."""

    def save(self, text: str, output_path: str | Path) -> Path:
        """
        Persist extracted prose, preserving paragraph spacing.

        Args:
            text: Non-table text content.
            output_path: Destination .txt path.

        Returns:
            Resolved path of the written file.
        """
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        except Exception as exc:
            raise IOError(f"Failed to write text file '{path}': {exc}") from exc

        logger.info("Saved text (%d chars) -> %s", len(text), path)
        return path
