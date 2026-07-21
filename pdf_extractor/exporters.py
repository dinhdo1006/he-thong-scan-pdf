"""Export parsed tables and text to Excel / CSV / Markdown / plain-text files."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


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
