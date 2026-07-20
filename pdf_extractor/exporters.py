"""Export parsed tables and text to Excel / plain-text files."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class TableExporter:
    """Write one or more DataFrames into a single multi-sheet Excel workbook."""

    def save(
        self,
        tables: list[pd.DataFrame],
        output_path: str | Path,
    ) -> Path:
        """
        Save each DataFrame to a separate worksheet (Table_1, Table_2, ...).

        If `tables` is empty, creates a workbook with one empty sheet
        named 'No_Tables' so the output file always exists.

        Args:
            tables: List of pandas DataFrames.
            output_path: Destination .xlsx path.

        Returns:
            Resolved path of the written file.
        """
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                if not tables:
                    pd.DataFrame({"info": ["No tables found in document."]}).to_excel(
                        writer, sheet_name="No_Tables", index=False
                    )
                    logger.warning("No tables to export; wrote placeholder sheet.")
                else:
                    for idx, df in enumerate(tables, start=1):
                        sheet_name = f"Table_{idx}"
                        # Excel sheet names max 31 chars
                        sheet_name = sheet_name[:31]
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        logger.debug("Wrote sheet %s (%s)", sheet_name, df.shape)
        except Exception as exc:
            raise IOError(f"Failed to write Excel file '{path}': {exc}") from exc

        logger.info("Saved %d table(s) -> %s", len(tables), path)
        return path


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
