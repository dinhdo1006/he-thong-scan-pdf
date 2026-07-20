"""End-to-end offline PDF → tables (Excel) + text pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .exporters import TableExporter, TextExporter
from .marker_extractor import MarkerExtractor
from .markdown_parser import MarkdownBlockParser

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Paths produced by a successful pipeline run."""

    markdown_path: Path | None
    tables_path: Path
    text_path: Path
    table_count: int


class PDFPipeline:
    """
    Orchestrates:
      1. Marker PDF → Markdown (local, no LLM)
      2. Markdown → table blocks + text blocks
      3. Tables → output_tables.xlsx
      4. Text   → output_text.txt
    """

    def __init__(
        self,
        extractor: MarkerExtractor | None = None,
        parser: MarkdownBlockParser | None = None,
        table_exporter: TableExporter | None = None,
        text_exporter: TextExporter | None = None,
    ) -> None:
        self.extractor = extractor or MarkerExtractor()
        self.parser = parser or MarkdownBlockParser()
        self.table_exporter = table_exporter or TableExporter()
        self.text_exporter = text_exporter or TextExporter()

    def run(
        self,
        pdf_path: str | Path,
        output_dir: str | Path = ".",
        *,
        save_markdown: bool = True,
        tables_filename: str = "output_tables.xlsx",
        text_filename: str = "output_text.txt",
        markdown_filename: str = "output_markdown.md",
    ) -> PipelineResult:
        """
        Process one PDF and write outputs into `output_dir`.

        Args:
            pdf_path: Input PDF path.
            output_dir: Directory for output files (created if missing).
            save_markdown: Also write the intermediate Markdown file.
            tables_filename: Excel workbook name.
            text_filename: Plain text file name.
            markdown_filename: Intermediate Markdown file name.

        Returns:
            PipelineResult with output paths and table count.
        """
        out = Path(output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)

        # Step 1 — Extraction
        markdown = self.extractor.extract_markdown(pdf_path)

        markdown_path: Path | None = None
        if save_markdown:
            markdown_path = out / markdown_filename
            markdown_path.write_text(markdown, encoding="utf-8")
            logger.info("Saved intermediate Markdown -> %s", markdown_path)

        # Step 2 — Parsing
        parsed = self.parser.parse(markdown)

        # Step 3 — Table export
        tables_path = self.table_exporter.save(parsed.tables, out / tables_filename)

        # Step 4 — Text export
        text_path = self.text_exporter.save(parsed.text, out / text_filename)

        return PipelineResult(
            markdown_path=markdown_path,
            tables_path=tables_path,
            text_path=text_path,
            table_count=len(parsed.tables),
        )
