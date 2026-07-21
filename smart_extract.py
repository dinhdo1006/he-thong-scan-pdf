#!/usr/bin/env python3
"""
CLI entrypoint: Unified, content-driven PDF pipeline.

Every input PDF goes through the SAME sequence, regardless of template,
keyword, or filename:
    1. Marker            -> general text (output_text.txt / output_markdown.md)
    2. pdfplumber scan   -> which pages physically contain a table
    3. If (and only if) step 2 found tables:
         VLM (Qwen2-VL) -> PaddleOCR PP-Structure -> pdfplumber grid
       (first backend that produces usable tables wins)
       -> output_tables.xlsx (+ CSV / Markdown preview companions)
    4. If step 2 found nothing anywhere, table extraction (and the GPU) is
       skipped entirely.

No filename keyword, template name, or per-document configuration is used
anywhere in this decision path.

Usage:
    python smart_extract.py --input path/to/file.pdf --output-dir ./output
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pdf_extractor.unified_pipeline import UnifiedPDFPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Content-driven PDF pipeline: text always, tables only if physically present."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to the input PDF file.")
    parser.add_argument("--output-dir", "-o", default="./output", help="Directory for output files.")
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Do not save the intermediate Markdown file.",
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help=(
            "Fast test mode: run Marker + the cheap table-page scan only, "
            "skip VLM/PaddleOCR/grid extraction entirely (no GPU model load). "
            "Use this to quickly sanity-check the text pipeline/wiring."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    pdf_path = Path(args.input)
    if not pdf_path.is_file():
        logging.error("Input PDF not found: %s", pdf_path)
        return 1

    try:
        result = UnifiedPDFPipeline().run(
            pdf_path,
            output_dir=args.output_dir,
            save_markdown=not args.no_markdown,
            skip_tables=args.skip_tables,
        )
    except Exception as exc:
        logging.error("Pipeline failed: %s", exc)
        return 1

    print("Done.")
    print(f"  Text:                 {result.text_path}")
    if result.markdown_path:
        print(f"  Markdown:             {result.markdown_path}")

    if result.pages_with_tables:
        print(f"  Tables on page(s):    {[p + 1 for p in result.pages_with_tables]}")
        print(f"  Backend used:         {result.table_backend_used}")
        print(f"  Tables ({result.table_count}):           {result.tables_path}")
        preview_path = result.tables_path.with_name(result.tables_path.stem + "_preview.md")
        print(f"  Preview (VS Code):    {preview_path}")
        print("  Note: evaluate TABLE quality from the xlsx/preview above,")
        print("        NOT from output_markdown.md (that file is Marker prose).")
    else:
        print("  No tables detected -- table extraction skipped (GPU untouched).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
