#!/usr/bin/env python3
"""
CLI entrypoint: extract text + tables from a PDF (100% local / offline).

Dependencies (install once):
    pip install marker-pdf
    pip install pandas openpyxl

Usage:
    python extract_pdf.py --input path/to/file.pdf --output-dir ./output
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pdf_extractor import PDFPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract text and tables from a PDF using marker-pdf (local only). "
            "Tables -> output_tables.xlsx | Text -> output_text.txt"
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input PDF file.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./output",
        help="Directory for output files (default: ./output).",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Do not save the intermediate Markdown file.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
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
        result = PDFPipeline().run(
            pdf_path=pdf_path,
            output_dir=args.output_dir,
            save_markdown=not args.no_markdown,
        )
    except Exception as exc:
        logging.error("Pipeline failed: %s", exc)
        return 1

    print("Done.")
    print(f"  Tables ({result.table_count}): {result.tables_path}")
    print(f"  Text:                 {result.text_path}")
    if result.markdown_path:
        print(f"  Markdown:             {result.markdown_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
