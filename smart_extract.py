#!/usr/bin/env python3
"""
CLI entrypoint: Unified PDF -> single plain-text .txt output.

Usage:
    python smart_extract.py -i path/to/file.pdf -o result.txt
    python smart_extract.py -i path/to/file.pdf -o ./output   # writes ./output/output.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pdf_extractor.exporters import DEFAULT_OUTPUT_TXT
from pdf_extractor.unified_pipeline import UnifiedPDFPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract PDF to a single plain-text .txt file (prose + tables)."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to the input PDF file.")
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT_TXT,
        help="Output .txt path, or a directory (writes output.txt inside).",
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Fast test: Marker only, skip VLM/Paddle/grid (no GPU model load).",
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
            output=args.output,
            skip_tables=args.skip_tables,
        )
    except Exception as exc:
        logging.error("Pipeline failed: %s", exc)
        return 1

    print("Done.")
    print(f"  Output:             {result.output_path}")
    if result.pages_with_tables:
        print(f"  Table page(s):      {[p + 1 for p in result.pages_with_tables]}")
        print(f"  Backend:            {result.table_backend_used}")
        if result.used_marker_table_fallback:
            print("  Note: ML backends failed/skipped -- tables taken from Marker OCR.")
    else:
        print("  No tables detected -- text-only extraction.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
