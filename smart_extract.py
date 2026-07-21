#!/usr/bin/env python3
"""
CLI entrypoint: Smart Router for PDF extraction.

Classifies the input PDF (via `pdf_extractor.router.classify_pdf`) and dispatches
to the correct extraction pipeline:

    - Keyword "Mẫu số: B06" found on page 1 -> local Vision-Language Model
      table pipeline (Qwen2-VL running on CUDA, see vlm_extractor.py) + OCR
      cleanup -> writes an .xlsx file (one sheet per table).
    - Keyword not found -> generic Marker pipeline
      (pdf_extractor.pipeline.PDFPipeline), which writes its own Markdown /
      tables / text output files.

Usage:
    python smart_extract.py --input path/to/file.pdf --output output_tables.xlsx
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pdf_extractor.router import PipelineChoice, classify_pdf, route_and_extract


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify a PDF and route it to the correct extraction pipeline."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to the input PDF file.")
    parser.add_argument(
        "--output",
        "-o",
        default="output_tables.xlsx",
        help="Path to the output .xlsx file (only used if routed to the VLM pipeline).",
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

    choice = classify_pdf(pdf_path)
    print(f"Router decision: {choice}")

    dataframes = route_and_extract(pdf_path, output_path=args.output)

    if choice == PipelineChoice.VLM_TABLE and dataframes is not None:
        print(f"Done. Exported {len(dataframes)} table(s) to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
