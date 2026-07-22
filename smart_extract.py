#!/usr/bin/env python3
"""
CLI entrypoint: Unified PDF -> output.txt + output_tables.xlsx.

Usage:
    python smart_extract.py -i path/to/file.pdf -o ./output
    python smart_extract.py -i path/to/file.pdf -o ./output --docx
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pdf_extractor.exporters import DEFAULT_OUTPUT_TXT
from pdf_extractor.logging_config import configure_app_logging
from pdf_extractor.unified_pipeline import UnifiedPDFPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract PDF to output.txt + output_tables.xlsx (optional --docx)."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to the input PDF file.")
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT_TXT,
        help="Output .txt path, or a directory (writes output.txt + output_tables.xlsx).",
    )
    parser.add_argument(
        "--docx",
        action="store_true",
        help="Also write output.docx with editable Word tables.",
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Skip VLM/Paddle/grid table backends (text/prose only).",
    )
    parser.add_argument(
        "--skip-marker",
        action="store_true",
        help="Skip Marker (default anyway on CPU). Use PyMuPDF prose + Paddle/VLM tables.",
    )
    parser.add_argument(
        "--force-marker",
        action="store_true",
        help="Force-load Marker/Surya (slow; needs working torch). Overrides --skip-marker.",
    )
    parser.add_argument(
        "--vlm-model",
        default=None,
        help=(
            "HuggingFace VLM id for table extraction. "
            "Default: Qwen/Qwen2-VL-2B-Instruct (fits ~16 GiB GPUs). "
            "Example larger: Qwen/Qwen2-VL-7B-Instruct"
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_app_logging(verbose=args.verbose)

    pdf_path = Path(args.input)
    if not pdf_path.is_file():
        logging.error("Input PDF not found: %s", pdf_path)
        return 1

    skip_marker: bool | None = True if args.skip_marker else None
    if args.force_marker:
        skip_marker = False

    from pdf_extractor.vlm_extractor import VLMConfig, VLMTableExtractor

    vlm_config = VLMConfig(crop_to_table=False)
    if args.vlm_model:
        vlm_config.model_name = args.vlm_model

    try:
        result = UnifiedPDFPipeline(
            skip_marker=skip_marker,
            force_marker=args.force_marker,
            vlm_extractor=VLMTableExtractor(vlm_config),
        ).run(
            pdf_path,
            output=args.output,
            skip_tables=args.skip_tables,
            skip_marker=skip_marker,
            force_marker=args.force_marker,
            write_docx=args.docx,
        )
    except Exception as exc:
        logging.error("Pipeline failed: %s", exc)
        return 1

    print("Done.")
    if result.txt_path:
        print(f"  Text:               {result.txt_path}")
    if result.xlsx_path:
        print(f"  Tables (xlsx):      {result.xlsx_path}")
    if result.docx_path:
        print(f"  DOCX:               {result.docx_path}")
    if result.pages_with_tables:
        print(f"  Table page(s):      {[p + 1 for p in result.pages_with_tables]}")
        print(f"  Backend:            {result.table_backend_used}")
        if result.used_marker_table_fallback:
            print("  WARNING: Marker OCR fallback -- fix VLM/Paddle for real table quality.")
        if result.table_count <= 0 or result.table_backend_used == "failed":
            print("  ERROR: detected table pages but extracted 0 tables.")
            return 2
    else:
        print("  No tables detected -- text-only extraction.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
