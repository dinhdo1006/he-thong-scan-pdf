#!/usr/bin/env python3
"""
Local smoke (no Kafka): structured tables JSON -> DOCX.

Usage (from repo root):
  python ocr_pipeline_backup/ocr_pipeline/scripts/smoke_structured_docx.py
  python ocr_pipeline_backup/ocr_pipeline/scripts/smoke_structured_docx.py --pdf path/to.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ocr_pipeline_backup" / "ocr_pipeline" / "convert_to_docx"))

from structured_docx import create_document_from_pages_and_tables  # noqa: E402


def from_fixture(out_dir: Path) -> Path:
    pages = [
        {
            "page_number": 1,
            "lines": [{"line_text": "BAO CAO THU TIEN - smoke test"}],
            "tables": [],
        }
    ]
    payload = {
        "version": 1,
        "tables": [
            {
                "table_id": 0,
                "columns": ["STT", "Tieu chi", "Tong"],
                "header_matrix": [
                    ["STT", "Tieu chi", "Tong so"],
                    ["A", "B", "1"],
                ],
                "data": [
                    ["I", "Tong phai thu", "100"],
                    ["1", "Chi tiet", "40"],
                ],
                "page": 0,
            }
        ],
    }
    doc = create_document_from_pages_and_tables(pages, payload)
    out = out_dir / "smoke_fixture.docx"
    doc.save(str(out))
    (out_dir / "structured_tables.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out


def from_pdf(pdf: Path, out_dir: Path) -> Path:
    from pdf_extractor.pipeline_bridge import tables_to_payload, write_structured_tables_json
    from pdf_extractor.unified_pipeline import UnifiedPDFPipeline

    pipe = UnifiedPDFPipeline(skip_marker=True, skip_docling=True)
    result = pipe.run(pdf, out_dir, write_txt=True, write_xlsx=True, write_docx=False)
    payload = tables_to_payload(result.tables)
    write_structured_tables_json(result.tables, out_dir / "structured_tables.json")
    pages = [
        {
            "page_number": 1,
            "lines": [
                {"line_text": line}
                for line in (result.txt_path.read_text(encoding="utf-8").splitlines()[:30]
                             if result.txt_path and result.txt_path.is_file()
                             else [])
            ],
            "tables": [],
        }
    ]
    doc = create_document_from_pages_and_tables(pages, payload)
    out = out_dir / f"{pdf.stem}_smoke.docx"
    doc.save(str(out))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke structured tables -> DOCX")
    parser.add_argument("--pdf", type=Path, default=None)
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.output or Path(tempfile.mkdtemp(prefix="smoke_docx_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pdf:
        path = from_pdf(args.pdf.expanduser().resolve(), out_dir)
    else:
        path = from_fixture(out_dir)

    print(f"OK -> {path}")
    print(f"Artifacts dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
