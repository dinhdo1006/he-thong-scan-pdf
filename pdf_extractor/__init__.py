"""Offline PDF text/table extraction: content-driven unified pipeline (no keyword routing)."""

from __future__ import annotations

from . import vlm_extractor  # noqa: F401 — deprecated; kept for import-path compatibility
from .grid_table_extractor import detect_table_pages, extract_pdf_tables_to_excel
from .ocr_cleanup import clean_ocr_errors
from .pipeline import PDFPipeline
from .router import PipelineChoice, classify_pdf, route_and_extract
from .unified_pipeline import UnifiedPDFPipeline, UnifiedResult

try:
    from .docling_extractor import DoclingTableExtractor, extract_tables_from_pdf
except ImportError:  # docling not installed yet
    DoclingTableExtractor = None  # type: ignore[misc, assignment]
    extract_tables_from_pdf = None  # type: ignore[misc, assignment]

__all__ = [
    "PDFPipeline",
    "UnifiedPDFPipeline",
    "UnifiedResult",
    "extract_pdf_tables_to_excel",
    "detect_table_pages",
    "DoclingTableExtractor",
    "extract_tables_from_pdf",
    "vlm_extractor",
    "clean_ocr_errors",
    "classify_pdf",
    "route_and_extract",
    "PipelineChoice",
]
__version__ = "6.5.0"
