"""Offline PDF text/table extraction: content-driven unified pipeline (no keyword routing)."""

from __future__ import annotations

# Keep package import light for workers (table_extract). Heavy optional modules
# are imported lazily below / by callers.
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

# Deprecated compatibility alias — do not import at module load (needs Pillow/torch).
vlm_extractor = None  # type: ignore[misc, assignment]


def __getattr__(name: str):
    if name == "vlm_extractor":
        from . import vlm_extractor as _vlm

        return _vlm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
__version__ = "6.8.1"
