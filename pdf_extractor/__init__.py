"""Offline PDF text/table extraction: content-driven unified pipeline (no keyword routing)."""

from . import vlm_extractor
from .grid_table_extractor import detect_table_pages, extract_pdf_tables_to_excel
from .ocr_cleanup import clean_ocr_errors
from .pipeline import PDFPipeline
from .router import PipelineChoice, classify_pdf, route_and_extract
from .unified_pipeline import UnifiedPDFPipeline, UnifiedResult

__all__ = [
    "PDFPipeline",
    "UnifiedPDFPipeline",
    "UnifiedResult",
    "extract_pdf_tables_to_excel",
    "detect_table_pages",
    "vlm_extractor",
    "clean_ocr_errors",
    "classify_pdf",
    "route_and_extract",
    "PipelineChoice",
]
__version__ = "6.0.0"
