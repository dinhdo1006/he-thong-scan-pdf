"""Offline PDF text/table extraction: local VLM (table-heavy docs) + Marker (generic) pipelines."""

from . import vlm_extractor
from .grid_table_extractor import extract_pdf_tables_to_excel
from .ocr_cleanup import clean_ocr_errors
from .pipeline import PDFPipeline
from .router import PipelineChoice, classify_pdf, route_and_extract

__all__ = [
    "PDFPipeline",
    "extract_pdf_tables_to_excel",
    "vlm_extractor",
    "clean_ocr_errors",
    "classify_pdf",
    "route_and_extract",
    "PipelineChoice",
]
__version__ = "5.0.0"
