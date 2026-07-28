"""Offline PDF text/table extraction: content-driven unified pipeline (no keyword routing)."""

from __future__ import annotations

from typing import Any

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
__version__ = "6.8.2"


def __getattr__(name: str) -> Any:
    """Lazy exports so lightweight workers do not import heavy optional deps at import time."""
    if name == "PDFPipeline":
        from .pipeline import PDFPipeline

        return PDFPipeline
    if name == "UnifiedPDFPipeline":
        from .unified_pipeline import UnifiedPDFPipeline

        return UnifiedPDFPipeline
    if name == "UnifiedResult":
        from .unified_pipeline import UnifiedResult

        return UnifiedResult
    if name in {"extract_pdf_tables_to_excel", "detect_table_pages"}:
        from . import grid_table_extractor as g

        return getattr(g, name)
    if name in {"DoclingTableExtractor", "extract_tables_from_pdf"}:
        try:
            from . import docling_extractor as d
        except ImportError as exc:  # pragma: no cover
            raise AttributeError(name) from exc
        return getattr(d, name)
    if name == "vlm_extractor":
        from . import vlm_extractor as _vlm

        return _vlm
    if name == "clean_ocr_errors":
        from .ocr_cleanup import clean_ocr_errors

        return clean_ocr_errors
    if name in {"classify_pdf", "route_and_extract", "PipelineChoice"}:
        from . import router as r

        return getattr(r, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
