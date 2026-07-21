"""
Smart Router / Classifier.

Decides which extraction pipeline should handle a given PDF.

--------------------------------------------------------------------------
MIGRATION NOTE (this version):
The old coordinate-based table pipeline (`coordinate_extractor.py`) has been
removed entirely -- it required hardcoded/anchor-derived x0-y0 coordinates
that only worked for one fixed document layout at a time.

Table-heavy documents are now routed to `vlm_extractor.py`: a local
Vision-Language Model (Qwen2-VL by default) running on CUDA. The page image
is handed to the VLM directly, with a strict prompt asking for the table
back as JSON -- no coordinate calibration, no anchor text, and no dedicated
table-structure-recognition step is required; the general-purpose VLM reads
the table the same way a human would.

(An earlier iteration of this router used `paddle_extractor.py`, a
PaddleOCR/PP-Structure pipeline. That module is still present in this package
for standalone use, but the router no longer dispatches to it -- the VLM
pipeline is now the primary table route, with Marker as the fallback.)

Current routing rule (kept intentionally simple for now):
    - If the keyword `DEFAULT_KEYWORD` ("Mẫu số: B06") is found on the first
      page's text -> route to the local VLM table pipeline
      (`vlm_extractor.extract`).
    - Otherwise -> fall back to the generic Marker pipeline
      (`pdf_extractor.pipeline.PDFPipeline`), which handles running text and
      any table shapes it can recover on its own.
--------------------------------------------------------------------------

The VLM pipeline requires a CUDA-equipped GPU server (see `vlm_extractor.VLMConfig`).
The Marker fallback runs on CPU.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from . import vlm_extractor
from .pipeline import PDFPipeline, PipelineResult

logger = logging.getLogger(__name__)

# The literal keyword used to detect this specific table-heavy form. Extend/
# replace this with a smarter (e.g. structural) signal later -- see the
# MIGRATION NOTE above for why this is still keyword-based for now.
DEFAULT_KEYWORD = "Mẫu số: B06"

DEFAULT_OUTPUT_PATH = "output_tables.xlsx"


class PipelineChoice:
    """Simple string constants identifying which pipeline was selected."""

    VLM_TABLE = "PIPELINE_VLM_TABLE"
    GENERIC_MARKER = "PIPELINE_MARKER_GENERIC"


def _first_page_text(pdf_path: str | Path) -> str:
    """
    Return the raw text of the first page of a PDF.

    Uses PyMuPDF (already a project dependency) purely for cheap text
    extraction -- this is NOT the coordinate-based extraction that was
    removed; it is just a keyword lookup.
    """
    import fitz  # PyMuPDF

    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    try:
        if doc.page_count == 0:
            return ""
        return doc[0].get_text()
    finally:
        doc.close()


def classify_pdf(pdf_path: str | Path, keyword: str = DEFAULT_KEYWORD) -> str:
    """
    Inspect the PDF's first page and decide which pipeline should process it.

    Args:
        pdf_path: Path to the PDF to classify.
        keyword: Literal string to search for on page 1 (e.g. "Mẫu số: B06",
            or any other table-form marker). If found, the file is
            considered "table-heavy" and routed to the VLM pipeline.

    Returns:
        PipelineChoice.VLM_TABLE if the keyword was found; otherwise
        PipelineChoice.GENERIC_MARKER.
    """
    try:
        page_text = _first_page_text(pdf_path)
    except Exception as exc:
        # If we can't even read the text layer (e.g. a pure-image scan with
        # no embedded text), fall back to Marker rather than crashing the
        # whole classification step.
        logger.warning("Router: failed to read page 1 text of %s (%s) -- falling back to Marker.", pdf_path, exc)
        return PipelineChoice.GENERIC_MARKER

    if keyword in page_text:
        logger.info("Router: keyword '%s' found -> Pipeline VLM_TABLE (local VLM).", keyword)
        return PipelineChoice.VLM_TABLE

    logger.info("Router: keyword '%s' not found -> Pipeline GENERIC_MARKER.", keyword)
    return PipelineChoice.GENERIC_MARKER


def run_pipeline_marker(pdf_path: str | Path, output_dir: str | Path = ".") -> PipelineResult:
    """Fallback pipeline: generic Marker-based Markdown -> tables/text extraction."""
    logger.info("Dispatching to Marker pipeline: %s", pdf_path)
    return PDFPipeline().run(pdf_path=pdf_path, output_dir=output_dir)


def route_and_extract(
    pdf_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    keyword: str = DEFAULT_KEYWORD,
) -> Optional[List[pd.DataFrame]]:
    """
    End-to-end entrypoint: classify the PDF, then dispatch to the matching pipeline.

    Args:
        pdf_path: Path to the input PDF.
        output_path: Destination .xlsx path used ONLY when routed to the
            VLM table pipeline.
        keyword: Forwarded to `classify_pdf`.

    Returns:
        The list of extracted DataFrames (one per page that yielded a usable
        table) if routed to the VLM pipeline, or `None` if routed to the
        Marker pipeline (which writes its own output files instead).
    """
    choice = classify_pdf(pdf_path, keyword=keyword)

    if choice == PipelineChoice.VLM_TABLE:
        try:
            return vlm_extractor.extract(pdf_path, output_path=output_path)
        except vlm_extractor.VLMExtractionError as exc:
            # Graceful degradation: if the VLM/GPU stack itself is
            # unavailable (e.g. no CUDA, missing weights) or fails outright,
            # don't crash the whole run -- fall back to Marker so the user
            # still gets *something* useful.
            logger.error("VLM pipeline failed (%s) -- falling back to Marker pipeline.", exc)
            run_pipeline_marker(pdf_path)
            return None

    run_pipeline_marker(pdf_path)
    return None
