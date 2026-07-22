"""
Zero-shot table extraction via PaddleOCR PP-Structure (v3 preferred).

PaddleOCR 3.x removed the old `PPStructure` class. This module uses
`PPStructureV3` when available, with a narrow compatibility shim for the
legacy `PPStructure` API so older environments still work.

Pipeline:
    1. Load PPStructureV3 (or legacy PPStructure).
    2. Run layout + table recognition on the PDF / page images.
    3. Collect each table's HTML (`pred_html` / `html["pred"]`).
    4. Parse HTML -> DataFrame WITHOUT pandas numeric coercion
       (so "250.000" stays "250.000").
    5. Optional OCR cleanup + Excel export.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .ocr_cleanup import clean_ocr_errors

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = "output_tables.xlsx"
DEFAULT_DPI = 200


class PaddleExtractionError(Exception):
    """Raised when the PP-Structure pipeline cannot be initialized or run."""


# ---------------------------------------------------------------------------
# PDF -> page images (legacy PPStructure path only)
# ---------------------------------------------------------------------------
def render_pdf_to_images(pdf_path: str | Path, dpi: int = DEFAULT_DPI) -> List[np.ndarray]:
    """Rasterize every PDF page to a BGR numpy array (OpenCV convention)."""
    import fitz

    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    images: List[np.ndarray] = []
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            images.append(rgb[:, :, ::-1].copy())
    finally:
        doc.close()

    logger.info("Rendered %d page(s) from %s at %d DPI.", len(images), pdf_path, dpi)
    return images


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
_engine_cache: Dict[str, Any] = {}


def get_engine():
    """
    Lazily construct and cache a PP-Structure / table-recognition engine.

    Prefers PaddleOCR 3.x `PPStructureV3`, then `TableRecognitionPipelineV2`,
    then legacy `PPStructure`. Always passes `enable_mkldnn=False` on the
    3.x path to avoid the known PaddlePaddle 3.3 + oneDNN CPU crash.
    """
    if "engine" in _engine_cache:
        return _engine_cache["engine"]

    engine: Any
    api = "v3"
    common_kwargs = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "enable_mkldnn": False,
    }

    try:
        from paddleocr import PPStructureV3

        logger.info("Loading PPStructureV3 (mobile OCR + table recognition)...")
        # Mobile det/rec: lower RAM than server models; full server stack has
        # AVed on this Windows CPU host when combined with Marker/torch.
        engine = PPStructureV3(
            **common_kwargs,
            use_seal_recognition=False,
            use_formula_recognition=False,
            use_chart_recognition=False,
            use_table_recognition=True,
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
        )
    except Exception as v3_exc:
        logger.warning("PPStructureV3 unavailable (%s) -- trying TableRecognitionPipelineV2.", v3_exc)
        try:
            from paddleocr import TableRecognitionPipelineV2

            logger.info("Loading TableRecognitionPipelineV2...")
            engine = TableRecognitionPipelineV2(**common_kwargs)
            api = "table_v2"
        except Exception as v2_exc:
            logger.warning("TableRecognitionPipelineV2 unavailable (%s) -- trying legacy PPStructure.", v2_exc)
            try:
                from paddleocr import PPStructure
            except ImportError as exc:
                raise PaddleExtractionError(
                    "paddleocr is not installed or too incomplete. "
                    "Run: pip install 'paddlepaddle' 'paddleocr>=3.0' 'paddlex[ocr]'"
                ) from exc

            logger.info("Loading legacy PPStructure (show_log=False, layout=True)...")
            engine = PPStructure(show_log=False, layout=True)
            api = "legacy"

    _engine_cache["engine"] = engine
    _engine_cache["api"] = api
    logger.info("PP-Structure engine ready (api=%s).", api)
    return engine


def _engine_api() -> str:
    get_engine()
    return str(_engine_cache.get("api", "v3"))


# ---------------------------------------------------------------------------
# HTML -> DataFrame (keep cell text as strings)
# ---------------------------------------------------------------------------
def html_to_dataframe(html: str) -> Optional[pd.DataFrame]:
    """
    Parse one HTML `<table>` into a DataFrame without numeric coercion.

    Reuses the VLM extractor's colspan/rowspan-aware parser so "250.000"
    is not silently turned into 250.0 by `pandas.read_html`.
    """
    if not html or not str(html).strip():
        return None
    from .vlm_extractor import VLMTableExtractor

    fragment = str(html).strip()
    if "<table" not in fragment.lower():
        fragment = f"<table>{fragment}</table>"
    return VLMTableExtractor._html_table_to_dataframe(fragment)


def _pred_html_from_table_res(table_res: Any) -> Optional[str]:
    """Pull HTML string out of a PPStructureV3 / legacy table result object."""
    if table_res is None:
        return None

    if isinstance(table_res, dict):
        html = table_res.get("pred_html")
        if html:
            return str(html)
        nested = table_res.get("html")
        if isinstance(nested, dict) and nested.get("pred"):
            return str(nested["pred"])
        if isinstance(nested, str) and nested.strip():
            return nested
        res = table_res.get("res")
        if isinstance(res, dict) and res.get("html"):
            return str(res["html"])
        return None

    html_attr = getattr(table_res, "html", None)
    if isinstance(html_attr, dict) and html_attr.get("pred"):
        return str(html_attr["pred"])
    if isinstance(html_attr, str) and html_attr.strip():
        return html_attr

    try:
        pred = table_res["pred_html"]  # type: ignore[index]
        if pred:
            return str(pred)
    except Exception:
        pass

    return None


def _iter_table_htmls_from_page(page_result: Any) -> List[str]:
    """Collect every table HTML string from one PPStructureV3 page result."""
    htmls: List[str] = []

    to_html = getattr(page_result, "_to_html", None)
    if callable(to_html):
        try:
            html_map = to_html() or {}
            for value in html_map.values():
                if value and str(value).strip():
                    htmls.append(str(value))
            if htmls:
                return htmls
        except Exception as exc:
            logger.debug("page_result._to_html() failed: %s", exc)

    tables: List[Any] = []
    try:
        tables = list(page_result["table_res_list"] or [])
    except Exception:
        getter = getattr(page_result, "get", None)
        if callable(getter):
            tables = list(getter("table_res_list") or [])

    for table_res in tables:
        html = _pred_html_from_table_res(table_res)
        if html:
            htmls.append(html)
    return htmls


# ---------------------------------------------------------------------------
# Legacy block helpers (PPStructure 2.x)
# ---------------------------------------------------------------------------
def analyze_page(image: np.ndarray, engine=None) -> List[Dict[str, Any]]:
    """Run legacy PPStructure on one page image."""
    engine = engine or get_engine()
    try:
        return engine(image) or []
    except Exception as exc:  # pragma: no cover
        logger.warning("PP-Structure failed to analyze a page: %s", exc)
        return []


def filter_table_blocks(layout_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only legacy layout blocks classified as `table`."""
    return [block for block in layout_blocks if block.get("type") == "table"]


def table_block_to_dataframe(table_block: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Convert one legacy PPStructure table block into a DataFrame."""
    html = _pred_html_from_table_res(table_block)
    if not html:
        logger.warning("Table block on page %s has no HTML output -- skipping.", table_block.get("page"))
        return None
    return html_to_dataframe(html)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_to_excel(dataframes: List[pd.DataFrame], output_path: str | Path) -> Path:
    """Write every recovered table to its own sheet in one .xlsx workbook."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not dataframes:
        pd.DataFrame({"info": ["No tables were detected by PP-Structure in this document."]}).to_excel(
            output_path, index=False, sheet_name="Info", engine="openpyxl"
        )
        logger.warning("No tables detected -- wrote placeholder workbook to %s", output_path)
        return output_path

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for i, df in enumerate(dataframes, start=1):
            sheet_name = f"Table_{i}"[:31]
            df.to_excel(writer, index=False, sheet_name=sheet_name)

    logger.info("Exported %d table(s) to %s", len(dataframes), output_path)
    return output_path


# ---------------------------------------------------------------------------
# End-to-end entrypoint
# ---------------------------------------------------------------------------
def _extract_with_v3(
    pdf_path: Path,
    engine: Any,
    apply_ocr_cleanup: bool,
) -> List[pd.DataFrame]:
    """
    PPStructureV3 / TableRecognitionPipelineV2 path.

    Prefer page-image inputs over the whole PDF: feeding multi-page PDFs into
    the heavy layout stack has crashed (access violation) on Windows CPU builds.
    """
    dataframes: List[pd.DataFrame] = []
    try:
        page_images = render_pdf_to_images(pdf_path, dpi=DEFAULT_DPI)
    except Exception as exc:
        raise PaddleExtractionError(f"Failed to render '{pdf_path}' to images: {exc}") from exc

    for page_index, image in enumerate(page_images):
        logger.info("Running table recognition on page %d/%d ...", page_index + 1, len(page_images))
        try:
            page_results = engine.predict(
                input=image,
                use_wired_table_cells_trans_to_html=True,
                use_wireless_table_cells_trans_to_html=False,
                use_table_orientation_classify=True,
                use_ocr_results_with_table_cells=True,
            )
        except TypeError:
            # TableRecognitionPipelineV2 accepts fewer kwargs.
            try:
                page_results = engine.predict(input=image)
            except Exception as exc:
                logger.warning("Page %d predict failed: %s", page_index + 1, exc)
                continue
        except Exception as exc:
            logger.warning("Page %d predict failed: %s", page_index + 1, exc)
            continue

        page_htmls: List[str] = []
        for page_result in page_results or []:
            page_htmls.extend(_iter_table_htmls_from_page(page_result))

        if not page_htmls:
            logger.info("Page %d: no tables detected -- skipping.", page_index + 1)
            continue

        for table_index, html in enumerate(page_htmls):
            df = html_to_dataframe(html)
            if df is None or df.empty:
                logger.info(
                    "Page %d, table %d: empty/unparsable -- skipping.",
                    page_index + 1,
                    table_index + 1,
                )
                continue
            if apply_ocr_cleanup:
                df = clean_ocr_errors(df)
            dataframes.append(df)
            logger.info(
                "Page %d, table %d: extracted shape %s.",
                page_index + 1,
                table_index + 1,
                df.shape,
            )
    return dataframes


def _extract_with_legacy(
    pdf_path: Path,
    engine: Any,
    dpi: int,
    apply_ocr_cleanup: bool,
) -> List[pd.DataFrame]:
    """Legacy PPStructure path: render pages, then call engine(image)."""
    try:
        page_images = render_pdf_to_images(pdf_path, dpi=dpi)
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise PaddleExtractionError(f"Failed to render '{pdf_path}' to images: {exc}") from exc

    dataframes: List[pd.DataFrame] = []
    for page_index, image in enumerate(page_images):
        layout_blocks = analyze_page(image, engine=engine)
        table_blocks = filter_table_blocks(layout_blocks)
        if not table_blocks:
            logger.info("Page %d: no tables detected -- skipping.", page_index + 1)
            continue
        for table_index, table_block in enumerate(table_blocks):
            table_block["page"] = page_index + 1
            df = table_block_to_dataframe(table_block)
            if df is None or df.empty:
                logger.info(
                    "Page %d, table %d: empty/unparsable -- skipping.",
                    page_index + 1,
                    table_index + 1,
                )
                continue
            if apply_ocr_cleanup:
                df = clean_ocr_errors(df)
            dataframes.append(df)
    return dataframes


def extract(
    pdf_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    dpi: int = DEFAULT_DPI,
    apply_ocr_cleanup: bool = True,
) -> List[pd.DataFrame]:
    """
    Full pipeline: PDF -> PP-Structure -> table HTML -> DataFrame -> Excel.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    engine = get_engine()
    api = _engine_api()

    if api in ("v3", "table_v2"):
        dataframes = _extract_with_v3(pdf_path, engine, apply_ocr_cleanup)
    else:
        dataframes = _extract_with_legacy(pdf_path, engine, dpi, apply_ocr_cleanup)

    export_to_excel(dataframes, output_path)
    return dataframes


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot table extraction via PaddleOCR PP-StructureV3 "
            "(layout analysis + table structure recognition)."
        )
    )
    parser.add_argument("--input", "-i", required=True, help="Path to the input PDF file.")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_PATH, help="Path to the output .xlsx file.")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Page rendering resolution (legacy API only).")
    parser.add_argument("--no-ocr-cleanup", action="store_true", help="Skip the OCR post-processing replacement pass.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        dataframes = extract(
            args.input,
            output_path=args.output,
            dpi=args.dpi,
            apply_ocr_cleanup=not args.no_ocr_cleanup,
        )
    except (FileNotFoundError, PaddleExtractionError) as exc:
        logging.error(str(exc))
        return 1

    print(f"Done. Exported {len(dataframes)} table(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
