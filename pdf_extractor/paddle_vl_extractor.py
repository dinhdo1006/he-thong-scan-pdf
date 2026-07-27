"""
PaddleOCR-VL table extraction (Wave 2 Tier 1).

Uses the official ``paddleocr.PaddleOCRVL`` pipeline to parse PDF pages into
Markdown, then converts GFM pipe-tables into DataFrames (VND strings preserved).

Requires a recent ``paddleocr`` build that ships ``PaddleOCRVL`` plus model
weights (GPU strongly recommended; VRAM note: pause other VLMs/Ollama).
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd

from .markdown_parser import MarkdownBlockParser
from .ocr_cleanup import clean_ocr_errors

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = "output_tables_vl.xlsx"


class PaddleVLExtractionError(Exception):
    """Raised when PaddleOCR-VL cannot be initialized or returns no tables."""


_engine_cache: dict[str, Any] = {}


def paddle_vl_available() -> bool:
    """True when ``PaddleOCRVL`` can be imported (models may still download later)."""
    try:
        from paddleocr import PaddleOCRVL  # noqa: F401

        return True
    except Exception:
        return False


def get_engine(*, force_reload: bool = False) -> Any:
    """Lazily construct and cache a ``PaddleOCRVL`` pipeline instance."""
    if not force_reload and "engine" in _engine_cache:
        return _engine_cache["engine"]

    try:
        from paddleocr import PaddleOCRVL
    except Exception as exc:  # pragma: no cover - optional dependency
        raise PaddleVLExtractionError(
            "PaddleOCR-VL is not installed. "
            "Install a recent paddleocr (3.x) that exports PaddleOCRVL, e.g. "
            "`pip install 'paddleocr>=3.0' paddlepaddle`."
        ) from exc

    kwargs: dict[str, Any] = {}
    # Prefer lighter CPU-safe defaults when explicitly requested.
    version = os.environ.get("PADDLE_VL_PIPELINE_VERSION", "").strip()
    if version:
        kwargs["pipeline_version"] = version

    logger.info("Loading PaddleOCR-VL pipeline%s...", f" ({kwargs})" if kwargs else "")
    try:
        engine = PaddleOCRVL(**kwargs) if kwargs else PaddleOCRVL()
    except TypeError:
        # Older signatures may not accept pipeline_version.
        engine = PaddleOCRVL()
    except Exception as exc:
        raise PaddleVLExtractionError(f"Failed to init PaddleOCR-VL: {exc}") from exc

    _engine_cache["engine"] = engine
    return engine


def unload_engine() -> None:
    """Drop the cached VL engine (best-effort; frees Python refs only)."""
    _engine_cache.pop("engine", None)


def _result_to_markdown(res: Any) -> str:
    """Best-effort extract Markdown text from one PaddleOCR-VL page result."""
    if res is None:
        return ""

    # Common paddlex result helpers.
    for attr in ("markdown", "md", "text"):
        val = getattr(res, attr, None)
        if isinstance(val, str) and val.strip():
            return val
        if isinstance(val, dict):
            for key in ("markdown", "text", "content"):
                inner = val.get(key)
                if isinstance(inner, str) and inner.strip():
                    return inner

    if isinstance(res, dict):
        for key in ("markdown", "md", "text", "content"):
            val = res.get(key)
            if isinstance(val, str) and val.strip():
                return val

    # save_to_markdown → temp file
    save = getattr(res, "save_to_markdown", None)
    if callable(save):
        with tempfile.TemporaryDirectory(prefix="paddle_vl_") as tmp:
            try:
                save(save_path=tmp)
            except TypeError:
                try:
                    save(tmp)
                except Exception as exc:
                    logger.debug("save_to_markdown failed: %s", exc)
                    return ""
            except Exception as exc:
                logger.debug("save_to_markdown failed: %s", exc)
                return ""
            md_files = sorted(Path(tmp).rglob("*.md"))
            if md_files:
                return md_files[0].read_text(encoding="utf-8", errors="replace")

    # str() last resort
    text = str(res)
    if "|" in text and re.search(r"\|[-: ]+\|", text):
        return text
    return ""


def markdown_tables_to_dataframes(markdown: str) -> List[pd.DataFrame]:
    """Parse GFM pipe-tables from VL Markdown into DataFrames."""
    if not (markdown or "").strip():
        return []
    blocks = MarkdownBlockParser().parse_blocks(markdown)
    out: list[pd.DataFrame] = []
    for block in blocks:
        if block.kind == "table" and isinstance(block.content, pd.DataFrame):
            if not block.content.empty and len(block.content.columns) > 0:
                out.append(block.content)
    return out


def _page_index_from_result(res: Any, fallback: int) -> int:
    for attr in ("page_index", "page_idx", "page_no", "page"):
        val = getattr(res, attr, None)
        if val is None and isinstance(res, dict):
            val = res.get(attr)
        if val is None:
            continue
        try:
            idx = int(val)
        except (TypeError, ValueError):
            continue
        # Some APIs are 1-based.
        if idx >= 1 and attr in {"page_no", "page"}:
            return idx - 1
        return max(0, idx)
    return fallback


def extract_with_pages(
    pdf_path: str | Path,
    *,
    apply_ocr_cleanup: bool = True,
    page_indices: Optional[List[int]] = None,
) -> List[Tuple[int, pd.DataFrame]]:
    """
    Run PaddleOCR-VL on a PDF; return ``(page_index, DataFrame)`` pairs.

    When a page yields multiple tables, they are returned in reading order
    (same page_index repeated). ``page_indices`` optionally filters which
    0-based pages to keep.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    engine = get_engine()
    try:
        raw_out = engine.predict(input=str(pdf_path))
    except TypeError:
        raw_out = engine.predict(str(pdf_path))
    except Exception as exc:
        raise PaddleVLExtractionError(f"PaddleOCR-VL predict failed: {exc}") from exc

    if raw_out is None:
        return []

    # Normalize to a list of page results.
    if hasattr(raw_out, "__iter__") and not isinstance(raw_out, (str, bytes, dict)):
        pages = list(raw_out)
    else:
        pages = [raw_out]

    allow = set(page_indices) if page_indices is not None else None
    tagged: list[tuple[int, pd.DataFrame]] = []

    for fallback_i, res in enumerate(pages):
        page_no = _page_index_from_result(res, fallback_i)
        if allow is not None and page_no not in allow:
            continue
        md = _result_to_markdown(res)
        tables = markdown_tables_to_dataframes(md)
        if not tables:
            logger.debug("PaddleOCR-VL page %d: no pipe-tables in markdown.", page_no + 1)
            continue
        for df in tables:
            prepared = df.copy()
            if apply_ocr_cleanup:
                prepared = clean_ocr_errors(prepared)
            prepared.attrs["page_no"] = page_no
            prepared.attrs["backend"] = "paddle-vl"
            tagged.append((page_no, prepared))

    logger.info("PaddleOCR-VL extracted %d table(s) from %s.", len(tagged), pdf_path.name)
    return tagged


def extract(
    pdf_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    apply_ocr_cleanup: bool = True,
    page_indices: Optional[List[int]] = None,
) -> List[pd.DataFrame]:
    """PDF -> PaddleOCR-VL -> DataFrames (+ optional Excel)."""
    from .exporters import export_tables_preview

    page_dfs = extract_with_pages(
        pdf_path,
        apply_ocr_cleanup=apply_ocr_cleanup,
        page_indices=page_indices,
    )
    dataframes = [df for _, df in page_dfs]
    export_tables_preview(dataframes, output_path, write_csv=False, write_markdown=False)
    return dataframes
