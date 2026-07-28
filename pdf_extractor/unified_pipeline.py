"""
Unified Generic Pipeline: content-driven orchestration for ANY input PDF.

Every PDF goes through the same fixed sequence:
    Step A (always):     Marker (or PyMuPDF when skipped) -> Markdown / prose.
    Step B (always):     pdfplumber/visual scan -> which pages contain a table.
    Step C (if B found): PaddleOCR-VL (Tier 1) -> Docling -> Paddle PP-Structure
                         -> pdfplumber grid; pick per page via semantic+geometry.

On Windows CPU, Marker (torch) and Paddle collide in one process -- Step A and
the Paddle branch of Step C run in isolated subprocesses.

Primary outputs: `output.txt` + `output_tables.xlsx` (form grids).
Optional: `output.docx` with real Word tables.
If GPU/ML table backends fail, Marker's table blocks are kept with a loud warning.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .exporters import (
    DEFAULT_OUTPUT_DOCX,
    DEFAULT_OUTPUT_TXT,
    DEFAULT_OUTPUT_XLSX,
    compose_document_txt,
    export_document_from_markdown,
    export_tables_preview,
    resolve_output_path,
    save_unified_txt,
)
from .grid_table_extractor import (
    DEFAULT_TABLE_SETTINGS,
    detect_table_pages,
    extract_pdf_tables_to_excel,
)
from .isolated_backends import (
    _skip_marker,
    extract_markdown_isolated,
    extract_paddle_tables_isolated,
    extract_paddle_tables_isolated_with_pages,
)
from .marker_extractor import MarkerExtractor
from .markdown_parser import MarkdownBlockParser
from .paddle_extractor import PaddleExtractionError
from .paddle_extractor import extract as paddle_extract
from .paddle_extractor import extract_with_pages as paddle_extract_with_pages
from .text_fallback import (
    extract_page_prose_regions,
    extract_plaintext_fallback,
    plaintext_as_markdown,
)
from .table_quality import (
    LOW_QUALITY_THRESHOLD,
    pick_better_table,
    rebind_table_tokens_to_grid,
    score_table_quality,
)
from .semantic_validator import annotate_tables

# Docling (+ transformers + torch) is imported lazily at first use so that
# `smart_extract.py` starts up without hanging on Linux CPU machines that have
# a broken/slow torch install.  _DOCLING_AVAILABLE is resolved on first call
# to _extract_tables(), not at module import time.
DoclingExtractionError: type = RuntimeError  # placeholder; overwritten lazily
DoclingTableExtractor = None  # placeholder; overwritten lazily
_DOCLING_AVAILABLE: bool | None = None  # None = not yet probed


def _probe_docling() -> None:
    """Lazy one-time import of Docling so torch doesn't load at startup."""
    global _DOCLING_AVAILABLE, DoclingExtractionError, DoclingTableExtractor
    if _DOCLING_AVAILABLE is not None:
        return
    try:
        from .docling_extractor import (  # noqa: PLC0415
            DoclingExtractionError as _DocErr,
            DoclingTableExtractor as _DocCls,
        )
        DoclingExtractionError = _DocErr  # type: ignore[assignment]
        DoclingTableExtractor = _DocCls  # type: ignore[assignment]
        _DOCLING_AVAILABLE = True
    except Exception:  # ImportError or torch crash on Linux CPU
        _DOCLING_AVAILABLE = False


try:
    from .paddle_vl_extractor import (
        PaddleVLExtractionError,
        extract_with_pages as paddle_vl_extract_with_pages,
        paddle_vl_available,
    )

    _PADDLE_VL_IMPORTABLE = True
except ImportError:  # pragma: no cover
    PaddleVLExtractionError = RuntimeError  # type: ignore[misc, assignment]
    paddle_vl_extract_with_pages = None  # type: ignore[misc, assignment]
    paddle_vl_available = lambda: False  # type: ignore[misc, assignment]
    _PADDLE_VL_IMPORTABLE = False

logger = logging.getLogger(__name__)

BACKEND_NONE = "none"
BACKEND_DOCLING = "docling"
BACKEND_VLM = "vlm"  # retained for backward-compatible result labels only
BACKEND_PADDLE_VL = "paddle-vl"
BACKEND_PADDLE = "paddle"
BACKEND_HYBRID = "docling+paddle"  # some pages from Docling, gaps filled by Paddle
BACKEND_GRID = "grid"
BACKEND_MARKER = "marker"
BACKEND_FAILED = "failed"


@dataclass
class UnifiedResult:
    """Result of one `UnifiedPDFPipeline.run()` call."""

    output_path: Path
    table_count: int
    pages_with_tables: List[int] = field(default_factory=list)
    table_backend_used: str = BACKEND_NONE
    used_marker_table_fallback: bool = False
    docx_path: Optional[Path] = None
    xlsx_path: Optional[Path] = None
    txt_path: Optional[Path] = None
    tables: List[pd.DataFrame] = field(default_factory=list)


def _needs_process_isolation() -> bool:
    """True when Marker/torch and Paddle must not share one interpreter."""
    return sys.platform.startswith("win")


def _tag_page(df: pd.DataFrame, page_no: int) -> pd.DataFrame:
    out = df.copy()
    out.attrs["page_no"] = int(page_no)
    return out


def _backend_label(sources: Dict[int, str]) -> str:
    used = set(sources.values())
    if not used:
        return BACKEND_NONE
    if used == {BACKEND_DOCLING}:
        return BACKEND_DOCLING
    if used == {BACKEND_PADDLE}:
        return BACKEND_PADDLE
    if used == {BACKEND_PADDLE_VL}:
        return BACKEND_PADDLE_VL
    if used == {BACKEND_GRID}:
        return BACKEND_GRID
    if BACKEND_PADDLE_VL in used and used <= {BACKEND_PADDLE_VL, BACKEND_DOCLING, BACKEND_PADDLE}:
        return "paddle-vl+hybrid" if len(used) > 1 else BACKEND_PADDLE_VL
    if used <= {BACKEND_DOCLING, BACKEND_PADDLE}:
        return BACKEND_HYBRID
    return "+".join(sorted(used))


def _run_paddle_fallback(pdf_path: Path, out_dir: Path) -> List[pd.DataFrame]:
    if _needs_process_isolation():
        return extract_paddle_tables_isolated(pdf_path, out_dir)

    tmp_path = out_dir / ".tmp_paddle_fallback.xlsx"
    try:
        return paddle_extract(pdf_path, output_path=tmp_path, apply_ocr_cleanup=True)
    finally:
        tmp_path.unlink(missing_ok=True)


def _run_paddle_fallback_with_pages(pdf_path: Path, out_dir: Path) -> List[Tuple[int, pd.DataFrame]]:
    """Same as `_run_paddle_fallback`, but tags each table with its 0-indexed page."""
    if _needs_process_isolation():
        return extract_paddle_tables_isolated_with_pages(pdf_path, out_dir)
    return paddle_extract_with_pages(pdf_path, apply_ocr_cleanup=True)


def _run_grid_fallback(pdf_path: Path, out_dir: Path, table_settings: dict) -> List[pd.DataFrame]:
    tmp_path = out_dir / ".tmp_grid_fallback.xlsx"
    try:
        return extract_pdf_tables_to_excel(
            pdf_path,
            output_path=tmp_path,
            table_settings=table_settings,
            apply_ocr_cleanup=True,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


class UnifiedPDFPipeline:
    """Content-driven orchestrator: txt + xlsx primary, optional docx."""

    def __init__(
        self,
        marker_extractor: Optional[MarkerExtractor] = None,
        table_settings: dict = DEFAULT_TABLE_SETTINGS,
        *,
        skip_marker: Optional[bool] = None,
        force_marker: bool = False,
        docling_no_ocr: bool = False,
        skip_paddle_vl: Optional[bool] = None,
        force_paddle_vl: bool = False,
        skip_docling: bool = False,
        rebind_tokens: Optional[bool] = None,
    ) -> None:
        self.marker = marker_extractor or MarkerExtractor()
        self.table_settings = table_settings
        self.skip_marker = skip_marker
        self.force_marker = force_marker
        self.docling_no_ocr = bool(docling_no_ocr)
        self.skip_paddle_vl = skip_paddle_vl
        self.force_paddle_vl = bool(force_paddle_vl)
        self.skip_docling = bool(skip_docling)
        if rebind_tokens is None:
            rebind_tokens = _env_flag("REBIND_TABLE_TOKENS")
        self.rebind_tokens = bool(rebind_tokens)

    def _ensure_page_tag(self, df: pd.DataFrame, page_no: int) -> pd.DataFrame:
        if getattr(df, "attrs", {}).get("page_no") is None:
            df.attrs["page_no"] = int(page_no)
        return df

    def _detect_table_pages(self, pdf_path: Path) -> List[int]:
        return detect_table_pages(pdf_path, table_settings=self.table_settings)

    def _extract_markdown(self, pdf_path: Path, work_dir: Path) -> str:
        if _skip_marker(skip_marker=self.skip_marker, force_marker=self.force_marker):
            logger.info(
                "Skipping Marker (default on CPU / SKIP_MARKER). "
                "Using PyMuPDF prose; tables still come from Docling/Paddle/VL. "
                "Set FORCE_MARKER=1 or pass --force-marker to enable Marker."
            )
            return plaintext_as_markdown(extract_plaintext_fallback(pdf_path))

        if _needs_process_isolation():
            logger.info("Windows: running Marker in an isolated subprocess (torch/paddle DLL split).")
            return extract_markdown_isolated(
                pdf_path,
                work_dir,
                skip_marker=False,
                force_marker=True,
            )
        try:
            return self.marker.extract_markdown(pdf_path)
        except Exception as exc:
            logger.warning("Marker failed (%s) -- using PyMuPDF text fallback.", exc)
            return plaintext_as_markdown(extract_plaintext_fallback(pdf_path))

    def _should_try_paddle_vl(self) -> bool:
        """Tier 1 when VL is importable, unless explicitly skipped."""
        if getattr(self, "force_paddle_vl", False) or _env_flag("FORCE_PADDLE_VL"):
            return True
        if _env_flag("SKIP_PADDLE_VL"):
            return False
        skip_vl = getattr(self, "skip_paddle_vl", None)
        if skip_vl is True:
            return False
        # skip_vl False or None: attempt when package is available (Wave 2 Tier 1).
        if not _PADDLE_VL_IMPORTABLE or paddle_vl_extract_with_pages is None:
            return False
        try:
            return bool(paddle_vl_available())
        except Exception:
            return False

    def _try_paddle_vl(
        self,
        pdf_path: Path,
        page_indices: List[int],
    ) -> Dict[int, pd.DataFrame]:
        if not self._should_try_paddle_vl():
            return {}
        try:
            logger.info(
                "Table backend: trying PaddleOCR-VL (Tier 1) on page(s) %s...",
                [p + 1 for p in page_indices],
            )
            pairs = paddle_vl_extract_with_pages(pdf_path, page_indices=page_indices)
            by_page: Dict[int, pd.DataFrame] = {}
            for page_no, df in pairs:
                if df is None or df.empty:
                    continue
                by_page[int(page_no)] = self._ensure_page_tag(df, int(page_no))
            logger.info("PaddleOCR-VL extracted %d page table(s).", len(by_page))
            return by_page
        except (PaddleVLExtractionError, FileNotFoundError, OSError, RuntimeError) as exc:
            logger.warning("PaddleOCR-VL unavailable/failed (%s) -- continuing cascade.", exc)
        except Exception as exc:
            logger.warning("PaddleOCR-VL unavailable/failed (%s) -- continuing cascade.", exc)
        return {}

    def _maybe_rebind(
        self,
        pdf_path: Path,
        tables: List[pd.DataFrame],
        *,
        bbox_by_page: Optional[Dict[int, tuple[float, float, float, float]]] = None,
    ) -> List[pd.DataFrame]:
        if not getattr(self, "rebind_tokens", False):
            return tables
        out: List[pd.DataFrame] = []
        for df in tables:
            page_no = getattr(df, "attrs", {}).get("page_no")
            if page_no is None:
                out.append(df)
                continue
            try:
                page_index = int(page_no)
            except (TypeError, ValueError):
                out.append(df)
                continue
            table_bbox = (bbox_by_page or {}).get(page_index)
            rebound = rebind_table_tokens_to_grid(
                pdf_path,
                page_index,
                df,
                table_bbox,
            )
            out.append(rebound if rebound is not None else df)
        return out

    def _merge_vl_semantic(
        self,
        merged: Dict[int, pd.DataFrame],
        sources: Dict[int, str],
        vl_by_page: Dict[int, pd.DataFrame],
        out_dir: Path,
    ) -> None:
        if not vl_by_page:
            return
        for page_no, vl_df in vl_by_page.items():
            base = merged.get(page_no)
            if base is None:
                merged[page_no] = self._ensure_page_tag(vl_df, page_no)
                sources[page_no] = BACKEND_PADDLE_VL
                continue
            base_label = sources.get(page_no, "base")
            chosen, winner = pick_better_table(
                base,
                vl_df,
                use_semantic=True,
                left_label=base_label,
                right_label="paddle-vl",
                conflict_dir=out_dir,
                page_no=page_no,
            )
            if winner == "right":
                sources[page_no] = BACKEND_PADDLE_VL
            merged[page_no] = self._ensure_page_tag(chosen, page_no)

    def _extract_tables(
        self,
        pdf_path: Path,
        out_dir: Path,
        page_indices: List[int],
    ) -> Tuple[List[pd.DataFrame], str]:
        vl_by_page = self._try_paddle_vl(pdf_path, page_indices)

        docling_by_page: Dict[int, pd.DataFrame] = {}
        docling_unplaced: List[pd.DataFrame] = []
        skip_docling = bool(getattr(self, "skip_docling", False) or _env_flag("SKIP_DOCLING"))
        if skip_docling:
            logger.info(
                "Skipping Docling (--backend paddle / SKIP_DOCLING). "
                "Using PaddleOCR PP-Structure as primary table backend."
            )
        else:
            _probe_docling()  # lazy import of Docling/torch — safe on Linux CPU
            if not _DOCLING_AVAILABLE or DoclingTableExtractor is None:
                logger.warning(
                    "Docling is not installed -- skipping TableFormer. "
                    "Install with: pip install 'docling>=2.0.0' 'docling-core>=2.0.0'"
                )
            else:
                try:
                    logger.info(
                        "Table backend: trying Docling TableFormer on page(s) %s...",
                        [p + 1 for p in page_indices],
                    )
                    extractor = DoclingTableExtractor(
                        pdf_path,
                        disable_ocr=self.docling_no_ocr,
                    )
                    for page_no, df in extractor.extract_with_pages():
                        if page_no is None:
                            docling_unplaced.append(df)
                        else:
                            docling_by_page[page_no] = df
                    logger.info(
                        "Docling extracted %d table(s).",
                        len(docling_by_page) + len(docling_unplaced),
                    )
                except (DoclingExtractionError, FileNotFoundError, OSError, RuntimeError) as exc:
                    logger.warning(
                        "Docling unavailable/failed (%s) -- falling back to PaddleOCR.",
                        exc,
                    )
                except Exception as exc:
                    logger.warning(
                        "Docling unavailable/failed (%s) -- falling back to PaddleOCR.",
                        exc,
                    )

        missing_pages = [p for p in page_indices if p not in docling_by_page]
        weak_pages = []
        for p, df in docling_by_page.items():
            scored, breakdown = score_table_quality(df, return_breakdown=True)
            if scored < LOW_QUALITY_THRESHOLD:
                weak_pages.append(p)
            logger.debug(
                "Docling pre-Paddle page %d score=%.2f shape=%s breakdown=%s head=%s",
                p + 1,
                scored,
                getattr(df, "shape", None),
                breakdown,
                df.head(5).to_dict("records") if df is not None and not df.empty else [],
            )
        docling_widths = [len(df.columns) for df in docling_by_page.values()]
        width_inconsistent = (
            bool(docling_widths)
            and (max(docling_widths) - min(docling_widths)) >= 2
        )
        if width_inconsistent:
            max_w = max(docling_widths)
            for p, df in docling_by_page.items():
                if len(df.columns) <= max_w - 2 and p not in weak_pages:
                    weak_pages.append(p)
            logger.warning(
                "Docling page widths inconsistent %s -- treating narrower page(s) "
                "as weak for Paddle comparison.",
                {p + 1: len(df.columns) for p, df in docling_by_page.items()},
            )
        need_paddle = bool(missing_pages or weak_pages or not docling_by_page)

        if docling_by_page and not need_paddle:
            logger.info(
                "Docling covered every detected table page with acceptable quality."
            )
            merged: Dict[int, pd.DataFrame] = {
                p: self._ensure_page_tag(docling_by_page[p], p) for p in docling_by_page
            }
            sources: Dict[int, str] = {p: BACKEND_DOCLING for p in docling_by_page}
            if vl_by_page:
                logger.info(
                    "Merging PaddleOCR-VL via semantic pick (Docling baseline)."
                )
                self._merge_vl_semantic(merged, sources, vl_by_page, out_dir)
            ordered = [merged[p] for p in sorted(merged)] + docling_unplaced
            ordered = self._maybe_rebind(pdf_path, ordered)
            return ordered, _backend_label(sources)

        if weak_pages:
            logger.warning(
                "Docling page(s) %s scored below quality threshold %.2f "
                "-- running PaddleOCR for a form-agnostic comparison.",
                [p + 1 for p in weak_pages],
                LOW_QUALITY_THRESHOLD,
            )
        if missing_pages and docling_by_page:
            logger.warning(
                "Docling covered %d/%d table page(s); page(s) %s missing "
                "-- filling gaps with PaddleOCR.",
                len(docling_by_page),
                len(page_indices),
                [p + 1 for p in missing_pages],
            )
        elif not docling_by_page:
            logger.warning("Docling returned 0 usable tables -- falling back to PaddleOCR.")

        try:
            logger.info("Table backend: trying PaddleOCR PP-Structure...")
            paddle_pages = _run_paddle_fallback_with_pages(pdf_path, out_dir)
            paddle_by_page = {page_no: df for page_no, df in paddle_pages}
            merged = {}
            sources = {}

            all_pages = sorted(
                set(page_indices) | set(docling_by_page) | set(paddle_by_page) | set(vl_by_page)
            )
            for page_no in all_pages:
                d_df = docling_by_page.get(page_no)
                p_df = paddle_by_page.get(page_no)
                if d_df is None and p_df is None:
                    if page_no in vl_by_page:
                        merged[page_no] = self._ensure_page_tag(vl_by_page[page_no], page_no)
                        sources[page_no] = BACKEND_PADDLE_VL
                    continue
                if d_df is not None and p_df is not None:
                    chosen, winner = pick_better_table(
                        d_df,
                        p_df,
                        use_semantic=True,
                        left_label="docling",
                        right_label="paddle",
                        conflict_dir=out_dir,
                        page_no=page_no,
                    )
                    sources[page_no] = BACKEND_PADDLE if winner == "right" else BACKEND_DOCLING
                    merged[page_no] = chosen
                    d_score, d_breakdown = score_table_quality(d_df, return_breakdown=True)
                    p_score, p_breakdown = score_table_quality(p_df, return_breakdown=True)
                    logger.info(
                        "Page %d: Docling score=%.2f, Paddle score=%.2f -> %s "
                        "(semantic+geometric pick)",
                        page_no + 1,
                        d_score,
                        p_score,
                        sources[page_no],
                    )
                    logger.debug(
                        "Page %d Docling candidate shape=%s columns=%s",
                        page_no + 1,
                        getattr(d_df, "shape", None),
                        [str(c) for c in list(d_df.columns)[:20]],
                    )
                    try:
                        head_records = d_df.head(5).to_dict("records")
                    except Exception as exc:  # pragma: no cover - defensive
                        head_records = [{"_error": str(exc)}]
                    logger.debug("Page %d Docling head(5)=%s", page_no + 1, head_records)
                    logger.debug("Page %d Docling score breakdown=%s", page_no + 1, d_breakdown)
                    logger.debug("Page %d Paddle score breakdown=%s", page_no + 1, p_breakdown)
                elif p_df is not None:
                    merged[page_no] = p_df
                    sources[page_no] = BACKEND_PADDLE
                else:
                    merged[page_no] = d_df  # type: ignore[assignment]
                    sources[page_no] = BACKEND_DOCLING

            self._merge_vl_semantic(merged, sources, vl_by_page, out_dir)

            still_missing = [p for p in page_indices if p not in merged]
            if merged:
                backend = _backend_label(sources)
                if still_missing:
                    logger.error(
                        "Page(s) %s still have no valid table after Docling + "
                        "PaddleOCR -- shipping the %d page(s) that did succeed.",
                        [p + 1 for p in still_missing],
                        len(merged),
                    )
                else:
                    logger.info("Per-page quality merge complete (backend=%s).", backend)
                ordered = [merged[p] for p in sorted(merged)] + docling_unplaced
                ordered = self._maybe_rebind(pdf_path, ordered)
                return ordered, backend
            logger.warning("PaddleOCR found no tables -- falling back to pdfplumber grid.")
        except (PaddleExtractionError, RuntimeError, OSError) as exc:
            logger.warning(
                "PaddleOCR unavailable/failed (%s) -- falling back to pdfplumber grid. "
                "Install with: pip install paddlepaddle==3.2.2 'paddleocr>=3.0' 'paddlex[ocr]'",
                exc,
            )

        if docling_by_page or vl_by_page:
            merged = {
                p: self._ensure_page_tag(docling_by_page[p], p) for p in docling_by_page
            }
            sources = {p: BACKEND_DOCLING for p in docling_by_page}
            self._merge_vl_semantic(merged, sources, vl_by_page, out_dir)
            if merged:
                logger.warning(
                    "PaddleOCR fallback failed -- shipping %d merged page(s); "
                    "page(s) %s may remain missing.",
                    len(merged),
                    [p + 1 for p in missing_pages],
                )
                ordered = [merged[p] for p in sorted(merged)] + docling_unplaced
                ordered = self._maybe_rebind(pdf_path, ordered)
                return ordered, _backend_label(sources)

        logger.info("Table backend: trying pdfplumber ruled-line grid...")
        dataframes = _run_grid_fallback(pdf_path, out_dir, self.table_settings)
        if dataframes:
            logger.info("pdfplumber grid extracted %d table(s).", len(dataframes))
            tagged = []
            for i, df in enumerate(dataframes):
                page_guess = page_indices[i] if i < len(page_indices) else None
                if page_guess is not None:
                    tagged.append(self._ensure_page_tag(df, page_guess))
                else:
                    tagged.append(df)
            tagged = self._maybe_rebind(pdf_path, tagged)
            return tagged, BACKEND_GRID

        logger.error(
            "All image/grid table backends returned 0 tables on page(s) %s. "
            "Will fall back to Marker OCR tables (often garbled on scans).",
            [p + 1 for p in page_indices],
        )
        return [], BACKEND_GRID

    def run(
        self,
        pdf_path: str | Path,
        output: str | Path = DEFAULT_OUTPUT_TXT,
        *,
        output_filename: str = DEFAULT_OUTPUT_TXT,
        skip_tables: bool = False,
        skip_marker: Optional[bool] = None,
        force_marker: bool = False,
        docling_no_ocr: Optional[bool] = None,
        skip_paddle_vl: Optional[bool] = None,
        force_paddle_vl: bool = False,
        skip_docling: Optional[bool] = None,
        write_txt: bool = True,
        write_xlsx: bool = True,
        write_docx: bool = False,
    ) -> UnifiedResult:
        """
        Process one PDF.

        Default deliverables: `output.txt` + `output_tables.xlsx`.
        Set `write_docx=True` (or pass a `.docx` `-o` path) for Word output.
        """
        if skip_marker is not None:
            self.skip_marker = skip_marker
        if force_marker:
            self.force_marker = True
        if docling_no_ocr is not None:
            self.docling_no_ocr = bool(docling_no_ocr)
        if skip_paddle_vl is not None:
            self.skip_paddle_vl = skip_paddle_vl
        if force_paddle_vl:
            self.force_paddle_vl = True
        if skip_docling is not None:
            self.skip_docling = bool(skip_docling)

        pdf_path = Path(pdf_path)
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        requested = Path(output).expanduser()
        suffix = requested.suffix.lower()
        want_docx_primary = suffix == ".docx"

        if suffix == ".docx":
            docx_path = resolve_output_path(requested, default_name=DEFAULT_OUTPUT_DOCX)
            work_dir = docx_path.parent
            txt_path = work_dir / DEFAULT_OUTPUT_TXT
            write_docx = True
        elif suffix == ".txt":
            txt_path = resolve_output_path(requested, default_name=DEFAULT_OUTPUT_TXT)
            work_dir = txt_path.parent
            docx_path = work_dir / DEFAULT_OUTPUT_DOCX
        else:
            work_dir = requested
            work_dir.mkdir(parents=True, exist_ok=True)
            txt_path = work_dir / DEFAULT_OUTPUT_TXT
            docx_path = work_dir / DEFAULT_OUTPUT_DOCX

        xlsx_path = work_dir / DEFAULT_OUTPUT_XLSX

        marker_skipped = _skip_marker(skip_marker=self.skip_marker, force_marker=self.force_marker)
        page_regions = None
        prefer_extracted = False
        bbox_by_page: Optional[Dict[int, tuple[float, float, float, float]]] = None

        markdown = self._extract_markdown(pdf_path, work_dir)
        pages_with_tables = self._detect_table_pages(pdf_path)
        marker_table_count = sum(
            1 for b in MarkdownBlockParser().parse_blocks(markdown) if b.kind == "table"
        )
        use_page_regions = marker_skipped or marker_table_count == 0
        if use_page_regions:
            try:
                page_regions = extract_page_prose_regions(pdf_path)
                prefer_extracted = True
                bbox_by_page = {
                    r.page_index: r.table_bbox
                    for r in page_regions
                    if getattr(r, "table_bbox", None) is not None
                }
                logger.info(
                    "Assemble mode: page bbox regions (%d page(s)); prefer extracted grids.",
                    len(page_regions),
                )
            except Exception as exc:
                logger.warning(
                    "Page prose regions failed (%s) -- falling back to markdown assemble.",
                    exc,
                )
                page_regions = None
                prefer_extracted = marker_skipped

        dataframes: List[pd.DataFrame] = []
        backend = BACKEND_NONE
        used_marker_fallback = False

        if not pages_with_tables:
            logger.info("No tables detected -- skipping GPU table extraction.")
            self.marker.unload()
        elif skip_tables:
            logger.info("skip_tables=True -- using Marker's tables only.")
            self.marker.unload()
            used_marker_fallback = True
            backend = BACKEND_MARKER
        else:
            self.marker.unload()
            table_error: Optional[str] = None
            try:
                dataframes, backend = self._extract_tables(pdf_path, work_dir, pages_with_tables)
            except Exception as exc:
                table_error = str(exc)
                logger.error("Table extraction failed: %s", exc)
                dataframes, backend = [], BACKEND_FAILED

            if getattr(self, "rebind_tokens", False) and dataframes and bbox_by_page is not None:
                dataframes = self._maybe_rebind(pdf_path, dataframes, bbox_by_page=bbox_by_page)

            if not dataframes:
                used_marker_fallback = False
                backend = BACKEND_FAILED
                logger.error(
                    "Table pages %s were detected but every backend returned 0 tables. "
                    "Install/fix: pip install paddlepaddle==3.2.2 'paddleocr>=3.0' 'paddlex[ocr]'. "
                    "Detail: %s",
                    [p + 1 for p in pages_with_tables],
                    table_error or "empty result",
                )

        final_tables: List[pd.DataFrame] = []
        written_docx: Optional[Path] = None
        if write_docx or want_docx_primary:
            written_docx, final_tables = export_document_from_markdown(
                markdown,
                dataframes,
                docx_path,
                apply_ocr_cleanup=True,
                prefer_extracted=prefer_extracted,
                page_regions=page_regions,
            )
        else:
            from .exporters import assemble_document_blocks, assemble_document_by_page_regions

            if page_regions is not None:
                blocks = assemble_document_by_page_regions(
                    page_regions,
                    dataframes,
                    apply_ocr_cleanup=True,
                )
            else:
                blocks = assemble_document_blocks(
                    markdown,
                    dataframes,
                    apply_ocr_cleanup=True,
                    prefer_extracted=prefer_extracted,
                )
            final_tables = [
                b.content for b in blocks if b.kind == "table" and isinstance(b.content, pd.DataFrame)
            ]

        if dataframes and not final_tables:
            logger.warning(
                "Document assembler dropped %d extracted table(s); exporting them anyway.",
                len(dataframes),
            )
            final_tables = list(dataframes)

        if final_tables:
            from .outline_enrichment import enrich_tables_from_pdf
            from .table_layout import merge_outline_form_tables

            # Page-region assemble can leave outline fragments split by prose.
            final_tables = merge_outline_form_tables(final_tables)
            final_tables = enrich_tables_from_pdf(pdf_path, final_tables)
            final_tables = annotate_tables(final_tables)

        written_xlsx: Optional[Path] = None
        if write_xlsx:
            empty_reason = None
            if not final_tables and pages_with_tables and not skip_tables:
                empty_reason = (
                    "Table pages were detected but extraction backends returned 0 tables. "
                    "Check PaddleOCR install / logs (not a Marker OCR table)."
                )
            written = export_tables_preview(
                final_tables,
                xlsx_path,
                write_csv=True,
                write_markdown=False,
                empty_reason=empty_reason,
            )
            written_xlsx = written["xlsx"]

        written_txt: Optional[Path] = None
        if write_txt or not want_docx_primary:
            txt_content = compose_document_txt(
                markdown,
                dataframes,
                apply_ocr_cleanup=True,
                prefer_extracted=prefer_extracted,
                page_regions=page_regions,
                pdf_path=pdf_path,
            )
            written_txt = save_unified_txt(txt_content, txt_path)

        marker_table_count = sum(1 for _ in MarkdownBlockParser().parse_blocks(markdown) if _.kind == "table")
        table_count = len(final_tables) if final_tables else marker_table_count

        primary = written_docx if want_docx_primary and written_docx else (written_txt or written_docx or xlsx_path)
        return UnifiedResult(
            output_path=primary if primary is not None else txt_path,
            table_count=table_count,
            pages_with_tables=pages_with_tables,
            table_backend_used=backend,
            used_marker_table_fallback=used_marker_fallback,
            docx_path=written_docx,
            xlsx_path=written_xlsx,
            txt_path=written_txt,
            tables=list(final_tables),
        )


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Content-driven PDF pipeline -> output.txt + output_tables.xlsx."
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
        help="Also write output.docx with real Word tables.",
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Fast test: Marker only, skip Docling/Paddle/grid extraction.",
    )
    parser.add_argument(
        "--docling-no-ocr",
        action="store_true",
        help="Force Docling do_ocr=False (debug column-collapse without OCR anchors).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    from .logging_config import configure_app_logging

    args = _build_arg_parser().parse_args(argv)
    configure_app_logging(verbose=args.verbose)

    try:
        result = UnifiedPDFPipeline(docling_no_ocr=args.docling_no_ocr).run(
            args.input,
            output=args.output,
            skip_tables=args.skip_tables,
            write_docx=args.docx,
            docling_no_ocr=args.docling_no_ocr,
        )
    except Exception as exc:
        logging.error("Pipeline failed: %s", exc)
        return 1

    print("Done.")
    if result.txt_path:
        print(f"  Text:              {result.txt_path}")
    if result.xlsx_path:
        print(f"  Tables (xlsx):     {result.xlsx_path}")
    if result.docx_path:
        print(f"  DOCX:              {result.docx_path}")
    if result.pages_with_tables:
        print(f"  Table page(s):     {[p + 1 for p in result.pages_with_tables]}")
        print(f"  Backend:           {result.table_backend_used}")
        if result.used_marker_table_fallback:
            print("  WARNING: tables from Marker OCR only -- install/fix Docling or PaddleOCR.")
    else:
        print("  No tables detected in PDF structure scan.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
