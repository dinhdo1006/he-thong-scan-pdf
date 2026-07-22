"""
Run Marker / Paddle in isolated Python processes.

On Windows CPU hosts, importing both `torch` (Marker) and `paddle` in the
SAME process often breaks with WinError 127 / access violations because their
native DLLs collide. Marker.unload() does not unload torch DLLs, so the table
backend must start in a fresh interpreter.

Marker itself may also AV while loading Surya on some Windows CPU setups; the
markdown helper then falls back to PyMuPDF prose so the pipeline still ships.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

# Marker model load that hangs/AVs should fail fast so PyMuPDF fallback can run.
MARKER_TIMEOUT_S = 90
PADDLE_TIMEOUT_S = 1800


def _skip_marker(*, skip_marker: bool | None = None, force_marker: bool = False) -> bool:
    """
    Decide whether to bypass Marker (Surya/transformers).

    Marker import+load is slow/fragile on CPU (Linux hangs on transformers;
    Windows often AVs). Tables come from Paddle/VLM; prose uses PyMuPDF.

    Priority: explicit args > FORCE_MARKER / SKIP_MARKER env > default skip.
    """
    if force_marker:
        return False
    if skip_marker is True:
        return True
    if skip_marker is False:
        return False

    force_env = os.environ.get("FORCE_MARKER", "").strip().lower() in {"1", "true", "yes"}
    if force_env:
        return False
    explicit = os.environ.get("SKIP_MARKER", "").strip().lower()
    if explicit in {"1", "true", "yes"}:
        return True
    if explicit in {"0", "false", "no"}:
        return False
    # Default: skip Marker. Opt-in with FORCE_MARKER=1 or --force-marker.
    return True


def _run_worker(payload: str, *, label: str, timeout_s: int = 1800) -> None:
    """Execute a short Python worker script in a fresh interpreter."""
    logger.info("Starting isolated %s worker (timeout=%ss)...", label, timeout_s)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", payload],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Isolated {label} worker timed out after {timeout_s}s") from exc

    if completed.stdout.strip():
        logger.info("[%s stdout]\n%s", label, completed.stdout.strip()[-4000:])
    if completed.stderr.strip():
        logger.info("[%s stderr]\n%s", label, completed.stderr.strip()[-4000:])
    if completed.returncode != 0:
        raise RuntimeError(
            f"Isolated {label} worker failed (exit {completed.returncode}). "
            f"stderr tail: {completed.stderr.strip()[-1500:]}"
        )
    logger.info("Isolated %s worker finished OK.", label)


def extract_markdown_isolated(
    pdf_path: Path,
    work_dir: Path,
    *,
    skip_marker: bool | None = None,
    force_marker: bool = False,
) -> str:
    """
    Try Marker in a subprocess; on skip/crash/timeout use PyMuPDF prose fallback.

    Returns Markdown/plain text suitable for `compose_document_txt` (prose only
    when falling back -- tables come from Paddle).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    out_md = work_dir / ".tmp_marker_isolated.md"
    if out_md.exists():
        out_md.unlink()

    payload = f"""
from pathlib import Path
from pdf_extractor.logging_config import configure_app_logging
from pdf_extractor.marker_extractor import MarkerExtractor

configure_app_logging(verbose=False)
pdf = Path(r'''{pdf_path}''')
out = Path(r'''{out_md}''')
md = MarkerExtractor().extract_markdown(pdf)
out.write_text(md, encoding='utf-8')
print(f'MARKER_OK chars={{len(md)}}')
"""
    try:
        if _skip_marker(skip_marker=skip_marker, force_marker=force_marker):
            raise RuntimeError("SKIP_MARKER default: bypass Marker, use PyMuPDF prose")
        _run_worker(payload, label="marker", timeout_s=MARKER_TIMEOUT_S)
        if out_md.is_file():
            return out_md.read_text(encoding="utf-8")
        raise RuntimeError(f"Marker worker did not write {out_md}")
    except Exception as exc:
        logger.warning(
            "Marker isolated worker unavailable (%s) -- using PyMuPDF text fallback.",
            exc,
        )
        from .text_fallback import extract_plaintext_fallback, plaintext_as_markdown

        prose = extract_plaintext_fallback(pdf_path)
        md = plaintext_as_markdown(prose)
        out_md.write_text(md, encoding="utf-8")
        return md


def extract_paddle_tables_isolated(pdf_path: Path, work_dir: Path) -> List[pd.DataFrame]:
    """Run PaddleOCR PP-Structure in a subprocess; return table DataFrames."""
    work_dir.mkdir(parents=True, exist_ok=True)
    out_xlsx = work_dir / ".tmp_paddle_isolated.xlsx"
    if out_xlsx.exists():
        out_xlsx.unlink()

    payload = f"""
from pathlib import Path
from pdf_extractor.logging_config import configure_app_logging
from pdf_extractor.paddle_extractor import extract

configure_app_logging(verbose=False)
pdf = Path(r'''{pdf_path}''')
out = Path(r'''{out_xlsx}''')
dfs = extract(pdf, output_path=out, apply_ocr_cleanup=True)
print(f'PADDLE_OK tables={{len(dfs)}}')
"""
    _run_worker(payload, label="paddle", timeout_s=PADDLE_TIMEOUT_S)
    if not out_xlsx.is_file():
        return []

    frames: List[pd.DataFrame] = []
    xl = pd.ExcelFile(out_xlsx)
    for sheet in xl.sheet_names:
        df = pd.read_excel(out_xlsx, sheet_name=sheet, dtype=str).fillna("")
        if list(df.columns) == ["info"] and len(df) == 1 and "No tables" in str(df.iloc[0, 0]):
            continue
        if not df.empty:
            frames.append(df)
    return frames
