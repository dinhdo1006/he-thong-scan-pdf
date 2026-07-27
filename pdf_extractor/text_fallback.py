"""Lightweight PDF text extraction without Marker/torch.

Used when Marker crashes (common on Windows CPU with heavy Surya models)
so the pipeline can still emit prose + Paddle tables.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_PAGE_BREAK_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class PageProseRegion:
    """Prose above/below a table bbox on one PDF page (0-indexed)."""

    page_index: int
    above: str
    below: str
    table_bbox: tuple[float, float, float, float] | None = None


def extract_page_prose_regions(pdf_path: str | Path) -> list[PageProseRegion]:
    """
    Per-page prose split around the densest table bbox.

    Used by Wave 1 assemble: emit above → table → below in reading order
    instead of concatenating footer into the header then appending tables.
    """
    import fitz

    from .grid_table_extractor import find_table_bbox

    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    regions: list[PageProseRegion] = []
    doc = fitz.open(pdf_path)
    try:
        for page_index, page in enumerate(doc):
            page_rect = page.rect
            table_box = None
            try:
                table_box = find_table_bbox(pdf_path, page_index)
            except Exception as exc:
                logger.debug("table bbox probe failed on page %d: %s", page_index + 1, exc)

            above = ""
            below = ""
            if table_box is None:
                above = _header_lines_only(page.get_text("text") or "")
            else:
                x0, y0, x1, y1 = table_box
                if y0 > page_rect.y0 + 8:
                    clip_above = fitz.Rect(
                        page_rect.x0, page_rect.y0, page_rect.x1, max(y0 - 2, page_rect.y0)
                    )
                    if clip_above.height >= 4:
                        above = _header_lines_only(page.get_text("text", clip=clip_above) or "")
                if y1 < page_rect.y1 - 8:
                    clip_below = fitz.Rect(
                        page_rect.x0, min(y1 + 2, page_rect.y1), page_rect.x1, page_rect.y1
                    )
                    if clip_below.height >= 4:
                        below = _clean_footer(page.get_text("text", clip=clip_below) or "")

            regions.append(
                PageProseRegion(
                    page_index=page_index,
                    above=(above or "").strip(),
                    below=(below or "").strip(),
                    table_bbox=table_box,
                )
            )
    finally:
        doc.close()

    logger.info(
        "Page prose regions: %d page(s) from %s (with_bbox=%d).",
        len(regions),
        pdf_path.name,
        sum(1 for r in regions if r.table_bbox is not None),
    )
    return regions


def extract_plaintext_fallback(pdf_path: str | Path) -> str:
    """
    Extract prose with PyMuPDF, preferring text *outside* detected table boxes.

    Avoids dumping the PDF's embedded (often garbled) table OCR into the
    final document when Paddle already supplies structured tables.

    Note: for document assembly when Marker is skipped, prefer
    ``extract_page_prose_regions`` so footers stay *after* tables.
    """
    regions = extract_page_prose_regions(pdf_path)
    parts: list[str] = []
    for region in regions:
        page_chunks = [c for c in (region.above, region.below) if c]
        text = "\n\n".join(page_chunks).strip()
        if text:
            parts.append(text)
            logger.debug("Fallback text page %d: %d chars.", region.page_index + 1, len(text))

    body = "\n\n".join(parts)
    body = _PAGE_BREAK_RE.sub("\n\n", body).strip()
    body = _finalize_prose(body)
    logger.info("Fallback PyMuPDF text extraction: %d chars from %s", len(body), Path(pdf_path).name)
    return body


_MONEY_RE = re.compile(r"\d{1,3}(?:\.\d{3}){2,}")
_OUTLINE_RE = re.compile(r"^(?:[IVX]+|\d+(?:[./⁄]\d+)*)(?:\s|$)")
_PIPE_RE = re.compile(r"\|")
_NAME_LIKE_RE = re.compile(r"[A-Za-zÀ-ỹĂÂÊÔƠƯăâêôơưĐđ]{3,}")


def _is_tableish_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _MONEY_RE.search(s) or _OUTLINE_RE.match(s) or _PIPE_RE.search(s):
        return True
    if re.fullmatch(r"[A-G1-7]", s):
        return True
    return False


def _header_lines_only(text: str, max_lines: int = 24) -> str:
    """Keep leading prose lines; stop at money/outline/pipe table rows."""
    kept: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if kept:
                kept.append("")
            continue
        if _is_tableish_line(line):
            break
        kept.append(line)
        if len([x for x in kept if x]) >= max_lines:
            break
    return "\n".join(kept).strip()


def _clean_footer(text: str) -> str:
    """Keep signature / stamp lines; drop table rows and OCR junk."""
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or _is_tableish_line(line):
            continue
        letters = "".join(_NAME_LIKE_RE.findall(line))
        if len(letters) < 6:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _finalize_prose(text: str) -> str:
    """Global pass: header prose + trailing signature-like lines only."""
    header = _header_lines_only(text)
    footer_rev: list[str] = []
    for raw in reversed(text.splitlines()):
        line = raw.strip()
        if not line:
            if footer_rev:
                break
            continue
        if _is_tableish_line(line):
            break
        letters = "".join(_NAME_LIKE_RE.findall(line))
        if len(letters) < 6:
            continue
        footer_rev.append(line)
        if len(footer_rev) >= 5:
            break
    footer = "\n".join(reversed(footer_rev)).strip()
    parts = [p for p in (header, footer) if p]
    if len(parts) == 2 and parts[1] in parts[0]:
        return parts[0]
    return "\n\n".join(parts)


def plaintext_as_markdown(text: str) -> str:
    """Wrap plain prose so MarkdownBlockParser treats it as text-only (no tables)."""
    return (text or "").strip()
