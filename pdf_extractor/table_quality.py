"""
Generic table quality scoring and spatial cell assignment helpers.

Form-agnostic: scores any DataFrame grid and (Phase 4) rebinds OCR tokens
into cell boxes by coordinates so content stays in the correct (row, col).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# Below this score, Docling's page table is considered weak enough that
# PaddleOCR should be run for a quality comparison (generic, not form-locked).
LOW_QUALITY_THRESHOLD = 0.42

# When two backends disagree on width by this many columns or more, prefer
# the wider grid so multi-page stitch can keep column alignment. A near-zero
# wider table (total collapse) can still lose to a cleaner narrower one.
WIDTH_PREFER_GAP = 2
WIDTH_MIN_SCORE = 0.25

_GARBAGE_HEADER_RE = re.compile(r"\|_|_{2,}|\|{2,}")
_WEIRD_SHORT_RE = re.compile(r"^[^\w\dÀ-ỹ]+$", re.IGNORECASE)


def _cell_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def score_table_quality(df: pd.DataFrame | None) -> float:
    """
    Return a 0..1 quality score for a table grid (higher is better).

    Heuristics are form-agnostic: they penalize garbled headers and reward
    a reasonable non-empty body fill rate. They do NOT assume a fixed
    column count or header template.
    """
    if df is None or len(getattr(df, "columns", [])) == 0:
        return 0.0

    score = 0.55
    cols = [_cell_text(c) for c in df.columns]

    for col in cols:
        if not col:
            score -= 0.04
            continue
        if _GARBAGE_HEADER_RE.search(col) or col.count("|") >= 2:
            score -= 0.25
        if len(col) > 70:
            score -= 0.18
        elif len(col) > 40:
            score -= 0.08
        if len(col) <= 2 and _WEIRD_SHORT_RE.match(col):
            score -= 0.06
        # Collapsed multi-header soup often joins levels with underscores.
        if col.count("_") >= 3 and len(col) > 25:
            score -= 0.12

    if df.empty:
        score -= 0.15
    else:
        total = int(df.shape[0] * df.shape[1])
        nonempty = sum(1 for v in df.to_numpy().ravel() if _cell_text(v))
        fill = nonempty / total if total else 0.0
        score += 0.35 * fill
        # Mild preference for having some structure, not for a magic width.
        score += min(0.08, 0.008 * len(cols))

    return float(max(0.0, min(1.0, score)))


def pick_better_table(
    left: pd.DataFrame | None,
    right: pd.DataFrame | None,
) -> tuple[pd.DataFrame, str]:
    """
    Choose the better table between two backends for the same page.

    Rules (form-agnostic):
    1. Only one side present -> that side wins.
    2. If widths differ by ``WIDTH_PREFER_GAP`` or more, prefer the **wider**
       grid when it is not near-collapsed (score >= ``WIDTH_MIN_SCORE``).
       This prevents a clean 3-column fragment from beating a usable 9-column
       table and breaking later stitch/merge.
    3. Otherwise pick the higher ``score_table_quality``.

    Returns:
        ``(dataframe, winner)`` where winner is ``"left"`` or ``"right"``.
    """
    if left is None and right is None:
        raise ValueError("pick_better_table() needs at least one DataFrame")
    if left is None:
        return right, "right"  # type: ignore[return-value]
    if right is None:
        return left, "left"

    left_cols = len(left.columns)
    right_cols = len(right.columns)
    left_score = score_table_quality(left)
    right_score = score_table_quality(right)
    gap = left_cols - right_cols

    if gap >= WIDTH_PREFER_GAP and left_score >= WIDTH_MIN_SCORE:
        return left, "left"
    if -gap >= WIDTH_PREFER_GAP and right_score >= WIDTH_MIN_SCORE:
        return right, "right"

    if right_score > left_score:
        return right, "right"
    return left, "left"


# ---------------------------------------------------------------------------
# Phase 4 foundation: coordinate / bbox re-bind
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BBox:
    """Axis-aligned box in page/image coordinates (any consistent unit)."""

    x0: float
    y0: float
    x1: float
    y1: float

    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    def contains_point(self, x: float, y: float, *, pad: float = 0.0) -> bool:
        return (
            (self.x0 - pad) <= x <= (self.x1 + pad)
            and (self.y0 - pad) <= y <= (self.y1 + pad)
        )

    def iou(self, other: "BBox") -> float:
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)
        iw = max(0.0, ix1 - ix0)
        ih = max(0.0, iy1 - iy0)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)
        area_b = max(0.0, other.x1 - other.x0) * max(0.0, other.y1 - other.y0)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class TextToken:
    """OCR word or PDF character run with a bounding box."""

    text: str
    bbox: BBox


@dataclass(frozen=True)
class TableCellBox:
    """One physical table cell (structure layer)."""

    row: int
    col: int
    bbox: BBox
    row_span: int = 1
    col_span: int = 1


def assign_tokens_to_cells(
    tokens: Sequence[TextToken],
    cells: Sequence[TableCellBox],
    *,
    n_rows: int | None = None,
    n_cols: int | None = None,
) -> list[list[str]]:
    """
    Build a dense string grid by assigning each token to the best cell.

    Assignment rule (generic):
    1. Prefer the cell whose bbox contains the token center.
    2. Else pick the cell with the highest IoU against the token bbox.
    3. Tokens with no match are skipped (logged at debug).

    Spanned slots: text is written only at the anchor ``(row, col)`` of the
    matching cell; covered span slots stay empty so the physical grid does
    not collapse.
    """
    if not cells:
        return []

    max_r = max(c.row + max(1, c.row_span) - 1 for c in cells)
    max_c = max(c.col + max(1, c.col_span) - 1 for c in cells)
    rows = n_rows if n_rows is not None else max_r + 1
    cols = n_cols if n_cols is not None else max_c + 1
    grid: list[list[str]] = [["" for _ in range(cols)] for _ in range(rows)]

    for token in tokens:
        text = (token.text or "").strip()
        if not text:
            continue
        cx, cy = token.bbox.center()
        chosen: TableCellBox | None = None
        for cell in cells:
            if cell.bbox.contains_point(cx, cy):
                chosen = cell
                break
        if chosen is None:
            best_iou = 0.0
            for cell in cells:
                iou = cell.bbox.iou(token.bbox)
                if iou > best_iou:
                    best_iou = iou
                    chosen = cell
            if best_iou <= 0:
                logger.debug("Token %r unmatched to any cell bbox.", text[:40])
                continue
        assert chosen is not None
        r, c = chosen.row, chosen.col
        if 0 <= r < rows and 0 <= c < cols:
            if grid[r][c]:
                grid[r][c] = f"{grid[r][c]} {text}".strip()
            else:
                grid[r][c] = text
    return grid


def grid_to_dataframe(
    grid: Sequence[Sequence[str]],
    headers: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Convert a dense string grid into a DataFrame (optional header row)."""
    if not grid:
        return pd.DataFrame()
    width = max((len(row) for row in grid), default=0)
    if width == 0:
        return pd.DataFrame()

    if headers is not None:
        cols = [str(h) for h in headers]
        while len(cols) < width:
            cols.append(f"col_{len(cols)}")
        cols = cols[:width]
        body = [list(row) + [""] * (width - len(row)) for row in grid]
        body = [row[:width] for row in body]
        return pd.DataFrame(body, columns=cols)

    # First row as header when it looks non-empty; else synthetic names.
    first = list(grid[0]) + [""] * (width - len(grid[0]))
    first = first[:width]
    if any(cell.strip() for cell in first):
        cols = [c.strip() or f"col_{i}" for i, c in enumerate(first)]
        body = grid[1:]
    else:
        cols = [f"col_{i}" for i in range(width)]
        body = grid
    normalized = [list(row) + [""] * (width - len(row)) for row in body]
    normalized = [row[:width] for row in normalized]
    return pd.DataFrame(normalized, columns=cols)
