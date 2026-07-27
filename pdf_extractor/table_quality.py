"""
Generic table quality scoring and spatial cell assignment helpers.

Form-agnostic: scores any DataFrame grid and (Phase 4) rebinds OCR tokens
into cell boxes by coordinates so content stays in the correct (row, col).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
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


def score_table_quality(
    df: pd.DataFrame | None,
    *,
    return_breakdown: bool = False,
) -> float | tuple[float, dict]:
    """
    Return a 0..1 quality score for a table grid (higher is better).

    Heuristics are form-agnostic: they penalize garbled headers and reward
    a reasonable non-empty body fill rate. They do NOT assume a fixed
    column count or header template.

    Args:
        df: Table grid to score.
        return_breakdown: When True, also return a dict of score components
            for DEBUG logging (scoring math is unchanged).

    Returns:
        ``score`` (float), or ``(score, breakdown)`` when
        ``return_breakdown=True``.
    """
    breakdown: dict = {
        "early_exit": None,
        "base": 0.55,
        "n_rows": 0,
        "n_cols": 0,
        "empty_header_penalty": 0.0,
        "garbage_header_penalty": 0.0,
        "long_header_penalty": 0.0,
        "weird_short_header_penalty": 0.0,
        "underscore_soup_penalty": 0.0,
        "empty_body_penalty": 0.0,
        "fill_rate": 0.0,
        "fill_bonus": 0.0,
        "width_structure_bonus": 0.0,
        "raw_before_clamp": 0.0,
        "score": 0.0,
    }

    if df is None or len(getattr(df, "columns", [])) == 0:
        breakdown["early_exit"] = "none_or_zero_columns"
        breakdown["score"] = 0.0
        if return_breakdown:
            return 0.0, breakdown
        return 0.0

    score = 0.55
    cols = [_cell_text(c) for c in df.columns]
    breakdown["n_rows"] = int(df.shape[0])
    breakdown["n_cols"] = int(len(cols))

    empty_header_penalty = 0.0
    garbage_header_penalty = 0.0
    long_header_penalty = 0.0
    weird_short_header_penalty = 0.0
    underscore_soup_penalty = 0.0

    for col in cols:
        if not col:
            score -= 0.04
            empty_header_penalty -= 0.04
            continue
        if _GARBAGE_HEADER_RE.search(col) or col.count("|") >= 2:
            score -= 0.25
            garbage_header_penalty -= 0.25
        if len(col) > 70:
            score -= 0.18
            long_header_penalty -= 0.18
        elif len(col) > 40:
            score -= 0.08
            long_header_penalty -= 0.08
        if len(col) <= 2 and _WEIRD_SHORT_RE.match(col):
            score -= 0.06
            weird_short_header_penalty -= 0.06
        # Collapsed multi-header soup often joins levels with underscores.
        if col.count("_") >= 3 and len(col) > 25:
            score -= 0.12
            underscore_soup_penalty -= 0.12

    breakdown["empty_header_penalty"] = empty_header_penalty
    breakdown["garbage_header_penalty"] = garbage_header_penalty
    breakdown["long_header_penalty"] = long_header_penalty
    breakdown["weird_short_header_penalty"] = weird_short_header_penalty
    breakdown["underscore_soup_penalty"] = underscore_soup_penalty

    if df.empty:
        score -= 0.15
        breakdown["empty_body_penalty"] = -0.15
    else:
        total = int(df.shape[0] * df.shape[1])
        nonempty = sum(1 for v in df.to_numpy().ravel() if _cell_text(v))
        fill = nonempty / total if total else 0.0
        fill_bonus = 0.35 * fill
        width_bonus = min(0.08, 0.008 * len(cols))
        score += fill_bonus
        # Mild preference for having some structure, not for a magic width.
        score += width_bonus
        breakdown["fill_rate"] = fill
        breakdown["fill_bonus"] = fill_bonus
        breakdown["width_structure_bonus"] = width_bonus

    breakdown["raw_before_clamp"] = float(score)
    final = float(max(0.0, min(1.0, score)))
    breakdown["score"] = final
    if return_breakdown:
        return final, breakdown
    return final


def _geometric_pick(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_score: float,
    right_score: float,
) -> tuple[str, str]:
    """Width-then-score tie-break. Returns (winner, reason)."""
    left_cols = len(left.columns)
    right_cols = len(right.columns)
    gap = left_cols - right_cols

    if gap >= WIDTH_PREFER_GAP and left_score >= WIDTH_MIN_SCORE:
        return "left", (
            f"geometric width prefer left ({left_cols} vs {right_cols} cols, "
            f"score={left_score:.2f}>={WIDTH_MIN_SCORE})"
        )
    if -gap >= WIDTH_PREFER_GAP and right_score >= WIDTH_MIN_SCORE:
        return "right", (
            f"geometric width prefer right ({right_cols} vs {left_cols} cols, "
            f"score={right_score:.2f}>={WIDTH_MIN_SCORE})"
        )
    if right_score > left_score:
        return "right", f"geometric score right {right_score:.2f} > left {left_score:.2f}"
    return "left", f"geometric score left {left_score:.2f} >= right {right_score:.2f}"


def _shared_semantic_conflicts(left_report: dict, right_report: dict) -> list[dict]:
    """
    STT rows where BOTH candidates failed the same kind of check.

    These must not be auto-resolved; they go to a debug CSV.
    """
    left_by = left_report.get("by_stt") or {}
    right_by = right_report.get("by_stt") or {}
    conflicts: list[dict] = []
    for stt in sorted(set(left_by) & set(right_by)):
        if not stt:
            continue
        lrow = left_by[stt]
        rrow = right_by[stt]
        if lrow.get("stt_valid") is False and rrow.get("stt_valid") is False:
            conflicts.append(
                {
                    "stt": stt,
                    "kind": "stt",
                    "left_issue": lrow.get("stt_issue"),
                    "right_issue": rrow.get("stt_issue"),
                    "left_value": dict(lrow),
                    "right_value": dict(rrow),
                    "flag": "CONFLICT_NEEDS_MANUAL_CHECK",
                }
            )
        if lrow.get("sum_valid") is False and rrow.get("sum_valid") is False:
            conflicts.append(
                {
                    "stt": stt,
                    "kind": "sum",
                    "left_issue": lrow.get("sum_issue"),
                    "right_issue": rrow.get("sum_issue"),
                    "left_sum_diff": lrow.get("sum_diff"),
                    "right_sum_diff": rrow.get("sum_diff"),
                    "left_value": dict(lrow),
                    "right_value": dict(rrow),
                    "flag": "CONFLICT_NEEDS_MANUAL_CHECK",
                }
            )
    return conflicts


def write_conflict_csv(
    conflicts: list[dict],
    conflict_dir: str | Path,
    *,
    page_no: int | None = None,
    left_label: str = "left",
    right_label: str = "right",
) -> Path | None:
    """Persist shared semantic conflicts for manual review. Returns written path."""
    if not conflicts:
        return None
    out_dir = Path(conflict_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    page_tag = f"page{page_no + 1}" if page_no is not None else "pageNA"
    path = out_dir / f"conflict_{page_tag}_{left_label}_vs_{right_label}.csv"

    flat_rows: list[dict] = []
    for c in conflicts:
        flat_rows.append(
            {
                "page": (page_no + 1) if page_no is not None else "",
                "stt": c.get("stt"),
                "kind": c.get("kind"),
                f"{left_label}_issue": c.get("left_issue"),
                f"{right_label}_issue": c.get("right_issue"),
                f"{left_label}_sum_diff": c.get("left_sum_diff", ""),
                f"{right_label}_sum_diff": c.get("right_sum_diff", ""),
                f"{left_label}_row": str(c.get("left_value")),
                f"{right_label}_row": str(c.get("right_value")),
                "flag": c.get("flag") or "CONFLICT_NEEDS_MANUAL_CHECK",
            }
        )
    pd.DataFrame(flat_rows).to_csv(path, index=False, encoding="utf-8-sig")
    logger.warning(
        "Wrote %d CONFLICT_NEEDS_MANUAL_CHECK row(s) -> %s",
        len(flat_rows),
        path,
    )
    return path


def pick_better_table(
    left: pd.DataFrame | None,
    right: pd.DataFrame | None,
    *,
    use_semantic: bool = True,
    left_label: str = "left",
    right_label: str = "right",
    conflict_dir: str | Path | None = None,
    page_no: int | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Choose the better table between two backends for the same page.

    Rules (form-agnostic):
    1. Only one side present -> that side wins.
    2. **Semantic (primary):** run outline/sum validators on BOTH candidates;
       fewer ``stt_valid=False`` / ``sum_valid=False`` events wins, even if the
       geometric score is lower.
    3. Shared failures on the same STT -> write debug CSV tagged
       ``CONFLICT_NEEDS_MANUAL_CHECK`` (both values kept); do not treat that
       row as a semantic win for either side.
    4. **Geometric (secondary):** width gap prefer, then ``score_table_quality``.

    Returns:
        ``(dataframe, winner)`` where winner is ``"left"`` or ``"right"``.
    """
    if left is None and right is None:
        raise ValueError("pick_better_table() needs at least one DataFrame")
    if left is None:
        logger.debug("pick_better_table: only %s present -> right", right_label)
        return right, "right"  # type: ignore[return-value]
    if right is None:
        logger.debug("pick_better_table: only %s present -> left", left_label)
        return left, "left"

    left_score = float(score_table_quality(left))
    right_score = float(score_table_quality(right))
    reason = ""
    winner = "left"

    if use_semantic:
        # Local import avoids a hard cycle at module load time.
        from .semantic_validator import evaluate_table_semantics

        left_rep = evaluate_table_semantics(left, emit_warnings=False)
        right_rep = evaluate_table_semantics(right, emit_warnings=False)
        left_fails = int(left_rep["fail_count"])
        right_fails = int(right_rep["fail_count"])
        conflicts = _shared_semantic_conflicts(left_rep, right_rep)

        logger.debug(
            "pick_better_table semantic: %s fails=%d (stt=%d sum=%d), "
            "%s fails=%d (stt=%d sum=%d), shared_conflicts=%d",
            left_label,
            left_fails,
            left_rep["stt_fail_count"],
            left_rep["sum_fail_count"],
            right_label,
            right_fails,
            right_rep["stt_fail_count"],
            right_rep["sum_fail_count"],
            len(conflicts),
        )

        if conflicts:
            if conflict_dir is not None:
                write_conflict_csv(
                    conflicts,
                    conflict_dir,
                    page_no=page_no,
                    left_label=left_label,
                    right_label=right_label,
                )
            else:
                logger.warning(
                    "Semantic CONFLICT_NEEDS_MANUAL_CHECK on %d STT row(s) "
                    "(no conflict_dir provided): %s",
                    len(conflicts),
                    [c.get("stt") for c in conflicts],
                )

        if left_fails < right_fails:
            winner = "left"
            reason = (
                f"semantic prefer {left_label}: fails {left_fails} < {right_fails} "
                f"(geom scores L={left_score:.2f} R={right_score:.2f})"
            )
        elif right_fails < left_fails:
            winner = "right"
            reason = (
                f"semantic prefer {right_label}: fails {right_fails} < {left_fails} "
                f"(geom scores L={left_score:.2f} R={right_score:.2f})"
            )
        else:
            # Equal semantic fail counts (including both-zero, or tied with conflicts).
            # Do not invent a semantic winner for shared conflict rows — fall back
            # to geometric for shipping a whole-page grid.
            winner, geo_reason = _geometric_pick(left, right, left_score, right_score)
            reason = (
                f"semantic tied (fails={left_fails}"
                f"{', shared_conflicts=' + str(len(conflicts)) if conflicts else ''}); "
                f"fallback {geo_reason}"
            )
    else:
        winner, reason = _geometric_pick(left, right, left_score, right_score)

    logger.info(
        "pick_better_table -> %s (%s). reason: %s",
        left_label if winner == "left" else right_label,
        winner,
        reason,
    )
    if winner == "right":
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


def uniform_cell_boxes(
    table_bbox: tuple[float, float, float, float],
    n_rows: int,
    n_cols: int,
) -> list[TableCellBox]:
    """Build a regular row×col cell grid inside ``table_bbox`` (PDF points)."""
    if n_rows <= 0 or n_cols <= 0:
        return []
    x0, y0, x1, y1 = table_bbox
    width = max(1e-6, x1 - x0)
    height = max(1e-6, y1 - y0)
    cell_w = width / n_cols
    cell_h = height / n_rows
    cells: list[TableCellBox] = []
    for r in range(n_rows):
        for c in range(n_cols):
            cells.append(
                TableCellBox(
                    row=r,
                    col=c,
                    bbox=BBox(
                        x0 + c * cell_w,
                        y0 + r * cell_h,
                        x0 + (c + 1) * cell_w,
                        y0 + (r + 1) * cell_h,
                    ),
                )
            )
    return cells


def extract_page_text_tokens(
    pdf_path: str | Path,
    page_index: int,
    clip: tuple[float, float, float, float] | None = None,
) -> list[TextToken]:
    """Collect PyMuPDF word tokens (with bboxes) for one page, optional clip."""
    import fitz

    pdf_path = Path(pdf_path)
    tokens: list[TextToken] = []
    doc = fitz.open(pdf_path)
    try:
        if page_index < 0 or page_index >= doc.page_count:
            return []
        page = doc[page_index]
        words = page.get_text("words")  # x0, y0, x1, y1, word, block, line, word_no
        clip_rect = fitz.Rect(*clip) if clip is not None else None
        for w in words:
            x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], str(w[4] or "")
            if not text.strip():
                continue
            if clip_rect is not None and not clip_rect.intersects(fitz.Rect(x0, y0, x1, y1)):
                continue
            tokens.append(TextToken(text=text, bbox=BBox(x0, y0, x1, y1)))
    finally:
        doc.close()
    return tokens


def rebind_table_tokens_to_grid(
    pdf_path: str | Path,
    page_index: int,
    structure_df: pd.DataFrame,
    table_bbox: tuple[float, float, float, float] | None = None,
    *,
    include_header_row: bool = True,
) -> pd.DataFrame | None:
    """
    Wave 1.3: rebuild cell text by assigning page tokens into a uniform grid.

    Structure (row/col counts + headers) comes from ``structure_df`` (Docling /
    Paddle / grid). Token positions come from PyMuPDF. Returns None when the
    page has no usable tokens or bbox cannot be resolved.
    """
    from .grid_table_extractor import find_table_bbox

    if structure_df is None or structure_df.empty or len(structure_df.columns) == 0:
        return None

    pdf_path = Path(pdf_path)
    bbox = table_bbox
    if bbox is None:
        try:
            bbox = find_table_bbox(pdf_path, page_index)
        except Exception as exc:
            logger.debug("rebind: bbox probe failed page %d: %s", page_index + 1, exc)
            bbox = None
    if bbox is None:
        return None

    n_cols = len(structure_df.columns)
    n_body = len(structure_df)
    n_rows = n_body + (1 if include_header_row else 0)
    cells = uniform_cell_boxes(bbox, n_rows, n_cols)
    tokens = extract_page_text_tokens(pdf_path, page_index, clip=bbox)
    if not tokens:
        logger.debug("rebind: no tokens on page %d inside table bbox.", page_index + 1)
        return None

    grid = assign_tokens_to_cells(tokens, cells, n_rows=n_rows, n_cols=n_cols)
    if not grid:
        return None

    if include_header_row and len(grid) >= 1:
        # Keep backend headers when token header row is empty/noisy.
        header_cells = [c.strip() for c in grid[0]]
        if sum(1 for c in header_cells if c) < max(1, n_cols // 3):
            rebound = grid_to_dataframe(grid[1:], headers=[str(c) for c in structure_df.columns])
        else:
            rebound = grid_to_dataframe(grid)
    else:
        rebound = grid_to_dataframe(grid, headers=[str(c) for c in structure_df.columns])

    if rebound.empty:
        return None
    rebound.attrs.update(getattr(structure_df, "attrs", {}))
    rebound.attrs["page_no"] = page_index
    rebound.attrs["rebound_tokens"] = True
    logger.info(
        "Token→cell rebind page %d: structure %s -> rebound %s (%d tokens).",
        page_index + 1,
        structure_df.shape,
        rebound.shape,
        len(tokens),
    )
    return rebound
