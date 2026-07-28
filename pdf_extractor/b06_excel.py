"""
Excel layout for B06/CD_02 forms: multi-row parent/child headers + row hierarchy.

Horizontal (header):
  Row1 group title ``Trong đó Chấp hành viên...`` merges over cols 2–5
  Row2 sub-labels Ủy thác / Trả đơn / Đình chỉ / Miễn, giảm
  Row3 column codes A B 1 2 3 4 5 6 7

Vertical (body):
  Bold parent outline rows; Excel indent by outline depth (no Cap column).
  Semantic helper columns (Cấp / STT_valid / Sum_check / STT_gap) are never written.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Sequence

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, Side
from openpyxl.utils import get_column_letter

from .semantic_validator import outline_level
from .table_layout import (
    _B06_CANONICAL_HEADERS,
    is_outline_code,
    looks_like_b06_cd02_table,
    normalize_outline_token,
)

logger = logging.getLogger(__name__)

_HELPER_COLS = {"cấp", "cap", "stt_valid", "sum_check", "stt_gap"}

_B06_TOP = [
    "Số TT",
    "Tiêu chí trong quyết định thi hành án",
    "Tổng số tiền, giá trị tài sản phải thi hành",
    "Trong đó Chấp hành viên đã giải quyết bằng các biện pháp",
    "",
    "",
    "",
    "Số thực thu thi hành án\n(bao gồm cả số giao nhận tháng)",
    "Số đã nộp Nhà nước và chi trả đương sự\n(bao gồm cả số giao nhận tháng)",
]
_B06_MID = [
    "",
    "",
    "",
    "Ủy thác THA",
    "Trả đơn THA",
    "Đình chỉ THA",
    "Miễn, giảm THA",
    "",
    "",
]
_B06_CODE = ["A", "B", "1", "2", "3", "4", "5", "6", "7"]

_THIN = Border(
    left=Side(style="thin"),
    right=Side(style="thin"),
    top=Side(style="thin"),
    bottom=Side(style="thin"),
)
_HEADER_FONT = Font(bold=True, size=10)
_PARENT_FONT = Font(bold=True, size=11)
_BODY_FONT = Font(size=10)
_CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)

# Light label cleanups seen on B06 scans (display-only; OCR map still applies upstream).
_LABEL_FIXES = (
    (re.compile(r"^số phải thu", re.I), "Số phải thu"),
    (re.compile(r"khân\s+cấp", re.I), "khẩn cấp"),
    (re.compile(r"tiêu hủy", re.I), "tiêu huỷ"),
)


def drop_semantic_helpers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove Cap / validator helper columns from a table destined for export."""
    if df is None or df.empty:
        return df
    drop = [c for c in df.columns if str(c).lower().strip() in _HELPER_COLS]
    out = df.drop(columns=drop, errors="ignore").copy()
    # Strip space-indent that annotate may have injected into the description col.
    if len(out.columns) >= 2:
        desc = out.columns[1]
        out[desc] = [
            _clean_label(v) for v in out[desc].tolist()
        ]
    return out


def _clean_label(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = " ".join(str(value).replace("\n", " ").split())
    for pat, repl in _LABEL_FIXES:
        text = pat.sub(repl, text)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def is_b06_excel_candidate(df: pd.DataFrame) -> bool:
    """True when the frame carries the B06 9-column schema (possibly + helpers)."""
    if df is None or df.empty:
        return False
    cols = [str(c) for c in df.columns]
    form_cols = [c for c in cols if c.lower().strip() not in _HELPER_COLS]
    if len(form_cols) < 9:
        return False
    joined = " | ".join(form_cols).lower()
    hits = sum(
        1
        for key in ("số tt", "tiêu chí", "ủy thác", "trả đơn", "đình chỉ", "miễn")
        if key in joined
    )
    if hits >= 4:
        return True
    slim = df[form_cols[:9]].copy()
    if len(slim.columns) == 9:
        slim.columns = list(_B06_CANONICAL_HEADERS)
        return looks_like_b06_cd02_table(slim)
    return False


def write_b06_workbook(tables: Sequence[pd.DataFrame], path: str | Path) -> Path:
    """Write one or more B06 tables with hierarchical headers into ``path``."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    default = wb.active
    wb.remove(default)

    written = 0
    for idx, df in enumerate(tables, start=1):
        if not is_b06_excel_candidate(df):
            ws = wb.create_sheet(title=f"Table_{idx}"[:31])
            _write_flat_sheet(ws, drop_semantic_helpers(df))
        else:
            ws = wb.create_sheet(title=f"B06_Table_{idx}"[:31])
            _write_b06_sheet(ws, df)
            written += 1
    if not wb.sheetnames:
        ws = wb.create_sheet("No_Tables")
        ws["A1"] = "No tables found in document."
    wb.save(path)
    logger.info("Wrote B06-aware Excel (%d hierarchical sheet(s)) -> %s", written, path)
    return path


def _form_body(df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c) for c in df.columns]
    form_cols: list[str] = []
    for want in _B06_CANONICAL_HEADERS:
        if want in cols:
            form_cols.append(want)
    if len(form_cols) < 9:
        form_cols = [c for c in cols if c.lower().strip() not in _HELPER_COLS][:9]
    body = df[form_cols].copy()
    if len(body.columns) == 9:
        body.columns = list(_B06_CANONICAL_HEADERS)
    # Clean labels in-place for display.
    if len(body.columns) >= 2:
        desc = body.columns[1]
        body[desc] = [_clean_label(v) for v in body[desc].tolist()]
    return body


def _write_b06_sheet(ws, df: pd.DataFrame) -> None:
    body = _form_body(df)
    n_form = len(body.columns)
    header_rows = 3

    for i, text in enumerate(_B06_TOP[:n_form]):
        ws.cell(1, i + 1, text)
    for i, text in enumerate(_B06_MID[:n_form]):
        ws.cell(2, i + 1, text)
    for i, text in enumerate(_B06_CODE[:n_form]):
        ws.cell(3, i + 1, text)

    def _merge(rr1: int, cc1: int, rr2: int, cc2: int) -> None:
        ws.merge_cells(start_row=rr1, start_column=cc1, end_row=rr2, end_column=cc2)

    for col in (1, 2, 3):
        if n_form >= col:
            _merge(1, col, 2, col)
    if n_form >= 7:
        _merge(1, 4, 1, 7)
    for col in (8, 9):
        if n_form >= col:
            _merge(1, col, 2, col)

    for r in range(1, header_rows + 1):
        for c in range(1, n_form + 1):
            cell = ws.cell(r, c)
            cell.font = _HEADER_FONT
            cell.alignment = _CENTER
            cell.border = _THIN

    for row_i in range(len(body)):
        excel_r = header_rows + 1 + row_i
        stt = normalize_outline_token(body.iloc[row_i, 0]) if n_form else ""
        depth = outline_level(stt)
        is_parent = is_outline_code(stt) and "." not in stt
        for c in range(n_form):
            val = body.iloc[row_i, c]
            if val is None or (isinstance(val, float) and pd.isna(val)):
                text = ""
            else:
                text = str(val)
            cell = ws.cell(excel_r, c + 1, text)
            cell.border = _THIN
            cell.font = _PARENT_FONT if is_parent else _BODY_FONT
            if c == 0:
                cell.alignment = _CENTER
            elif c == 1:
                # Visual parent/child without a Cap column.
                indent = max(0, min(depth - 1, 4))
                cell.alignment = Alignment(
                    horizontal="left", vertical="center", wrap_text=True, indent=indent
                )
            else:
                cell.alignment = Alignment(
                    horizontal="right" if text else "center",
                    vertical="center",
                    wrap_text=True,
                )

    widths = [10, 44, 14, 12, 12, 12, 12, 14, 16]
    for i, w in enumerate(widths[:n_form], start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.row_dimensions[1].height = 36
    ws.row_dimensions[2].height = 22
    ws.row_dimensions[3].height = 18
    ws.freeze_panes = "A4"


def _write_flat_sheet(ws, df: pd.DataFrame) -> None:
    """Fallback: plain header + body (non-B06 tables)."""
    for c, name in enumerate(df.columns, start=1):
        cell = ws.cell(1, c, str(name))
        cell.font = _HEADER_FONT
        cell.border = _THIN
        cell.alignment = _CENTER
    for r in range(len(df)):
        for c in range(len(df.columns)):
            val = df.iloc[r, c]
            text = "" if val is None or (isinstance(val, float) and pd.isna(val)) else str(val)
            cell = ws.cell(r + 2, c + 1, text)
            cell.border = _THIN
            cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
