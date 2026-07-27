"""
Post-process extracted form tables to fix common layout mistakes.

Generic (no form-id / B06 template): repairs cell shifts around outline codes
(I, 1, 1.1, 1.1.2), demotes data rows wrongly used as headers on continuation
pages, and stitches multi-page continuation tables.
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# Roman numerals or dotted outline codes used as row identifiers on forms.
_OUTLINE_CODE_RE = re.compile(
    r"^(?:[IVXLCDM]+|\d+(?:\.\d+)*)$",
    re.IGNORECASE,
)
# Vietnamese / EU style money: 10.620.000 or 12.50 (exactly 2 decimals).
# Do NOT match 1.1 / 1.3 — those are outline codes.
_MONEY_LIKE_RE = re.compile(
    r"^-?\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?$|^-?\d+[.,]\d{2}$"
)
_GENERIC_HEADER_RE = re.compile(r"^(?:col(?:_\d+)?|Column_\d+)$", re.IGNORECASE)
_COLUMN_CODE_RE = re.compile(r"^(?:[A-Za-z]|[0-9]{1,2})$")


def _clean(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ")
    return " ".join(text.split())


def normalize_outline_token(text: object) -> str:
    """
    Normalize OCR-mangled outline codes before matching.

    Examples: ``1⁄2`` / ``1/2`` -> ``1.2`` (only when both sides are digits).
    """
    s = _clean(text)
    if not s:
        return ""
    # Unicode fraction slash / ASCII slash between digit groups -> dots.
    s = s.replace("⁄", ".").replace("∕", ".")
    if re.fullmatch(r"\d+(?:[./]\d+)+", s):
        s = s.replace("/", ".")
    return s


def is_money_like(text: object) -> bool:
    s = _clean(text)
    return bool(s) and bool(_MONEY_LIKE_RE.fullmatch(s))


def is_outline_code(text: object) -> bool:
    """True for Roman/dotted outline ids — never for money or dates."""
    s = normalize_outline_token(text)
    if not s or is_money_like(s):
        return False
    # Dates like 17.6.2010 / 12.10.2011
    if re.search(r"(?:^|\.)\d{4}(?:\.|$)", s):
        return False
    if not _OUTLINE_CODE_RE.fullmatch(s):
        return False
    # Bare day numbers from "ngày 12 tháng ..." — keep 1-9 only as root codes.
    if re.fullmatch(r"\d{2,}", s):
        return False
    return True


_GROUP_HEADER_RE = re.compile(
    r"trong\s*đ[oó]|chấp\s*hành\s*viên|đã\s*giải\s*quyết|bao\s*gồm|"
    r"trong\s*do|chap\s*hanh\s*vien",
    re.IGNORECASE,
)

# Official B06/CD_02 data columns (A B 1..7).
# "Trong đó Chấp hành viên đã giải quyết..." is a GROUP title over cols 2–5,
# not a data column of its own.
_B06_CANONICAL_HEADERS = [
    "Số TT A",
    "Tiêu chí trong quyết định thi hành án B",
    "Tổng số tiền, giá trị tài sản phải thi hành 1",
    "Ủy thác THA 2",
    "Trả đơn THA 3",
    "Đình chỉ THA 4",
    "Miễn, giảm THA 5",
    "Số thực thu thi hành án 6",
    "Số đã nộp Nhà nước và chi trả 7",
]


def is_group_header_label(text: object) -> bool:
    """True for spanning group titles (not a single data-column name)."""
    s = _clean(text)
    if len(s) < 12:
        return False
    return bool(_GROUP_HEADER_RE.search(s))


def is_generic_header(name: object) -> bool:
    return bool(_GENERIC_HEADER_RE.fullmatch(_clean(name)))


def _headers_are_generic(columns: Sequence[object]) -> bool:
    cols = [_clean(c) for c in columns]
    return bool(cols) and all(is_generic_header(c) for c in cols)


def _looks_like_column_code_row(cells: Sequence[str]) -> bool:
    """True for rows like A | B | 1 | 2 | 3 | 4 | 5 | 6 | 7."""
    non_empty = [c.strip() for c in cells if c.strip()]
    if len(non_empty) < 3:
        return False
    hits = sum(1 for c in non_empty if _COLUMN_CODE_RE.fullmatch(c))
    return hits / len(non_empty) >= 0.8


def _looks_like_header_label_row(cells: Sequence[str]) -> bool:
    """
    Leading body row that is still header text (no outline code / money in col0).
    Typical: Ủy thác THA | Trả đơn THA | Đình chỉ THA | ...
    """
    if not cells:
        return False
    first = cells[0].strip()
    if is_outline_code(first) or is_money_like(first):
        return False
    non_empty = [c.strip() for c in cells if c.strip()]
    if len(non_empty) < 2:
        return False
    if any(is_money_like(c) for c in non_empty):
        return False
    if any(is_outline_code(c) for c in non_empty):
        return False
    textish = sum(1 for c in non_empty if len(c) >= 3)
    return textish >= 2


def _matrix_from_df(df: pd.DataFrame) -> tuple[list[str], list[list[str]]]:
    cols = [_clean(c) for c in df.columns]
    body = [[_clean(df.iloc[r, c]) for c in range(len(df.columns))] for r in range(len(df))]
    return cols, body


def _df_from_matrix(headers: list[str], body: list[list[str]]) -> pd.DataFrame:
    width = len(headers)
    normalized: list[list[str]] = []
    for row in body:
        padded = list(row) + [""] * max(0, width - len(row))
        normalized.append(padded[:width])
    return pd.DataFrame(normalized, columns=headers)


def demote_false_data_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Continuation pages often promote the first data row to DataFrame columns
    (e.g. headers = ['1.3.2', 'Thu tài sản...', 'Column_3', ...]).

    Detect and push that row back into the body with generic column names.
    Synthetic Column_* header cells become empty body cells (not literal text).
    """
    if df is None or len(df.columns) == 0:
        return df

    cols = [_clean(c) for c in df.columns]
    if not is_outline_code(cols[0]):
        return df

    # Second header cell should look like a description, not another code / money.
    second = cols[1] if len(cols) > 1 else ""
    if not second or is_outline_code(second) or is_money_like(second):
        return df
    if len(second) < 6:
        return df

    # Remaining headers mostly generic/empty → strong signal of false promotion.
    tail = cols[2:]
    if tail:
        generic_or_empty = sum(1 for c in tail if not c or is_generic_header(c))
        if generic_or_empty / len(tail) < 0.5:
            return df

    first_row = ["" if is_generic_header(c) else c for c in cols]
    body = [first_row] + [[_clean(df.iloc[r, c]) for c in range(len(cols))] for r in range(len(df))]
    new_headers = [f"Column_{i + 1}" for i in range(len(cols))]
    logger.info(
        "Demoted false data-header row starting with outline code %r.",
        cols[0],
    )
    return _df_from_matrix(new_headers, body)


def left_compact_sparse_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    If the first header cells are blank but later cells hold labels
    (só TT / Tiêu chí shifted right), slide header text left to fill gaps.
    Body cells are left untouched.
    """
    if df is None or len(df.columns) == 0:
        return df

    cols = [_clean(c) for c in df.columns]
    if cols[0] or _headers_are_generic(cols):
        return df

    non_empty = [c for c in cols if c]
    if len(non_empty) < 2:
        return df

    # Only compact when a clear run of leading blanks exists.
    leading_blanks = 0
    for c in cols:
        if c:
            break
        leading_blanks += 1
    if leading_blanks < 1:
        return df

    compacted = non_empty + [""] * (len(cols) - len(non_empty))
    if compacted == cols:
        return df

    out = df.copy()
    out.columns = compacted
    logger.info("Left-compacted sparse table headers by %d slot(s).", leading_blanks)
    return out


def absorb_leading_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fold leading body rows that are still header labels / A-B-1-2 codes into
    the column names so real data starts at the first outline/money row.

    Label rows (Ủy thác / Trả đơn / ...) are packed into *empty* header slots
    after left-compacting a sparse primary header — this recovers the common
    multi-row form header without smashing labels onto STT/Tiêu chí.
    """
    if df is None or df.empty or len(df.columns) == 0:
        return df

    cols, body = _matrix_from_df(df)
    label_rows: list[list[str]] = []
    code_rows: list[list[str]] = []
    while body:
        row = body[0]
        if _looks_like_column_code_row(row):
            code_rows.append(row)
            body = body[1:]
            continue
        if _looks_like_header_label_row(row):
            label_rows.append(row)
            body = body[1:]
            continue
        break

    if not label_rows and not code_rows:
        return df

    headers = list(cols)

    # Left-compact sparse primary headers before placing labels into empties.
    if headers and not headers[0].strip() and not _headers_are_generic(headers):
        non_empty = [c for c in headers if c.strip()]
        if len(non_empty) >= 2:
            headers = non_empty + [""] * (len(headers) - len(non_empty))

    for row in label_rows:
        headers = _merge_header_label_row(headers, row)

    for row in code_rows:
        for i, cell in enumerate(row):
            if i >= len(headers) or not cell.strip():
                continue
            code = cell.strip()
            cur = headers[i].strip()
            if not cur or is_generic_header(cur):
                headers[i] = code
            elif code not in cur.split():
                headers[i] = f"{cur} {code}"

    headers = [
        h if h.strip() else f"Column_{i + 1}" for i, h in enumerate(headers)
    ]
    logger.info(
        "Absorbed %d label row(s) + %d column-code row(s) into headers.",
        len(label_rows),
        len(code_rows),
    )
    return _df_from_matrix(headers, body)


def _merge_header_label_row(headers: list[str], label_row: list[str]) -> list[str]:
    """
    Merge one sub-header label row into ``headers``.

    Prefer *positional* placement (label cell i → header i) so Ủy thác stays
    under column 2. Fall back to packing into empty/group slots when the OCR
    row is left-clustered while empties sit further right.
    """
    out = list(headers)
    width = len(out)
    row = list(label_row) + [""] * max(0, width - len(label_row))
    row = row[:width]

    non_empty_idxs = [i for i, c in enumerate(row) if c.strip()]
    if not non_empty_idxs:
        return out

    def _slot_open(i: int) -> bool:
        cur = out[i].strip()
        return (not cur) or is_generic_header(cur) or is_group_header_label(cur)

    positional_hits = sum(1 for i in non_empty_idxs if _slot_open(i))
    use_positional = positional_hits / len(non_empty_idxs) >= 0.5

    if use_positional:
        for i in non_empty_idxs:
            cell = row[i].strip()
            cur = out[i].strip()
            if not cur or is_generic_header(cur) or is_group_header_label(cur):
                out[i] = cell
            elif cell not in cur:
                # Prefer concrete sub-label over keeping only the group title.
                if is_group_header_label(cur) and not is_group_header_label(cell):
                    out[i] = cell
                else:
                    out[i] = f"{cur} {cell}"
        return out

    # Fallback: pack non-empty tokens into open slots (legacy sparse headers).
    label_tokens = [row[i].strip() for i in non_empty_idxs]
    empty_idxs = [i for i in range(width) if _slot_open(i)]
    for token, idx in zip(label_tokens, empty_idxs):
        out[idx] = token
    return out


def realign_shifted_outline_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix two common OCR/VLM cell-shift patterns:

    1) Money leaked into STT:  [250.000 | 1.1.2 | Mô tả | ...]
       -> [1.1.2 | Mô tả | ... | 250.000 in *rightmost* empty amount slot]

    2) STT shifted right:      [     | 1.1.3 | Mô tả | ...]
       -> [1.1.3 | Mô tả | ...]
    """
    if df is None or df.empty or len(df.columns) < 2:
        return df

    cols, body = _matrix_from_df(df)
    width = len(cols)
    fixed = 0
    new_body: list[list[str]] = []

    for row in body:
        cells = list(row) + [""] * max(0, width - len(row))
        cells = cells[:width]
        c0, c1 = cells[0].strip(), cells[1].strip()

        # Case 1: money in col0, outline in col1
        if is_money_like(c0) and is_outline_code(c1):
            money = c0
            outline = normalize_outline_token(c1)
            rest = cells[2:]
            rebuilt = [outline] + rest
            while len(rebuilt) < width:
                rebuilt.append("")
            rebuilt = rebuilt[:width]
            # Prefer RIGHTMOST empty amount slot: leaked values are usually
            # wrap from cols 6–7 of a missing previous row — putting them in
            # the first amount column (Tổng) invents false totals (e.g. 1.1.2).
            placed = False
            for i in range(width - 1, 1, -1):
                if not rebuilt[i].strip():
                    rebuilt[i] = money
                    placed = True
                    break
            if not placed and width >= 3:
                rebuilt[width - 1] = money
            new_body.append(rebuilt)
            fixed += 1
            continue

        # Case 2: blank col0, outline in col1, description-ish in col2
        if (
            not c0
            and is_outline_code(c1)
            and width >= 3
            and cells[2].strip()
            and not is_outline_code(cells[2])
            and not is_money_like(cells[2])
        ):
            rebuilt = cells[1:] + [""]
            new_body.append(rebuilt[:width])
            fixed += 1
            continue

        # Case 3: outline in col0, money leaked into description (col1),
        # real label sits in col2 — push money into first amount slot.
        if (
            is_outline_code(c0)
            and is_money_like(c1)
            and width >= 3
            and cells[2].strip()
            and not is_outline_code(cells[2])
            and not is_money_like(cells[2])
        ):
            money = c1
            label = cells[2].strip()
            rest = cells[3:]
            rebuilt = [normalize_outline_token(c0), label] + rest
            while len(rebuilt) < width:
                rebuilt.append("")
            rebuilt = rebuilt[:width]
            placed = False
            for i in range(2, width):
                if not rebuilt[i].strip():
                    rebuilt[i] = money
                    placed = True
                    break
            if not placed:
                rebuilt[width - 1] = money
            new_body.append(rebuilt)
            fixed += 1
            continue

        # Normalize outline token in col0 when already well-placed.
        if is_outline_code(c0):
            norm = normalize_outline_token(c0)
            if norm != c0:
                cells = [norm] + cells[1:]
                fixed += 1

        new_body.append(cells)

    if fixed:
        logger.info("Realigned %d shifted outline row(s).", fixed)
    return _df_from_matrix(cols, new_body)


def drop_empty_spacer_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop body rows where every cell is blank (common OCR phantom lines)."""
    if df is None or df.empty:
        return df
    cols, body = _matrix_from_df(df)
    kept = [row for row in body if any(c.strip() for c in row)]
    dropped = len(body) - len(kept)
    if dropped:
        logger.info("Dropped %d empty spacer row(s).", dropped)
    return _df_from_matrix(cols, kept)


def looks_like_b06_cd02_table(df: pd.DataFrame) -> bool:
    """
    Heuristic for the B06/CD_02 9-column enforcement form.

    We only need a coarse detector: width 9 plus strong header/body hints.
    """
    if df is None or len(getattr(df, "columns", [])) != 9:
        return False

    cols, body = _matrix_from_df(df)
    haystack = " | ".join(cols + [cell for row in body[:6] for cell in row]).lower()
    hits = 0
    if "số tt" in haystack or "so tt" in haystack:
        hits += 1
    if "tiêu chí" in haystack or "tieu chi" in haystack:
        hits += 1
    if "ủy thác" in haystack or "uy thac" in haystack:
        hits += 1
    if "trả đơn" in haystack or "tra don" in haystack:
        hits += 1
    if "đình chỉ" in haystack or "dinh chi" in haystack:
        hits += 1
    if "miễn" in haystack or "mien" in haystack:
        hits += 1

    # Also allow strong structural cues even if OCR text is noisy.
    first_col_hits = sum(1 for row in body[:8] if row and is_outline_code(row[0]))
    return hits >= 3 or first_col_hits >= 3


def canonicalize_b06_cd02_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Force the known B06/CD_02 9-column header order when detected."""
    if not looks_like_b06_cd02_table(df):
        return df
    if list(df.columns) == _B06_CANONICAL_HEADERS:
        return df
    out = df.copy()
    out.columns = _B06_CANONICAL_HEADERS
    logger.info("Canonicalized B06/CD_02 headers to the standard 9-column schema.")
    return out


def _stt_cell(row: Sequence[object], stt_idx: int = 0) -> str:
    if stt_idx >= len(row):
        return ""
    return normalize_outline_token(row[stt_idx])


def merge_b06_cd02_tables(tables: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
    Force-merge all detected B06/CD_02 tables into one grid.

    Page-region assemble often inserts prose between page-1 and page-2 tables,
    which blocks consecutive-table stitching. Deduplicate by STT, keeping the
    richer row when the same code appears twice.
    """
    if len(tables) <= 1:
        return tables

    b06_idxs = [i for i, df in enumerate(tables) if looks_like_b06_cd02_table(df)]
    if len(b06_idxs) <= 1:
        return tables

    repaired = [canonicalize_b06_cd02_headers(repair_form_table(tables[i])) for i in b06_idxs]
    width = len(_B06_CANONICAL_HEADERS)
    by_stt: dict[str, list[str]] = {}
    order: list[str] = []
    extras: list[list[str]] = []  # rows without outline codes (footer notes)

    def _richer(a: list[str], b: list[str]) -> list[str]:
        score_a = sum(1 for c in a if c.strip())
        score_b = sum(1 for c in b if c.strip())
        if score_b > score_a:
            # Prefer non-empty cells from the richer row, fill gaps from the other.
            out = list(b)
            for i, cell in enumerate(a):
                if i < len(out) and not out[i].strip() and cell.strip():
                    out[i] = cell
            return out
        out = list(a)
        for i, cell in enumerate(b):
            if i < len(out) and not out[i].strip() and cell.strip():
                out[i] = cell
        return out

    for df in repaired:
        _, body = _matrix_from_df(df)
        for row in body:
            padded = list(row) + [""] * max(0, width - len(row))
            padded = padded[:width]
            code = _stt_cell(padded, 0)
            if is_outline_code(code):
                if code not in by_stt:
                    by_stt[code] = padded
                    order.append(code)
                else:
                    by_stt[code] = _richer(by_stt[code], padded)
            elif any(c.strip() for c in padded):
                extras.append(padded)

    merged_body = [by_stt[c] for c in order] + extras
    merged = _df_from_matrix(list(_B06_CANONICAL_HEADERS), merged_body)
    # Preserve attrs from the first B06 table.
    merged.attrs.update(getattr(tables[b06_idxs[0]], "attrs", {}))
    logger.info(
        "Merged %d B06/CD_02 tables into 1 (%d unique STT rows).",
        len(b06_idxs),
        len(order),
    )

    out: list[pd.DataFrame] = []
    consumed = set(b06_idxs)
    placed = False
    for i, df in enumerate(tables):
        if i in consumed:
            if not placed:
                out.append(merged)
                placed = True
            continue
        out.append(df)
    if not placed:
        out.append(merged)
    return out


def repair_form_table(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full generic layout repair pipeline on one table."""
    if df is None or len(df.columns) == 0:
        return df
    out = demote_false_data_headers(df)
    # absorb left-compacts when label rows exist; left_compact is a no-op
    # afterward if col0 is already filled.
    out = absorb_leading_header_rows(out)
    out = left_compact_sparse_headers(out)
    out = realign_shifted_outline_rows(out)
    out = drop_empty_spacer_rows(out)
    out = canonicalize_b06_cd02_headers(out)
    return out


def _first_data_outline(df: pd.DataFrame) -> str | None:
    cols, body = _matrix_from_df(df)
    # Also check demoted-style headers.
    if is_outline_code(cols[0]):
        return cols[0]
    for row in body:
        if row and is_outline_code(row[0]):
            return row[0]
        if len(row) > 1 and not row[0].strip() and is_outline_code(row[1]):
            return row[1]
    return None


def _last_data_outline(df: pd.DataFrame) -> str | None:
    _, body = _matrix_from_df(df)
    for row in reversed(body):
        if row and is_outline_code(row[0]):
            return row[0]
        if len(row) > 1 and not row[0].strip() and is_outline_code(row[1]):
            return row[1]
    return None


def _outline_sort_key(code: str) -> tuple:
    """Rough order key so 1.3.1 < 1.3.2 < 2 < II."""
    s = code.strip().upper()
    if _OUTLINE_CODE_RE.fullmatch(s) and s[0].isdigit():
        parts = tuple(int(p) for p in s.split("."))
        return (0, parts)
    # Romans / letters after numeric outlines — keep stable but allow stitch.
    return (1, s)


def looks_like_outline_continuation(prev: pd.DataFrame, nxt: pd.DataFrame) -> bool:
    """
    True when nxt likely continues prev's outline form across a page break.

    Requires same width (or nxt width after demote), an outline code on both
    sides, and nxt starting at a code that does not jump *before* prev's last.
    """
    if prev is None or nxt is None:
        return False
    if len(prev.columns) == 0 or len(nxt.columns) == 0:
        return False
    if len(prev.columns) != len(nxt.columns):
        return False

    last = _last_data_outline(prev)
    first = _first_data_outline(nxt)
    if not last or not first:
        return False

    # Same code twice is suspicious (duplicate page) — still allow if nxt
    # headers are generic / demoted-looking; caller may choose.
    try:
        if _outline_sort_key(first) < _outline_sort_key(last):
            return False
    except Exception:
        return False

    # Prefer when nxt looks headerless / generic.
    nxt_cols = [_clean(c) for c in nxt.columns]
    if is_outline_code(nxt_cols[0]) or _headers_are_generic(nxt_cols):
        return True
    # Or when a large share of nxt headers are generic placeholders.
    generic_share = sum(1 for c in nxt_cols if is_generic_header(c) or not c) / len(nxt_cols)
    return generic_share >= 0.4


def concat_continuation_tables(prev: pd.DataFrame, nxt: pd.DataFrame) -> pd.DataFrame:
    """Concatenate nxt under prev, keeping prev headers."""
    left = repair_form_table(prev)
    right = repair_form_table(nxt)
    # Align width to left.
    width = len(left.columns)
    cols, body = _matrix_from_df(right)
    if is_outline_code(cols[0]) and not _headers_are_generic(cols):
        # Still had data-as-header; demote already handled in repair, but if
        # headers are real outline leftovers, push into body again.
        right = demote_false_data_headers(right)
        cols, body = _matrix_from_df(right)

    aligned_rows: list[list[str]] = []
    for row in body:
        padded = list(row) + [""] * max(0, width - len(row))
        aligned_rows.append(padded[:width])

    cont = pd.DataFrame(aligned_rows, columns=list(left.columns))
    return pd.concat([left, cont], ignore_index=True)


def stitch_outline_continuation_tables(tables: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
    Merge consecutive tables that look like multi-page outline-form continuations.
    Safe no-op when widths differ or outline order goes backwards.
    Always force-merge B06/CD_02 fragments into one table when detected.
    """
    if len(tables) <= 1:
        return merge_b06_cd02_tables(tables)

    stitched: list[pd.DataFrame] = [repair_form_table(tables[0])]
    for nxt in tables[1:]:
        nxt_fixed = repair_form_table(nxt)
        prev = stitched[-1]
        if looks_like_outline_continuation(prev, nxt_fixed):
            stitched[-1] = concat_continuation_tables(prev, nxt_fixed)
            logger.info(
                "Stitched outline-continuation tables (%d + %d rows -> %d).",
                len(prev),
                len(nxt_fixed),
                len(stitched[-1]),
            )
        else:
            stitched.append(nxt_fixed)
    return merge_b06_cd02_tables(stitched)