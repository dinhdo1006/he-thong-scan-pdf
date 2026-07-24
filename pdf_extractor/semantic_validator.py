"""
Form-agnostic semantic checks for hierarchical outline tables.

Validates STT outline sequencing (I / 1 / 1.1 / 1.1.2) and parent/child
amount consistency for VND-style numbers. No model / template dependency.

Roman section codes (I, II, …) and arabic dotted codes (1, 1.1, …) are
tracked as separate sequences so ``I`` then ``1`` is legal on VN forms.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, Literal, Optional, Sequence

import pandas as pd

from .table_layout import is_outline_code

logger = logging.getLogger(__name__)

OutlineKind = Literal["roman", "arabic"]
# Path: (kind, *levels) e.g. ("roman", 1) or ("arabic", 1, 1, 2)
OutlinePath = tuple

_ROMAN_VALUES: dict[str, int] = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
    "XI": 11,
    "XII": 12,
    "XIII": 13,
    "XIV": 14,
    "XV": 15,
    "XVI": 16,
    "XVII": 17,
    "XVIII": 18,
    "XIX": 19,
    "XX": 20,
}
_VALUE_TO_ROMAN = {v: k for k, v in _ROMAN_VALUES.items()}

_ROMAN_RE = re.compile(r"^[IVXLCDM]+$", re.IGNORECASE)
_DOTTED_RE = re.compile(r"^\d+(?:\.\d+)*$")
_VND_GROUPED_RE = re.compile(r"^\d{1,3}([.,]\d{3})+$")
_VND_DECIMAL_RE = re.compile(r"^\d+[.,]\d{1,2}$")

ABS_TOLERANCE_VND = 1000.0
REL_TOLERANCE = 0.001  # 0.1%

_STT_HEADER_HINTS = re.compile(
    r"stt|số\s*tt|so\s*tt|^\s*a\s*$|số\s*thứ",
    re.IGNORECASE,
)


def parse_stt_path(stt: object) -> Optional[OutlinePath]:
    """
    Parse an outline code into a comparable path.

    Examples:
        ``"I"`` -> ``("roman", 1)``
        ``"1.1.2"`` -> ``("arabic", 1, 1, 2)``
    """
    if stt is None or (isinstance(stt, float) and pd.isna(stt)):
        return None
    text = str(stt).replace("\n", " ").strip()
    if not text:
        return None
    if _ROMAN_RE.fullmatch(text):
        value = _ROMAN_VALUES.get(text.upper())
        return ("roman", value) if value is not None else None
    if _DOTTED_RE.fullmatch(text):
        parts = tuple(int(p) for p in text.split("."))
        return ("arabic", *parts)
    return None


def format_stt_path(path: OutlinePath) -> str:
    """Format a path back to outline text."""
    if not path:
        return ""
    kind = path[0]
    levels = path[1:]
    if kind == "roman" and len(levels) == 1:
        return _VALUE_TO_ROMAN.get(levels[0], str(levels[0]))
    return ".".join(str(p) for p in levels)


def _numeric_levels(path: OutlinePath) -> tuple[int, ...]:
    return path[1:]


def _is_valid_successor(prev: OutlinePath, curr: OutlinePath) -> bool:
    """
    True if ``curr`` is a legal next step after ``prev`` within the same kind.

    Allowed:
      - first child: levels(prev) + (1,)
      - next sibling / ancestor sibling: increment exactly one component
    """
    if prev[0] != curr[0]:
        return False
    p = _numeric_levels(prev)
    c = _numeric_levels(curr)
    if c == p + (1,):
        return True
    for depth in range(len(p), 0, -1):
        ancestor = p[:depth]
        expected = ancestor[:-1] + (ancestor[-1] + 1,)
        if c == expected:
            return True
    return False


def _expected_next_sibling(prev: OutlinePath) -> OutlinePath:
    """Immediate next sibling at the deepest level (e.g. 1.1.5 -> 1.1.6)."""
    levels = _numeric_levels(prev)
    nxt = levels[:-1] + (levels[-1] + 1,)
    return (prev[0], *nxt)


def _is_plausible_first(path: OutlinePath) -> bool:
    """First code in a kind should start at 1 (I or 1 / 1.x rooted at 1)."""
    levels = _numeric_levels(path)
    return bool(levels) and levels[0] == 1


def validate_stt_outline(rows: list[dict]) -> list[dict]:
    """
    Validate hierarchical STT outline sequencing in reading order.

    Each input row must already contain ``\"stt\"``. Returns copies with
    ``stt_valid`` (bool) and ``stt_issue`` (str|None). Blank / non-outline
    STT cells are skipped for sequence purposes (marked valid).

    Roman and arabic codes are validated on separate chains so a form that
    goes ``I`` then ``1`` then ``1.1`` is accepted.
    """
    out: list[dict] = []
    prev_by_kind: dict[OutlineKind, OutlinePath] = {}
    seen: set[OutlinePath] = set()

    for raw in rows:
        row = dict(raw)
        stt_raw = row.get("stt", "")
        path = parse_stt_path(stt_raw)

        if path is None:
            row["stt_valid"] = True
            row["stt_issue"] = None
            out.append(row)
            continue

        kind: OutlineKind = path[0]  # type: ignore[assignment]
        levels = _numeric_levels(path)
        issue: Optional[str] = None

        # Parent (same kind) must already have appeared for depth > 1.
        if len(levels) > 1:
            parent: OutlinePath = (kind, *levels[:-1])
            if parent not in seen:
                issue = (
                    f"parent {format_stt_path(parent)} not yet seen before "
                    f"{format_stt_path(path)}"
                )

        prev = prev_by_kind.get(kind)
        if issue is None:
            if prev is None:
                if not _is_plausible_first(path) and len(levels) == 1:
                    # Allow starting mid-document at a root other than 1 only
                    # with a soft note? Spec: generic syntax — flag skip of 1.
                    if levels[0] != 1:
                        issue = (
                            f"expected {format_stt_path((kind, 1))} before "
                            f"{format_stt_path(path)}, missing"
                        )
                elif len(levels) > 1 and levels != (1,) + (1,) * (len(levels) - 1):
                    # First arabic entry should not jump straight to 1.1.5 etc.
                    # without parents — already covered by parent-not-seen.
                    pass
            elif not _is_valid_successor(prev, path):
                p_levels = _numeric_levels(prev)
                c_levels = levels
                # Prefer a precise "missing N" hint:
                # - child gap under prev (1.1 -> 1.1.2) => expected 1.1.1
                # - sibling gap (1.1.5 -> 1.1.7) => expected 1.1.6
                if len(c_levels) == len(p_levels) + 1 and c_levels[:-1] == p_levels:
                    expected: OutlinePath = (kind, *p_levels, 1)
                elif len(c_levels) == len(p_levels) and c_levels[:-1] == p_levels[:-1]:
                    expected = (kind, *p_levels[:-1], p_levels[-1] + 1)
                else:
                    expected = _expected_next_sibling(prev)
                issue = (
                    f"expected {format_stt_path(expected)} before "
                    f"{format_stt_path(path)}, missing"
                )

        row["stt_valid"] = issue is None
        row["stt_issue"] = issue
        out.append(row)

        seen.add(path)
        for depth in range(1, len(levels)):
            seen.add((kind, *levels[:depth]))
        prev_by_kind[kind] = path

    return out


def parse_vnd_amount(value: object) -> float:
    """
    Parse a VND-style cell to float.

    Empty / non-numeric -> 0. Parentheses or leading ``-`` -> negative.
    Thousand separators ``.`` or ``,`` are stripped for grouped values.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    text = str(value).replace("\n", " ").replace("\xa0", " ").strip()
    if not text or text.lower() in {"nan", "none", "-"}:
        return 0.0

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1].strip()
    if text.startswith("-"):
        negative = True
        text = text[1:].strip()
    text = text.replace(" ", "")
    if not text:
        return 0.0

    if _VND_GROUPED_RE.fullmatch(text):
        amount = float(re.sub(r"[.,]", "", text))
        return -amount if negative else amount

    if _VND_DECIMAL_RE.fullmatch(text):
        amount = float(text.replace(",", "."))
        return -amount if negative else amount

    if re.fullmatch(r"\d+", text):
        amount = float(text)
        return -amount if negative else amount

    digits = re.sub(r"[^\d]", "", text)
    if not digits:
        return 0.0
    amount = float(digits)
    return -amount if negative else amount


def _amounts_match(parent_val: float, children_sum: float) -> tuple[bool, float]:
    diff = children_sum - parent_val
    if abs(diff) < ABS_TOLERANCE_VND:
        return True, diff
    denom = max(abs(parent_val), 1.0)
    if abs(diff) / denom < REL_TOLERANCE:
        return True, diff
    return False, diff


def _is_direct_child(parent_stt: str, child_stt: str) -> bool:
    """True when child is exactly one outline level deeper under parent (same kind)."""
    parent_path = parse_stt_path(parent_stt)
    child_path = parse_stt_path(child_stt)
    if parent_path is None or child_path is None:
        return False
    if parent_path[0] != child_path[0]:
        return False
    p = _numeric_levels(parent_path)
    c = _numeric_levels(child_path)
    return len(c) == len(p) + 1 and c[:-1] == p


def validate_sum_consistency(
    rows: list[dict],
    amount_columns: list[str],
) -> list[dict]:
    """
    Compare each parent outline row's amounts to the sum of direct children.

    Adds ``sum_valid`` (bool) and ``sum_diff`` (float|None). Rows that are not
    parents (no direct children) get ``sum_valid=True`` and ``sum_diff=None``.
    Optional ``is_total=True`` still requires ≥1 direct child to apply.
    """
    prepared = [dict(r) for r in rows]
    stt_list = [str(r.get("stt", "") or "").strip() for r in prepared]

    for idx, row in enumerate(prepared):
        stt = stt_list[idx]
        path = parse_stt_path(stt)
        force_parent = bool(row.get("is_total"))

        child_indices = [
            j
            for j, other in enumerate(stt_list)
            if j != idx and _is_direct_child(stt, other)
        ]

        if path is None or (not child_indices and not force_parent):
            row["sum_valid"] = True
            row["sum_diff"] = None
            continue

        if not child_indices:
            row["sum_valid"] = True
            row["sum_diff"] = None
            continue

        worst_diff = 0.0
        all_ok = True
        for col in amount_columns:
            parent_val = parse_vnd_amount(row.get(col))
            children_sum = sum(
                parse_vnd_amount(prepared[j].get(col)) for j in child_indices
            )
            ok, diff = _amounts_match(parent_val, children_sum)
            if abs(diff) >= abs(worst_diff):
                worst_diff = diff
            if not ok:
                all_ok = False

        row["sum_valid"] = all_ok
        row["sum_diff"] = float(worst_diff)
        if not all_ok:
            row["sum_issue"] = (
                f"children sum differs from parent {stt} by {worst_diff:g} "
                f"on columns {amount_columns}"
            )
        else:
            row.pop("sum_issue", None)

    return prepared


def _clean_header(name: object) -> str:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    return " ".join(str(name).replace("\n", " ").split())


def detect_stt_column(df: pd.DataFrame) -> Optional[str]:
    """Pick the column most likely to hold outline STT codes."""
    if df is None or df.empty or len(df.columns) == 0:
        return None

    best_col: Optional[str] = None
    best_hits = -1
    for col in df.columns:
        name = _clean_header(col)
        hits = sum(
            1
            for v in df[col]
            if is_outline_code(v) or parse_stt_path(v) is not None
        )
        header_boost = 2 if _STT_HEADER_HINTS.search(name) or name.lower() in {"a", "stt"} else 0
        score = hits + header_boost
        if score > best_hits:
            best_hits = score
            best_col = str(col)
    if best_hits <= 0:
        return str(df.columns[0])
    return best_col


def detect_amount_columns(df: pd.DataFrame, stt_column: str) -> list[str]:
    """Heuristic: columns (other than STT) that look money-like in ≥1 cell."""
    cols: list[str] = []
    for col in df.columns:
        if str(col) == stt_column:
            continue
        money_hits = 0
        for v in df[col].tolist():
            text = _clean_header(v)
            if not text:
                continue
            compact = text.replace(" ", "")
            if _VND_GROUPED_RE.fullmatch(compact.lstrip("-").strip("()")):
                money_hits += 1
            elif re.fullmatch(r"-?\d+", compact):
                money_hits += 1
            elif parse_vnd_amount(text) != 0.0 and re.search(r"\d", text):
                money_hits += 1
        if money_hits >= 1:
            cols.append(str(col))
    return cols


def dataframe_to_rows(df: pd.DataFrame, stt_column: str) -> list[dict]:
    """Convert a DataFrame to list[dict] with a normalized ``stt`` field."""
    rows: list[dict] = []
    for _, series in df.iterrows():
        item = {str(c): series[c] for c in df.columns}
        item["stt"] = series[stt_column] if stt_column in df.columns else ""
        rows.append(item)
    return rows


def annotate_dataframe_semantics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run outline + sum validators and append ``STT_valid`` / ``Sum_check`` columns.

    Existing columns are preserved. Invalid rows emit a WARNING log / console line.
    """
    if df is None or len(getattr(df, "columns", [])) == 0:
        return df

    out = df.copy()
    stt_col = detect_stt_column(out)
    if stt_col is None:
        out["STT_valid"] = True
        out["Sum_check"] = "N/A"
        return out

    amount_cols = detect_amount_columns(out, stt_col)
    rows = dataframe_to_rows(out, stt_col)
    rows = validate_stt_outline(rows)
    rows = validate_sum_consistency(rows, amount_cols)

    stt_valid_flags: list[bool] = []
    sum_checks: list[str] = []
    for row in rows:
        stt = str(row.get("stt", "") or "").strip()
        stt_ok = bool(row.get("stt_valid", True))
        stt_valid_flags.append(stt_ok)
        if not stt_ok:
            issue = row.get("stt_issue") or "outline invalid"
            logger.warning("[WARN] STT_valid=False | stt=%s | %s", stt, issue)
            print(f"[WARN] STT_valid=False | stt={stt} | {issue}", flush=True)

        sum_ok = bool(row.get("sum_valid", True))
        sum_diff = row.get("sum_diff")
        if sum_diff is None:
            sum_checks.append("N/A")
        elif sum_ok:
            sum_checks.append("OK")
        else:
            label = f"FAIL:Δ={sum_diff:g}"
            sum_checks.append(label)
            issue = row.get("sum_issue") or label
            logger.warning("[WARN] Sum_check=FAIL | stt=%s | %s", stt, issue)
            print(f"[WARN] Sum_check=FAIL | stt={stt} | {issue}", flush=True)

    out["STT_valid"] = stt_valid_flags
    out["Sum_check"] = sum_checks
    return out


def annotate_tables(tables: Sequence[pd.DataFrame]) -> list[pd.DataFrame]:
    """Annotate every assembled table DataFrame with semantic columns."""
    return [annotate_dataframe_semantics(df) for df in tables]


def list_invalid_rows(rows: Iterable[dict]) -> list[dict]:
    """Return compact invalid summaries for reporting."""
    invalid: list[dict] = []
    for row in rows:
        stt = str(row.get("stt", "") or "").strip()
        if row.get("stt_valid") is False:
            invalid.append({"stt": stt, "kind": "stt", "issue": row.get("stt_issue")})
        if row.get("sum_valid") is False:
            invalid.append(
                {
                    "stt": stt,
                    "kind": "sum",
                    "issue": row.get("sum_issue") or f"sum_diff={row.get('sum_diff')}",
                    "sum_diff": row.get("sum_diff"),
                }
            )
    return invalid
