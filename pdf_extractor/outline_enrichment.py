"""
Enrich outline form tables using the PDF embedded text layer.

When PP-Structure drops rows (e.g. 1.1.1 Án phí) but PyMuPDF still sees the
outline codes + labels + amounts, merge those back into the repaired grid.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from .semantic_validator import detect_stt_column, parse_stt_path
from .table_layout import (
    is_money_like,
    is_outline_code,
    normalize_outline_token,
)

logger = logging.getLogger(__name__)

_Y_TOL = 4.0
_LABEL_MIN_CHARS = 4


@dataclass(frozen=True)
class PdfOutlineRow:
    stt: str
    label: str
    amounts: tuple[str, ...]


def extract_outline_rows_from_pdf(pdf_path: str | Path) -> list[PdfOutlineRow]:
    """
    Parse outline rows (STT + label + money tokens) from PDF word positions.
    """
    import fitz

    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    rows: list[PdfOutlineRow] = []
    seen: set[str] = set()
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            for cluster in _cluster_words_by_y(page.get_text("words")):
                parsed = _parse_outline_cluster(cluster)
                if parsed is None:
                    continue
                if parsed.stt in seen:
                    continue
                # Prefer the first rich occurrence (with label / amounts).
                seen.add(parsed.stt)
                rows.append(parsed)
    finally:
        doc.close()

    logger.info(
        "PDF outline probe: %d unique STT row(s) from %s.",
        len(rows),
        pdf_path.name,
    )
    return rows


def enrich_table_from_pdf_outline(
    df: pd.DataFrame,
    outline_rows: Sequence[PdfOutlineRow],
) -> pd.DataFrame:
    """
    Fill missing STT rows / empty labels / empty amounts from PDF outline rows.
    """
    if df is None or df.empty or not outline_rows:
        return df

    stt_col = detect_stt_column(df)
    if stt_col is None:
        return df

    cols = [str(c) for c in df.columns]
    stt_idx = cols.index(stt_col)
    desc_idx = stt_idx + 1 if stt_idx + 1 < len(cols) else None
    amount_idxs = list(range(stt_idx + (2 if desc_idx is not None else 1), len(cols)))

    by_stt = {r.stt: r for r in outline_rows}
    body = [[_cell(df.iloc[r, c]) for c in range(len(cols))] for r in range(len(df))]

    existing: dict[str, int] = {}
    for i, row in enumerate(body):
        code = normalize_outline_token(row[stt_idx])
        if is_outline_code(code):
            existing[code] = i

    filled_labels = 0
    filled_amounts = 0
    for code, idx in list(existing.items()):
        src = by_stt.get(code)
        if src is None:
            continue
        row = body[idx]
        if desc_idx is not None and not row[desc_idx].strip() and src.label:
            row[desc_idx] = src.label
            filled_labels += 1
        if src.amounts and amount_idxs:
            if not any(row[j].strip() for j in amount_idxs):
                spread = _spread_amounts(list(src.amounts), len(amount_idxs))
                for j, val in zip(amount_idxs, spread):
                    row[j] = val
                filled_amounts += 1
        elif amount_idxs:
            # PDF row has no money: drop orphan single trailing amount from Case-1 realign.
            vals = [row[j].strip() for j in amount_idxs]
            nonempty = [v for v in vals if v]
            if len(nonempty) == 1 and vals[-1] and is_money_like(vals[-1]):
                for j in amount_idxs:
                    row[j] = ""
                filled_amounts += 1

    # Insert missing codes using PDF reading order (not roman-before-all-arabic).
    inserted = 0
    ordered = [r for r in outline_rows if is_outline_code(r.stt)]
    ordered_stts = [r.stt for r in ordered]
    for src in ordered:
        if src.stt in existing:
            continue
        if not src.label and not src.amounts:
            continue
        insert_at = _insertion_index_by_order(ordered_stts, existing, src.stt, len(body))
        new_row = [""] * len(cols)
        new_row[stt_idx] = src.stt
        if desc_idx is not None:
            new_row[desc_idx] = src.label
        if src.amounts and amount_idxs:
            spread = _spread_amounts(list(src.amounts), len(amount_idxs))
            for j, val in zip(amount_idxs, spread):
                new_row[j] = val
        body.insert(insert_at, new_row)
        existing = {}
        for i, row in enumerate(body):
            code = normalize_outline_token(row[stt_idx])
            if is_outline_code(code):
                existing[code] = i
        inserted += 1

    if inserted or filled_labels or filled_amounts:
        logger.info(
            "PDF outline enrich: inserted=%d filled_labels=%d filled_amounts=%d.",
            inserted,
            filled_labels,
            filled_amounts,
        )
    out = pd.DataFrame(body, columns=cols)
    # Preserve attrs (page_no, backend, ...).
    out.attrs.update(getattr(df, "attrs", {}))
    return out


def enrich_tables_from_pdf(
    pdf_path: str | Path,
    tables: Sequence[pd.DataFrame],
) -> list[pd.DataFrame]:
    """Enrich every table; no-op when PDF has no usable outline rows."""
    try:
        outline_rows = extract_outline_rows_from_pdf(pdf_path)
    except Exception as exc:
        logger.warning("PDF outline enrich skipped (%s).", exc)
        return list(tables)
    if not outline_rows:
        return list(tables)
    return [enrich_table_from_pdf_outline(df, outline_rows) for df in tables]


def _cell(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return " ".join(str(value).replace("\n", " ").split())


def _cluster_words_by_y(words: list) -> list[list[tuple[float, str]]]:
    clusters: list[dict] = []
    for item in words:
        x0, y0 = float(item[0]), float(item[1])
        text = str(item[4])
        placed = False
        for c in clusters:
            if abs(c["y"] - y0) < _Y_TOL:
                c["items"].append((x0, text))
                n = c["n"]
                c["y"] = (c["y"] * n + y0) / (n + 1)
                c["n"] = n + 1
                placed = True
                break
        if not placed:
            clusters.append({"y": y0, "n": 1, "items": [(x0, text)]})
    clusters.sort(key=lambda c: c["y"])
    return [sorted(c["items"], key=lambda t: t[0]) for c in clusters]


def _parse_outline_cluster(items: list[tuple[float, str]]) -> PdfOutlineRow | None:
    texts = [t for _, t in items]
    outline = None
    oi = None
    for i, raw in enumerate(texts):
        code = normalize_outline_token(raw)
        if not is_outline_code(code):
            continue
        if parse_stt_path(code) is None:
            continue
        outline, oi = code, i
        break
    if outline is None or oi is None:
        return None

    rest = texts[oi + 1 :]
    amounts = tuple(t for t in rest if is_money_like(t))
    label_parts = [
        t
        for t in rest
        if not is_money_like(t) and t not in {"|", ":", "-", "_"}
    ]
    label = " ".join(label_parts)
    label = re.sub(r"\s*\|\s*", " ", label)
    label = re.sub(r"\s+", " ", label).strip(" |_")
    # Drop trailing underscore artifacts from OCR.
    label = label.rstrip("_").strip()

    if not label and not amounts:
        return None

    letter_count = len(re.findall(r"[A-Za-zÀ-ỹ]", label, flags=re.UNICODE))
    # Reject column-code debris ("+ 5 6 Fị") and date fragments without a real label.
    if letter_count < 4:
        return None
    # Reject labels that are almost only digits/symbols.
    if len(re.sub(r"[\d\s|+\-.,]", "", label)) < 3:
        return None

    return PdfOutlineRow(stt=outline, label=label, amounts=amounts)


def _spread_amounts(moneys: list[str], n_cols: int) -> list[str]:
    """Map N money tokens into N_cols (first half left, remainder right)."""
    if n_cols <= 0:
        return []
    if not moneys:
        return [""] * n_cols
    if len(moneys) >= n_cols:
        return moneys[:n_cols]
    if len(moneys) <= 2:
        return moneys + [""] * (n_cols - len(moneys))
    left = (len(moneys) + 1) // 2
    mid = n_cols - len(moneys)
    return moneys[:left] + [""] * mid + moneys[left:]


def _insertion_index_by_order(
    ordered_stts: Sequence[str],
    existing: dict[str, int],
    new_code: str,
    body_len: int,
) -> int:
    """Insert after the nearest previous STT that already exists in the table."""
    if new_code not in ordered_stts:
        return body_len
    pos = ordered_stts.index(new_code)
    for prev in reversed(list(ordered_stts[:pos])):
        if prev in existing:
            return existing[prev] + 1
    for nxt in ordered_stts[pos + 1 :]:
        if nxt in existing:
            return existing[nxt]
    return body_len


def _outline_sort_key(code: str) -> tuple:
    path = parse_stt_path(code)
    if path is None:
        return (99, code)
    kind = 0 if path[0] == "roman" else 1
    return (kind, *path[1:])