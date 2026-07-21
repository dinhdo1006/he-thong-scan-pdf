"""
OCR post-processing: fix known, recurring OCR/text-recognition typos in an already
extracted Pandas DataFrame.

This is intentionally NOT an AI/LLM spell-checker -- it is a plain, deterministic
find-and-replace pass over a curated dictionary of known errors. You keep 100%
control over what gets "corrected", and the mapping is easy to audit/extend as new
OCR error patterns are discovered in real documents.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Known OCR error -> correct text. Extend this dictionary as new typos are found
# in real scanned documents (e.g. via manual review of the "Unmatched" column or
# spot-checking exported rows).
DEFAULT_OCR_REPLACEMENTS: Dict[str, str] = {
    "hoyết": "quyết định",
    "Erở nh": "Trong đó",
    "thăng": "thẳng",
    "qhyết": "quyết",
    "hãnh": "hành",
    "thí": "thi",
}


def _replace_all(text: str, replacements: Dict[str, str]) -> str:
    """Apply every (wrong -> right) pair in `replacements` to a single string."""
    for wrong, right in replacements.items():
        if wrong in text:
            text = re.sub(re.escape(wrong), right, text)
    return text


def clean_ocr_errors(
    df: pd.DataFrame,
    replacements: Optional[Dict[str, str]] = None,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Return a copy of `df` with known OCR typos replaced in its text columns.

    Args:
        df: Input DataFrame (e.g. produced by `build_dataframe`).
        replacements: Mapping of wrong-text -> correct-text. Defaults to
            `DEFAULT_OCR_REPLACEMENTS`.
        columns: Optional subset of column names to clean. Defaults to every
            object-dtype (string-like) column, so numeric/page/y0 columns are
            left untouched automatically.

    Returns:
        A new DataFrame; the input `df` is never mutated in place.
    """
    replacements = replacements if replacements is not None else DEFAULT_OCR_REPLACEMENTS
    if not replacements:
        return df.copy()

    cleaned = df.copy()
    target_columns = columns if columns is not None else list(cleaned.select_dtypes(include="object").columns)

    total_fixes = 0
    for col in target_columns:
        if col not in cleaned.columns:
            continue

        def _fix_cell(value):
            nonlocal total_fixes
            if not isinstance(value, str):
                return value
            fixed = _replace_all(value, replacements)
            if fixed != value:
                total_fixes += 1
            return fixed

        cleaned[col] = cleaned[col].apply(_fix_cell)

    logger.info("OCR cleanup: applied %d replacement(s) across %d column(s).", total_fixes, len(target_columns))
    return cleaned
