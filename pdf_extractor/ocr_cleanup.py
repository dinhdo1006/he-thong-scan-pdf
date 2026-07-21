"""
OCR post-processing: fix known, recurring OCR/text-recognition typos in an already
extracted Pandas DataFrame.

This is intentionally NOT an AI/LLM spell-checker -- it is a plain, deterministic
find-and-replace pass. Unlike earlier versions of this module, the wrong->right
mapping is NOT hardcoded in Python anymore: it is loaded dynamically from an
external `ocr_corrections.json` file (config-driven). This means:
  - Corrections can be added/edited/removed WITHOUT touching any Python code.
  - Different deployments/documents can ship their own corrections file.
  - If the file is missing, cleanup is simply a no-op (returns the DataFrame
    unchanged) instead of raising -- the pipeline never breaks over a missing
    or malformed config.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Resolved relative to the project root (one level up from this package),
# so it works the same whether the caller's cwd is the project root or not.
DEFAULT_CORRECTIONS_PATH: Path = Path(__file__).resolve().parent.parent / "ocr_corrections.json"

# Cache of {resolved_path: corrections_dict} so repeated calls (e.g. once per
# table/page) don't re-read + re-parse the JSON file from disk every time.
_corrections_cache: Dict[str, Dict[str, str]] = {}


def load_corrections(path: str | Path = DEFAULT_CORRECTIONS_PATH) -> Dict[str, str]:
    """
    Load the wrong->right text mapping from a JSON file.

    Args:
        path: Path to a JSON file containing a flat object, e.g.
            `{"hoyết": "quyết định", "thăng": "thẳng"}`.

    Returns:
        The parsed mapping, or an empty dict (never raises) if the file does
        not exist, is not valid JSON, or is not a JSON object -- callers
        should treat an empty dict as "no corrections configured" rather
        than an error.
    """
    path = Path(path)
    cache_key = str(path)
    if cache_key in _corrections_cache:
        return _corrections_cache[cache_key]

    if not path.is_file():
        logger.info(
            "OCR corrections file not found at %s -- OCR cleanup will be a no-op "
            "until the file is created.",
            path,
        )
        _corrections_cache[cache_key] = {}
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load OCR corrections from %s (%s) -- treating as empty.", path, exc)
        _corrections_cache[cache_key] = {}
        return {}

    if not isinstance(raw, dict):
        logger.warning("OCR corrections file %s did not contain a JSON object -- treating as empty.", path)
        _corrections_cache[cache_key] = {}
        return {}

    corrections = {str(k): str(v) for k, v in raw.items()}
    _corrections_cache[cache_key] = corrections
    logger.info("Loaded %d OCR correction(s) from %s.", len(corrections), path)
    return corrections


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
    corrections_path: str | Path = DEFAULT_CORRECTIONS_PATH,
) -> pd.DataFrame:
    """
    Return a copy of `df` with known OCR typos replaced in its text columns.

    Args:
        df: Input DataFrame (e.g. produced by any table extractor).
        replacements: Explicit wrong-text -> correct-text mapping. If `None`
            (default), the mapping is loaded dynamically via
            `load_corrections(corrections_path)` -- pass an explicit dict
            only if you need to override/bypass the JSON config file.
        columns: Optional subset of column names to clean. Defaults to every
            object-dtype (string-like) column, so numeric columns are left
            untouched automatically.
        corrections_path: Path to the JSON corrections file, used only when
            `replacements` is `None`. Defaults to `ocr_corrections.json` at
            the project root.

    Returns:
        A new DataFrame; the input `df` is never mutated in place. If no
        corrections are configured (missing/empty file and no explicit
        `replacements`), this is a no-op copy.
    """
    replacements = replacements if replacements is not None else load_corrections(corrections_path)
    if not replacements:
        return df.copy()

    cleaned = df.copy()
    target_columns = columns if columns is not None else list(cleaned.select_dtypes(include="object").columns)

    total_fixes = 0
    for col in target_columns:
        if col not in cleaned.columns:
            continue

        def _fix_cell(value, _replacements=replacements):
            nonlocal total_fixes
            if not isinstance(value, str):
                return value
            fixed = _replace_all(value, _replacements)
            if fixed != value:
                total_fixes += 1
            return fixed

        cleaned[col] = cleaned[col].apply(_fix_cell)

    logger.info("OCR cleanup: applied %d replacement(s) across %d column(s).", total_fixes, len(target_columns))
    return cleaned


def clean_text_ocr_errors(
    text: str,
    replacements: Optional[Dict[str, str]] = None,
    corrections_path: str | Path = DEFAULT_CORRECTIONS_PATH,
) -> str:
    """
    Apply the same JSON-driven find-and-replace pass to a plain string.

    Used for Marker prose before writing the final `.txt` output.
    """
    replacements = replacements if replacements is not None else load_corrections(corrections_path)
    if not replacements or not text:
        return text
    return _replace_all(text, replacements)
