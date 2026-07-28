"""
Serialize / deserialize extracted tables for OCR pipeline workers.

Form-agnostic JSON schema (no template / form-id fields):
  {
    "version": 1,
    "tables": [
      {
        "table_id": 0,
        "columns": [...],
        "data": [[...], ...],
        "header_matrix": [[...], ...] | null,
        "page": int | null
      }
    ]
  }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

SCHEMA_VERSION = 1


def _cell_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def dataframe_to_structured(df: pd.DataFrame, *, table_id: int = 0) -> dict[str, Any]:
    """Convert one DataFrame (+ optional header_matrix attrs) to a JSON object."""
    if df is None:
        return {
            "table_id": table_id,
            "columns": [],
            "data": [],
            "header_matrix": None,
            "page": None,
        }
    columns = [_cell_str(c) for c in df.columns]
    data = [[_cell_str(v) for v in row] for row in df.itertuples(index=False, name=None)]
    matrix = getattr(df, "attrs", {}).get("header_matrix")
    header_matrix = None
    if isinstance(matrix, list) and matrix:
        header_matrix = [[_cell_str(c) for c in row] for row in matrix]
    page = getattr(df, "attrs", {}).get("page_no")
    if page is not None:
        try:
            page = int(page)
        except (TypeError, ValueError):
            page = None
    return {
        "table_id": table_id,
        "columns": columns,
        "data": data,
        "header_matrix": header_matrix,
        "page": page,
    }


def tables_to_payload(tables: Sequence[pd.DataFrame]) -> dict[str, Any]:
    return {
        "version": SCHEMA_VERSION,
        "tables": [dataframe_to_structured(df, table_id=i) for i, df in enumerate(tables)],
    }


def structured_to_dataframe(obj: dict[str, Any]) -> pd.DataFrame:
    """Rebuild a DataFrame from one structured table object."""
    columns = [str(c) for c in obj.get("columns") or []]
    rows = obj.get("data") or []
    width = len(columns)
    body: list[list[str]] = []
    for row in rows:
        cells = [str(c) if c is not None else "" for c in row]
        if width:
            cells = (cells + [""] * width)[:width]
        body.append(cells)
    if not columns and body:
        width = max(len(r) for r in body)
        columns = [f"Column_{i + 1}" for i in range(width)]
        body = [(r + [""] * width)[:width] for r in body]
    df = pd.DataFrame(body, columns=columns or None)
    matrix = obj.get("header_matrix")
    if isinstance(matrix, list) and matrix:
        df.attrs["header_matrix"] = [
            [str(c) if c is not None else "" for c in row] for row in matrix
        ]
    page = obj.get("page")
    if page is not None:
        try:
            df.attrs["page_no"] = int(page)
        except (TypeError, ValueError):
            pass
    return df


def payload_to_tables(payload: dict[str, Any]) -> list[pd.DataFrame]:
    raw = payload.get("tables") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []
    return [structured_to_dataframe(t) for t in raw if isinstance(t, dict)]


def write_structured_tables_json(
    tables: Sequence[pd.DataFrame],
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = tables_to_payload(tables)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_structured_tables_json(path: str | Path) -> list[pd.DataFrame]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload_to_tables(payload)


def load_structured_payload(raw: bytes | str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)
