"""Export parsed tables and text to plain-text / Word (and optional debug formats)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from .markdown_parser import MarkdownBlockParser
from .ocr_cleanup import clean_ocr_errors, clean_text_ocr_errors

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_TXT = "output.txt"
DEFAULT_OUTPUT_DOCX = "output.docx"
DEFAULT_OUTPUT_XLSX = "output_tables.xlsx"

_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_GARBLE_HINT_RE = re.compile(
    r"hoyết|qhyết|Erở|ĐƑT|thăng\)|gôm cả|thí hãnh|ĐÓI CHIẾU",
    re.IGNORECASE,
)


def resolve_output_path(output: str | Path, default_name: str = DEFAULT_OUTPUT_DOCX) -> Path:
    """
    Resolve CLI `-o` to a concrete file path.

    - `result.docx` / `result.txt` -> that file
    - `./output` (directory)       -> `./output/<default_name>`
    """
    path = Path(output).expanduser()
    suffix = path.suffix.lower()
    if suffix in {".txt", ".docx", ".xlsx"}:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return (path / default_name).resolve()


def _markdown_text_to_plain(text: str) -> str:
    """Light cleanup: headings, images, and inline HTML from Marker -> plain text."""
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        # Drop Marker-emitted image placeholders (stamps/signatures as .jpeg).
        if _MD_IMAGE_RE.search(stripped) and not _MD_IMAGE_RE.sub("", stripped).strip():
            continue
        stripped = _MD_IMAGE_RE.sub("", stripped)
        if stripped.startswith("#"):
            stripped = re.sub(r"^#+\s*", "", stripped)
        stripped = stripped.replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ")
        stripped = stripped.strip()
        if stripped:
            lines.append(stripped)
    # Collapse excess blank lines
    body = "\n".join(lines)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


_GENERIC_HEADER_RE = re.compile(r"^(?:col(?:_\d+)?|Column_\d+)$", re.IGNORECASE)
_NUMBER_LIKE_RE = re.compile(r"^[\d.,]+$")
# Minimum similarity before an extracted table may replace a Marker block.
# Below this we keep Marker's grid so wrong VLM/Paddle output cannot scramble form.
_MIN_TABLE_MATCH_SCORE = 0.45


def _clean_cell_text(value: object) -> str:
    """Normalize one cell for display / comparison without dropping empties."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    text = text.replace("\n", " ").replace("\r", " ")
    text = text.replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ")
    text = text.replace("\t", " ")
    return " ".join(text.split())


def _is_generic_header(name: str) -> bool:
    return bool(_GENERIC_HEADER_RE.fullmatch(name.strip()))


def _headers_are_generic(columns: list[str]) -> bool:
    return bool(columns) and all(_is_generic_header(c) for c in columns)


def _dataframe_matrix(df: pd.DataFrame, *, include_header: bool) -> list[list[str]]:
    """Flatten a DataFrame to a 2D string grid, preserving empty cells."""
    cols = [_clean_cell_text(c) for c in df.columns]
    body = [
        [_clean_cell_text(df.iloc[r, c]) for c in range(len(df.columns))]
        for r in range(len(df))
    ]
    if include_header and not _headers_are_generic(cols):
        return [cols, *body]
    return body


def _token_set(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[\w./%+-]+", text, flags=re.UNICODE) if t}


def _content_overlap_score(a: pd.DataFrame, b: pd.DataFrame, sample_rows: int = 8) -> float:
    """Cheap bag-of-tokens overlap over the first few data rows."""
    def _sample(df: pd.DataFrame) -> set[str]:
        tokens: set[str] = set()
        for r in range(min(sample_rows, len(df))):
            for c in range(len(df.columns)):
                tokens |= _token_set(_clean_cell_text(df.iloc[r, c]))
        for c in df.columns:
            tokens |= _token_set(_clean_cell_text(c))
        return tokens

    ta, tb = _sample(a), _sample(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def table_shape_similarity(marker_df: pd.DataFrame, extracted_df: pd.DataFrame) -> float:
    """
    Score how safely an extracted table can replace a Marker table block.

    Weighted by column count (form structure), row count, and token overlap.
    """
    m_cols, e_cols = len(marker_df.columns), len(extracted_df.columns)
    m_rows, e_rows = len(marker_df), len(extracted_df)
    if m_cols == 0 or e_cols == 0:
        return 0.0

    col_ratio = min(m_cols, e_cols) / max(m_cols, e_cols)
    # Exact column match is critical for form layout.
    if abs(m_cols - e_cols) == 0:
        col_score = 1.0
    elif abs(m_cols - e_cols) == 1:
        col_score = 0.7 * col_ratio
    else:
        col_score = 0.35 * col_ratio

    if max(m_rows, e_rows) == 0:
        row_score = 1.0
    else:
        row_score = min(m_rows, e_rows) / max(m_rows, e_rows)
        # Stitched multi-page tables are taller than one Marker block — soft-penalize only.
        if e_rows > m_rows * 2 and m_rows > 0:
            row_score = max(row_score, 0.55)

    content = _content_overlap_score(marker_df, extracted_df)
    return 0.5 * col_score + 0.25 * row_score + 0.25 * content


def _pick_best_extracted_index(
    marker_df: pd.DataFrame,
    candidates: list[pd.DataFrame | None],
) -> tuple[int, float] | None:
    """Return (index, score) of the best unused candidate, or None if none match."""
    best_idx: int | None = None
    best_score = -1.0
    for idx, cand in enumerate(candidates):
        if cand is None:
            continue
        score = table_shape_similarity(marker_df, cand)
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_idx is None or best_score < _MIN_TABLE_MATCH_SCORE:
        return None
    return best_idx, best_score


def _slice_stitched_table(
    marker_df: pd.DataFrame,
    extracted_df: pd.DataFrame,
    row_cursor: dict[int, int],
    extracted_index: int,
    *,
    share_across_blocks: bool,
) -> pd.DataFrame:
    """
    When a backend stitches multi-page rows into one tall table AND Marker has
    multiple table blocks, carve out a Marker-sized slice so each document
    position keeps its own block. Single-block docs keep the full extracted
    table (no silent truncation).
    """
    if not share_across_blocks or len(extracted_df.columns) != len(marker_df.columns):
        row_cursor[extracted_index] = len(extracted_df)
        return extracted_df

    start = row_cursor.get(extracted_index, 0)
    need = max(len(marker_df), 1)
    if start >= len(extracted_df):
        return extracted_df.iloc[0:0].copy()

    end = min(len(extracted_df), start + need)
    remaining = len(extracted_df) - start
    if remaining <= need + 2:
        end = len(extracted_df)

    sliced = extracted_df.iloc[start:end].copy()
    marker_headers = [_clean_cell_text(c) for c in marker_df.columns]
    if not _headers_are_generic(marker_headers):
        sliced.columns = list(marker_df.columns)
    row_cursor[extracted_index] = end
    return sliced.reset_index(drop=True)


def _headers_look_garbled(columns: list[str]) -> bool:
    """Heuristic: Marker OCR headers that should not win over a clean backend grid."""
    if not columns:
        return True
    joined = " ".join(columns)
    if _GARBLE_HINT_RE.search(joined):
        return True
    # Many tiny / punctuation-heavy tokens → typical OCR header soup.
    tokens = [t for t in re.findall(r"\S+", joined) if t]
    if len(tokens) >= 4:
        tiny = sum(1 for t in tokens if len(t) <= 2)
        if tiny / len(tokens) >= 0.45:
            return True
    return False


def merge_table_preserving_form(
    marker_df: pd.DataFrame,
    extracted_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prefer extracted cell values when shapes align; keep Marker headers / width
    when they better describe the printed form. If Marker headers look
    OCR-garbled, prefer extracted headers instead.
    """
    marker_cols = [_clean_cell_text(c) for c in marker_df.columns]
    extracted_cols = [_clean_cell_text(c) for c in extracted_df.columns]

    # Same width: take extracted body, choose the cleaner header set.
    if len(marker_cols) == len(extracted_cols) and len(marker_cols) > 0:
        body = [
            [_clean_cell_text(extracted_df.iloc[r, c]) for c in range(len(extracted_cols))]
            for r in range(len(extracted_df))
        ]
        marker_garbled = _headers_look_garbled(marker_cols)
        extracted_generic = _headers_are_generic(extracted_cols)
        marker_generic = _headers_are_generic(marker_cols)

        if marker_garbled and not extracted_generic:
            headers = extracted_cols
        elif extracted_generic and not marker_generic and not marker_garbled:
            headers = marker_cols
        elif not extracted_generic:
            headers = extracted_cols
        elif not marker_generic and not marker_garbled:
            headers = marker_cols
        else:
            headers = extracted_cols if not extracted_generic else marker_cols
        return pd.DataFrame(body, columns=headers)

    # Extracted is narrower but Marker collapsed numbers into one cell — expand Marker.
    expanded_marker = _split_multi_value_cells(marker_df)
    if len(expanded_marker.columns) == len(extracted_cols):
        return merge_table_preserving_form(expanded_marker, extracted_df)

    # Width mismatch: prefer the extracted grid when Marker headers are garbled,
    # otherwise keep the wider grid (closer to printed form).
    if _headers_look_garbled(marker_cols):
        return extracted_df.copy()
    if len(marker_cols) >= len(extracted_cols):
        return marker_df.copy()
    return extracted_df.copy()


def _split_multi_value_cells(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-process a Marker-fallback DataFrame where one cell contains two
    formatted-number values separated by spaces (e.g. "10.620.000   10.370.000").

    For each column, if the column contains ANY cell that looks like
    "NUMBER  NUMBER" (2+ whitespace-separated number-like tokens), the
    column is replaced by two sub-columns named "col" and "col_2".
    Generic: no column-name or template assumption.
    """

    def _try_split(cell: str) -> list[str] | None:
        """Return [part1, part2, ...] if cell is 2+ number tokens, else None."""
        parts = cell.strip().split()
        if len(parts) >= 2 and all(_NUMBER_LIKE_RE.match(p) for p in parts):
            return parts
        return None

    rows = len(df)
    new_cols: list[str] = []
    column_data: list[list[str]] = []

    # Iterate by position to avoid duplicate-column issues with df[name]
    for col_pos, col in enumerate(df.columns):
        raw: list[str] = [_clean_cell_text(v) for v in df.iloc[:, col_pos]]
        splits: list[list[str] | None] = [_try_split(v) for v in raw]
        max_parts = max((len(s) for s in splits if s is not None), default=1)

        if max_parts >= 2:
            for part_idx in range(max_parts):
                if part_idx == 0:
                    sub = col
                else:
                    # Ensure the generated name is unique across ALL columns
                    candidate = f"{col}_{part_idx + 1}"
                    suffix = part_idx + 1
                    while candidate in new_cols:
                        suffix += 1
                        candidate = f"{col}_{suffix}"
                    sub = candidate
                new_cols.append(sub)
                col_vals: list[str] = []
                for i in range(rows):
                    s = splits[i]
                    if s is not None and part_idx < len(s):
                        col_vals.append(s[part_idx])
                    elif s is None and part_idx == 0:
                        col_vals.append(raw[i])
                    else:
                        col_vals.append("")
                column_data.append(col_vals)
        else:
            new_cols.append(col)
            column_data.append(raw)

    # Build using a list-of-lists to avoid duplicate-column confusion
    result = pd.DataFrame(dict(enumerate(column_data)))
    result.columns = pd.Index(new_cols)
    return result


def dataframe_to_plaintext_table(df: pd.DataFrame) -> str:
    """
    Render a DataFrame as a fixed-width pipe grid so columns stay aligned.

    Empty cells are preserved as blank slots (critical for form layouts).
    Generic synthetic headers (`col`, `Column_1`, ...) are omitted.
    """
    if df.empty or len(df.columns) == 0:
        return ""

    matrix = _dataframe_matrix(df, include_header=True)
    if not matrix:
        return ""

    width = max(len(row) for row in matrix)
    normalized: list[list[str]] = []
    for row in matrix:
        padded = list(row) + [""] * (width - len(row))
        normalized.append(padded[:width])

    col_widths = [1] * width
    for row in normalized:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    lines: list[str] = []
    for row in normalized:
        cells = [cell.ljust(col_widths[i]) for i, cell in enumerate(row)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _prepare_table_df(df: pd.DataFrame, *, apply_ocr_cleanup: bool) -> pd.DataFrame:
    prepared = _split_multi_value_cells(df)
    if apply_ocr_cleanup:
        prepared = clean_ocr_errors(prepared)
    return prepared


@dataclass
class AssembledBlock:
    """One ordered document slice after table matching/merge."""

    kind: Literal["text", "table"]
    content: str | pd.DataFrame


def assemble_document_blocks(
    markdown: str,
    extracted_tables: list[pd.DataFrame],
    *,
    apply_ocr_cleanup: bool = True,
) -> list[AssembledBlock]:
    """
    Walk Marker blocks in reading order and produce final text/table slices.

    Same matching rules as the plain-text composer: shape score, stitch slice,
    merge preserving form. Shared by `.txt` and `.docx` exporters so position
    stays consistent across formats.
    """
    blocks = MarkdownBlockParser().parse_blocks(markdown)
    candidates: list[pd.DataFrame | None] = list(extracted_tables)
    used_flags = [False] * len(candidates)
    row_cursors: dict[int, int] = {}
    parts: list[AssembledBlock] = []
    marker_table_count = sum(1 for b in blocks if b.kind == "table")
    share_across_blocks = marker_table_count > 1
    seen_marker_tables = 0

    # When Marker found no pipe-tables but backends returned grids (common on
    # scans where Marker OCR fails), emit prose then append extracted tables.
    for block in blocks:
        if block.kind == "text":
            plain = _markdown_text_to_plain(str(block.content))
            if apply_ocr_cleanup:
                plain = clean_text_ocr_errors(plain)
            if plain.strip():
                parts.append(AssembledBlock(kind="text", content=plain))
            continue

        seen_marker_tables += 1
        marker_df = block.content
        assert isinstance(marker_df, pd.DataFrame)

        pick = _pick_best_extracted_index(marker_df, candidates)
        marker_garbled = _headers_look_garbled(
            [_clean_cell_text(c) for c in marker_df.columns]
        )
        if pick is None and marker_garbled and any(c is not None for c in candidates):
            # Garbled Marker OCR soup: prefer image-backend table by order.
            for idx, cand in enumerate(candidates):
                if cand is not None:
                    pick = (idx, 0.4)
                    logger.info(
                        "Table block %d: forcing extracted[%d] over garbled Marker headers.",
                        seen_marker_tables,
                        idx,
                    )
                    break

        if pick is None:
            df = marker_df
            if any(c is not None for c in candidates):
                logger.warning(
                    "Table block %d: extracted tables remain but shape mismatch "
                    "and Marker headers look OK -- keeping Marker %s.",
                    seen_marker_tables,
                    marker_df.shape,
                )
            else:
                logger.warning(
                    "Table block %d: no extracted table left -- keeping Marker grid %s.",
                    seen_marker_tables,
                    marker_df.shape,
                )
        else:
            idx, score = pick
            extracted = candidates[idx]
            assert extracted is not None
            sliced = _slice_stitched_table(
                marker_df,
                extracted,
                row_cursors,
                idx,
                share_across_blocks=share_across_blocks,
            )
            if row_cursors.get(idx, 0) >= len(extracted):
                candidates[idx] = None
            used_flags[idx] = True
            if sliced.empty:
                df = marker_df
            else:
                df = merge_table_preserving_form(marker_df, sliced)
                logger.debug(
                    "Table block %d: matched extracted[%d] score=%.2f -> %s",
                    seen_marker_tables,
                    idx,
                    score,
                    df.shape,
                )

        df = _prepare_table_df(df, apply_ocr_cleanup=apply_ocr_cleanup)
        parts.append(AssembledBlock(kind="table", content=df))

    # Append unused extracted tables only when Marker had no table anchors OR
    # Marker headers were garbled (image backend is the source of truth).
    marker_garbled_any = any(
        b.kind == "table"
        and isinstance(b.content, pd.DataFrame)
        and _headers_look_garbled([_clean_cell_text(c) for c in b.content.columns])
        for b in blocks
    )
    for flag, cand in zip(used_flags, candidates):
        if flag or cand is None:
            continue
        if marker_table_count == 0 or marker_garbled_any:
            df = _prepare_table_df(cand, apply_ocr_cleanup=apply_ocr_cleanup)
            parts.append(AssembledBlock(kind="table", content=df))
            logger.info("Appended leftover extracted table %s.", df.shape)
        else:
            logger.info(
                "Dropped unmatched extracted table %s to preserve Marker reading order.",
                cand.shape,
            )

    return parts


def compose_document_txt(
    markdown: str,
    extracted_tables: list[pd.DataFrame],
    *,
    apply_ocr_cleanup: bool = True,
) -> str:
    """Build ONE plain-text document (prose + aligned pipe tables) in reading order."""
    parts: list[str] = []
    for block in assemble_document_blocks(
        markdown, extracted_tables, apply_ocr_cleanup=apply_ocr_cleanup
    ):
        if block.kind == "text":
            parts.append(str(block.content))
        else:
            assert isinstance(block.content, pd.DataFrame)
            table_txt = dataframe_to_plaintext_table(block.content)
            if table_txt.strip():
                parts.append(table_txt)
    body = "\n\n".join(parts)
    return body.rstrip() + ("\n" if body.strip() else "")


def save_unified_txt(content: str, output_path: str | Path) -> Path:
    """Write the final unified document to a UTF-8 `.txt` file."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")
    except Exception as exc:
        raise IOError(f"Failed to write text file '{path}': {exc}") from exc
    logger.info("Saved unified text (%d chars) -> %s", len(content), path)
    return path


def export_tables_preview(
    tables: list[pd.DataFrame],
    output_xlsx: str | Path,
    *,
    write_csv: bool = True,
    write_markdown: bool = True,
    empty_reason: str | None = None,
) -> dict[str, Path]:
    """
    Write tables for both Excel and VS Code-friendly previews.

    VS Code does not render .xlsx well; companions:
      - `output_tables_preview.md`  — Markdown tables (best preview in VS Code)
      - `output_table_1.csv`, ...   — plain CSV (open as text / Excel)

    Returns paths of files that were written (keys: xlsx, markdown, csv_dir).
    """
    xlsx_path = Path(output_xlsx).expanduser().resolve()
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    placeholder = empty_reason or "No tables found in document."

    # --- Excel (open with LibreOffice / Excel, not VS Code) ---
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            if not tables:
                pd.DataFrame({"info": [placeholder]}).to_excel(
                    writer, sheet_name="No_Tables", index=False
                )
            else:
                for idx, df in enumerate(tables, start=1):
                    df.to_excel(writer, sheet_name=f"Table_{idx}"[:31], index=False)
    except Exception as exc:
        raise IOError(f"Failed to write Excel file '{xlsx_path}': {exc}") from exc
    written["xlsx"] = xlsx_path
    logger.info("Saved %d table(s) -> %s", len(tables), xlsx_path)

    stem = xlsx_path.stem  # e.g. output_tables
    out_dir = xlsx_path.parent

    # --- CSV (one file per table; UTF-8 with BOM for Excel on Windows) ---
    if write_csv:
        if not tables:
            csv_path = out_dir / f"{stem}_empty.csv"
            pd.DataFrame({"info": [placeholder]}).to_csv(
                csv_path, index=False, encoding="utf-8-sig"
            )
            written["csv"] = csv_path
        else:
            csv_paths: list[Path] = []
            for idx, df in enumerate(tables, start=1):
                csv_path = out_dir / f"{stem}_table_{idx}.csv"
                df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                csv_paths.append(csv_path)
            written["csv"] = csv_paths[0]
            logger.info("Saved %d CSV file(s) under %s", len(csv_paths), out_dir)

    # --- Markdown preview (easiest to read inside VS Code) ---
    if write_markdown:
        md_path = out_dir / f"{stem}_preview.md"
        if not tables:
            md_body = f"# Tables preview\n\n_{placeholder}_\n"
        else:
            parts = ["# Tables preview\n"]
            for idx, df in enumerate(tables, start=1):
                parts.append(f"\n## Table {idx}\n\n")
                parts.append(_dataframe_to_markdown(df))
                parts.append("\n")
            md_body = "".join(parts)
        md_path.write_text(md_body, encoding="utf-8")
        written["markdown"] = md_path
        logger.info("Saved Markdown preview -> %s", md_path)

    return written


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GFM pipe table (no `tabulate` dependency)."""
    columns = [str(c) for c in df.columns]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    rows: list[str] = []
    for _, row in df.iterrows():
        cells = [str(row[c]).replace("\n", " ").replace("|", "\\|") for c in df.columns]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows]) if columns else "_empty table_"


class TableExporter:
    """Write one or more DataFrames into Excel + VS Code-friendly companions."""

    def save(
        self,
        tables: list[pd.DataFrame],
        output_path: str | Path,
    ) -> Path:
        """
        Save each DataFrame to Excel, plus CSV and Markdown previews.

        Returns:
            Path of the written .xlsx file (companions sit next to it).
        """
        written = export_tables_preview(tables, output_path)
        return written["xlsx"]


def _set_cell_text(cell, text: str, *, bold: bool = False) -> None:
    """Write plain text into a python-docx cell, preserving empty strings."""
    cell.text = ""
    paragraph = cell.paragraphs[0]
    run = paragraph.add_run(text)
    run.bold = bold


def _add_dataframe_as_word_table(doc, df: pd.DataFrame) -> None:
    """Insert one form-faithful Word table from a DataFrame (empty cells kept)."""
    if df is None or len(df.columns) == 0:
        return

    cols = [_clean_cell_text(c) for c in df.columns]
    include_header = not _headers_are_generic(cols)
    n_cols = len(cols)
    n_body = len(df)
    n_rows = n_body + (1 if include_header else 0)
    if n_rows == 0:
        # Header-only scaffold
        n_rows = 1
        include_header = True

    table = doc.add_table(rows=n_rows, cols=n_cols)
    table.style = "Table Grid"

    row_offset = 0
    if include_header:
        for col_idx, name in enumerate(cols):
            _set_cell_text(table.rows[0].cells[col_idx], name, bold=True)
        row_offset = 1

    for r in range(n_body):
        for c in range(n_cols):
            _set_cell_text(
                table.rows[row_offset + r].cells[c],
                _clean_cell_text(df.iloc[r, c]),
            )


def export_ordered_docx(
    blocks: list[AssembledBlock],
    output_path: str | Path,
) -> Path:
    """
    Write a `.docx` with prose and Word tables in document reading order.

    Each table is a real `python-docx` grid (editable cells), not an image.
    """
    from docx import Document

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = Document()
    table_count = 0

    for block in blocks:
        if block.kind == "text":
            text = str(block.content).strip()
            if not text:
                continue
            for paragraph in text.split("\n\n"):
                stripped = paragraph.strip()
                if not stripped:
                    continue
                for line in stripped.splitlines():
                    if line.strip():
                        doc.add_paragraph(line.strip())
            continue

        assert isinstance(block.content, pd.DataFrame)
        df = block.content
        if df is None or (df.empty and len(df.columns) == 0):
            continue
        _add_dataframe_as_word_table(doc, df)
        table_count += 1
        doc.add_paragraph("")

    if table_count == 0 and not any(b.kind == "text" for b in blocks):
        doc.add_paragraph("(Không có nội dung trích xuất được.)")

    try:
        doc.save(str(output_path))
    except Exception as exc:
        raise IOError(f"Failed to write Word document '{output_path}': {exc}") from exc

    logger.info("Saved ordered Word document (%d table(s)) -> %s", table_count, output_path)
    return output_path


def export_document_from_markdown(
    markdown: str,
    extracted_tables: list[pd.DataFrame],
    output_path: str | Path,
    *,
    apply_ocr_cleanup: bool = True,
) -> tuple[Path, list[pd.DataFrame]]:
    """Assemble reading-order DOCX from Marker markdown + extracted tables."""
    blocks = assemble_document_blocks(
        markdown, extracted_tables, apply_ocr_cleanup=apply_ocr_cleanup
    )
    path = export_ordered_docx(blocks, output_path)
    tables = [
        b.content for b in blocks if b.kind == "table" and isinstance(b.content, pd.DataFrame)
    ]
    return path, tables


def export_document(
    text: str,
    tables: list[pd.DataFrame],
    output_path: str | Path,
) -> Path:
    """
    Assemble one readable Word (.docx) file from prose + tables.

    Prefer `export_document_from_markdown` when Marker reading order matters.
    This helper remains for callers that already split prose/tables.
    """
    blocks: list[AssembledBlock] = []
    if text and text.strip():
        blocks.append(AssembledBlock(kind="text", content=text.strip()))
    for df in tables:
        blocks.append(AssembledBlock(kind="table", content=df))
    return export_ordered_docx(blocks, output_path)


class DocumentExporter:
    """Assemble prose + extracted tables into one readable `.docx` file."""

    def save(self, text: str, tables: list[pd.DataFrame], output_path: str | Path) -> Path:
        return export_document(text, tables, output_path)


class TextExporter:
    """Write non-table Markdown text to a UTF-8 plain text file."""

    def save(self, text: str, output_path: str | Path) -> Path:
        """
        Persist extracted prose, preserving paragraph spacing.

        Args:
            text: Non-table text content.
            output_path: Destination .txt path.

        Returns:
            Resolved path of the written file.
        """
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        except Exception as exc:
            raise IOError(f"Failed to write text file '{path}': {exc}") from exc

        logger.info("Saved text (%d chars) -> %s", len(text), path)
        return path
